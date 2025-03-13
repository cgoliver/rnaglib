import logging
from collections import Counter
from dataclasses import dataclass

from pulp import COIN_CMD,PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value

logger = logging.getLogger(__name__)


@dataclass
class SplitMetrics:
    size_deviation: dict[str, float]
    label_deviation: dict[str, float]


class ValidationError(Exception):
    pass


def validate_inputs(
    clusters: list[list[str]],
    cluster_counters: list[Counter],
    split_ratios: tuple[float, float, float],
    ratio_tolerance: float,
) -> None:
    """Validate all input parameters."""
    if not clusters or not cluster_counters:
        raise ValidationError("Clusters and cluster_counters cannot be empty")

    if len(clusters) != len(cluster_counters):
        raise ValidationError("Clusters and cluster_counters must have same length")

    if not (0.99 <= sum(split_ratios) <= 1.01):  # Allow small floating point errors
        raise ValidationError("Split ratios must sum to 1.0")

    if not (0 < ratio_tolerance < 1):
        raise ValidationError("Ratio tolerance must be between 0 and 1")


def calculate_balance_metrics(
    splits: list[list[list[str]]],
    clusters: list[list[str]],
    target_sizes: list[float],
    cluster_sizes: list[int],
    cluster_counters: list[Counter],
    split_ratios: tuple[float, float, float],
) -> SplitMetrics:
    """Calculate metrics showing how well the splits achieve balance targets."""
    size_errors = []
    label_errors = []
    per_split_label_errors = [[] for _ in range(3)]

    # Calculate size deviations
    for split_idx, split in enumerate(splits):
        actual_size = sum(cluster_sizes[clusters.index(cluster)] for cluster in split)
        if target_sizes[split_idx] > 0:
            relative_error = abs(actual_size - target_sizes[split_idx]) / target_sizes[split_idx]
            size_errors.append(relative_error)

    # Calculate label distribution deviations
    all_labels = list(cluster_counters[0].keys())
    for label in all_labels:
        label_total = sum(counter[label] for counter in cluster_counters)
        if label_total > 0:
            for split_idx, split in enumerate(splits):
                split_count = sum(cluster_counters[clusters.index(cluster)][label] for cluster in split)
                target_count = label_total * split_ratios[split_idx]
                if target_count > 0:
                    relative_error = abs(split_count - target_count) / target_count
                    label_errors.append(relative_error)
                    per_split_label_errors[split_idx].append(relative_error)

    # Calculate per-split averages
    per_split_avg = [sum(errors) / len(errors) if errors else 0.0 for errors in per_split_label_errors]

    return SplitMetrics(
        size_deviation={
            "mean": sum(size_errors) / len(size_errors) if size_errors else 0.0,
            "max": max(size_errors) if size_errors else 0.0,
            "per_split": size_errors,
        },
        label_deviation={
            "mean": sum(label_errors) / len(label_errors) if label_errors else 0.0,
            "max": max(label_errors) if label_errors else 0.0,
            "total": label_errors,
            "per_split": per_split_avg,
        },
    )


def assign_clusters(
    clusters: list[list[str]],
    cluster_counters: list[Counter],
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ratio_tolerance: float = 0.5,
    size_weight: float = 1.0,
    label_weight: float = 1.0,
    verbose=False
) -> tuple[
    list[list[str]] | None,
    list[list[str]] | None,
    list[list[str]] | None,
    SplitMetrics | None,
]:
    """Split clusters into train/val/test sets optimizing for balance."""
    try:
        validate_inputs(clusters, cluster_counters, split_ratios, ratio_tolerance)
    except ValidationError as e:
        logger.error("Input validation failed: %s", e)
        return None, None, None, None

    cluster_sizes = [len(c) for c in clusters]
    total_elements = sum(cluster_sizes)
    if total_elements == 0:
        logger.error("Total cluster size is 0")
        return None, None, None, None

    target_sizes = [total_elements * ratio for ratio in split_ratios]

    prob = LpProblem("ClusterSplit", LpMinimize)
    n_clusters = len(clusters)
    splits = range(3)



    # Decision variables
    x = LpVariable.dicts(
        "split",
        ((i, j) for i in range(n_clusters) for j in splits),
        cat="Binary",
    )

    # Constraints
    for i in range(n_clusters):
        prob += lpSum(x[i, j] for j in splits) == 1

    # Size bounds with tolerance
    for j, target in enumerate(target_sizes):
        min_size = max(1, target * (1 - ratio_tolerance))
        max_size = target * (1 + ratio_tolerance)
        split_size = lpSum(cluster_sizes[i] * x[i, j] for i in range(n_clusters))
        prob += split_size >= min_size
        prob += split_size <= max_size

    # Objective components
    size_diff_vars = []
    label_diff_vars = []

    # Size objective
    for j, target in enumerate(target_sizes):
        size_diff = LpVariable(f"size_diff_{j}", lowBound=0)
        split_size = lpSum(cluster_sizes[i] * x[i, j] for i in range(n_clusters))

        prob += split_size - target <= size_diff
        prob += target - split_size <= size_diff
        size_diff_vars.append(size_diff)

    # Label distribution objective
    all_labels = list(cluster_counters[0].keys())
    for label in all_labels:
        label_total = sum(counter[label] for counter in cluster_counters)
        if label_total > 0:
            for j in splits:
                diff = LpVariable(f"ratio_diff_{label}_{j}", lowBound=0)
                split_count = lpSum(cluster_counters[i][label] * x[i, j] for i in range(n_clusters))
                target_count = label_total * split_ratios[j]

                prob += split_count - target_count <= diff * label_total
                prob += target_count - split_count <= diff * label_total
                label_diff_vars.append(diff)

    # Combined weighted objective
    if total_elements > 0:
        normalized_size_diff = size_weight * lpSum(size_diff_vars) / total_elements
        normalized_label_diff = label_weight * lpSum(label_diff_vars)
        prob += normalized_size_diff + normalized_label_diff

    # Solve

    # This allows the solver to accept non perfect solutions
    solver = PULP_CBC_CMD(msg=verbose, timeLimit=300, gapRel=0.05)
    # solver = PULP_CBC_CMD(msg=verbose, timeLimit=300, gapRel=0.05, path="/opt/homebrew/bin/cbc") 
    # solver = COIN_CMD(msg=verbose, timeLimit=300, gapRel=0.05, path="/opt/homebrew/bin/cbc") 

    status = prob.solve(solver)
    if status != 1:
        logger.error("Failed to find optimal solution. Status: %s", LpStatus[status])
        return None, None, None, None

    # Extract results
    result = [[] for _ in range(3)]
    for i in range(n_clusters):
        for j in splits:
            if value(x[i, j]) == 1:
                result[j].append(clusters[i])
                break

    # Calculate metrics
    metrics = calculate_balance_metrics(
        result,
        clusters,
        target_sizes,
        cluster_sizes,
        cluster_counters,
        split_ratios,
    )

    return result[0], result[1], result[2], metrics


def print_split_metrics(metrics: SplitMetrics) -> None:
    """Print formatted metrics for the splits."""
    print("\nBalance Metrics:")
    print(
        f"Size deviation - Mean: {metrics.size_deviation['mean'] * 100:.1f}%, "
        f"Max: {metrics.size_deviation['max'] * 100:.1f}%",
    )
    print(
        f"Label deviation - Mean: {metrics.label_deviation['mean'] * 100:.1f}%, "
        f"Max: {metrics.label_deviation['max'] * 100:.1f}%",
    )

    print("\nPer-split deviations:")
    split_names = ["Train", "Validation", "Test"]
    for i in range(3):
        size_dev = metrics.size_deviation["per_split"][i]
        label_dev = metrics.label_deviation["per_split"][i]
        print(f"{split_names[i]}:")
        print(f"  Size deviation: {size_dev * 100:.1f}%")
        print(f"  Label deviation: {label_dev * 100:.1f}%")


if __name__ == "__main__":
    # Example with error handling
    try:
        # Sample data
        clusters = [["example1"], ["example2", "example3"], ["example4"], ["example5"]]
        cluster_counters = [
            Counter({(0.0,): 32, (1.0,): 2}),
            Counter({(0.0,): 28, (1.0,): 3}),
            Counter({(0.0,): 30, (1.0,): 1}),
            Counter({(0.0,): 25, (1.0,): 4}),
        ]

        train, val, test, metrics = assign_clusters(
            clusters,
            cluster_counters,
            size_weight=1.0,
            label_weight=1.0,
        )

        if all(x is not None for x in (train, val, test, metrics)):
            print("\nTrain clusters:", train)
            print("Val clusters:", val)
            print("Test clusters:", test)
            print_split_metrics(metrics)
        else:
            print("Failed to assign clusters")

    except Exception as e:
        logger.error(f"Error in cluster assignment: {e}")
