import collections

import numpy as np
from scipy.sparse.csgraph import connected_components

from rnaglib.dataset_transforms import DSTransform


class PrepareDataset(DSTransform):
    """
    Dataset transform removing redundancy in a dataset by performing clustering on the dataset, then keeping the pocket
    with the highest resolution within each cluster. This is the RNA_Ligand counterpart of RedundancyRemover, from which
    it differs on a single point: a cluster whose pockets do not all bind the same ligand is dropped entirely.

    The members of such a cluster are superimposable sites binding different compounds in different depositions, so
    structure does not determine the ligand there and no representative can be picked without asserting an arbitrary
    answer: which ligand the benchmark would claim binds that site would come down to which deposition happened to be
    better resolved. Dropping the cluster asserts nothing instead. This restricts the task to the sites where the label
    is well defined, and the resulting score should be read as such, since the sites removed here are the hard ones.

    The mixed-label drop and the collapse-to-one-representative step are independent: the former is a correctness
    guard (a superimposable site can't be given two different answers), the latter is a redundancy/sample-count
    tradeoff. ``collapse_to_representative=False`` keeps that guard while keeping every member of a label-consistent
    cluster instead of discarding all but the best-resolution one - useful when a class doesn't have enough
    post-redundancy-removal samples to learn from and the near-duplicate structures being discarded aren't actually
    ambiguous, just redundant.

    Two thresholds, two passes: ``threshold`` (loose - "similar enough to be redundant") and ``ambiguity_threshold``
    (strict - "similar enough to plausibly be the same site") are deliberately decoupled, because
    ``connected_components`` clustering is transitive: at a single shared threshold, pocket A can chain into pocket
    C's cluster through an intermediate B (A-B and B-C both cross the cutoff) even when A and C themselves are not
    that similar. Under the old single-threshold scheme, if A and C carry different labels, that transitive chaining
    drops the *entire* neighborhood -- including A and B, whose own label was never actually ambiguous -- just
    because a loosely-related third party C happened to bridge into the group. Splitting the check into two passes
    fixes this:

    1. **Ambiguity pass** (``ambiguity_threshold``, strict): cluster at the strict cutoff first. Only a neighborhood
       built from genuinely near-identical structures can veto its members on label disagreement; everything that
       survives this pass has an unambiguous label.
    2. **Redundancy pass** (``threshold``, loose): re-cluster survivors at the loose cutoff to find near-duplicates
       worth collapsing, but grouped *by label* within each loose neighborhood -- since two different labels can
       legitimately co-occur in a loose (merely similar-fold) neighborhood without being ambiguous about each
       other, every label present gets its own representative instead of the whole neighborhood being merged (or
       dropped) into one.

    If ``ambiguity_threshold`` is left as ``None``, it defaults to ``threshold``, reproducing the old single-pass
    behavior (one shared cutoff for both concerns).

    :param str distance_name: the name of the distance metric which has to be used to perform clustering. The distance
    must have been computed on the dataset (see DistanceComputer)
    :param float threshold: the similarity threshold (considering similarity as 1-distance) used for the redundancy
    (collapse-to-representative) pass -- how similar two same-label pockets must be to be considered duplicates
    :param float ambiguity_threshold: the (typically stricter) similarity threshold used for the mixed-label
    correctness guard -- how similar two pockets must be to be considered plausibly-the-same-site. Defaults to
    ``threshold`` (single-pass behavior) when None.
    :param str target_name: the graph-level attribute holding the ligand a pocket is labelled with (default "ligand")
    :param bool collapse_to_representative: if True (default), each label within a label-consistent redundancy group
        is reduced to its highest-resolution member. If False, every member is kept; the ambiguity guard still drops
        genuinely mixed-label (near-identical) groups entirely either way.
    """

    def __init__(
        self,
        distance_name: str = "USalign",
        threshold: float = 0.95,
        ambiguity_threshold: float = None,
        target_name: str = "ligand",
        collapse_to_representative: bool = True,
    ):
        self.distance_name = distance_name
        self.threshold = threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.target_name = target_name
        self.collapse_to_representative = collapse_to_representative

    def __call__(self, dataset):
        """
        Applies the redundancy removal transformation to a dataset.

        :param RNADataset dataset: the initial dataset (without redundancy removal)
        :return: the dataset with redundancy removed according to specified criteria
        :rtype: RNADataset
        """
        if dataset.distances is None or not self.distance_name in dataset.distances:
            raise ValueError(f"The distance matrix using distances {self.distance_name} has not been computed")

        distance_matrix = dataset.distances[self.distance_name]

        def neighborhoods(similarity_threshold):
            adjacency_matrix = (distance_matrix <= 1 - similarity_threshold).astype(int)
            n_components, labels = connected_components(adjacency_matrix)
            return [np.where(labels == i)[0].tolist() for i in range(n_components)]

        def bound_ligand(rna_idx):
            try:
                return dataset[rna_idx]["rna"].graph[self.target_name]
            except Exception:
                return None

        ambiguity_threshold = self.threshold if self.ambiguity_threshold is None else self.ambiguity_threshold

        # Pass 1 (strict): a neighborhood of near-identical pockets that disagree on label is genuinely
        # ambiguous -- superimposable sites can't be given two different answers -- so it's dropped whole.
        # Everything that survives has an unambiguous, known label.
        survivor_label = {}
        for neighborhood in neighborhoods(ambiguity_threshold):
            bound_ligands = {lig for lig in (bound_ligand(i) for i in neighborhood) if lig is not None}
            if len(bound_ligands) > 1:
                continue
            if not bound_ligands:
                continue
            label = next(iter(bound_ligands))
            for rna_idx in neighborhood:
                if bound_ligand(rna_idx) is not None:
                    survivor_label[rna_idx] = label

        # Pass 2 (loose): re-cluster survivors to find near-duplicates worth collapsing, grouped by label
        # within each loose neighborhood -- two different labels can share a loose (merely similar-fold)
        # neighborhood without being ambiguous about each other, since pass 1 already resolved ambiguity.
        final_list_ids = []
        for neighborhood in neighborhoods(self.threshold):
            by_label = collections.defaultdict(list)
            for rna_idx in neighborhood:
                if rna_idx in survivor_label:
                    by_label[survivor_label[rna_idx]].append(rna_idx)

            for members in by_label.values():
                if not self.collapse_to_representative:
                    final_list_ids.extend(members)
                    continue

                highest_resolution = 100
                highest_resolution_idx = members[0]
                for rna_idx in members:
                    rna_dict = dataset[rna_idx]
                    try:
                        resolution = rna_dict["rna"].graph["resolution_high"]
                        if resolution < highest_resolution:
                            highest_resolution = resolution
                            highest_resolution_idx = rna_idx
                    except Exception:
                        continue
                final_list_ids.append(highest_resolution_idx)

        dataset = dataset.subset(list_of_ids=final_list_ids)
        return dataset
