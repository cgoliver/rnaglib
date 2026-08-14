"""Defines the RNA-Ligand classes as Tanimoto-similarity clusters of ligands, instead of the hand-curated
ChEBI/ClassyFire families in data/family_map.json.

Why: the family scheme only covers 121 of the 345 ligand codes seen in binding pockets, and even restricted to the
3 families with enough splittable sites, the post-redundancy-removal dataset is 88% one class (aminoglycoside:
161/182). Tanimoto clustering lets every ligand with a resolvable SMILES contribute to a class, and clusters are
built with a homogeneity guarantee (bounded max pairwise distance) that is at least as strong as manual curation.

Method:
1. Fingerprint every ligand code appearing in a binding pocket (ECFP4 / Morgan r=2, 2048 bits) from
   data/ligand_smiles.json (241 codes from rna_smiles.txt + 104 pulled from the RCSB Chemical Component
   Dictionary, see fetch_missing_smiles.py).
2. Complete-linkage hierarchical clustering on 1-Tanimoto distance, cut at `cut_dist`. Complete linkage bounds
   the *maximum* pairwise distance within a cluster (unlike the connected-components / single-linkage approach
   used elsewhere for redundancy collapsing), which is the actual homogeneity guarantee wanted here: every pair
   of ligands in a class is chemically similar, not merely chain-connected through intermediates.
3. Clusters are weighted by their raw occurrence count in ligands_dict.json (residue-level appearances) as a
   fast proxy for how many binding-pocket samples a cluster will contribute. This is only a proxy: the ground
   truth is the post-CD-Hit/US-align pocket count, which requires running the full task pipeline (see
   ligand_identity.py) and is expensive enough (pairwise structural alignment) that it belongs in a separate,
   slower loop rather than this script.
4. Clusters below `n_min` (proxy-weighted) are merged into their nearest neighboring cluster (smallest average
   inter-cluster Tanimoto distance) as long as the merge keeps mean intra-cluster similarity above
   `homogeneity_floor`. A cluster that cannot be merged without breaking the floor is dropped (its ligands are
   excluded from the task, mirroring how ligand codes outside `family_map.json` are silently excluded today).

This script does NOT cap or subsample any resulting cluster's size: structural diversity within a class is left
entirely to the existing redundancy-removal / splitting pipeline (CD-Hit, US-align, ClusterSplitter).

Outputs (written to data/):
- ligand_to_cluster.json: {ligand_code: cluster_id}, only for ligands kept in an admissible cluster.
- cluster_report.json: per-cluster members, proxy weight, mean intra-cluster similarity, and the family_map
  labels its members used to carry (for a human sanity check against the old scheme).
"""
import argparse
import json
import os
import collections

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_ligand_counts():
    with open(os.path.join(DATA_DIR, "ligands_dict.json")) as f:
        ligands_dict = json.load(f)
    return collections.Counter(ligands_dict.values())


def load_family_map():
    with open(os.path.join(DATA_DIR, "family_map.json")) as f:
        return json.load(f)


def compute_fingerprints(smiles_map, radius=2, n_bits=2048):
    fps = {}
    for code, smiles in smiles_map.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fps[code] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fps


def tanimoto_distance_matrix(fps):
    codes = list(fps.keys())
    n = len(codes)
    dist = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[codes[i]], [fps[codes[j]] for j in range(n)])
        dist[i] = 1 - np.array(sims)
    np.fill_diagonal(dist, 0)
    # BulkTanimotoSimilarity is symmetric in theory; average out floating-point asymmetry
    dist = (dist + dist.T) / 2
    return codes, dist


def initial_clusters(codes, dist, cut_dist):
    """Complete-linkage clustering: every pair within a cluster has distance <= cut_dist."""
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, t=cut_dist, criterion="distance")
    clusters = collections.defaultdict(list)
    for code, label in zip(codes, labels):
        clusters[int(label)].append(code)
    return list(clusters.values())


def mean_intra_similarity(members, code_to_idx, dist):
    if len(members) < 2:
        return 1.0
    idxs = [code_to_idx[m] for m in members]
    sims = [1 - dist[i, j] for a, i in enumerate(idxs) for j in idxs[a + 1:]]
    return float(np.mean(sims))


def mean_inter_distance(members_a, members_b, code_to_idx, dist):
    idx_a = [code_to_idx[m] for m in members_a]
    idx_b = [code_to_idx[m] for m in members_b]
    return float(np.mean([dist[i, j] for i in idx_a for j in idx_b]))


def merge_small_clusters(clusters, codes, dist, counts, n_min, homogeneity_floor):
    """Iteratively merge the proxy-weighted-smallest cluster into its nearest neighbor, as long as the merge
    keeps mean intra-cluster similarity above homogeneity_floor. Clusters that cannot be merged without
    breaking the floor are moved to `excluded` and taken out of further consideration.
    """
    code_to_idx = {c: i for i, c in enumerate(codes)}
    active = [list(c) for c in clusters]
    excluded = []

    def weight(cluster):
        return sum(counts.get(c, 0) for c in cluster)

    changed = True
    while changed:
        changed = False
        weights = [weight(c) for c in active]
        # smallest-first so we resolve the most under-populated clusters first
        order = sorted(range(len(active)), key=lambda i: weights[i])
        for i in order:
            if weights[i] >= n_min:
                continue
            cluster = active[i]
            others = [j for j in range(len(active)) if j != i]
            if not others:
                excluded.append(active.pop(i))
                changed = True
                break
            best_j, best_d = None, None
            for j in others:
                d = mean_inter_distance(cluster, active[j], code_to_idx, dist)
                if best_d is None or d < best_d:
                    best_j, best_d = j, d
            merged = cluster + active[best_j]
            if mean_intra_similarity(merged, code_to_idx, dist) >= homogeneity_floor:
                new_active = [c for k, c in enumerate(active) if k not in (i, best_j)]
                new_active.append(merged)
                active = new_active
                changed = True
                break
            else:
                # closest neighbor already violates the floor -> no merge can save this cluster
                excluded.append(active.pop(i))
                changed = True
                break

    return active, excluded


def build_report(final_clusters, excluded, codes, dist, counts, family_map):
    code_to_idx = {c: i for i, c in enumerate(codes)}
    report = {"clusters": [], "excluded": []}
    for i, members in enumerate(sorted(final_clusters, key=lambda c: -sum(counts.get(m, 0) for m in c))):
        families = collections.Counter(family_map.get(m, "(uncurated)") for m in members)
        report["clusters"].append({
            "cluster_id": f"cluster_{i:02d}",
            "n_ligands": len(members),
            "proxy_weighted_count": sum(counts.get(m, 0) for m in members),
            "mean_intra_similarity": round(mean_intra_similarity(members, code_to_idx, dist), 3),
            "members": sorted(members, key=lambda m: -counts.get(m, 0)),
            "legacy_family_composition": dict(families),
        })
    for members in excluded:
        report["excluded"].append({
            "n_ligands": len(members),
            "proxy_weighted_count": sum(counts.get(m, 0) for m in members),
            "members": members,
        })
    return report


def main(cut_dist, n_min, homogeneity_floor, out_dir):
    with open(os.path.join(DATA_DIR, "ligand_smiles.json")) as f:
        smiles_map = json.load(f)
    counts = load_ligand_counts()
    family_map = load_family_map()

    fps = compute_fingerprints(smiles_map)
    codes, dist = tanimoto_distance_matrix(fps)

    clusters = initial_clusters(codes, dist, cut_dist)
    final_clusters, excluded = merge_small_clusters(clusters, codes, dist, counts, n_min, homogeneity_floor)

    report = build_report(final_clusters, excluded, codes, dist, counts, family_map)

    ligand_to_cluster = {}
    for entry in report["clusters"]:
        for member in entry["members"]:
            ligand_to_cluster[member] = entry["cluster_id"]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ligand_to_cluster.json"), "w") as f:
        json.dump(ligand_to_cluster, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "cluster_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    n_admissible = len(report["clusters"])
    total_kept = sum(e["proxy_weighted_count"] for e in report["clusters"])
    total_excluded = sum(e["proxy_weighted_count"] for e in report["excluded"])
    print(f"{n_admissible} admissible clusters, {len(report['excluded'])} excluded singleton/unmergeable groups")
    print(f"proxy-weighted count kept: {total_kept}, excluded: {total_excluded}")
    for entry in report["clusters"]:
        top_families = ", ".join(f"{k}={v}" for k, v in
                                  sorted(entry["legacy_family_composition"].items(), key=lambda kv: -kv[1])[:3])
        print(f"  {entry['cluster_id']:>12s}  n_ligands={entry['n_ligands']:3d}  "
              f"proxy_weight={entry['proxy_weighted_count']:6d}  "
              f"mean_sim={entry['mean_intra_similarity']:.2f}  legacy=[{top_families}]  "
              f"top_members={entry['members'][:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut_dist", type=float, default=0.5,
                         help="Complete-linkage cut distance (1-Tanimoto similarity) for the initial clustering")
    parser.add_argument("--n_min", type=int, default=150,
                         help="Minimum proxy-weighted (raw ligand occurrence) count for a cluster to stand alone")
    parser.add_argument("--homogeneity_floor", type=float, default=0.35,
                         help="Minimum mean intra-cluster Tanimoto similarity a merge is allowed to produce")
    parser.add_argument("--out_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()
    main(args.cut_dist, args.n_min, args.homogeneity_floor, args.out_dir)
