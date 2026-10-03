"""Ligand-group identification from RNA binding pockets, split at the RNA-molecule level.

Pocket-level multi-class task: given the structure of an RNA binding pocket, predict which curated
ligand group (recognition mode) it binds. Differences from LigandIdentification (ligand_identity.py):

* **Pockets and labels come from the current structure graphs.** Every ligand copy annotated in the
  graphs (``binding_small-molecule-4.0A``; rnaglib's annotation already requires the ligand to be
  RNA-bound) with at least ``min_rna_contacts`` RNA residues within 4 A defines one pocket: the RNA
  residues within ``pocket_cutoff`` A of that copy. No pre-computed pocket dictionary is used, so
  pockets cannot carry a stale label.
* **Curated labels.** Ligand codes are mapped to biological ligand groups (data/ligand_groups.json);
  unmapped and excluded codes (polyamines, buffers, tRNA-attached amino acids, ...) are dropped, and
  so are the groups in ``excluded_groups`` (default DEFAULT_EXCLUDED_GROUPS: nucleotides, nucleotide
  cofactors and cyclic dinucleotides).
  ``group_merges`` merges groups into coarser classes (default DEFAULT_GROUP_MERGES: nucleotide-like
  cofactors and nucleobase-like heteroaromatics; pass {} to keep the fine groups).
* **Every binding site is kept once (duplicate sites removed).** Two pockets are at the same site
  if both bind the same (fine) ligand group on the same RNA molecule and their sets of contacted
  positions overlap with Jaccard > ``site_min_jaccard`` (default 0: any shared contact residue). The
  relation is made transitive: a site is a connected component of such overlaps. Positions are chain indices for
  ordinary RNAs (same molecule = identical sequence) and E. coli rRNA numbers for large rRNAs (chains
  with >= ``rrna_min_length`` modelled residues, aligned to data/ecoli_rrna_reference.json), so the
  same ribosomal site is recognised across entries and organisms. An entry with any large rRNA chain is
  a ribosome: all its pockets get the rRNA key, including those whose main chain is a small rRNA
  (5S, 5.8S), keyed by the large rRNA with most residues in the pocket shell (LSU if none) and sited
  by their contacts on it plus their contacts on the main chain. One pocket per site is kept, the one
  with the best resolution. This covers NCS copies, several chains or ribosomes per asymmetric unit, and repeated
  depositions of the same complex. No site is removed for any other reason (no primary-site rule),
  except the per-molecule cap below.
* **At most ``max_sites_per_molecule`` sites per molecule** (default 3), so that an RNA with many
  modelled ligand copies cannot dominate a class. Applied after the class filter (which is then
  re-applied): a molecule key (one sequence, or one rRNA subunit) above the cap keeps the sites that
  are the least structurally redundant with the rest of the dataset. Pockets of all over-cap
  molecules are picked greedily, each time the one whose highest pocket US-align similarity to the
  pockets kept so far is the lowest. No rule on which site is biologically relevant is used.
* **Split at the RNA-molecule level.** The RNA molecules carrying the pockets (main chain) are
  clustered by whole-molecule structural similarity with US-align: two molecules are linked if their
  TM-score normalised by the longer chain is >= ``molecule_tm`` (similar molecules), or if the
  shorter one (>= ``fragment_min_len`` nt) matches the longer with TM normalised by the shorter
  >= ``fragment_tm`` (fragment / model of the other, e.g. an A-site oligonucleotide and a 16S rRNA).
  All small-subunit rRNAs share one molecule key and all large-subunit rRNAs another (one reference
  structure each, which fragments are aligned against). Two non-rRNA molecules are also linked if their
  sequence identity over the shorter chain is >= ``molecule_seq_identity`` (default 0.9; shorter chain
  >= ``seq_link_min_len`` nt and length ratio >= ``seq_link_min_ratio``): TM-score stays low on 15-50 nt
  chains even between near-identical structures. Linked molecules only share a split; every conformer
  is kept. The connected components are the split
  groups: no RNA molecule cluster is shared between train, validation and test. Whole clusters are
  assigned to the splits so that every class is split close to 70/15/15, in pockets and in molecule
  clusters, with at least 3 clusters per class in validation and in test, none holding more than half of
  a class's validation or test pockets (MoleculeClusterStratifiedSplitter).
* **Class filter.** Ligand groups (after merging) with fewer than ``min_class_pockets`` pockets, or
  present in fewer than ``min_class_clusters`` molecule clusters, are dropped.
* **Training weights for sequence variants.** Pockets are kept for every distinct sequence, but
  sequence variants of one RNA (point mutants, construct and loop swaps: identity >=
  ``variant_identity`` over the shorter chain, same molecule cluster) binding the same fine ligand
  group at the same site (any shared contact residue after mapping to a common numbering, transitive)
  form one variant site. Each pocket gets ``sample_weight`` = (pockets of its class / variant sites of
  its class) / (pockets of its variant site): within a class every site weighs the same, whatever its
  number of variants, and each class keeps its total weight (= its pocket count), so the class
  balance is unchanged. Weights are stored on the graphs and in
  ``{root}/sample_weights.json``; evaluation is unweighted.

Task type: multi-class classification. Task level: substructure (pocket) level.
"""

from __future__ import annotations

import collections
import hashlib
import json
import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from Bio import Align
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from rnaglib.dataset import RNADataset
from rnaglib.dataset_transforms.splitters import Splitter
from rnaglib.encoders import IntMappingEncoder
from rnaglib.tasks import RNAClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.utils.graph_io import load_json

DATA_DIR = Path(__file__).parent / "data"


_WORKER_TASK = None  # set by LigandGroupIdentification.process() before forking its worker pool


def _worker_entry_pockets(fname):
    return _WORKER_TASK._entry_pockets(fname)


def _as_list(value):
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def _ligand_key(entry):
    return entry.get("name"), json.dumps(entry.get("id"))


# ligand groups left out of the task: nucleotides, nucleotide cofactors and cyclic dinucleotides. Their
# recognition is not the target here, and in-chain nucleotides can still be mislabelled as free ligands
DEFAULT_EXCLUDED_GROUPS = frozenset({"nucleotide", "nad_nmn", "cyclic_dinucleotide"})

DEFAULT_GROUP_MERGES = {
    "nucleotide": "nucleotide_dinucleotide",
    "nad_nmn": "nucleotide_dinucleotide",
    "cyclic_dinucleotide": "nucleotide_dinucleotide",
    "guanine_like_nucleobase": "heteroaromatic_nucleobase_like",
    "folate_pterin": "heteroaromatic_nucleobase_like",
    "flavin": "heteroaromatic_nucleobase_like",
}


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 1.0


class MoleculeClusterStratifiedSplitter(Splitter):
    """Train/val/test split by whole molecule clusters, balanced per class.

    Each graph carries ``molecule_cluster`` and the class in ``label_key``. Every class must have at
    least ``min_per_split`` pockets and at least ``min_clusters_per_split`` molecule clusters in validation
    and in test, and no single cluster may hold more than ``max_cluster_share`` of a class's validation or
    test pockets (None: no limit); among such assignments of clusters to train/val/test, keep the one whose
    per-class fractions of clusters and of pockets are closest (sum of squared errors) to the target
    fractions. Both cluster constraints keep one large cluster from deciding a class's validation or test
    score, which would make balanced accuracy rest on very few independent predictions.
    Search (seeded): ``n_starts`` random assignments, each improved by moving one cluster at a time to
    another split while the score, penalised by the constraint shortfall, decreases.
    """

    def __init__(self, label_key="ligand_group", n_starts=300, min_per_split=5, min_clusters_per_split=3,
                 max_cluster_share=0.5, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.label_key, self.n_starts, self.min_per_split, self.seed = label_key, n_starts, min_per_split, seed
        self.min_clusters_per_split = min_clusters_per_split
        self.max_cluster_share = max_cluster_share

    def forward(self, dataset):
        graphs = [dataset[i]["rna"].graph for i in range(len(dataset))]
        clusters = sorted({g["molecule_cluster"] for g in graphs})
        classes = sorted({g[self.label_key] for g in graphs})
        c_index, k_index = {c: i for i, c in enumerate(clusters)}, {k: i for i, k in enumerate(classes)}
        counts = np.zeros((len(clusters), len(classes)))  # pockets per (cluster, class)
        for g in graphs:
            counts[c_index[g["molecule_cluster"]], k_index[g[self.label_key]]] += 1
        present = (counts > 0).astype(float)  # whether a cluster holds pockets of a class
        target = np.array([self.split_train, self.split_valid, self.split_test])
        rng = np.random.default_rng(self.seed)

        def evaluate(assign):
            """(constraint shortfall, score) of an assignment"""
            per_split = np.stack([counts[assign == s].sum(0) for s in range(3)])  # pockets per (split, class)
            per_split_clusters = np.stack([present[assign == s].sum(0) for s in range(3)])  # clusters per (split, class)
            shortfall = (np.clip(self.min_per_split - per_split[1:], 0, None).sum()
                         + np.clip(self.min_clusters_per_split - per_split_clusters[1:], 0, None).sum())
            if self.max_cluster_share is not None:
                # pockets by which the largest cluster of a class exceeds its allowed share of val / test
                for s in (1, 2):
                    largest = np.where(assign[:, None] == s, counts, 0).max(0)
                    shortfall += np.clip(largest - self.max_cluster_share * per_split[s], 0, None).sum()
            score = (((per_split / counts.sum(0) - target[:, None]) ** 2).sum()
                     + ((per_split_clusters / present.sum(0) - target[:, None]) ** 2).sum())
            return shortfall, score

        best, best_score = None, np.inf
        for _ in range(self.n_starts):
            assign = rng.choice(3, size=len(clusters), p=target)
            shortfall, score = evaluate(assign)
            current = 100 * shortfall + score  # any shortfall outweighs any fraction error (score <= 4 per class)
            improved = True
            while improved:
                improved = False
                for c in rng.permutation(len(clusters)):
                    for s in range(3):
                        if s == assign[c]:
                            continue
                        old, assign[c] = assign[c], s
                        shortfall, score = evaluate(assign)
                        if 100 * shortfall + score < current - 1e-12:
                            current, improved = 100 * shortfall + score, True
                        else:
                            assign[c] = old
            shortfall, score = evaluate(assign)
            if shortfall == 0 and score < best_score:
                best, best_score = assign.copy(), score
        if best is None:
            raise ValueError(f"no cluster assignment gives >= {self.min_per_split} pockets and >= "
                             f"{self.min_clusters_per_split} molecule clusters per class in val and test")
        split_of = {c: best[i] for i, c in enumerate(clusters)}
        out = [[i for i, g in enumerate(graphs) if split_of[g["molecule_cluster"]] == s] for s in range(3)]
        return out[0], out[1], out[2]


class LigandGroupIdentification(RNAClassificationTask):
    """See module docstring.

    :param graph_path: folder of all-atom rnaglib graphs (one JSON per PDB entry)
    :param min_rna_contacts: fewest RNA residues within 4 A of a ligand copy for it to define a pocket
    :param pocket_cutoff: pocket = RNA residues within this distance of the copy ("4.0", "6.0" or "8.0")
    :param rrna_min_length: chains with at least this many modelled residues are treated as large
        (SSU/LSU) rRNA and numbered in E. coli coordinates
    :param site_min_jaccard: contact-position overlap (Jaccard) above which two pockets of one ligand group on one
        molecule are the same binding site
    :param molecule_tm: TM-score (normalised by the longer chain) linking two RNA molecules
    :param fragment_tm: TM-score (normalised by the shorter chain) linking a fragment to a molecule
    :param fragment_min_len: shortest chain allowed to be linked as a fragment
    :param molecule_seq_identity: sequence identity (over the shorter chain) linking two non-rRNA molecules
        (None: no sequence links)
    :param seq_link_min_len: shortest chain allowed in a sequence link
    :param seq_link_min_ratio: smallest length ratio (shorter / longer) allowed in a sequence link
    :param excluded_groups: (fine) ligand groups whose pockets are dropped (None: DEFAULT_EXCLUDED_GROUPS,
        {}: none)
    :param group_merges: {ligand group: merged class} applied before the class filter
        (None: DEFAULT_GROUP_MERGES, {}: no merge)
    :param min_class_pockets: drop classes with fewer pockets than this
    :param min_class_clusters: drop classes present in fewer molecule clusters than this
    :param variant_identity: sequence identity above which two molecules of one cluster are variants
    :param max_sites_per_molecule: keep at most this many binding sites per molecule key, the least
        structurally redundant with the rest of the dataset (None: no cap)
    :param n_jobs: worker processes building the pockets (one entry graph per task)
    """

    input_var = "nt_code"
    target_var = "ligand_group"
    name = "rna_ligand_group"
    default_metric = "balanced_accuracy"
    version = "3.0.0"

    def __init__(
        self,
        graph_path=None,
        size_thresholds=(5, 500),
        min_rna_contacts=6,
        pocket_cutoff="8.0",
        rrna_min_length=700,
        site_min_jaccard=0.0,
        molecule_tm=0.5,
        fragment_tm=0.7,
        fragment_min_len=20,
        molecule_seq_identity=0.9,
        seq_link_min_len=18,
        seq_link_min_ratio=0.5,
        excluded_groups=None,
        group_merges=None,
        min_class_pockets=30,
        min_class_clusters=2,
        variant_identity=0.9,
        max_sites_per_molecule=3,
        n_jobs=1,
        **kwargs,
    ):
        self.graph_path = graph_path
        self.min_rna_contacts = min_rna_contacts
        self.pocket_cutoff = pocket_cutoff
        self.rrna_min_length = rrna_min_length
        self.site_min_jaccard = site_min_jaccard
        self.molecule_tm = molecule_tm
        self.fragment_tm = fragment_tm
        self.fragment_min_len = fragment_min_len
        self.molecule_seq_identity = molecule_seq_identity
        self.seq_link_min_len = seq_link_min_len
        self.seq_link_min_ratio = seq_link_min_ratio
        self.excluded_groups = DEFAULT_EXCLUDED_GROUPS if excluded_groups is None else frozenset(excluded_groups)
        self.group_merges = DEFAULT_GROUP_MERGES if group_merges is None else group_merges
        self.min_class_pockets = min_class_pockets
        self.min_class_clusters = min_class_clusters
        self.variant_identity = variant_identity
        self.max_sites_per_molecule = max_sites_per_molecule
        self.n_jobs = n_jobs

        groups = json.load(open(DATA_DIR / "ligand_groups.json"))["groups"]
        self.code_to_group = {code: group for group, codes in groups.items() for code in codes}
        ref = json.load(open(DATA_DIR / "ecoli_rrna_reference.json"))
        self.rrna_ref = {"SSU": ref["16S"], "LSU": ref["23S"]}

        self._aligner = Align.PairwiseAligner()
        self._aligner.mode = "global"
        self._aligner.match_score, self._aligner.mismatch_score = 2, -1
        self._aligner.open_gap_score, self._aligner.extend_gap_score = -3, -1
        self._aligner.end_gap_score = 0  # fragments align anywhere on the reference
        self._rrna_cache = {}

        super().__init__(additional_metadata={"multi_label": False}, size_thresholds=size_thresholds, **kwargs)

    # ------------------------------------------------------------------ helpers
    def _chains(self, graph):
        """{chain: (ordered node ids, sequence)} for one entry graph."""
        by_chain = collections.defaultdict(list)
        for node, data in graph.nodes(data=True):
            seq_id = data.get("label_seq_id")
            by_chain[data.get("chain_id")].append((seq_id if isinstance(seq_id, int) else 10**9, node, data.get("nt_code") or "N"))
        out = {}
        for chain, items in by_chain.items():
            items.sort()
            out[chain] = ([n for _, n, _ in items], "".join(c.upper() if c else "N" for _, _, c in items))
        return out

    def _map_rrna(self, seq):
        """(subunit, {chain index -> E. coli number}) for a large rRNA chain sequence (cached)."""
        if seq in self._rrna_cache:
            return self._rrna_cache[seq]
        best = None
        for subunit, ref in self.rrna_ref.items():
            aln = self._aligner.align(seq, ref)[0]
            score = aln.score / min(len(seq), len(ref))
            if best is None or score > best[0]:
                best = (score, subunit, aln)
        _, subunit, aln = best
        mapping = {}
        for (q0, q1), (r0, r1) in zip(*aln.aligned):
            for k in range(q1 - q0):
                mapping[int(q0 + k)] = int(r0 + k + 1)
        self._rrna_cache[seq] = (subunit, mapping)
        return subunit, mapping

    def _write_chain_pdb(self, graph, nodes, path):
        lines, atom_id = [], 1
        for res_id, node in enumerate(nodes, start=1):
            data = graph.nodes[node]
            resname = (data.get("nt_full") or data.get("nt_code") or "N")[:3].rjust(3)
            for atom, xyz in (data.get("heavy_atoms") or {}).items():
                if xyz is None:
                    continue
                name = atom if len(atom) == 4 else f" {atom:<3}"
                lines.append(f"ATOM  {atom_id:5d} {name} {resname} A{res_id:4d}    "
                             f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {atom[0]:>2}")
                atom_id += 1
        Path(path).write_text("\n".join(lines) + "\nEND\n")

    def _variant_sites(self, graphs):
        """Variant-site id of each graph (see module docstring, "Training weights")."""
        mols = collections.defaultdict(set)
        for g in graphs:
            if not g.graph["molecule"].startswith("rRNA_"):
                mols[g.graph["molecule_cluster"]].add(g.graph["molecule"])
        ref_of, pos_map = {}, {}
        for members in mols.values():
            refs = []
            for m in sorted(members, key=lambda m: (-len(self.molecule_seqs[m]), m)):
                seq = self.molecule_seqs[m]
                for r in refs:
                    aln = self._aligner.align(seq, self.molecule_seqs[r])[0]
                    pairs = {int(q): int(t) for (q0, q1), (t0, t1) in zip(*aln.aligned)
                             for q, t in zip(range(q0, q1), range(t0, t1))}
                    identity = sum(seq[q] == self.molecule_seqs[r][t] for q, t in pairs.items())
                    if identity / min(len(seq), len(self.molecule_seqs[r])) >= self.variant_identity:
                        ref_of[m], pos_map[m] = r, pairs
                        break
                else:
                    refs.append(m)
                    ref_of[m], pos_map[m] = m, None
        by_ref = collections.defaultdict(list)
        for k, g in enumerate(graphs):
            m, site = g.graph["molecule"], set(g.graph["site_positions"])
            if not m.startswith("rRNA_") and pos_map[m] is not None:
                site = {pos_map[m][p] for p in site if p in pos_map[m]}
            by_ref[(g.graph["ligand_group_fine"], ref_of.get(m, m))].append((k, site))
        site_of = [None] * len(graphs)
        for key, items in by_ref.items():
            n = len(items)
            overlap = np.eye(n, dtype=int)
            for a in range(n):
                for b in range(a):
                    if items[a][1] & items[b][1]:
                        overlap[a, b] = overlap[b, a] = 1
            _, comp = connected_components(overlap, directed=False)
            for (k, _), c in zip(items, comp):
                site_of[k] = f"{key[0]}|{key[1]}|{c}"
        return site_of

    def _pocket_similarities(self, graphs, candidates, references):
        """{(candidate, reference): US-align similarity} between pocket graphs (indices into ``graphs``).
        Each pocket is written as one chain, residues ordered by (chain, label_seq_id); the similarity is
        the larger of the two TM-scores, as in rnaglib's US_align_wrapper. Failed alignments count as 0."""
        from concurrent.futures import ThreadPoolExecutor

        def order(g):
            key = lambda n: (str(g.nodes[n].get("chain_id")), g.nodes[n].get("label_seq_id") if isinstance(g.nodes[n].get("label_seq_id"), int) else 10**9, str(n))
            return sorted(g.nodes(), key=key)

        sims = {}
        with tempfile.TemporaryDirectory() as tmp:
            for i in set(candidates) | set(references):
                self._write_chain_pdb(graphs[i], order(graphs[i]), Path(tmp) / f"{i}.pdb")
            listing = Path(tmp) / "references.txt"
            listing.write_text("\n".join(f"{j}.pdb" for j in references) + "\n")

            def align(i):
                out = subprocess.run(["USalign", "-dir1", f"{tmp}/", str(listing), str(Path(tmp) / f"{i}.pdb"),
                                      "-mol", "RNA", "-outfmt", "2"], capture_output=True, text=True).stdout
                result = {}
                for line in out.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    result[(i, int(Path(fields[0].split(":")[0]).stem))] = max(float(fields[2]), float(fields[3]))
                return result

            with ThreadPoolExecutor(max_workers=max(1, self.n_jobs)) as pool:
                for result in tqdm(pool.map(align, candidates), total=len(candidates), desc="pocket US-align"):
                    sims.update(result)
        return sims

    def _cap_sites_per_molecule(self, graphs, kept):
        """Keeps at most ``max_sites_per_molecule`` pockets per molecule key. Molecules under the cap keep all
        their pockets. For the others, pockets are picked greedily across all such molecules: at each step the
        candidate (of a molecule with slots left) whose highest US-align similarity to every pocket kept so far
        is the lowest, i.e. the least redundant with the rest of the dataset."""
        cap = self.max_sites_per_molecule
        by_molecule = collections.defaultdict(list)
        for i in kept:
            by_molecule[graphs[i].graph["molecule"]].append(i)
        over = {m: members for m, members in by_molecule.items() if len(members) > cap}
        if not over:
            return kept
        selected = [i for m, members in by_molecule.items() if m not in over for i in members]
        candidates = sorted(i for members in over.values() for i in members)
        sims = self._pocket_similarities(graphs, candidates, sorted(kept))
        redundancy = {i: max((sims.get((i, j), 0.0) for j in selected), default=0.0) for i in candidates}
        slots = {m: cap for m in over}
        remaining = set(candidates)
        while remaining:
            best = min(remaining, key=lambda i: (redundancy[i], graphs[i].name))
            remaining.discard(best)
            molecule = graphs[best].graph["molecule"]
            if slots[molecule] == 0:
                continue
            slots[molecule] -= 1
            selected.append(best)
            for i in remaining:
                redundancy[i] = max(redundancy[i], sims.get((i, best), 0.0))
        return sorted(selected)

    @staticmethod
    def _resolution(graph):
        try:
            return float(graph.graph.get("resolution_high"))
        except (TypeError, ValueError):
            return float("inf")

    # ------------------------------------------------------------------ build
    def _entry_pockets(self, fname):
        """Pockets of one entry graph file: (pocket names or graphs, drop Counter,
        {molecule key: (sequence, main chain)} of the molecules its pockets sit on)."""
        drops, molecules, pockets = collections.Counter(), {}, []
        path = Path(self.graph_path) / fname
        if '"binding_small-molecule-4.0A": {' not in path.read_text():
            return pockets, drops, molecules, fname
        rna = load_json(path)
        pdbid = rna.graph.get("pdbid", rna.graph.get("name"))
        copies = collections.defaultdict(lambda: {"c4": [], "shell": []})
        for node, data in rna.nodes(data=True):
            for entry in _as_list(data.get("binding_small-molecule-4.0A")):
                if isinstance(entry, dict):
                    copies[_ligand_key(entry)]["c4"].append(node)
            for entry in _as_list(data.get(f"binding_small-molecule-{self.pocket_cutoff}A")):
                if isinstance(entry, dict):
                    copies[_ligand_key(entry)]["shell"].append(node)
        chains = None
        seen_sites = set()
        k = 0
        for (code, _), members in sorted(copies.items()):
            group = self.code_to_group.get(code)
            if group is None:
                drops["unmapped or excluded ligand"] += 1
                continue
            if group in self.excluded_groups:
                drops["excluded ligand group"] += 1
                continue
            if len(members["c4"]) < self.min_rna_contacts:
                drops["too few RNA contacts"] += 1
                continue
            if chains is None:
                chains = self._chains(rna)
                # an entry is a ribosome if any of its chains is a large rRNA; all its pockets then get an rRNA key
                large_chains = {c: self._map_rrna(s) for c, (_, s) in chains.items() if len(s) >= self.rrna_min_length}
            main_chain = collections.Counter(rna.nodes[n]["chain_id"] for n in members["c4"]).most_common(1)[0][0]
            order, seq = chains[main_chain]
            index = {n: i for i, n in enumerate(order)}
            positions = sorted(index[n] for n in members["c4"] if n in index)
            mol_seq, mol_chain = seq, main_chain
            if len(seq) >= self.rrna_min_length:
                subunit, mapping = large_chains[main_chain]
                molecule = f"rRNA_{subunit}"
                site = sorted({mapping[p] for p in positions if p in mapping})
            elif large_chains:
                # pocket of a ribosome entry whose main chain is a small rRNA (5S, 5.8S), mRNA or tRNA: keyed by the
                # large rRNA with most residues in the pocket shell (LSU, which carries 5S and 5.8S, if none), so that
                # it falls under the same per-molecule cap. Site: E. coli numbers of its contacts on that large chain,
                # plus "<chain sequence hash>:<index>" for its contacts on the main chain
                shell_chains = collections.Counter(rna.nodes[n]["chain_id"] for n in members["shell"] if rna.nodes[n]["chain_id"] in large_chains)
                if shell_chains:
                    large = shell_chains.most_common(1)[0][0]
                else:
                    lsu = sorted(c for c, (sub, _) in large_chains.items() if sub == "LSU")
                    large = lsu[0] if lsu else sorted(large_chains)[0]
                subunit, mapping = large_chains[large]
                molecule = f"rRNA_{subunit}"
                large_index = {n: i for i, n in enumerate(chains[large][0])}
                seq_hash = hashlib.md5(seq.encode()).hexdigest()[:12]
                site = (sorted({mapping[large_index[n]] for n in members["c4"] if n in large_index and large_index[n] in mapping})
                        + [f"{seq_hash}:{p}" for p in positions])
                mol_seq, mol_chain = chains[large][1], large
            else:
                molecule = hashlib.md5(seq.encode()).hexdigest()[:12]
                site = positions
            molecules.setdefault(molecule, (mol_seq, mol_chain))
            site_key = (group, molecule, tuple(site))
            if site_key in seen_sites:  # exact repeat within the entry (NCS / multiple copies)
                drops["duplicate site within entry"] += 1
                continue
            seen_sites.add(site_key)

            pocket = rna.subgraph(members["shell"] or members["c4"]).copy()
            pocket.name = f"{pdbid}_{k}"
            k += 1
            pocket.graph.update({
                "name": pocket.name, "pdbid": pdbid, "ligand_code": code, "ligand_group": group,
                "ligand_group_fine": group, "molecule": molecule, "main_chain": main_chain,
                "site_positions": site, "resolution_high": rna.graph.get("resolution_high"),
            })
            if self.size_thresholds is not None and not self.size_filter.forward({"rna": pocket}):
                drops["pocket size outside thresholds"] += 1
                continue
            self.add_rna_to_building_list(all_rnas=pockets, rna=pocket)
        return pockets, drops, molecules, fname

    def process(self) -> RNADataset:
        files = sorted(f for f in os.listdir(self.graph_path) if f.endswith(".json"))
        if self.debug:
            files = files[:300]
        self.molecule_dir = Path(self.root) / "molecules"
        self.molecule_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)

        if self.n_jobs > 1:  # fork: workers inherit the task (aligner, curated groups, rRNA references)
            global _WORKER_TASK
            _WORKER_TASK = self
            pool = multiprocessing.get_context("fork").Pool(self.n_jobs)
            results = pool.imap(_worker_entry_pockets, files, chunksize=4)
        else:
            pool, results = None, map(self._entry_pockets, files)

        self.molecule_seqs = {}  # molecule key -> sequence
        molecule_source = {}  # molecule key -> (entry file, chain) of its first occurrence (file order)
        self.drop_log = collections.Counter()
        all_pockets = []
        for pockets, drops, molecules, fname in tqdm(results, total=len(files), desc="building pockets"):
            all_pockets += pockets
            self.drop_log.update(drops)
            for molecule, (seq, chain) in molecules.items():
                if molecule not in self.molecule_seqs:
                    self.molecule_seqs[molecule] = seq
                    molecule_source[molecule] = (fname, chain)
        if pool is not None:
            pool.close()
            pool.join()

        # one PDB per molecule key (the chain of its first occurrence), for the US-align clustering
        by_file = collections.defaultdict(list)
        for molecule, (fname, chain) in molecule_source.items():
            by_file[fname].append((molecule, chain))
        for fname, items in tqdm(sorted(by_file.items()), desc="writing molecule PDBs"):
            rna = load_json(Path(self.graph_path) / fname)
            chains = self._chains(rna)
            for molecule, chain in items:
                self._write_chain_pdb(rna, chains[chain][0], self.molecule_dir / f"{molecule}.pdb")
        print("dropped ligand copies:", dict(self.drop_log))
        return self.create_dataset_from_list(all_pockets)

    def _molecule_clusters(self, molecules):
        """{molecule key -> cluster id}: molecules linked by US-align (see module docstring), grouped
        into connected components; a component containing an rRNA subunit is named after it."""
        keys = sorted(molecules)
        n = len(keys)
        adjacency = np.eye(n, dtype=int)
        self.molecule_links = []
        if n > 1:
            with tempfile.TemporaryDirectory() as tmp:
                listing = Path(tmp) / "list.txt"
                listing.write_text("\n".join(f"{k}.pdb" for k in keys) + "\n")
                out = subprocess.run(
                    ["USalign", "-dir", f"{self.molecule_dir}/", str(listing), "-outfmt", "2", "-mol", "RNA"],
                    capture_output=True, text=True, check=True,
                ).stdout
            pos = {k: i for i, k in enumerate(keys)}
            for line in out.splitlines():
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                a, b = Path(fields[0].split(":")[0]).stem, Path(fields[1].split(":")[0]).stem
                tm1, tm2 = float(fields[2]), float(fields[3])
                la, lb = len(self.molecule_seqs[a]), len(self.molecule_seqs[b])
                short_tm = tm1 if la <= lb else tm2
                long_tm = tm2 if la <= lb else tm1
                linked = long_tm >= self.molecule_tm or (min(la, lb) >= self.fragment_min_len and short_tm >= self.fragment_tm)
                if linked:
                    adjacency[pos[a], pos[b]] = adjacency[pos[b], pos[a]] = 1
                    self.molecule_links.append((a, b, tm1, tm2))
            # sequence links: on 15-50 nt chains, TM-score stays low even between near-identical structures, so
            # sequence-identical molecules (e.g. A-site duplex constructs differing by an overhang) can fall in
            # different clusters. The length guards matter: with free end gaps a short chain aligns anywhere in a
            # long one and reaches high identity by chance
            if self.molecule_seq_identity is not None:
                self.molecule_seq_links = []
                for ia, a in enumerate(keys):
                    for b in keys[ia + 1:]:
                        if a.startswith("rRNA_") or b.startswith("rRNA_") or adjacency[pos[a], pos[b]]:
                            continue
                        sa, sb = self.molecule_seqs[a], self.molecule_seqs[b]
                        short, long_ = sorted((len(sa), len(sb)))
                        if short < self.seq_link_min_len or short / long_ < self.seq_link_min_ratio:
                            continue
                        aln = self._aligner.align(sa, sb)[0]
                        identity = sum(sa[q] == sb[t] for (q0, q1), (t0, t1) in zip(*aln.aligned)
                                       for q, t in zip(range(q0, q1), range(t0, t1)))
                        if identity / short >= self.molecule_seq_identity:
                            adjacency[pos[a], pos[b]] = adjacency[pos[b], pos[a]] = 1
                            self.molecule_seq_links.append((a, b, identity / short))
        _, labels = connected_components(adjacency, directed=False)
        names = {labels[i]: k for i, k in enumerate(keys) if k.startswith("rRNA_")}
        return {k: names.get(labels[i], f"mol_{labels[i]}") for i, k in enumerate(keys)}

    def post_process(self):
        graphs = [self.dataset[i]["rna"] for i in range(len(self.dataset))]
        clusters = self._molecule_clusters({g.graph["molecule"] for g in graphs})
        for g in graphs:
            g.graph["molecule_cluster"] = clusters[g.graph["molecule"]]

        # 1) duplicate binding sites: same fine ligand group, same molecule, overlapping contacts (transitive)
        by_mol = collections.defaultdict(list)
        for i, g in enumerate(graphs):
            by_mol[(g.graph["ligand_group_fine"], g.graph["molecule"])].append(i)
        kept = []
        for members in by_mol.values():
            n = len(members)
            overlap = np.eye(n, dtype=int)
            for a in range(n):
                for b in range(a):
                    if _jaccard(graphs[members[a]].graph["site_positions"], graphs[members[b]].graph["site_positions"]) > self.site_min_jaccard:
                        overlap[a, b] = overlap[b, a] = 1
            _, site_ids = connected_components(overlap, directed=False)
            sites = collections.defaultdict(list)
            for a, site in enumerate(site_ids):
                sites[site].append(members[a])
            kept.extend(min(site, key=lambda i: self._resolution(graphs[i])) for site in sites.values())
        kept.sort()
        self.n_duplicate_sites = len(graphs) - len(kept)
        print(f"duplicate binding sites removed: {self.n_duplicate_sites}")

        # 2) optional merges, then class filter
        for i in kept:
            g = graphs[i]
            g.graph["ligand_group"] = self.group_merges.get(g.graph["ligand_group_fine"], g.graph["ligand_group_fine"])

        def class_filter(kept):
            pockets = collections.Counter(graphs[i].graph["ligand_group"] for i in kept)
            cls_clusters = collections.defaultdict(set)
            for i in kept:
                cls_clusters[graphs[i].graph["ligand_group"]].add(graphs[i].graph["molecule_cluster"])
            dropped = {c: {"pockets": pockets[c], "molecule_clusters": len(cls_clusters[c])} for c in pockets
                       if pockets[c] < self.min_class_pockets or len(cls_clusters[c]) < self.min_class_clusters}
            print(f"classes dropped (< {self.min_class_pockets} pockets or < {self.min_class_clusters} molecule clusters):",
                  dropped)
            return [i for i in kept if graphs[i].graph["ligand_group"] not in dropped], dropped

        kept, self.dropped_groups = class_filter(kept)

        # 2b) at most max_sites_per_molecule sites per molecule, chosen to be the least structurally redundant
        # with the rest of the dataset; applied after the class filter so that sites of dropped classes take no
        # slot, then the class filter is applied again since the cap can bring a class below its thresholds
        if self.max_sites_per_molecule is not None:
            n_before = len(kept)
            kept = self._cap_sites_per_molecule(graphs, kept)
            self.n_capped_sites = n_before - len(kept)
            print(f"sites removed by the cap of {self.max_sites_per_molecule} per molecule: {self.n_capped_sites}")
            kept, dropped_after_cap = class_filter(kept)
            self.dropped_groups.update(dropped_after_cap)

        # 3) training weights: one unit per variant site
        site_of = self._variant_sites([graphs[i] for i in kept])
        site_size = collections.Counter(site_of)
        class_pockets = collections.Counter(graphs[i].graph["ligand_group"] for i in kept)
        class_sites = collections.Counter(c for c, _ in {(graphs[i].graph["ligand_group"], site) for i, site in zip(kept, site_of)})
        for i, site in zip(kept, site_of):
            c = graphs[i].graph["ligand_group"]
            graphs[i].graph["variant_site"] = site
            graphs[i].graph["sample_weight"] = class_pockets[c] / class_sites[c] / site_size[site]
        self.sample_weights = {graphs[i].name: graphs[i].graph["sample_weight"] for i in kept}
        print(f"variant sites: {len(site_size)} for {len(kept)} pockets")

        # the pockets must not carry their label outside the target: drop the residue-level ligand annotations
        # (they name the ligand) and the entry-level ligand SMILES inherited from the entry graph
        for i in kept:
            for _, data in graphs[i].nodes(data=True):
                for key in [key for key in data if key.startswith("binding_small-molecule")]:
                    del data[key]
            graphs[i].graph.pop("ligand_to_smiles", None)

        # molecule_cluster / merged labels / weights live only in memory on the graphs above: write them back
        if not self.in_memory:
            from rnaglib.utils.graph_io import dump_json
            for i in kept:
                dump_json(os.path.join(self.dataset_path, f"{graphs[i].name}.json"), graphs[i])
        self.dataset = self.dataset.subset(list_of_ids=kept)
        kept_graphs = [graphs[i] for i in kept]
        cluster_ids = np.array([g.graph["molecule_cluster"] for g in kept_graphs])
        self.dataset.add_distance("molecule", (cluster_ids[:, None] != cluster_ids[None, :]).astype(float))
        self.dataset.features_computer = self.get_task_vars()  # class mapping without the dropped classes
        if not self.in_memory:
            keep_names = {g.name for g in kept_graphs}
            for f in os.listdir(self.dataset.dataset_path):
                if f.endswith(".json") and Path(f).stem not in keep_names and f[:-5].split("_")[-1].isdigit():
                    os.remove(Path(self.dataset.dataset_path) / f)
            self.dataset.save_distances()
        with open(Path(self.root) / "sample_weights.json", "w") as f:
            json.dump(self.sample_weights, f, indent=1)

    def get_task_vars(self) -> FeaturesComputer:
        values = sorted({rna["rna"].graph[self.target_var] for rna in self.dataset})
        self.mapping = {value: i for i, value in enumerate(values)}
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(mapping=self.mapping)},
        )

    @property
    def default_splitter(self):
        return MoleculeClusterStratifiedSplitter(label_key=self.target_var)
