"""Unit tests for the RNA_Ligand task LigandIdentification: splitter, site / molecule logic and post-processing
with US-align mocked.

Tasks are not built: objects are created without running Task.__init__ (which would build the dataset), and the
external tools (US-align) are replaced by fixed similarity tables.
"""

import collections
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
from Bio import Align

from rnaglib.dataset import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.tasks.RNA_Ligand import ligand_identity as lgi
from rnaglib.tasks.RNA_Ligand.ligand_identity import (
    LigandIdentification,
    MoleculeClusterStratifiedSplitter,
    resolution_high,
)
from rnaglib.transforms import SizeFilter
from rnaglib.utils.graph_io import dump_json


def pocket(name, n_nodes=3, **graph_attrs):
    g = nx.Graph(name=name)
    g.add_nodes_from(f"{name}.A.{i}" for i in range(n_nodes))
    g.graph.update(graph_attrs)
    return g


def dataset_of(graphs, distances=None):
    ds = RNADataset(rnas=graphs)
    for name, mat in (distances or {}).items():
        ds.add_distance(name, np.asarray(mat, dtype=float))
    return ds


def names(ds):
    return {ds[i]["rna"].name for i in range(len(ds))}


def group_task(**kwargs):
    """LigandIdentification with its own __init__ run (curated groups, rRNA references, aligner) but without
    Task.__init__, which would build the dataset."""
    with mock.patch.object(RNAClassificationTask, "__init__", lambda self, **kw: None):
        return LigandIdentification(**kwargs)


# --------------------------------------------------------------------------- module helpers
class TestResolution(unittest.TestCase):
    def test_resolution_high(self):
        self.assertEqual(resolution_high(pocket("a", resolution_high="2.5")), 2.5)
        self.assertEqual(resolution_high(pocket("a", resolution_high=1.8)), 1.8)
        self.assertEqual(resolution_high(pocket("a", resolution_high=None)), float("inf"))
        self.assertEqual(resolution_high(pocket("a", resolution_high="N/A")), float("inf"))
        self.assertEqual(resolution_high(pocket("a")), float("inf"))


class TestModuleHelpers(unittest.TestCase):
    def test_aligned_positions(self):
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score, aligner.mismatch_score = 2, -1
        aligner.open_gap_score, aligner.extend_gap_score = -3, -1
        aln = aligner.align("ACGU", "AGU")[0]
        self.assertEqual(list(lgi._aligned_positions(aln)), [(0, 0), (2, 1), (3, 2)])

    def test_parse_usalign(self):
        out = ("#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\n"
               "/tmp/a/12.pdb:A\t/tmp/b/rRNA_SSU.pdb:A\t0.4120\t0.8100\t2.1\n"
               "\n"
               "mol.pdb\tother.pdb\t1.0\t0.5\t0.0\n")
        self.assertEqual(list(lgi._parse_usalign(out)),
                         [("12", "rRNA_SSU", 0.412, 0.81), ("mol", "other", 1.0, 0.5)])

    def test_jaccard(self):
        self.assertAlmostEqual(lgi._jaccard([1, 2], [2, 3]), 1 / 3)
        self.assertEqual(lgi._jaccard([1], [2]), 0.0)
        self.assertEqual(lgi._jaccard([1, 2], [2, 1, 1]), 1.0)
        self.assertEqual(lgi._jaccard([], [1]), 0.0)

    def test_as_list_and_ligand_key(self):
        entry = {"name": "TAC", "id": ["A", 101]}
        self.assertEqual(lgi._as_list(None), [])
        self.assertEqual(lgi._as_list(entry), [entry])
        self.assertEqual(lgi._as_list([entry, entry]), [entry, entry])
        # two copies of one ligand are distinct keys
        self.assertEqual(lgi._ligand_key(entry), ("TAC", json.dumps(["A", 101])))
        self.assertNotEqual(lgi._ligand_key(entry), lgi._ligand_key({"name": "TAC", "id": ["B", 101]}))


class TestSplitter(unittest.TestCase):
    @staticmethod
    def dataset():
        graphs, k = [], 0
        for c in range(24):
            label = "X" if c < 12 else "Y"
            for _ in range(c % 3 + 1):
                graphs.append(pocket(f"p{k}", molecule_cluster=f"c{c}", ligand_group=label))
                k += 1
        for c in range(24, 27):  # clusters holding both classes
            for label in ("X", "Y"):
                graphs.append(pocket(f"p{k}", molecule_cluster=f"c{c}", ligand_group=label))
                k += 1
        return dataset_of(graphs)

    def test_no_cluster_leakage_and_constraints(self):
        ds = self.dataset()
        splitter = MoleculeClusterStratifiedSplitter(n_starts=10, min_per_split=3, min_clusters_per_split=3)
        train, val, test = splitter(ds)
        self.assertEqual(sorted(train + val + test), list(range(len(ds))))
        split_of_cluster = collections.defaultdict(set)
        for s, idx in enumerate((train, val, test)):
            for i in idx:
                split_of_cluster[ds[i]["rna"].graph["molecule_cluster"]].add(s)
        self.assertTrue(all(len(v) == 1 for v in split_of_cluster.values()), "a molecule cluster spans splits")
        for idx in (val, test):
            for label in ("X", "Y"):
                members = [ds[i]["rna"].graph for i in idx if ds[i]["rna"].graph["ligand_group"] == label]
                per_cluster = collections.Counter(g["molecule_cluster"] for g in members)
                self.assertGreaterEqual(len(members), 3)
                self.assertGreaterEqual(len(per_cluster), 3)
                self.assertLessEqual(max(per_cluster.values()), 0.5 * len(members))
        self.assertGreater(len(train), len(val))

    def test_deterministic(self):
        ds = self.dataset()
        a = MoleculeClusterStratifiedSplitter(n_starts=5, seed=3)(ds)
        b = MoleculeClusterStratifiedSplitter(n_starts=5, seed=3)(ds)
        self.assertEqual(a, b)

    def test_infeasible_raises(self):
        graphs = [pocket(f"p{i}", molecule_cluster=f"c{i % 2}", ligand_group="X") for i in range(20)]
        with self.assertRaises(ValueError):
            MoleculeClusterStratifiedSplitter(n_starts=3)(dataset_of(graphs))


# --------------------------------------------------------------------------- LigandIdentification: methods
SEQ = "AUCGAUCCGUAGCUAGGCUAACGUUAGCAUGCUACGAUCA"  # 40 nt


class TestLigandIdentificationMethods(unittest.TestCase):
    def test_process_requires_graph_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                LigandIdentification(root=tmp, precomputed=False)

    def test_chains(self):
        g = nx.Graph()
        g.add_node("x.A.3", chain_id="A", label_seq_id=3, nt_code="c")
        g.add_node("x.A.1", chain_id="A", label_seq_id=1, nt_code="G")
        g.add_node("x.A.9", chain_id="A", label_seq_id=None, nt_code=None)
        g.add_node("x.B.1", chain_id="B", label_seq_id=1, nt_code="U")
        chains = group_task()._chains(g)
        self.assertEqual(chains["A"], (["x.A.1", "x.A.3", "x.A.9"], "GCN"))
        self.assertEqual(chains["B"], (["x.B.1"], "U"))

    def test_map_rrna_to_ecoli_numbering(self):
        t = group_task()
        for subunit, ref, start in (("SSU", t.rrna_ref["SSU"], 100), ("LSU", t.rrna_ref["LSU"], 1500)):
            fragment = ref[start:start + 750]
            sub, mapping = t._map_rrna(fragment)
            self.assertEqual(sub, subunit)
            # E. coli numbering is 1-based
            self.assertEqual(mapping[0], start + 1)
            self.assertEqual(mapping[len(fragment) - 1], start + len(fragment))
            self.assertIs(t._map_rrna(fragment)[1], mapping)  # cached

    def test_variant_sites(self):
        t = group_task()
        long_seq = "GG" + SEQ  # same RNA with a 2-nt 5' extension: positions shift by 2
        t.molecule_seqs = {"m1": SEQ, "m2": long_seq, "m3": "GCGCGCAUAUAUGCGCGCAUAUAU" * 2}

        def g(name, molecule, cluster, site, group="tetracycline"):
            return pocket(name, molecule=molecule, molecule_cluster=cluster, site_positions=site,
                          ligand_group_fine=group)

        graphs = [
            g("a", "m1", "mol_0", [7, 8]),
            g("b", "m2", "mol_0", [10]),  # = position 8 of m1: same variant site as a
            g("c", "m2", "mol_0", [30, 31]),  # another site of the same RNA
            g("d", "m1", "mol_0", [7, 8], group="flavin"),  # same positions, other ligand group
            g("e", "m3", "mol_1", [7, 8]),  # other molecule
            g("f", "rRNA_SSU", "rRNA_SSU", [1054, 1055]),
            g("h", "rRNA_SSU", "rRNA_SSU", [1055, 1056]),  # overlapping rRNA site
            g("i", "rRNA_SSU", "rRNA_SSU", [1492]),
        ]
        site = dict(zip("abcdefhi", t._variant_sites(graphs)))
        self.assertEqual(site["a"], site["b"])
        self.assertEqual(site["f"], site["h"])
        others = [site[k] for k in "acdefi"]
        self.assertEqual(len(set(others)), len(others))

    def test_cap_sites_per_molecule(self):
        t = group_task(max_sites_per_molecule=2)
        graphs = [pocket(f"m{i}", molecule="M") for i in range(5)] + [pocket("n0", molecule="N")]
        q = 5  # pocket of N, under the cap
        sims = {(0, q): 0.9, (1, q): 0.1, (2, q): 0.2, (3, q): 0.3, (4, q): 0.05,
                (1, 4): 0.95, (2, 4): 0.1, (3, 4): 0.1, (0, 4): 0.1}
        calls = []

        def fake_sims(graphs_, candidates, references):
            calls.append((sorted(candidates), sorted(references)))
            return sims

        with mock.patch.object(t, "_pocket_similarities", side_effect=fake_sims):
            kept = t._cap_sites_per_molecule(graphs, list(range(6)))
        # greedy: m4 (least redundant, 0.05), then m2 (m1 became redundant with m4), M then full
        self.assertEqual(kept, [2, 4, 5])
        self.assertEqual(calls, [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5])])

    def test_cap_sites_noop_under_cap(self):
        t = group_task(max_sites_per_molecule=3)
        graphs = [pocket(f"m{i}", molecule="M") for i in range(3)]
        with mock.patch.object(t, "_pocket_similarities", side_effect=AssertionError("no US-align needed")):
            self.assertEqual(t._cap_sites_per_molecule(graphs, [0, 1, 2]), [0, 1, 2])

    def test_entry_pockets(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = nx.DiGraph(name="9xyz", pdbid="9xyz", resolution_high=2.0)
            nodes = [f"9xyz.A.{i}" for i in range(1, 21)]
            for i, n in enumerate(nodes, start=1):
                g.add_node(n, chain_id="A", label_seq_id=i, nt_code="ACGU"[i % 4])

            def bind(code, chain, contacts4, contacts8):
                entry = {"name": code, "id": [chain, 101]}
                for n in contacts8:
                    g.nodes[n]["binding_small-molecule-8.0A"] = entry
                for n in contacts4:
                    g.nodes[n]["binding_small-molecule-4.0A"] = entry

            bind("TAC", "A", nodes[2:8], nodes[0:10])  # tetracycline, 6 contacts: one pocket
            bind("KSG", "B", nodes[12:15], nodes[11:16])  # 3 contacts: too few
            # residue-level annotations hold one ligand per residue, so the others use separate residues
            bind("GTP", "C", nodes[15:21], nodes[15:21])  # nucleotide: excluded group
            dump_json(Path(tmp) / "9xyz.json", g)
            h = nx.DiGraph(name="8xyz", pdbid="8xyz")
            h.add_node("8xyz.A.1", chain_id="A", label_seq_id=1, nt_code="A")
            dump_json(Path(tmp) / "8xyz.json", h)

            t = group_task(graph_path=tmp)
            t.in_memory, t.size_thresholds, t.size_filter = True, (5, 500), SizeFilter(5, 500)
            pockets, drops, molecules, _ = t._entry_pockets("9xyz.json")
            self.assertEqual(t._entry_pockets("8xyz.json")[0], [])  # no ligand annotation at all

        self.assertEqual(len(pockets), 1)
        p = pockets[0]
        self.assertEqual(set(p.nodes()), set(nodes[0:10]))  # pocket = 8 A shell
        self.assertEqual(p.graph["ligand_group"], "tetracycline")
        self.assertEqual(p.graph["ligand_code"], "TAC")
        self.assertEqual(p.graph["site_positions"], [2, 3, 4, 5, 6, 7])
        self.assertEqual(p.graph["main_chain"], "A")
        self.assertEqual(p.name, "9xyz_0")
        self.assertEqual(drops["too few RNA contacts"], 1)
        self.assertEqual(drops["excluded ligand group"], 1)
        self.assertEqual(list(molecules.values()), [("".join(g.nodes[n]["nt_code"] for n in nodes), "A")])

    def test_post_process_sites_weights_and_label_leakage(self):
        """Duplicate-site removal, class filter, sample weights and removal of ligand annotations, with the
        US-align molecule clustering replaced by a fixed table."""
        t = group_task(min_class_pockets=2, min_class_clusters=2, max_sites_per_molecule=None)
        long_seq = "GG" + SEQ
        t.molecule_seqs = {"mA": SEQ, "mA2": long_seq, "mB": "GCGCGCAUAUAUGCGCGCAUAUAU" * 2,
                           "mC": "UUUUGGGGCCCCAAAAUUUUGGGG" * 2, "rRNA_SSU": "A"}
        clusters = {"mA": "mol_0", "mA2": "mol_0", "mB": "mol_1", "mC": "mol_2", "rRNA_SSU": "rRNA_SSU"}

        def g(name, molecule, site, res, group="tetracycline"):
            p = pocket(name, molecule=molecule, site_positions=site, resolution_high=res, ligand_group=group,
                       ligand_group_fine=group, ligand_to_smiles={"TAC": "C"})
            for n in p.nodes():
                p.nodes[n]["binding_small-molecule-4.0A"] = {"name": "TAC", "id": ["A", 1]}
                p.nodes[n]["binding_small-molecule-8.0A"] = {"name": "TAC", "id": ["A", 1]}
                p.nodes[n]["nt_code"] = "A"
            return p

        graphs = [
            g("p0", "mA", [5, 6, 7], 2.0),
            g("p1", "mA", [7, 8], 1.5),  # same site as p0 (shares 7), better resolution: kept
            g("p2", "mA2", [10], 2.5),  # variant of mA, same site as p1 after numbering shift
            g("p3", "mB", [1, 2], 3.0),
            g("p4", "mC", [3], 3.0),
            g("p5", "mB", [10], 2.0, group="flavin"),  # merged class with a single pocket: dropped
            g("p6", "rRNA_SSU", [1054], 3.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            t.root, t.in_memory, t.dataset = tmp, True, dataset_of(graphs)
            with mock.patch.object(t, "_molecule_clusters", return_value=clusters):
                t.post_process()
            weights = json.loads((Path(tmp) / "sample_weights.json").read_text())

        self.assertEqual(names(t.dataset), {"p1", "p2", "p3", "p4", "p6"})
        out = {t.dataset[i]["rna"].name: t.dataset[i]["rna"] for i in range(len(t.dataset))}
        # 5 pockets, 4 variant sites ({p1, p2}, p3, p4, p6): every site weighs 5/4, the class total stays 5
        expected = {"p1": 0.625, "p2": 0.625, "p3": 1.25, "p4": 1.25, "p6": 1.25}
        for name, w in expected.items():
            self.assertAlmostEqual(out[name].graph["sample_weight"], w)
            self.assertAlmostEqual(weights[name], w)
        self.assertEqual(out["p1"].graph["variant_site"], out["p2"].graph["variant_site"])
        self.assertIn("heteroaromatic_nucleobase_like", t.dropped_groups)
        # the label must not leak through residue annotations or entry-level SMILES
        for gr in out.values():
            self.assertNotIn("ligand_to_smiles", gr.graph)
            for _, data in gr.nodes(data=True):
                self.assertFalse(any(k.startswith("binding_small-molecule") for k in data))
        # molecule-cluster distance: 0 within a cluster, 1 across
        order = [t.dataset[i]["rna"].name for i in range(len(t.dataset))]
        dist = t.dataset.distances["molecule"]
        self.assertEqual(dist[order.index("p1"), order.index("p2")], 0.0)
        self.assertEqual(dist[order.index("p1"), order.index("p3")], 1.0)
        self.assertEqual(t.mapping, {"tetracycline": 0})
