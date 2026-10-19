"""Unit tests for RNA_Site (BenchmarkBindingSite cutoff) and RNA_Prot (crystallization-chaperone relabelling).

process() is run on a tiny in-memory graph, with the dataset download and the structure-dependent transforms
replaced by stubs; the tasks themselves are not built.
"""

import tempfile
import unittest
from unittest import mock

import networkx as nx

from rnaglib.tasks import BenchmarkBindingSite, ProteinBindingSite
from rnaglib.tasks.RNA_Prot import protein_binding_site as prot_module
from rnaglib.tasks.RNA_Site import binding_site as site_module
from rnaglib.transforms import SizeFilter


class _FakeDataset(list):
    structures_path = None
    dataset_path = "fake"


class _PassFilter:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, rna):
        return True


class _Identity:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, rna):
        return rna


class TestBenchmarkBindingSiteCutoff(unittest.TestCase):
    @staticmethod
    def graph():
        g = nx.DiGraph(name="1abc", pdbid="1abc")
        lig = {"name": "TAC", "id": ["A", 1]}
        g.add_node("1abc.A.1", **{"binding_small-molecule-4.0A": lig, "binding_small-molecule-6.0A": lig,
                                   "binding_small-molecule-8.0A": lig})
        g.add_node("1abc.A.2", **{"binding_small-molecule-6.0A": lig, "binding_small-molecule-8.0A": lig})
        g.add_node("1abc.A.3", **{"binding_ion_6.0A": "MG", "binding_ion_8.0A": "MG"})
        g.add_node("1abc.A.4", **{"binding_small-molecule-8.0A": lig})
        g.add_node("1abc.A.5")
        g.add_edge("1abc.A.1", "1abc.A.2", LW="B35")
        return g

    def run_process(self, **kwargs):
        task = BenchmarkBindingSite.__new__(BenchmarkBindingSite)
        with mock.patch.object(site_module.ResidueClassificationTask, "__init__", lambda self, **kw: None):
            BenchmarkBindingSite.__init__(task, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            task.debug, task.in_memory, task.dataset_path = True, True, tmp
            with mock.patch.object(site_module, "RNADataset", lambda **kw: _FakeDataset([{"rna": self.graph()}])), \
                    mock.patch.object(site_module, "ChainFilter", _PassFilter), \
                    mock.patch.object(site_module, "SmallMoleculeBindingTransform", _Identity):
                dataset = task.process()
        g = dataset[0]["rna"]
        return {n for n, v in nx.get_node_attributes(g, "binding_site").items() if v}

    def test_cutoff_reaches_annotator(self):
        self.assertEqual(self.run_process(cutoff=4.0), {"1abc.A.1"})
        self.assertEqual(self.run_process(cutoff=6.0), {"1abc.A.1", "1abc.A.2", "1abc.A.3"})
        self.assertEqual(self.run_process(cutoff=8.0), {"1abc.A.1", "1abc.A.2", "1abc.A.3", "1abc.A.4"})

    def test_default_cutoff_is_6(self):
        self.assertEqual(self.run_process(), {"1abc.A.1", "1abc.A.2", "1abc.A.3"})


class TestChaperoneRelabelling(unittest.TestCase):
    @staticmethod
    def graph(pdbid, with_chains=True):
        """Two connected components: one with a mix of native and chaperone contacts, one contacted only by the
        chaperone (chain A is a listed chaperone of 1drz)."""
        g = nx.DiGraph(name=pdbid, pdbid=pdbid.upper())
        nodes = {
            # component 1
            "A.1": (True, ["A"]),  # chaperone only -> non-binding when excluded
            "A.2": (True, ["A", "Z"]),  # chaperone + native partner -> stays binding
            "A.3": (True, ["Z"]),
            "A.4": (False, []),
            # component 2
            "B.1": (True, ["A"]),
            "B.2": (False, []),
        }
        for n, (binding, chains) in nodes.items():
            attrs = {"protein_binding_chains": chains} if with_chains else {}
            g.add_node(f"{pdbid}.{n}", protein_binding=binding, nt_code="A", **attrs)
        for a, b in (("A.1", "A.2"), ("A.2", "A.3"), ("A.3", "A.4"), ("B.1", "B.2")):
            g.add_edge(f"{pdbid}.{a}", f"{pdbid}.{b}", LW="B35")
        return g

    def run_process(self, pdbid, exclude, with_chains=True):
        task = ProteinBindingSite.__new__(ProteinBindingSite)
        with mock.patch.object(prot_module.ResidueClassificationTask, "__init__", lambda self, **kw: None):
            ProteinBindingSite.__init__(task, exclude_chaperone_contacts=exclude)
        with tempfile.TemporaryDirectory() as tmp:
            task.debug, task.in_memory, task.dataset_path = False, True, tmp
            task.size_thresholds, task.size_filter = (1, 500), SizeFilter(1, 500)
            with mock.patch.object(prot_module, "RNADataset", lambda **kw: _FakeDataset([{"rna": self.graph(pdbid, with_chains)}])):
                dataset = task.process()
        labels = {}
        for i in range(len(dataset)):
            g = dataset[i]["rna"]
            labels.update(nx.get_node_attributes(g, "protein_binding"))
        return labels

    def test_chaperone_only_residues_relabelled(self):
        self.assertIn("A", prot_module.CHAPERONE_CHAINS["1drz"])
        labels = self.run_process("1drz", exclude=True)
        # component 2 is left without any binding residue and is filtered out
        self.assertEqual(labels, {"1drz.A.1": False, "1drz.A.2": True, "1drz.A.3": True, "1drz.A.4": False})

    def test_no_relabelling_by_default(self):
        labels = self.run_process("1drz", exclude=False)
        self.assertTrue(labels["1drz.A.1"])
        self.assertTrue(labels["1drz.B.1"])

    def test_unlisted_entry_untouched(self):
        self.assertNotIn("9zzz", prot_module.CHAPERONE_CHAINS)
        labels = self.run_process("9zzz", exclude=True)
        self.assertTrue(labels["9zzz.A.1"])
        self.assertTrue(labels["9zzz.B.1"])

    def test_missing_annotation_raises(self):
        # released 2.0.2 graphs lack protein_binding_chains: excluding chaperones must fail loudly, not silently
        with self.assertRaises(ValueError):
            self.run_process("1drz", exclude=True, with_chains=False)
        # not needed when the option is off, or for entries without a listed chaperone
        self.assertTrue(self.run_process("1drz", exclude=False, with_chains=False)["1drz.A.1"])
        self.assertTrue(self.run_process("9zzz", exclude=True, with_chains=False)["9zzz.A.1"])


if __name__ == "__main__":
    unittest.main()
