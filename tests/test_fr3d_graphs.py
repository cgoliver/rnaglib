"""Unit tests for the graph builder in rnaglib.prepare_data.fr3d_2_graphs.

Most tests use small synthetic inputs (hand-built mmCIF dictionaries, Bio.PDB residues, a tiny hand-written mmCIF
file with a mocked FR3D annotation). Two tests run the real pipeline (FR3D included) on small entries shipped with
the repository.
"""

import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import MMCIF2Dict
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from rnaglib.prepare_data import fr3d_2_graphs as f2g

REPO = Path(__file__).resolve().parents[1]
STRUCTURES = REPO / "tests" / "data"


# --------------------------------------------------------------------------- Bio.PDB helpers
def make_residue(num, resname="G", icode=" ", label_seq_id=None, atoms=None, het=" "):
    """Residue with the given atoms ({name: xyz}); label_seq_id stored in .xtra as fr3d_to_graph does."""
    res = Residue((het, num, icode), resname, "")
    for serial, (name, xyz) in enumerate((atoms or {}).items(), start=1):
        res.add(Atom(name, np.array(xyz, dtype="f"), 0.0, 1.0, " ", name.ljust(4), serial, name[0]))
    res.xtra["label_seq_id"] = label_seq_id
    return res


def backbone_atoms(k, offset=0.0, o3p=1.6):
    """P at x = 6k, O3' placed so that O3'(k)-P(k+1) is `o3p` A when consecutive residues use consecutive k."""
    return {"P": (6.0 * k + offset, 0.0, 0.0), "O3'": (6.0 * (k + 1) - o3p + offset, 0.0, 0.0)}


def make_structure(chains):
    """chains: {chain_id: [residues]} -> Bio.PDB Model (fr3d_to_graph works on the first model)."""
    structure = Structure("s")
    model = Model(0)
    structure.add(model)
    for cid, residues in chains.items():
        chain = Chain(cid)
        model.add(chain)
        for r in residues:
            chain.add(r)
    return model


class TestPolymerResidueSelection(unittest.TestCase):
    def mmcif_dict(self):
        # entity 1: RNA (chain A); entity 2: FMN ligand on author chain A; entity 3: protein (chain B)
        rows = [
            # auth_asym, auth_seq, ins, entity, label_seq
            ("A", "1", "?", "1", "1"),  # 5' GTP, a polymer residue typed non-polymer: must be kept
            ("A", "2", "?", "1", "2"),
            ("A", "2", "A", "1", "3"),  # insertion code
            ("A", "101", ".", "2", "."),  # FMN ligand sharing the author chain: no label_seq_id
            ("B", "1", "?", "3", "1"),  # protein residue
        ]
        return {
            "_entity_poly.entity_id": ["1", "3"],
            "_entity_poly.type": ["polyribonucleotide", "polypeptide(L)"],
            "_entity_poly.pdbx_strand_id": ["A", "B"],
            "_atom_site.auth_asym_id": [r[0] for r in rows],
            "_atom_site.auth_seq_id": [r[1] for r in rows],
            "_atom_site.pdbx_PDB_ins_code": [r[2] for r in rows],
            "_atom_site.label_entity_id": [r[3] for r in rows],
            "_atom_site.label_seq_id": [r[4] for r in rows],
        }

    def test_keeps_only_polymer_rna_residues(self):
        keys = f2g.get_polymer_rna_residue_keys(self.mmcif_dict())
        self.assertEqual(keys, {("A", 1, " "), ("A", 2, " "), ("A", 2, "A")})

    def test_scalar_entity_fields(self):
        # a single-entity entry may give scalars instead of lists
        d = self.mmcif_dict()
        d["_entity_poly.entity_id"] = "1"
        d["_entity_poly.type"] = "polyribonucleotide"
        keys = f2g.get_polymer_rna_residue_keys(d)
        self.assertIn(("A", 2, "A"), keys)
        self.assertNotIn(("A", 101, " "), keys)

    def test_missing_insertion_code_column(self):
        d = self.mmcif_dict()
        del d["_atom_site.pdbx_PDB_ins_code"]
        keys = f2g.get_polymer_rna_residue_keys(d)
        self.assertEqual(keys, {("A", 1, " "), ("A", 2, " ")})

    def test_get_rna_chains_splits_multi_chain_entities(self):
        d = {"_entity_poly.pdbx_strand_id": ["A,C", "B"],
             "_entity_poly.type": ["polyribonucleotide", "polypeptide(L)"]}
        self.assertEqual(f2g.get_rna_chains(d), ["A", "C"])


class TestResidueOrdering(unittest.TestCase):
    def test_sorted_by_label_seq_id_not_author_numbering(self):
        # author numbering 1, 1000 (inserted segment), 2, 2A, 3 in sequence order
        residues = [
            make_residue(1, label_seq_id=1),
            make_residue(2, label_seq_id=3),
            make_residue(2, icode="A", label_seq_id=4),
            make_residue(1000, label_seq_id=2),
            make_residue(3, label_seq_id=5),
        ]
        model = make_structure({"A": residues})
        poly = {("A", r.id[1], r.id[2]) for r in residues}
        ordered = f2g.get_residue_list(model["A"], poly)
        self.assertEqual([r.id[1:] for r in ordered], [(1, " "), (1000, " "), (2, " "), (2, "A"), (3, " ")])

    def test_excludes_non_polymer_and_unlabelled(self):
        residues = [
            make_residue(1, label_seq_id=1),
            make_residue(2, label_seq_id=None),  # e.g. a ligand residue: no label_seq_id
            make_residue(3, label_seq_id=2),  # not in the polymer keys
        ]
        model = make_structure({"A": residues})
        ordered = f2g.get_residue_list(model["A"], {("A", 1, " "), ("A", 2, " ")})
        self.assertEqual([r.id[1] for r in ordered], [1])


class TestBackbone(unittest.TestCase):
    def test_o3p_distance(self):
        r1 = make_residue(1, atoms={"O3'": (0, 0, 0)})
        r2 = make_residue(2, atoms={"P": (1.6, 0, 0)})
        self.assertAlmostEqual(f2g._o3p_distance(r1, r2), 1.6, places=3)
        self.assertIsNone(f2g._o3p_distance(r2, r1))  # r2 has no O3', r1 has no P
        self.assertIsNone(f2g._o3p_distance(r1, make_residue(3, atoms={"C1'": (0, 0, 0)})))

    def build(self):
        residues = [
            make_residue(1, label_seq_id=1, atoms=backbone_atoms(1)),
            make_residue(2, label_seq_id=2, atoms=backbone_atoms(2)),
            # residue 3 is 30 A away: physical chain break although label_seq_ids are consecutive
            make_residue(3, label_seq_id=3, atoms=backbone_atoms(3, offset=30.0)),
            # label gap 3 -> 5: no edge whatever the geometry
            make_residue(4, label_seq_id=5, atoms=backbone_atoms(5, offset=30.0)),
            # missing P on residue 5: edge kept on label adjacency, distance unmeasurable
            make_residue(5, label_seq_id=6, atoms={"O3'": backbone_atoms(6, offset=30.0)["O3'"]}),
        ]
        protein = [make_residue(1, resname="ALA", label_seq_id=1, atoms={"CA": (0, 0, 0)})]
        model = make_structure({"A": residues, "B": protein})
        poly = {("A", r.id[1], r.id[2]) for r in residues} | {("B", 1, " ")}
        return f2g.get_bb(model, ["A"], poly, pdbid="x")

    def test_get_bb_returns_five_values_and_residue_map(self):
        out = self.build()
        self.assertEqual(len(out), 5)
        bb, nt, nt_full, lsi, residues = out
        self.assertEqual(set(residues), {f"x.A.{i}" for i in range(1, 6)})
        self.assertTrue(all(isinstance(r, Residue) for r in residues.values()))
        self.assertEqual(residues["x.A.4"].id[1], 4)
        self.assertEqual(lsi["x.A.4"], 5)
        self.assertEqual(nt_full["x.A.1"], "G")
        # protein chain is not an RNA chain
        self.assertFalse(any(".B." in n for n in residues))

    def test_o3p_chain_break_rule(self):
        bb, _, _, _, residues = self.build()
        pairs = {(u, v): d for u, v, d in bb}
        # 1 -> 2: covalent linkage, both directions, with distance
        self.assertEqual(pairs[("x.A.2", "x.A.1")]["LW"], "B53")
        self.assertEqual(pairs[("x.A.1", "x.A.2")]["LW"], "B35")
        self.assertAlmostEqual(pairs[("x.A.1", "x.A.2")]["o3p_dist"], 1.6, places=2)
        self.assertGreater(30.0, f2g.MAX_O3P_DIST)
        # 2 -> 3: label-consecutive but O3'-P > MAX_O3P_DIST
        self.assertNotIn(("x.A.2", "x.A.3"), pairs)
        self.assertNotIn(("x.A.3", "x.A.2"), pairs)
        # 3 -> 4: label gap
        self.assertNotIn(("x.A.3", "x.A.4"), pairs)
        # 4 -> 5: P missing, kept with o3p_dist None
        self.assertIsNone(pairs[("x.A.4", "x.A.5")]["o3p_dist"])
        self.assertEqual(len(bb), 4)
        # residue 3, isolated by breaks on both sides, is still registered as a node
        self.assertIn("x.A.3", residues)
        self.assertFalse(any("x.A.3" in (u, v) for u, v, _ in bb))

    def test_threshold_boundary(self):
        for dist, expect_edge in ((f2g.MAX_O3P_DIST - 0.05, True), (f2g.MAX_O3P_DIST + 0.05, False)):
            residues = [make_residue(1, label_seq_id=1, atoms=backbone_atoms(1, o3p=dist)),
                        make_residue(2, label_seq_id=2, atoms=backbone_atoms(2, o3p=dist))]
            model = make_structure({"A": residues})
            bb = f2g.get_bb(model, ["A"], {("A", 1, " "), ("A", 2, " ")}, pdbid="x")[0]
            self.assertEqual(len(bb) == 2, expect_edge, dist)


class TestSymmetryHelpers(unittest.TestCase):
    def test_sym_code_and_nt_to_rgl(self):
        self.assertEqual(f2g._sym_code("1ABC|1|A|G|12"), "1_555")
        self.assertEqual(f2g._sym_code("1ABC|1|A|G|12||||7_555"), "7_555")
        self.assertEqual(f2g.nt_to_rgl("1ABC|1|A|G|12", "1ABC"), "1abc.A.12")
        self.assertEqual(f2g.nt_to_rgl("1ABC|1|A|G|12|||B", "1ABC"), "1abc.A.12B")
        self.assertEqual(f2g.nt_to_rgl("1ABC|1|A|G|12||||1_555", "1ABC"), "1abc.A.12")
        self.assertEqual(f2g.nt_to_rgl("1ABC|1|A|G|12|||B|7_555", "1ABC"), "1abc.A-7-555.12B")

    def oper_dict(self):
        rot = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        d = {
            "_pdbx_struct_oper_list.id": ["1", "2", "3"],
            "_pdbx_struct_oper_list.type": ["identity operation", "crystal symmetry operation",
                                            "crystal symmetry operation"],
            "_pdbx_struct_oper_list.name": ["1_555", "7_555", "4_655"],
        }
        mats = [np.eye(3), np.array(rot), np.eye(3)]
        vecs = [(0, 0, 0), (50, 0, 0), (10, 20, 30)]
        for r in range(1, 4):
            for c in range(1, 4):
                d[f"_pdbx_struct_oper_list.matrix[{r}][{c}]"] = [str(m[r - 1][c - 1]) for m in mats]
            d[f"_pdbx_struct_oper_list.vector[{r}]"] = [str(v[r - 1]) for v in vecs]
        return d

    def test_parse_oper_list(self):
        ops = f2g._parse_oper_list(self.oper_dict())
        self.assertEqual(set(ops), {"1_555", "7_555", "4_655"})
        R, t = ops["7_555"]
        np.testing.assert_allclose(R, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        np.testing.assert_allclose(t, [50, 0, 0])
        np.testing.assert_allclose(ops["4_655"][1], [10, 20, 30])
        self.assertEqual(f2g._parse_oper_list({}), {})

    def test_collect_sym_needs(self):
        d = self.oper_dict()
        d.update({
            # label asym A, C -> author chain X (RNA); label B -> author chain P (protein)
            "_atom_site.label_asym_id": ["A", "A", "B", "C"],
            "_atom_site.auth_asym_id": ["X", "X", "P", "X"],
            "_pdbx_struct_assembly_gen.oper_expression": ["1,2", "(1-60)(61)", "9", "3"],
            "_pdbx_struct_assembly_gen.asym_id_list": ["A,B", "A", "A", "B"],
        })
        needs = f2g._collect_sym_needs(d, ["X"])
        # identity skipped, product expression skipped, unknown operator id skipped, protein-only assembly skipped
        self.assertEqual(dict(needs), {"7_555": {"X"}})

    def test_add_sym_copy_chain(self):
        G = nx.MultiDiGraph()
        G.add_node("p.A.1", chain_id="A", xyz_P=[1.0, 2.0, 3.0], xyz_centroid=[0.0, 0.0, 0.0],
                   heavy_atoms={"P": [1.0, 2.0, 3.0]}, nt="G")
        G.add_node("p.A.2", chain_id="A", xyz_P=None, xyz_centroid=[1.0, 1.0, 1.0], heavy_atoms=None, nt="C")
        G.add_node("p.B.1", chain_id="B", xyz_P=[0.0, 0.0, 0.0], nt="A")
        G.add_edge("p.A.1", "p.A.2", LW="B35")
        G.add_edge("p.A.2", "p.A.1", LW="B53")
        G.add_edge("p.A.1", "p.B.1", LW="cWW")
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        t = np.array([50.0, 0.0, 0.0])
        f2g._add_sym_copy_chain(G, "A", "7_555", R, t)

        self.assertEqual(set(G.nodes()), {"p.A.1", "p.A.2", "p.B.1", "p.A-7-555.1", "p.A-7-555.2"})
        copy = G.nodes["p.A-7-555.1"]
        self.assertEqual(copy["chain_id"], "A-7-555")
        np.testing.assert_allclose(copy["xyz_P"], [49.0, -2.0, 3.0])
        np.testing.assert_allclose(copy["heavy_atoms"]["P"], [49.0, -2.0, 3.0])
        np.testing.assert_allclose(copy["xyz_centroid"], [50.0, 0.0, 0.0])
        self.assertIsNone(G.nodes["p.A-7-555.2"]["xyz_P"])
        # original chain untouched
        self.assertEqual(G.nodes["p.A.1"]["xyz_P"], [1.0, 2.0, 3.0])
        self.assertEqual(G.nodes["p.A.1"]["chain_id"], "A")
        # intra-chain edges replicated, inter-chain edges not
        self.assertEqual(G["p.A-7-555.1"]["p.A-7-555.2"][0]["LW"], "B35")
        self.assertEqual(G["p.A-7-555.2"]["p.A-7-555.1"][0]["LW"], "B53")
        self.assertFalse(G.has_edge("p.A-7-555.1", "p.B.1"))
        # the copy's node ids are those nt_to_rgl produces for FR3D unit ids carrying the operator
        self.assertIn(f2g.nt_to_rgl("P|1|A|G|1||||7_555", "p"), G)

    def test_add_sym_copy_unknown_chain_is_noop(self):
        G = nx.MultiDiGraph()
        G.add_node("p.A.1", chain_id="A")
        f2g._add_sym_copy_chain(G, "Z", "7_555", np.eye(3), np.zeros(3))
        self.assertEqual(list(G.nodes()), ["p.A.1"])


class TestForgiAnnotation(unittest.TestCase):
    def graph(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(["x.A.1", "x.A.2"])
        return G

    def assert_all_none(self, G):
        self.assertEqual(nx.get_node_attributes(G, "secondary_element"), {"x.A.1": None, "x.A.2": None})

    def test_unset(self):
        G = self.graph()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORGI_PYTHON", None)
            f2g._forgi_annotate_sse(G, "x.cif", "x")
        self.assert_all_none(G)

    def test_nonexistent_interpreter(self):
        G = self.graph()
        with mock.patch.dict(os.environ, {"FORGI_PYTHON": "/nonexistent/path/to/python"}):
            f2g._forgi_annotate_sse(G, "x.cif", "x")
        self.assert_all_none(G)

    def test_failing_and_garbage_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            for body in ("import sys; sys.exit(3)", "print('not json')"):
                helper = Path(tmp) / "helper.py"
                helper.write_text(body)
                G = self.graph()
                with mock.patch.dict(os.environ, {"FORGI_PYTHON": sys.executable}), \
                        mock.patch.object(f2g, "_FORGI_HELPER", helper):
                    f2g._forgi_annotate_sse(G, "x.cif", "x")
                self.assert_all_none(G)

    def test_working_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = Path(tmp) / "helper.py"
            helper.write_text('import json; print(json.dumps({"x.A.1": "stem", "x.Z.9": "hairpin"}))')
            G = self.graph()
            with mock.patch.dict(os.environ, {"FORGI_PYTHON": sys.executable}), \
                    mock.patch.object(f2g, "_FORGI_HELPER", helper):
                f2g._forgi_annotate_sse(G, "x.cif", "x")
        # unknown node from the helper ignored, unannotated node set to None
        self.assertEqual(nx.get_node_attributes(G, "secondary_element"), {"x.A.1": "stem", "x.A.2": None})


# --------------------------------------------------------------------------- end-to-end on a synthetic mmCIF file
_ATOM_COLUMNS = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
                 "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y",
                 "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
                 "auth_atom_id", "pdbx_PDB_model_num"]


def _synthetic_cif():
    """RNA chain A: 6 residues (5' GTP, author numbering out of sequence order, an insertion code, a chain break
    before the last residue), FMN ligand on author chain A, one protein residue on chain B, and a biological
    assembly made of the asymmetric unit plus a 7_555 copy."""
    # (group, comp, label_asym, entity, label_seq, ins, auth_seq, auth_asym, k, offset)
    residues = [
        ("HETATM", "GTP", "A", "1", "1", "?", "1", "A", 1, 0.0),
        ("ATOM", "G", "A", "1", "2", "?", "2", "A", 2, 0.0),
        ("ATOM", "C", "A", "1", "3", "A", "2", "A", 3, 0.0),
        ("ATOM", "C", "A", "1", "4", "?", "1000", "A", 4, 0.0),
        ("ATOM", "G", "A", "1", "5", "?", "3", "A", 5, 0.0),
        ("ATOM", "G", "A", "1", "6", "?", "4", "A", 6, 60.0),  # chain break
    ]
    lines = []
    serial = 1
    for group, comp, lasym, ent, lseq, ins, aseq, aasym, k, off in residues:
        for name, xyz in backbone_atoms(k, offset=off).items():
            elem = name[0]
            lines.append([group, str(serial), elem, f'"{name}"', ".", comp, lasym, ent, lseq, ins,
                          *(f"{c:.3f}" for c in xyz), "1.00", "0.00", aseq, comp, aasym, f'"{name}"', "1"])
            serial += 1
    lines.append(["HETATM", str(serial), "N", "N1", ".", "FMN", "C", "2", ".", "?", "10.000", "10.000", "10.000",
                  "1.00", "0.00", "101", "FMN", "A", "N1", "1"])
    serial += 1
    lines.append(["ATOM", str(serial), "C", "CA", ".", "ALA", "B", "3", "1", "?", "0.000", "20.000", "0.000",
                  "1.00", "0.00", "1", "ALA", "B", "CA", "1"])
    out = ["data_9XYZ", "#",
           "loop_", "_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.pdbx_strand_id",
           "1 polyribonucleotide A", "3 'polypeptide(L)' B", "#",
           "loop_"]
    oper_fields = ["id", "type", "name"] + [f"matrix[{r}][{c}]" for r in range(1, 4) for c in range(1, 4)] + \
                  [f"vector[{r}]" for r in range(1, 4)]
    out += [f"_pdbx_struct_oper_list.{f}" for f in oper_fields]
    out.append("1 'identity operation' 1_555 1 0 0 0 1 0 0 0 1 0 0 0")
    out.append("2 'crystal symmetry operation' 7_555 -1 0 0 0 -1 0 0 0 1 50 0 0")
    out += ["#", "loop_", "_pdbx_struct_assembly_gen.assembly_id", "_pdbx_struct_assembly_gen.oper_expression",
            "_pdbx_struct_assembly_gen.asym_id_list", "1 1,2 A,B,C", "#", "loop_"]
    out += [f"_atom_site.{c}" for c in _ATOM_COLUMNS]
    out += [" ".join(row) for row in lines]
    out.append("#")
    return "\n".join(out) + "\n"


def _annotations(rows):
    return pd.DataFrame(rows, columns=["source", "target", "interaction"])


class TestFr3dToGraphSynthetic(unittest.TestCase):
    """fr3d_to_graph on a hand-written mmCIF file, FR3D replaced by a fixed annotation table."""

    ANNOT = [
        ("9XYZ|1|A|G|2", "9XYZ|1|A|G|3", "cWW"),
        ("9XYZ|1|A|G|3", "9XYZ|1|A|G|2", "cWW"),
        ("9XYZ|2|A|G|2", "9XYZ|2|A|G|3", "cWW"),  # second model of an ensemble: must be ignored
        ("9XYZ|1|A|C|2|||A", "9XYZ|1|A|C|1000", "tWH"),  # insertion code
        ("9XYZ|1|A|GTP|1", "9XYZ|1|A|G|3||||7_555", "tWW"),  # contact with the symmetry mate
        ("9XYZ|1|A|G|2||||7_555", "9XYZ|1|A|G|3||||7_555", "cWW"),  # pair within the symmetry mate
        ("9XYZ|1|A|G|2", "9XYZ|1|A|G|99", "cWW"),  # unknown residue
        ("9XYZ|1|A|G|2", "9XYZ|1|A|C|1000", "s35"),  # stacking, not requested
    ]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "9xyz.cif"
        path.write_text(_synthetic_cif())
        with mock.patch.object(f2g, "generatePairwiseAnnotation_import", create=True,
                               return_value=_annotations(cls.ANNOT)), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORGI_PYTHON", None)
            cls.G = f2g.fr3d_to_graph(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_nodes(self):
        asu = {"9xyz.A.1", "9xyz.A.2", "9xyz.A.2A", "9xyz.A.1000", "9xyz.A.3", "9xyz.A.4"}
        sym = {n.replace(".A.", ".A-7-555.") for n in asu}
        self.assertEqual(set(self.G.nodes()), asu | sym)  # no FMN, no protein
        self.assertEqual(self.G.nodes["9xyz.A.1000"]["label_seq_id"], 4)
        self.assertEqual(self.G.nodes["9xyz.A.1"]["nt_full"], "GTP")
        self.assertEqual(self.G.nodes["9xyz.A.4"]["chain_id"], "A")
        self.assertIsNone(self.G.nodes["9xyz.A.1"]["secondary_element"])

    def test_backbone_follows_sequence_and_breaks(self):
        bb = {(u, v) for u, v, d in self.G.edges(data=True) if d["LW"] == "B35" and ".A." in u}
        self.assertEqual(bb, {("9xyz.A.1", "9xyz.A.2"), ("9xyz.A.2", "9xyz.A.2A"), ("9xyz.A.2A", "9xyz.A.1000"),
                              ("9xyz.A.1000", "9xyz.A.3")})
        self.assertEqual(self.G.degree("9xyz.A.4"), 0)  # isolated by the chain break, still a node

    def test_base_pairs(self):
        bps = Counter((u, v, d["LW"]) for u, v, d in self.G.edges(data=True) if d["LW"] not in ("B35", "B53"))
        self.assertEqual(bps, Counter({
            ("9xyz.A.2", "9xyz.A.3", "cWW"): 1,  # model 2 copy of the pair not added again
            ("9xyz.A.3", "9xyz.A.2", "cWW"): 1,
            ("9xyz.A.2A", "9xyz.A.1000", "tWH"): 1,
            ("9xyz.A.1", "9xyz.A-7-555.3", "tWW"): 1,
            ("9xyz.A-7-555.2", "9xyz.A-7-555.3", "cWW"): 1,
        }))

    def test_symmetry_copy_coordinates_and_backbone(self):
        np.testing.assert_allclose(self.G.nodes["9xyz.A-7-555.1"]["xyz_P"], [44.0, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(self.G.nodes["9xyz.A.1"]["xyz_P"], [6.0, 0.0, 0.0], atol=1e-3)
        self.assertTrue(self.G.has_edge("9xyz.A-7-555.2A", "9xyz.A-7-555.1000"))
        self.assertEqual(self.G.nodes["9xyz.A-7-555.1"]["chain_id"], "A-7-555")


class TestFr3dToGraphReal(unittest.TestCase):
    """Full pipeline (FR3D included) on the mmCIF fixtures of tests/data (1a1t reduced to its first 3 models)."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("FORGI_PYTHON", None)

    def tearDown(self):
        self._env.stop()

    def test_nmr_ensemble_first_model_only(self):
        # 1a1t: NMR ensemble; base pairs must come from the first model only, without duplicates
        path = STRUCTURES / "1a1t.cif"
        models = set(MMCIF2Dict.MMCIF2Dict(str(path))["_atom_site.pdbx_PDB_model_num"])
        self.assertGreater(len(models), 1)  # the fixture must keep several models for this test to be meaningful
        G = f2g.fr3d_to_graph(path)
        self.assertIsNotNone(G)
        self.assertEqual(G.number_of_nodes(), 20)
        self.assertEqual(G.number_of_edges(), 54)
        counts = Counter((u, v, d["LW"]) for u, v, d in G.edges(data=True))
        self.assertEqual(max(counts.values()), 1)
        self.assertTrue(all(n.split(".")[1] == "B" for n in G.nodes()))

    def test_ligand_entry_excludes_ligand(self):
        # 1fmn: FMN riboswitch aptamer; the FMN ligand must not become a node
        G = f2g.fr3d_to_graph(STRUCTURES / "1fmn.cif")
        self.assertIsNotNone(G)
        self.assertNotIn("FMN", set(nx.get_node_attributes(G, "nt_full").values()))
        self.assertTrue(all(len(G.nodes[n]["nt_full"]) <= 3 for n in G.nodes()))
        # every backbone edge is a measured covalent linkage, or unmeasurable
        for _, _, d in G.edges(data=True):
            if d["LW"] in ("B35", "B53") and d["o3p_dist"] is not None:
                self.assertLessEqual(d["o3p_dist"], f2g.MAX_O3P_DIST)


if __name__ == "__main__":
    unittest.main()
