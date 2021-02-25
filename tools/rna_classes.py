from collections import OrderedDict
import logging


class Atom(object):
    def __init__(self, atom_type, atom_label, x, y, z):
        self.atom_type = atom_type
        self.atom_label = atom_label
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"{self.atom_type} {self.atom_label} {self.x} {self.y} {self.z}"


class Nucleotide(object):
    def __init__(self, pos, nt, real_nt, chemically_modified, pdb_pos, pdb_pos_ins):
        self.pos = pos
        self.nt = nt
        self.real_nt = real_nt
        self.chemically_modified = chemically_modified
        self.pdb_pos = pdb_pos
        self.pdb_pos_ins = pdb_pos_ins
        self.auth_seq_id = None
        self.atoms = []


    def add_atom(self, atom):
        if not isinstance(atom, Atom):
            logging.debug(f"Trying to insert non Atom object in strand {strand} \
                          nucleotide at pos {pos}.\nThe atom to be inserted is:\n{atom}")
            raise Exception("This is not an Atom object")
        self.atoms.append(atom)


    def __repr__(self):
        return f"{self.nt} {self.real_nt} {self.chemically_modified} {self.pdb_pos} {self.pdb_pos_ins} {self.atoms}"


class Strand(dict):

    def __init__(self, *args, **kwargs):#, name, entity_id, description):
        super().__init__()
        if kwargs:
            self.name = kwargs.get('name', '')
            self.description = kwargs.get('description', '')
            self.entity_id = kwargs.get('entity_id', '')


    def __setitem__(self, key, val):
        if not isinstance(val, Nucleotide):
            logging.debug(f"Trying to add a non Nucleotide object to Strand.\n{val}")
            raise Exception(f"Trying to add a non Nucleotide object to RNA_Strand.\n{val}")
        if not isinstance(key, int):
            logging.debug(f"Trying to add a Nucleotide with non-integer key to strand\n{key}")
            raise Exception(f"Trying to add a Nucleotide with non-integer key to strand\n{key}")
        super().__setitem__(key, val)


class RNA_Molecule(dict):

    def __init__(self, pdb_id, title):
        self.pdb_id = pdb_id
        self.title = title
        self.fr3d_graph = None
        super().__init__()


    def __setitem__(self, key, val):
        if not isinstance(val, Strand):
            logging.debug(f"Trying to add a non Strand object to RNA_Molecule.\n{val}")
            raise Exception(f"Trying to add a non Strand object to RNA_Molecule.\n{val}")
        super().__setitem__(key, val)
