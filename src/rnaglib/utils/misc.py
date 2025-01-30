import json
import os
from pathlib import Path

import gemmi
from torch.utils.data import Subset


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def tonumpy(torch_array):
    """Routine to get any torch tensor back to numpy without cpu or gradient problems

    :param torch_array: A torch array

    :return: A numpy array
    """
    return torch_array.detach().cpu().numpy()


def get_dataset(loader):
    """Using torch subsets have a weird property that messes up the dataset accessing.
     This is the recommended fix in https://discuss.pytorch.org/t/how-to-get-a-part-of-datasets/82161/7

    :param dataset: either a dataset or a Subset
    :param node_sim:

    :return:
    """
    if isinstance(loader, Subset):
        return loader.dataset.dataset
    return loader.dataset


def load_index(redundancy="nr", version="1.0.0", glib_path=f"{os.path.expanduser('~')}/.rnaglib"):
    index_file = os.path.join(glib_path, f"indexes/rnaglib-{redundancy}-{version}.json")

    try:
        with open(index_file) as indx:
            return json.load(indx)
    except FileNotFoundError:
        print(f"Index file not found at {index_file}. Run rnaglib_index")
        return None


def cif_remove_residues(
    cif_path: str | os.PathLike,
    keep_residues: list | None,
    out_path: str | os.PathLike,
    file_type: str = "cif",
):
    """Remove all residues from a cif file except for those in `keep_residues` list.

    Save the new cif to `out_path`.
    :param cif_path: path to input cif
    :param keep_residues: list of residue IDs in format (chain_id, position) to keep.
    :param out_path: path to write new cif file
    """
    # Load the mmCIF file
    cif_model = gemmi.read_structure(str(cif_path))

    # Iterate through models and chains
    for model in cif_model:
        empty_chains = []
        for chain in model:
            for i in range(len(chain) - 1, -1, -1):
                res = chain[i]
                if (chain.name, int(res.seqid.num)) not in keep_residues:
                    del chain[i]
            # Mark chain for removal if it's empty
            if len(chain) == 0:
                empty_chains.append(chain.name)

        # Remove empty chains
        for chain_name in empty_chains:
            model.remove_chain(chain_name)
    # Save the modified structure to a new mmCIF or PDB file
    if file_type == "cif":
        cif_model.make_mmcif_document().write_file(str(out_path))
    else:
        cif_model.write_pdb(str(out_path))


def filter_cif_with_res(cif_path, keep_residues, out_path, file_type="cif"):
    # Read the CIF file and keep only the first model.
    structure = gemmi.read_structure(str(cif_path))
    if len(structure) > 1:
        del structure[1:]
    try:
        model = structure[0]
    except IndexError:
        model = structure

    # Hash things to keep
    keep_residues = set(keep_residues)
    keep_chains = set([x[0] for x in keep_residues])

    # Iterate over all residues and keep only the allowed ones
    # Use reverse to avoid index errors after deletions
    for i, chain in reversed(list(enumerate(model))):
        # Skip the whole chain if not using it
        if chain.name not in keep_chains:
            del model[i]
            continue
        for i, res in reversed(list(enumerate(chain))):
            if (chain.name, int(res.seqid.num)) not in keep_residues:
                del chain[i]

    structure.remove_empty_chains()
    # Save the modified structure to a new mmCIF or PDB file
    if file_type == "cif":
        structure.make_mmcif_document().write_file(str(out_path))
    else:
        structure.write_pdb(str(out_path))


def clean_mmcif(
    cif_path: str | os.PathLike,
    output_path: str | os.PathLike = ".",
    file_type: str = "cif",
):
    """Remove non-RNA entities.

    :param cif_path: path to input cif
    :param output_path: path to cleaned structure.
    """
    # Load the MMCIF file
    structure = gemmi.read_structure(str(cif_path))
    rna_residues = ["A", "U", "C", "G"]

    chain_structure = gemmi.Structure()
    chain_structure.add_model(gemmi.Model("X"))
    # Iterate over all models and chains
    paths = []
    for chain in structure[0]:
        if not any(res.name in rna_residues for res in chain):
            continue
        # Add the chain to the new structure
        chain_copy = chain.clone()
        chain_structure[0].add_chain(chain_copy)

    if file_type == "cif":
        with open(output_path, "w") as f:
            f.write(chain_structure.make_mmcif_document().as_string())
    else:
        with open(output_path, "w") as f:
            f.write(chain_structure.make_pdb_string())


def subset_mmcif_biopython(input_cif, allowed_residues, output_cif):
    from Bio.PDB import FastMMCIFParser, Select, PDBIO

    class ResidueSelector(Select):
        """Selector to filter residues by chain and residue ID."""

        def __init__(self, allowed_residues):
            super().__init__()
            self.allowed_residues = set(allowed_residues)  # List of (chain_id, res_id)

        def accept_residue(self, residue):
            chain_id = residue.parent.id  # Parent of residue is a chain
            res_id = residue.id[1]  # Residue ID is a tuple, use the numeric part
            return (chain_id, res_id) in self.allowed_residues

    # Parse the CIF file
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", input_cif)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_cif, ResidueSelector(allowed_residues))


def split_mmcif_by_chain(
    cif_path: str | os.PathLike,
    output_dir: str | os.PathLike = ".",
    prefix=None,
    min_length=0,
    max_length=1000,
):
    """Write one mmCIF for each chain in the input mmCIF.

    :param cif_path: path to input cif
    :param prefix: string to use as the name for new mmCIFs followed by chain ID. If None, use input file name.
    :param output_dir: Directory for dumping new mmCIFs.
    :param out_path: path to write new cif file

    :return: list of paths to new chain PDBs
    """
    if prefix is None:
        prefix = Path(cif_path).stem
    # Load the MMCIF file
    structure = gemmi.read_structure(str(cif_path))
    rna_residues = ["A", "U", "C", "G"]

    # Iterate over all models and chains
    paths = []
    for chain in structure[0]:
        if (len(chain) < min_length) or (len(chain) > max_length):
            continue
        if not any(res.name in rna_residues for res in chain):
            continue
        # Create a new structure for the chain
        chain_structure = gemmi.Structure()
        chain_structure.add_model(gemmi.Model(chain.name))

        # Add the chain to the new structure
        chain_copy = chain.clone()
        chain_structure[0].add_chain(chain_copy)

        # Output file name
        output_file = Path(output_dir) / f"{prefix}_{chain.name}.cif"
        # Write the chain to a new MMCIF file
        paths.append(output_file)
        with open(output_file, "w") as f:
            f.write(chain_structure.make_mmcif_document().as_string())
    return paths
