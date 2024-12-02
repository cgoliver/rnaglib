import os
import json
from typing import Union, Optional
from pathlib import Path
import gemmi
from torch.utils.data import Subset
import torch


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def tonumpy(torch_array):
    """
    Routine to get any torch tensor back to numpy without cpu or gradient problems

    :param torch_array: A torch array

    :return: A numpy array
    """
    return torch_array.detach().cpu().numpy()


def get_dataset(loader):
    """
    Using torch subsets have a weird property that messes up the dataset accessing.
     This is the recommended fix in https://discuss.pytorch.org/t/how-to-get-a-part-of-datasets/82161/7

    :param dataset: either a dataset or a Subset
    :param node_sim:

    :return:
    """
    if isinstance(loader, Subset):
        return loader.dataset.dataset
    else:
        return loader.dataset


def load_index(redundancy="nr", version="1.0.0", glib_path=f"{os.path.expanduser('~')}/.rnaglib"):
    index_file = os.path.join(glib_path, f"indexes/rnaglib-{redundancy}-{version}.json")

    try:
        with open(index_file, "r") as indx:
            return json.load(indx)
    except FileNotFoundError:
        print(f"Index file not found at {index_file}. Run rnaglib_index")
        return None


def cif_remove_residues(
    cif_path: Union[str, os.PathLike],
    keep_residues: Optional[list],
    out_path: Union[str, os.PathLike],
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
        for chain in model:
            for res in chain:
                if (chain.name, res.seqid.num) not in keep_residues:
                    del res

    # Save the modified structure to a new mmCIF file
    cif_model.make_mmcif_document().write_file(str(out_path))
    pass


def clean_mmcif(
    cif_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike] = ".",
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

    with open(output_path, "w") as f:
        f.write(chain_structure.make_mmcif_document().as_string())


def split_mmcif_by_chain(
    cif_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike] = ".",
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
