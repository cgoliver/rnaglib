import os
import json
from typing import Union, Optional
import gemmi
from torch.utils.data import Subset


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

def load_index(redundancy='nr', version='1.0.0', glib_path=f"{os.path.expanduser('~')}/.rnaglib"):
    index_file = os.path.join(glib_path, f"indexes/rnaglib-{redundancy}-{version}.json")

    try:
        with open(index_file, 'r') as indx:
            return json.load(indx)
    except FileNotFoundError:
        print(f"Index file not found at {index_file}. Run rnaglib_index")
        return None

def cif_remove_residues(cif_path: Union[str, os.PathLike],
                        keep_residues: Optional[list],
                        out_path: Union[str, os.PathLike]
                        ):
    """ Remove all residues from a cif file except for those in `keep_residues` list.
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
