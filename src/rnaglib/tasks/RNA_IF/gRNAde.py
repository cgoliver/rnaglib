import os
from pathlib import Path
from collections import defaultdict

from rnaglib.tasks import InverseFolding
from rnaglib.data_loading import RNADataset
from rnaglib.transforms import (
    DummyAnnotator,
    NameFilter,
    ChainFilter,
    ChainSplitTransform,
    ChainNameTransform,
    ComposeFilters,
)
from rnaglib.splitters import NameSplitter
from rnaglib.utils import dump_json



class gRNAde(InverseFolding):
    """This class is a subclass of InverseFolding and is used to train a model on the gRNAde dataset."""

    # everything is inherited except for process and splitter.

    def __init__(self, root, splitter=None, **kwargs):
        self.splits = {
            "train": [],
            "test": [],
            "val": [],
            "all": [],
            # Use sets instead of lists for chains since order doesn't matter
            "pdb_to_chain_train": defaultdict(set),
            "pdb_to_chain_test": defaultdict(set),
            "pdb_to_chain_val": defaultdict(set),
            "pdb_to_chain_all": defaultdict(set),
        }
        # Populate the structure
        data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
        for split in ["train", "test", "val"]:
            file_path = data_dir / f"{split}_ids_das.txt"
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    pdb_id = line.split("_")[0].lower()
                    chain = line.split("_")[-1]  # .upper()
                    chain_components = list(chain.split("-"))
                    # [c.upper() for c in chain.split("-")]

                    if pdb_id not in self.splits[split]:
                        self.splits[split].append(pdb_id)
                    if pdb_id not in self.splits["all"]:
                        self.splits["all"].append(pdb_id)

                    # Using update for sets automatically handles duplicates
                    self.splits[f"pdb_to_chain_{split}"][pdb_id].update(
                        chain_components
                    )
                    self.splits["pdb_to_chain_all"][pdb_id].update(chain_components)

        super().__init__(root=root, splitter=splitter, **kwargs)

    @property
    def default_splitter(self):
        train_names = [            f"{pdb.lower()}_{chain}"  # .upper()
            for pdb in self.splits["pdb_to_chain_train"]
            for chain in self.splits["pdb_to_chain_train"][pdb]
        ]

        val_names = [
            f"{pdb.lower()}_{chain}"  # .upper()
            for pdb in self.splits["pdb_to_chain_val"]
            for chain in self.splits["pdb_to_chain_val"][pdb]
        ]

        test_names = [
            f"{pdb.lower()}_{chain}"  # .upper()
            for pdb in self.splits["pdb_to_chain_test"]
            for chain in self.splits["pdb_to_chain_test"][pdb]
        ]

        return NameSplitter(train_names, val_names, test_names)

    def process(self) -> RNADataset:
        """
        Process the dataset in batches to avoid memory issues.
        Returns a filtered and processed RNADataset.
        """
        name_filter = NameFilter(self.splits["train"] + self.splits["test"] + self.splits["val"])
        chain_filter = ChainFilter(self.splits["pdb_to_chain_all"])
        filters = ComposeFilters([name_filter, chain_filter])

        annote_dummy = DummyAnnotator()
        split_chain = ChainSplitTransform()
        add_name_chains = ChainNameTransform()

        # Initialize dataset with in_memory=False to avoid loading everything at once
        source_dataset = RNADataset(debug=self.debug, redundancy="all", in_memory=False)

        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        import tqdm
        for rna in tqdm.tqdm(source_dataset):
            if filters.forward(rna):
                rna = annote_dummy(rna)
                rna_chains = split_chain(rna) # Split by chain
                renamed_chains = list(add_name_chains(rna_chains)) # Rename
                for rna_chain in renamed_chains:
                    rna_chain = rna_chain["rna"]
                    if self.in_memory:
                        all_rnas.append(rna_chain)
                    else:
                        all_rnas.append(rna_chain.name)
                        dump_json(os.path.join(self.dataset_path, f"{rna_chain.name}.json"), rna_chain)
        if self.in_memory:
            dataset = RNADataset(rnas=all_rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_rnas)
        return dataset