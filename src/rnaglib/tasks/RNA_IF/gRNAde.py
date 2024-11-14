import dill as pickle
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

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


class gRNAde(InverseFolding):
    """This class is a subclass of InverseFolding and is used to train a model on the gRNAde dataset."""

    # everything is inherited except for process and splitter.

    def __init__(self, root, splitter=None, **kwargs):
        data_dir = Path(os.getcwd()) / "data"
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
        data_dir = Path(os.getcwd()) / "data"
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
        train_names = [
            f"{pdb.lower()}_{chain}"  # .upper()
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
        name_filter = NameFilter(
            self.splits["train"] + self.splits["test"] + self.splits["val"]
        )
        chain_filter = ChainFilter(self.splits["pdb_to_chain_all"])
        filters = ComposeFilters([name_filter, chain_filter])

        print("Loading and processing dataset in batches")

        # Initialize dataset with in_memory=False to avoid loading everything at once
        source_dataset = RNADataset(debug=self.debug, redundancy="all", in_memory=False)
        processed_rnas = []

        # Process in batches
        batch_size = 100  # Adjust based on available memory
        total_items = len(source_dataset)

        for batch_start in range(0, total_items, batch_size):  # 0 instead o 1500
            batch_end = min(batch_start + batch_size, total_items)
            print(
                f"Processing batch {batch_start//batch_size + 1}, items {batch_start} to {batch_end}"
            )

            # Create a temporary batch dataset
            batch_rnas = []
            for idx in range(batch_start, batch_end):
                try:
                    item = source_dataset[idx]
                    batch_rnas.append(item)
                except Exception as e:
                    print(f"Error processing item {idx}: {str(e)}")
                    continue

            # Apply filters and transforms to the batch
            if batch_rnas:
                # Filter the batch
                filtered_batch = filters(batch_rnas)

                if filtered_batch:
                    # Create temporary dataset for the filtered batch
                    temp_dataset = RNADataset(rnas=[r["rna"] for r in filtered_batch])
                    # Apply annotations
                    annotated_batch = DummyAnnotator()(temp_dataset)
                    # Split by chain
                    chain_split_batch = ChainSplitTransform()(annotated_batch)
                    # Rename
                    renamed_batch = ChainNameTransform()(chain_split_batch)
                    # Add processed RNAs to our collection
                    processed_rnas.extend([r["rna"] for r in renamed_batch])

                    names = [x.name for x in processed_rnas]
                    duplicates = [x for x in set(names) if names.count(x) > 1]
                    assert not duplicates, f"Found duplicate names: {duplicates}"

            # Optional: Add periodic saving to disk
            if len(processed_rnas) >= 1000:  # Adjust threshold as needed
                temp_dataset = RNADataset(rnas=processed_rnas)
                temp_save_path = f"temp_processed_dataset_{batch_start}.pkl"
                with open(temp_save_path, "wb") as f:
                    pickle.dump(temp_dataset, f)
                processed_rnas = []  # Clear memory

        # If we have any remaining processed RNAs, create final dataset
        if processed_rnas:
            final_dataset = RNADataset(rnas=processed_rnas)
        else:
            # Load and combine all temporary datasets
            processed_rnas = []
            for temp_file in sorted(Path(".").glob("temp_processed_dataset_*.pkl")):
                with open(temp_file, "rb") as f:
                    temp_dataset = pickle.load(f)
                    processed_rnas.extend(temp_dataset.rnas)
                temp_file.unlink()  # Remove temporary file
            final_dataset = RNADataset(rnas=processed_rnas)

        print("Dataset processing completed")
        return final_dataset

    # The below process function is not memory efficient.
    # It should be used once memory is handled better and the above workaround no longer needed

    """
    def process(self) -> RNADataset:

        name_filter = NameFilter(
            self.splits["train"] + self.splits["test"] + self.splits["val"]
        )
        # TODO: return how many rnas are lost using name filter (how many of the ones that should have been in the dataset are not in the dataset)
        # TODO: same thing after the chain filter
        chain_filter = ChainFilter(self.splits["pdb_to_chain_all"])
        filters = ComposeFilters([name_filter, chain_filter])
        print("loading dataset (this can take a while)")
        dataset = RNADataset(debug=self.debug, redundancy="all")
        print("filtering dataset")
        rnas = filters(dataset)
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        print("annotating dataset")
        rnas = DummyAnnotator()(dataset)
        print("splitting dataset by chain")
        rnas = ChainSplitTransform()(rnas)
        print("renaming dataset")
        rnas = ChainNameTransform()(rnas)
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        print("dataset processed")
        return dataset
        """
