from collections import defaultdict
import requests

from rnaglib.dataset_transforms import random_split

SPLITTING_VARS = {
    "TR60": [
        "3sktA",
        "5u3gB",
        "5j02A",
        "2yieZ",
        "2fcyA",
        "3gx3A",
        "4nybA",
        "1hr2A",
        "4mgmB",
        "3oxeB",
        "1y90B",
        "2quwB",
        "4megB",
        "4lvxA",
        "4rgeB",
        "4pcjA",
        "3c44A",
        "5o69A",
        "2lwkA",
        "3vrsA",
        "2g5kA",
        "5fj1C",
        "5d5lD",
        "4frgX",
        "1ylsB",
        "3q50A",
        "4xw7A",
        "2ktzA",
        "4qlmA",
        "3fu2A",
        "5dh8B",
        "3meiB",
        "6fz0A",
        "2mxsA",
        "2nokC",
        "1ajuA",
        "1fypA",
        "4k31C",
        "1ntbA",
        "3bnqC",
        "5vciA",
        "3q3zV",
        "1uudB",
        "1byjA",
        "1lvjA",
        "1utsB",
        "1qd3A",
        "1arjN",
        "2l8hA",
        "6hagA",
        "1yrjA",
        "1tobA",
        "1f1tA",
        "3tzrA",
        "4qjhC",
        "2kgpA",
        "1rawA",
        "1ehtA",
        "1nbkA",
        "1ei2A",
    ],  # nok is a duplicate. RLBind uses chain C only. ,'2nokB'
    # 1f1tA is duplicated
    "TE18": [
        "2pwtA",
        "5v3fA",
        "379dB",
        "5bjoE",
        "4pqvA",
        "430dA",
        "1nemA",
        "1q8nA",
        "1f1tA",
        "2jukA",
        "4yazR",
        "364dC",
        "6ez0A",
        "2tobA",
        "1ddyA",
        "1fmnA",
        "2misA",
        "4f8uB",
    ],
}

SPLITTING_VARS["ID_TR60_TE18"] = set(SPLITTING_VARS["TR60"] + SPLITTING_VARS["TE18"])
id_to_chains = defaultdict(list)
for pdb_chain in SPLITTING_VARS["ID_TR60_TE18"]:
    pdb, chain = pdb_chain[:4], pdb_chain[4:]
    id_to_chains[pdb].append(chain)
SPLITTING_VARS["PDB_TO_CHAIN_TR60_TE18"] = id_to_chains


class Splitter:
    def __init__(self, split_train=0.7, split_valid=0.15, split_test=0.15,
                 debug=False):
        assert (
            sum([split_train, split_valid, split_test]) == 1
        ), "Splits don't sum to 1."
        self.split_train = split_train
        self.split_valid = split_valid
        self.split_test = split_test
        self.debug = debug
        pass

    def __call__(self, dataset):
        train, val, test = self.forward(dataset)
        if sum(map(len, [train, val, test])) != len(dataset):
            print(
                f"""WARNING: splitter dropped some data points.
                    Original dataset had: {len(dataset)}
                    Train split: {len(train)}
                    Validation split: {len(val)}
                    Test split: {len(test)}\n"""
            )

        return train, val, test

    def forward(self, dataset):
        raise NotImplementedError


class RandomSplitter(Splitter):
    """ " Just split a dataset randomly. Reproducible through the ``seed`` argument.

    :param seed: Seed for shuffling.
    """

    def __init__(self, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def forward(self, dataset):
        return random_split(
            dataset,
            split_train=self.split_train,
            split_valid=self.split_valid,
            seed=self.seed,
        )


class NameSplitter(Splitter):
    def __init__(self, train_names, val_names, test_names, **kwargs):
        super().__init__(**kwargs)
        self.train_names = train_names
        self.val_names = val_names
        self.test_names = test_names

    def forward(self, dataset):
        dataset_map = dataset.all_rnas
        train_ind = [
            dataset_map[name] for name in self.train_names if name in dataset_map
        ]
        val_ind = [dataset_map[name] for name in self.val_names if name in dataset_map]
        test_ind = [
            dataset_map[name] for name in self.test_names if name in dataset_map
        ]
        return train_ind, val_ind, test_ind


def default_splitter_tr60_tr18():
    train_names = [f"{name[:-1]}_{name[-1]}" for name in SPLITTING_VARS["TR60"][:-6]]
    val_names = [f"{name[:-1]}_{name[-1]}" for name in SPLITTING_VARS["TR60"][-6:]]
    test_names = [
        f"{name[:-1]}_{name[-1]}" for name in SPLITTING_VARS["TE18"] if name != "1f1tA"
    ]
    return NameSplitter(train_names, val_names, test_names)


def get_ribosomal_rnas():
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct_keywords.pdbx_keywords",
                "operator": "contains_phrase",
                "value": "ribosome",
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    response = requests.post(url, json=query)
    if response.status_code == 200:
        data = response.json()
        ribosomal_rnas = set([result["identifier"] for result in data["result_set"]])
        return ribosomal_rnas
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        print(response.text)
        return []
