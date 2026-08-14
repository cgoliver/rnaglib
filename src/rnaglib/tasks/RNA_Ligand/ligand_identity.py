
import os
import json
import collections
from pathlib import Path

from tqdm import tqdm

from rnaglib.tasks import RNAClassificationTask
from rnaglib.dataset import RNADataset
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import FeaturesComputer, PartitionFromDict, ResolutionFilter
from rnaglib.dataset_transforms import ClusterSplitter, CDHitComputer, StructureDistanceComputer
from rnaglib.tasks.RNA_Ligand.prepare_dataset import PrepareDataset


class LigandIdentification(RNAClassificationTask):
    """Binding pocket-level task where the job is to predict which ligand cluster the small molecule most likely
    to bind a binding pocket with a given structure belongs to.

    Rather than the exact ligand (a task dominated by a handful of over-deposited compounds and unsplittable for the
    rare ones) or a hand-curated ChEBI/ClassyFire family (which mixes chemically unrelated compounds under one label,
    e.g. free amino acids and peptide antibiotics both filed under "amino acid / peptide", and leaves most binding
    sites unlabelled since only 121 of the ~345 ligand codes seen in pockets are curated), the target is a
    Tanimoto-similarity cluster of the ligand's ECFP4 fingerprint, computed by ``cluster_ligands.py`` and stored in
    ``data/ligand_to_cluster.json``. Clusters are built with complete-linkage hierarchical clustering, which bounds
    the *maximum* pairwise distance within a cluster (a real chemical-homogeneity guarantee, unlike a
    connected-components / single-linkage scheme where a cluster is only chain-connected through intermediates), and
    small clusters are merged into their nearest neighbor as long as the merge keeps mean intra-cluster similarity
    above a floor; unmergeable groups are dropped, so the task stays closed (no catch-all class) exactly like the
    old family scheme, but the exclusion is now a systematic byproduct of chemistry rather than a hand pick.

    Task type: multi-class classification
    Task level: substructure-level

    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset (default (15, 500))
    :param tuple[str] admissible_clusters: cluster ids to keep as classes (default: all clusters present in
        ligand_to_cluster.json, i.e. every cluster that survived cluster_ligands.py's merge procedure).
    """
    input_var = "nt_code"
    target_var = "ligand_cluster"
    name = "rna_ligand"
    default_metric = "auc"
    version = "2.0.2"

    def __init__(self,
        size_thresholds=(15, 500),
        graph_path=None,
        admissible_clusters=None,
        **kwargs
    ):
        self.graph_path = graph_path
        meta = {"multi_label": False}

        # create a dict where key is RNA name and values are lists of lists [[residue 1 of binding pocket 1,...,residue N of BP 1],...,[residue 1 of BP k,...]]
        bp_dict_path = os.path.join(os.path.dirname(__file__), "data", "bp_dict.json")
        with open(bp_dict_path, "r") as bp_dict_json:
            self.bp_dict = json.load(bp_dict_json)
        self.nodes_keep = list(self.bp_dict.keys())

        # ligands_dict maps a binding-pocket seed node to the PDB chemical component code of the ligand it contacts
        ligands_dict_path = os.path.join(os.path.dirname(__file__), "data", "ligands_dict.json")
        with open(ligands_dict_path, "r") as ligands_dict_json:
            self.ligands_dict = json.load(ligands_dict_json)

        # cluster_map maps a PDB chemical component code to its Tanimoto-similarity cluster id (see cluster_ligands.py)
        cluster_map_path = os.path.join(os.path.dirname(__file__), "data", "ligand_to_cluster.json")
        with open(cluster_map_path, "r") as cluster_map_json:
            self.cluster_map = json.load(cluster_map_json)

        # default to every cluster that survived cluster_ligands.py's merge/homogeneity procedure
        self.admissible_clusters = admissible_clusters if admissible_clusters is not None \
            else sorted(set(self.cluster_map.values()))
        super().__init__(additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    def process(self) -> RNADataset:
        """
        Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """
        # Initialize dataset with in_memory=False to avoid loading everything at once
        dataset = RNADataset(dataset_path=self.graph_path, in_memory=False, redundancy='all', debug=self.debug, rna_id_subset=self.nodes_keep, version=self.version)

        # Instantiate filters to apply
        resolution_filter = ResolutionFilter(resolution_threshold=4.0)

        # Instantiate transforms to apply
        nt_partition = PartitionFromDict(partition_dict=self.bp_dict)

        # Run through database, applying our filters
        all_binding_pockets = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in tqdm(dataset):
            if resolution_filter.forward(rna):
                for binding_pocket_dict in nt_partition(rna):
                    if self.size_thresholds is not None:
                        if not self.size_filter.forward(binding_pocket_dict):
                            continue
                    pocket = binding_pocket_dict["rna"]
                    # the pocket's ligand is the chemical component labelling its seed nodes. A pocket is grown from a
                    # single ligand's >=10 contact residues, so that ligand dominates the seeds; the majority vote is
                    # robust to the few foreign seeds a BFS expansion may reach in a neighbouring site
                    codes = [self.ligands_dict[node] for node in pocket.nodes() if node in self.ligands_dict]
                    if not codes:
                        continue
                    ligand_code = collections.Counter(codes).most_common(1)[0][0]
                    cluster = self.cluster_map.get(ligand_code)
                    if cluster in self.admissible_clusters or self.debug:
                        pocket.graph[self.target_var] = cluster
                        self.add_rna_to_building_list(all_rnas=all_binding_pockets, rna=pocket)
        dataset = self.create_dataset_from_list(all_binding_pockets)
        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        """Specifies the `FeaturesComputer` object of the tasks which defines the features which have to be added to the RNAs
        (graphs) and nucleotides (graph nodes)
        
        :return: the features computer of the task
        :rtype: FeaturesComputer
        """
        represented_values = set()
        for rna in self.dataset:
            represented_values.add(rna['rna'].graph[self.target_var])
        # sorted, so that a given ligand always maps to the same class index: iterating the set directly makes the
        # mapping depend on string hash randomization, hence on the process, which silently invalidates any per-class
        # metric or checkpoint reloaded in another run
        self.mapping = {target_value: i for i, target_value in enumerate(sorted(represented_values))}
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(mapping=self.mapping)},
        )

    def post_process(self):
        """The task-specific post processing steps to remove redundancy and compute distances which will be used by the splitters.

        This mirrors the default Task.post_process, removing redundancy on sequence (CD-Hit) then on structure
        (US-align). The only difference is that PrepareDataset, rather than RedundancyRemover, is used: it additionally
        drops a structural-similarity cluster whose pockets do not all bind ligands from the same Tanimoto cluster,
        since structure cannot determine the label there and keeping any representative would assert an arbitrary one.
        """
        cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
        cd_hit_rr = PrepareDataset(distance_name="cd_hit", threshold=0.9, target_name=self.target_var)
        self.dataset = cd_hit_computer(self.dataset)
        if self.redundancy_removal:
            self.dataset = cd_hit_rr(self.dataset)

        us_align_computer = StructureDistanceComputer(name="USalign")
        us_align_rr = PrepareDataset(distance_name="USalign", threshold=0.8, target_name=self.target_var)
        self.dataset = us_align_computer(self.dataset)
        if self.redundancy_removal:
            self.dataset = us_align_rr(self.dataset)

        if not self.in_memory:
            # PATCH: delete graphs from dataset/ lost during redundancy removal
            for f in os.listdir(self.dataset.dataset_path):
                if Path(f).stem not in self.dataset.all_rnas:
                    os.remove(Path(self.dataset.dataset_path) / f)
            self.dataset.save_distances()

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        return ClusterSplitter(distance_name="USalign")
