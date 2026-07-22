import os

from rnaglib.dataset import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.encoders import MultiLabelOneHotEncoder
from rnaglib.transforms import FeaturesComputer
from rnaglib.dataset_transforms import ClusterSplitter, CDHitComputer, StructureDistanceComputer
from rnaglib.utils.rfam_utils import pdb_sel_to_rfam


class RNAGo(RNAClassificationTask):
    """Predict the functional class of a given RNA chain from its 3D structure.

    Individual Rfam-derived GO terms are, by construction, near-synonyms of Rfam family membership
    (e.g. GO:0005682 "U5 snRNP" is carried by exactly one family, RF00020), which makes single-GO-term
    prediction solvable from sequence/family identity alone rather than from structure. To get a task
    that actually probes 3D structural generalization, individual GO terms are grouped into 5 curated,
    multi-family functional classes, each spanning several Rfam families/species so a model has to
    generalize across genuine fold variation rather than fingerprint one family:

    - ``ribosome``: rRNA components (5S, 5.8S, SSU/LSU across bacteria/archaea/eukarya/microsporidia/
      trypanosome mitochondria). Rfam: RF00001, RF00002, RF00177, RF01959, RF01960, RF02540, RF02541,
      RF02542, RF02543, RF02545, RF02546. Source GO: 0003735, 0005840.
    - ``trna``: tRNA and tRNA-like/tRNA-derived elements. Rfam: RF00005 (tRNA), RF00233
      (Tymo_tRNA-like), RF00023 (tmRNA). Source GO: 0030533, 0006401.
    - ``spliceosome``: spliceosomal snRNAs (U1, U2, U4, U5, U6, U12, U6atac). Rfam: RF00003, RF00004,
      RF00007, RF00015, RF00020, RF00026, RF00619. Source GO: 0000244, 0000348, 0000353, 0045131,
      0046540, and the individual U-snRNA GO terms (e.g. 0005682, 0005688, 0030621).
    - ``riboswitch``: ligand-sensing cis-regulatory aptamers. Rfam: RF00162, RF00234, RF00379, RF00380,
      RF00442, RF00634, RF01689, RF01725, RF01763, RF01786, RF01826. Source GO: 0010468 (every Rfam
      family carrying this GO term, among those with solved PDB structures, happens to be a riboswitch
      class -- this is an empirical property of what's been deposited, not a restriction implied by the
      GO term itself).
    - ``ribozyme``: self-splicing introns and RNase P/MRP catalytic RNAs. Rfam: RF00028, RF00029,
      RF00030, RF00009, RF00010. Source GO: 0000372, 0000373, 0004526, 0008033.

    These 5 classes are pairwise disjoint at the Rfam-family level. Support in the PDB is uneven
    (``ribosome``/``trna`` are dominated by a couple of very frequent families such as 5S rRNA and
    cytoplasmic tRNA; ``riboswitch``/``ribozyme`` have the fewest structures but the most balanced
    per-family diversity) -- this is real biological/deposition imbalance, not filtered away here.

    Task type: multi-class (multi-label encoded) classification
    Task level: RNA-level

    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset(default (15, 500))
    """

    FUNCTION_CLASSES = {
        "ribosome": [
            "RF00001", "RF00002", "RF00177", "RF01959", "RF01960",
            "RF02540", "RF02541", "RF02542", "RF02543", "RF02545", "RF02546",
        ],
        "trna": ["RF00005", "RF00233", "RF00023"],
        "spliceosome": ["RF00003", "RF00004", "RF00007", "RF00015", "RF00020", "RF00026", "RF00619"],
        "riboswitch": [
            "RF00162", "RF00234", "RF00379", "RF00380", "RF00442",
            "RF00634", "RF01689", "RF01725", "RF01763", "RF01786", "RF01826",
        ],
        "ribozyme": ["RF00028", "RF00029", "RF00030", "RF00009", "RF00010"],
    }

    input_var = "nt_code"  # node level attribute
    target_var = "function_class"  # graph level attribute
    name = "rna_go"
    version = "2.0.2"
    default_metric = "jaccard"

    def __init__(self, size_thresholds=(15, 500), **kwargs):
        meta = {"multi_label": True}
        super().__init__(additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        return ClusterSplitter(similarity_threshold=0.5, distance_name="USalign")

    def get_task_vars(self):
        """Specifies the `FeaturesComputer` object of the tasks which defines the features which have to be added to the RNAs
        (graphs) and nucleotides (graph nodes)

        :return: the features computer of the task
        :rtype: FeaturesComputer
        """
        label_mapping = self.metadata["label_mapping"]
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var:
                                 MultiLabelOneHotEncoder(label_mapping)}, )

    def process(self) -> RNADataset:
        """
        Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """
        fam_to_class = {rfam: cls for cls, fams in self.FUNCTION_CLASSES.items() for rfam in fams}

        # Only pull pdb selections whose Rfam family belongs to one of the 5 curated function classes.
        df = pdb_sel_to_rfam()
        df = df[df['rfam_acc'].isin(fam_to_class)]

        dataset = RNADataset(redundancy='nr', debug=self.debug, in_memory=self.in_memory, rna_id_subset=df['pdb_id'].unique(), version=self.version)

        # Create dataset
        # Run through database, applying our filters
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            rna_graph = rna['rna']
            lines = df.loc[df['pdb_id'] == rna_graph.name]
            for pdbsel in lines['pdbsel'].unique():
                pdb = pdbsel.split('_')[0]
                # Subset the graph on the RFAM labeled part.
                # RFAM has a weird numbering convention: they give chain ids that seem like a range
                # and not the actual numbering. Hence, if you have residues [110, 111... 160], the RFAM
                # numbering can look like 2-16. I assume this is residues 111-125.
                _, chain, start, end = pdbsel.split('_')
                pdb_chain_numbers = [node_name for node_name in list(sorted(rna_graph.nodes())) if
                                     node_name.startswith(f'{pdb}.{chain}')]
                chunk_nodes = pdb_chain_numbers[int(start) - 1: int(end)]
                subgraph = rna_graph.subgraph(chunk_nodes).copy()
                subgraph.name = pdbsel

                # Get the function class(es) for this RFAM selection.
                # Needs a bit of caution because one pdbsel could have more than one rfam_id, but since
                # FUNCTION_CLASSES partitions Rfam families, this resolves to a single class in practice.
                rfams_pdbsel = lines.loc[lines['pdbsel'] == pdbsel]['rfam_acc'].values
                function_classes = sorted({fam_to_class[rfam_id] for rfam_id in rfams_pdbsel})

                # Finally, apply quality filters
                if len(subgraph) < 5 or len(subgraph.edges()) < 5:
                    continue
                # A feature dict (including structure path) is needed for the filtering.
                # SizeFilter.forward reads rna_dict["rna"], so that key (not "graph") must be
                # replaced with the extracted fragment, otherwise the filter silently sizes the
                # whole parent chain/assembly instead of this pdbsel's fragment.
                chunk_dict = {k: v for k, v in rna.items() if k != 'rna'}
                chunk_dict['rna'] = subgraph
                if self.size_thresholds is not None:
                    if not self.size_filter.forward(chunk_dict):
                        continue

                subgraph.graph['function_class'] = function_classes
                self.add_rna_to_building_list(all_rnas=all_rnas, rna=subgraph)
        dataset = self.create_dataset_from_list(all_rnas)

        # The label vocabulary is the fixed set of 5 curated classes, not derived from a frequency
        # threshold on whatever happened to get built (unlike the old per-GO-term scheme), so it is
        # stable across debug/full runs.
        self.metadata["label_mapping"] = {cls: i for i, cls in enumerate(sorted(self.FUNCTION_CLASSES))}
        return dataset

    def post_process(self):
        """
        Computes sequence similarity between all atom pairs using CD-Hit, and structure-based
        pairwise similarity using USalign (needed by the USalign-based ClusterSplitter used as
        this task's default splitter).
        """
        cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
        self.dataset = cd_hit_computer(self.dataset)

        us_align_computer = StructureDistanceComputer(name="USalign")
        self.dataset = us_align_computer(self.dataset)