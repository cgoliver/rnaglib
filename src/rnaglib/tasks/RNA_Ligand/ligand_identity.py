
import os
import json
import collections
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from scipy.sparse.csgraph import connected_components

from rnaglib.tasks import RNAClassificationTask
from rnaglib.dataset import RNADataset
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import FeaturesComputer, PartitionFromDict
from rnaglib.transforms.annotate.small_molecule import CRYSTALLIZATION_ADDITIVES, POLYAMINES
from rnaglib.dataset_transforms import ClusterSplitter, CDHitComputer, StructureDistanceComputer
from rnaglib.tasks.RNA_Ligand.prepare_dataset import PrepareDataset


class LigandIdentification(RNAClassificationTask):
    """Binding pocket-level task where the job is to predict which of a small set of ligand classes a binding
    pocket binds, given only its structure.

    Classes are ligand-chemistry clusters computed live from the data rather than hand-picked: every admissible
    ligand code present in the corpus is clustered against every other by Tanimoto similarity of its Morgan
    fingerprint (connected components at similarity >= ``tanimoto_threshold``), and every pocket is labelled with
    its ligand's cluster. Only the ``n_classes`` largest clusters -- by pocket count *after* the structural
    ambiguity/redundancy pass below, not raw chemical-cluster size -- are kept as classes; this two-stage
    filtering matters because a chemical cluster's raw size doesn't predict how many of its pockets survive as
    structurally unambiguous.

    This replaces an earlier, hand-curated version of this task (a fixed set of 7 individual antibiotic codes,
    e.g. PAR, LLL, NMY -- see git history) with a computed criterion. Classes don't need to match any
    expert/taxonomic category (e.g. "this is the SAM riboswitch family") to be a valid target here -- the only
    requirement is that pockets sharing a class are more structurally similar to each other than to a random
    pocket, since structure is all the model ever sees. That's directly measurable from the US-align distances
    already computed for redundancy removal: for a set of same-class pockets, compare their mean pairwise
    US-align similarity to the mean similarity of random pocket pairs (the "background"). Every class this task
    can produce clears that bar (see ``tanimoto_redesign/`` for the measurement), including ones that merge
    ligands from textbook-distinct RNA families (e.g. SAM/NAD/plain purine nucleotides all sharing an
    adenine-stacking sub-pocket) -- an earlier draft of this docstring wrongly called such a cluster
    "meaningless" on taxonomic grounds without checking this, before the check showed otherwise.

    Both parameters were chosen empirically, not picked a priori -- see ``src/rnaglib/tasks/RNA_Ligand/
    tanimoto_redesign/`` for the sweep (a threshold-vs-class-size analysis over thresholds 0.6-0.9, plus the
    intra-class-vs-background US-align similarity check) that motivated them:

    - ``tanimoto_threshold=0.6``: at the default ``n_classes=3``, thresholds 0.6/0.7/0.8 all produce classes that
      clear the background-similarity bar by a comparable margin (intra-class mean US-align similarity 0.23-0.40
      against a ~0.19-0.20 background at every threshold; the weakest class measured, the 0.6 purine/adenine
      cluster, still sits at 0.243, and a majority to two-thirds of its pairs individually beat the background
      mean depending on threshold). Since raising the threshold doesn't buy a clearly stronger structural signal
      here, 0.6 was picked instead for maximizing samples per class: its top-3 clusters total ~870
      post-redundancy-removal pockets, versus ~648 at 0.7 and ~539 at 0.8, because a lower threshold keeps more
      minor congeners merged into the two aminoglycoside-adjacent clusters. Re-run the similarity check in
      ``tanimoto_redesign/`` before lowering it further -- the margin over background is not guaranteed to hold
      at every threshold, only checked for 0.6-0.9.
    - ``n_classes=3``: with only a handful of classes, every one of them needs to be large enough to learn from.
      At ``tanimoto_threshold=0.6`` the top 3 clusters by post-redundancy-removal count are, in order: the core
      deoxystreptamine aminoglycoside family (paromomycin/PAR, neomycin/NMY, and congeners -- dominant, capped by
      ``max_per_class``), a cluster of purine/adenine-ring ligands (SAM, NAD, and plain adenine/guanine
      nucleotides), and the gentamicin/sisomicin aminoglycoside family (LLL, GET, and congeners). The next-largest
      cluster beyond that is a small dinucleotide-analog family (~47 pockets, well below the top 3); growing
      ``n_classes`` past 3 is possible but shrinks the smallest included class fast, so 3 was picked to keep every
      class reasonably sized. Changing ``tanimoto_threshold`` changes which exact clusters compete for the top
      ``n_classes`` slots, so re-run the sweep in ``tanimoto_redesign/`` before changing either parameter.

    Ligand codes that bind non-specifically (crystallization additives, polyamines -- see
    ``EXCLUDED_LIGAND_CODES``) are dropped before clustering: they bind the sugar-phosphate backbone through
    generic electrostatic contacts rather than a shape-complementary pocket (HARIBOSS's own annotation already
    flags polyamine pockets as atypically small/generic), so classes built from them would have little of the
    structural coherence this task is trying to select for. A code with no resolvable SMILES (missing from
    ``data/ligand_smiles.json``) can't be clustered and is dropped too.

    Task type: multi-class classification
    Task level: substructure-level

    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset (default (15, 500))
    :param float tanimoto_threshold: Tanimoto-similarity threshold (Morgan fingerprints, connected components)
        used to cluster ligand codes into classes (default 0.6, see class docstring for why)
    :param int n_classes: number of largest post-redundancy-removal Tanimoto clusters to keep as classes
        (default 3, see class docstring for why)
    :param bool collapse_structural_duplicates: if True, the US-align redundancy-removal step reduces each
        label-consistent structural cluster to its single highest-resolution representative. If False (default
        here -- unlike the general Task default), all members of a label-consistent cluster are kept (mixed-label
        clusters are still dropped either way).
    :param int max_per_class: after redundancy removal and class selection, cap any class exceeding this many
        pockets to its ``max_per_class`` highest-resolution members (default 120, matching the size of the
        smallest of the top-3 classes at the default threshold -- see class docstring). Set to None to disable
        capping. The dominant aminoglycoside cluster is expected to need this regardless of ``tanimoto_threshold``
        or ``n_classes``: even labelled by exact ligand code with no clustering at all, paromomycin (PAR) alone
        survives redundancy removal at ~280 pockets, more than every other single ligand combined.
    """
    input_var = "nt_code"
    target_var = "ligand_class"
    name = "rna_ligand"
    default_metric = "auc"
    version = "2.0.2"

    # Non-specific binders excluded from the ligand pool: these bind the sugar-phosphate backbone through
    # generic electrostatic contacts rather than a shape-complementary pocket (see class docstring), so
    # classes built from them would have little structural coherence.
    EXCLUDED_LIGAND_CODES = CRYSTALLIZATION_ADDITIVES | POLYAMINES

    MORGAN_RADIUS = 2
    MORGAN_NBITS = 2048

    def __init__(self,
        size_thresholds=(15, 500),
        graph_path=None,
        tanimoto_threshold=0.6,
        n_classes=3,
        collapse_structural_duplicates=False,
        max_per_class=120,
        **kwargs
    ):
        self.graph_path = graph_path
        self.tanimoto_threshold = tanimoto_threshold
        self.n_classes = n_classes
        self.collapse_structural_duplicates = collapse_structural_duplicates
        self.max_per_class = max_per_class
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

        # ligand_smiles maps a PDB chemical component code to a SMILES string, used to cluster ligands by
        # Tanimoto similarity (see _cluster_ligands_by_tanimoto)
        ligand_smiles_path = os.path.join(os.path.dirname(__file__), "data", "ligand_smiles.json")
        with open(ligand_smiles_path, "r") as ligand_smiles_json:
            self.ligand_smiles = json.load(ligand_smiles_json)

        # populated by process(): maps each ligand code present in the built corpus to its Tanimoto cluster
        # label, kept around for introspection (e.g. to see which real ligands ended up in a given class)
        self.tanimoto_clusters = {}

        super().__init__(additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    def _cluster_ligands_by_tanimoto(self, ligand_codes):
        """Clusters ligand codes by Tanimoto similarity of their Morgan fingerprints: connected components at
        similarity >= ``self.tanimoto_threshold``, the same clustering style ``PrepareDataset`` itself uses for
        structural neighborhoods (a single shared similarity threshold, transitive via connected components).

        :param ligand_codes: iterable of PDB chemical component codes, each with an entry in ``self.ligand_smiles``
        :return: dict mapping each ligand code to its integer cluster label
        :rtype: dict
        """
        codes = sorted(ligand_codes)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.MORGAN_RADIUS, fpSize=self.MORGAN_NBITS)
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(self.ligand_smiles[code])) for code in codes]

        n = len(codes)
        similarity = np.eye(n)
        for i in range(n):
            row = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            for offset, sim in enumerate(row):
                j = i + 1 + offset
                similarity[i, j] = similarity[j, i] = sim

        adjacency = (similarity >= self.tanimoto_threshold).astype(int)
        _, labels = connected_components(adjacency, directed=False)
        return {code: int(label) for code, label in zip(codes, labels)}

    def process(self) -> RNADataset:
        """
        Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """
        # Initialize dataset with in_memory=False to avoid loading everything at once
        dataset = RNADataset(dataset_path=self.graph_path, in_memory=False, redundancy='all', debug=self.debug, rna_id_subset=self.nodes_keep, version=self.version)

        # Instantiate filters to apply
        # ResolutionFilter (and RNAAttributeFilter's KeyError handling underneath it) rejects an RNA outright if
        # 'resolution_high' is missing, which silently drops every cryo-EM entry that doesn't populate that field
        # the way X-ray entries do: checked directly, recent large cryo-EM depositions here don't even have the
        # key in rna.graph, and this turned out to be the single biggest cause of binding pockets being discarded
        # pre-redundancy-removal for some ligand classes. A missing value is a metadata gap, not evidence of bad
        # quality, so it should pass; a present-but-too-coarse value should still be rejected. Written as a plain
        # function rather than RNAAttributeFilter since that class treats a missing key as instant rejection
        # before the value_checker ever runs, which is exactly the failure mode being fixed here.
        def resolution_ok(rna):
            val = rna["rna"].graph.get("resolution_high")
            try:
                return float(val) < 4.0
            except (TypeError, ValueError):
                return True

        # Instantiate transforms to apply
        nt_partition = PartitionFromDict(partition_dict=self.bp_dict)

        # Phase 1: build every admissible pocket (not yet labelled), tagging each with the PDB chemical
        # component code of its majority-contact ligand. Held in memory only transiently, so the Tanimoto
        # clustering below sees every candidate ligand before anything is written to disk.
        candidate_pockets = []
        for rna in tqdm(dataset):
            if not resolution_ok(rna):
                continue
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
                if ligand_code in self.EXCLUDED_LIGAND_CODES:
                    continue
                if self.ligand_smiles.get(ligand_code) is None:
                    continue
                pocket.graph["ligand_code"] = ligand_code
                candidate_pockets.append(pocket)

        # Phase 2: cluster the ligand codes actually present in this corpus by Tanimoto similarity, then label
        # every pocket with its ligand's cluster. Which clusters end up largest -- and hence which become the
        # final classes -- is only decided later, in post_process, after the structural ambiguity/redundancy
        # pass: a chemical cluster's raw size here doesn't predict how many of its pockets survive that pass.
        present_codes = {pocket.graph["ligand_code"] for pocket in candidate_pockets}
        self.tanimoto_clusters = self._cluster_ligands_by_tanimoto(present_codes)

        all_binding_pockets = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for pocket in candidate_pockets:
            cluster_label = self.tanimoto_clusters[pocket.graph["ligand_code"]]
            pocket.graph[self.target_var] = f"tanimoto_cluster_{cluster_label}"
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
        # sorted, so that a given ligand class always maps to the same class index: iterating the set directly makes
        # the mapping depend on string hash randomization, hence on the process, which silently invalidates any
        # per-class metric or checkpoint reloaded in another run
        self.mapping = {target_value: i for i, target_value in enumerate(sorted(represented_values))}
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(mapping=self.mapping)},
        )

    def post_process(self):
        """The task-specific post processing steps to remove redundancy, select classes and compute distances
        which will be used by the splitters.

        This mirrors the default Task.post_process, removing redundancy on sequence (CD-Hit) then on structure
        (US-align). As in the previous, hand-curated version of this task, PrepareDataset (rather than
        RedundancyRemover) is used for the structural pass: it additionally drops a structural-similarity cluster
        whose pockets do not all belong to the same Tanimoto-cluster class, since structure cannot determine the
        label there and keeping any representative would assert an arbitrary one.

        The US-align step's mixed-label ambiguity check uses a stricter threshold (0.95, "plausibly the same site")
        than its redundancy-collapse threshold (0.8, "similar enough to be a duplicate") -- see PrepareDataset's
        docstring for why the two are decoupled.

        After that structural pass, ``self.n_classes`` selects the largest surviving classes (see
        ``_keep_top_classes`` and the class docstring for why this has to happen after, not before, the
        structural pass), and ``self.max_per_class`` caps any of those that still dominates the dataset (see
        ``_cap_class_sizes`` and the class docstring for why the dominant aminoglycoside cluster needs this
        regardless of ``tanimoto_threshold``/``n_classes``).

        With only a handful of classes, a debug-mode build (a fixed ~50-structure sample) can easily end up with
        just a single matching pocket. CD-Hit/US-align can't build a pairwise similarity matrix from one sequence
        and silently drop it, collapsing the dataset to empty -- so both distance-computation/redundancy-removal
        passes are skipped outright below that size; this only matters for debug-mode smoke tests, since the real
        (non-debug) dataset has far more pockets per class.
        """
        if len(self.dataset) < 2:
            if not self.in_memory:
                self.dataset.save_distances()
            return

        cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
        us_align_rr = PrepareDataset(distance_name="USalign", threshold=0.8, ambiguity_threshold=0.95,
                                      target_name=self.target_var,
                                      collapse_to_representative=self.collapse_structural_duplicates)
        self.dataset = cd_hit_computer(self.dataset)

        us_align_computer = StructureDistanceComputer(name="USalign")
        self.dataset = us_align_computer(self.dataset)
        if self.redundancy_removal:
            self.dataset = us_align_rr(self.dataset)

        if self.n_classes is not None:
            self.dataset = self._keep_top_classes(self.dataset, self.n_classes)

        if self.max_per_class is not None:
            self.dataset = self._cap_class_sizes(self.dataset, self.max_per_class)

        if not self.in_memory:
            # PATCH: delete graphs from dataset/ lost during redundancy removal
            for f in os.listdir(self.dataset.dataset_path):
                if Path(f).stem not in self.dataset.all_rnas:
                    os.remove(Path(self.dataset.dataset_path) / f)
            self.dataset.save_distances()

    def _keep_top_classes(self, dataset, n_classes):
        """Keeps only the ``n_classes`` largest classes (by pocket count in ``dataset``), dropping every pocket
        belonging to a smaller class.

        :param RNADataset dataset: the (already structurally-filtered) dataset to select classes from
        :param int n_classes: how many of the largest classes to keep
        :return: the dataset restricted to those classes
        :rtype: RNADataset
        """
        counts = collections.Counter(dataset[idx]["rna"].graph[self.target_var] for idx in range(len(dataset)))
        top_labels = {label for label, _ in counts.most_common(n_classes)}
        kept_ids = [idx for idx in range(len(dataset)) if dataset[idx]["rna"].graph[self.target_var] in top_labels]
        return dataset.subset(list_of_ids=kept_ids)

    def _cap_class_sizes(self, dataset, max_per_class):
        """Caps any class exceeding ``max_per_class`` pockets down to its highest-resolution members.

        Resolution-based rather than random, for the same reason PrepareDataset's collapse-to-representative step
        picks the best-resolution structure: it's a deterministic, reproducible criterion that also happens to
        favor the more trustworthy structures, instead of leaving which pockets survive up to RNG seeding.

        :param RNADataset dataset: the (already redundancy-reduced) dataset to cap
        :param int max_per_class: the maximum number of pockets to keep per class
        :return: the capped dataset
        :rtype: RNADataset
        """
        by_class = collections.defaultdict(list)
        for idx in range(len(dataset)):
            label = dataset[idx]["rna"].graph[self.target_var]
            try:
                resolution = float(dataset[idx]["rna"].graph.get("resolution_high"))
            except (TypeError, ValueError):
                resolution = float("inf")
            by_class[label].append((resolution, idx))

        kept_ids = []
        for members in by_class.values():
            members.sort(key=lambda pair: pair[0])
            kept_ids.extend(idx for _, idx in members[:max_per_class])
        return dataset.subset(list_of_ids=kept_ids)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        return ClusterSplitter(distance_name="USalign")
