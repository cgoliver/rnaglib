
import os
import json
import collections
from pathlib import Path

from tqdm import tqdm

from rnaglib.tasks import RNAClassificationTask
from rnaglib.dataset import RNADataset
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import FeaturesComputer, PartitionFromDict
from rnaglib.dataset_transforms import ClusterSplitter, CDHitComputer, StructureDistanceComputer
from rnaglib.tasks.RNA_Ligand.prepare_dataset import PrepareDataset


class LigandIdentification(RNAClassificationTask):
    """Binding pocket-level task where the job is to predict which of a small, curated set of ligand classes a
    binding pocket binds, given only its structure.

    Rather than the exact ligand (a task dominated by a handful of over-deposited compounds and unsplittable for
    the rare ones) or a Tanimoto-similarity fingerprint cluster (which still let bulk-deposited, near-duplicate
    pockets dominate the class distribution and collapse every model to majority-class prediction), the target is
    one of a handful of hand-picked ligand classes, in the spirit of the MaSIF-ligand task for proteins (ADP / CoA
    / FAD / NAD / NADP / SAM pockets). Concretely, this is a "which antibiotic occupies this ribosomal RNA pocket"
    task: every class is an antibiotic bound to some part of the bacterial ribosome, so the task probes whether
    pocket geometry alone (the model never sees the ligand) can distinguish real, distinct binding sites rather
    than a chemically-motivated grouping:

    - **PAR**: paromomycin, capped at ``max_per_class`` (see below) -- bound to the 16S rRNA decoding-site (A-site)
    - **LLL**: gentamicin -- also decoding-site-proximal, but with enough surviving structural diversity after
      redundancy removal (unlike PAR-adjacent congeners AM2/84D/84G/KAN/AKN, which were tried and collapsed) to
      stand as a real class
    - **NMY**: neomycin
    - **8UZ**: aminoglycoside TC007
    - **GET**: an aminoglycoside congener distinct enough post-redundancy-removal to survive as its own class
    - **T1C**: tigecycline, a tetracycline-class antibiotic (different pocket architecture than the aminoglycosides)
    - **chloramphenicol**: the bacterial ribosomal peptidyl transferase center, structurally distinct from every
      aminoglycoside/tetracycline class above

    This list is the result of an empirical sweep, not an a-priori chemical judgment call: earlier iterations tried
    (a) every Tanimoto-similarity fingerprint cluster, (b) a hand-picked set of riboswitch/ribozyme cofactor ligands
    (SAM, NAD, arginine, c-di-AMP, PreQ1) analogous to MaSIF-ligand's cofactor classes -- both collapsed to under
    100 total post-redundancy-removal samples across their classes, several with single-digit counts, because these
    well-studied aptamers turn out to be re-solved as near-identical structures far more often than they're solved
    against genuinely different scaffolds. Individual antibiotic codes, by contrast, mostly survived redundancy
    removal with real counts (15-50 each) -- except PAR itself, which alone would still be ~47% of the resulting
    dataset if left uncapped, reproducing the exact majority-class problem this redesign exists to avoid. Hence the
    ``max_per_class`` cap: applied only where a class actually exceeds it (only PAR does, at the current sizes).

    The class -> ligand-code mapping lives in ``CLASS_MAP`` below and is a hand curation, not a computed
    clustering, so extending it (or swapping in a different set of ligand classes) is a matter of editing that
    dict and adjusting ``admissible_classes``. Any change should be validated the same way this set was: build the
    full (non-debug) task and check real post-redundancy-removal per-class counts, since raw pre-redundancy pocket
    counts are a poor predictor of how many samples a class will actually retain.

    Task type: multi-class classification
    Task level: substructure-level

    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset (default (15, 500))
    :param tuple[str] admissible_classes: class names to keep as targets (default: every class present in
        ``CLASS_MAP``).
    :param bool collapse_structural_duplicates: if True, the US-align redundancy-removal step reduces each
        label-consistent structural cluster to its single highest-resolution representative. If False (default
        here -- unlike the general Task default), all members of a label-consistent cluster are kept (mixed-label
        clusters are still dropped either way). Turning collapsing on is what silently created the earlier
        single-digit-count classes: e.g. chloramphenicol went from 3 samples (collapsed) to 25 (uncollapsed)
        because most of its near-duplicate structures were real, independently-solved sites, not redundant
        redepositions of one structure.
    :param int max_per_class: after redundancy removal, cap any class exceeding this many pockets to its
        ``max_per_class`` highest-resolution members (default 50, matching the size of the next-largest class
        after PAR at the time this set was curated). Set to None to disable capping.
    """
    input_var = "nt_code"
    target_var = "ligand_class"
    name = "rna_ligand"
    default_metric = "auc"
    version = "2.0.2"

    # maps a PDB chemical component code to its hand-curated ligand class (see class docstring)
    CLASS_MAP = {
        "PAR": "PAR",
        "LLL": "LLL",
        "NMY": "NMY",
        "8UZ": "8UZ",
        "GET": "GET",
        "T1C": "T1C",
        "CLM": "chloramphenicol",
    }

    def __init__(self,
        size_thresholds=(15, 500),
        graph_path=None,
        admissible_classes=None,
        collapse_structural_duplicates=False,
        max_per_class=50,
        **kwargs
    ):
        self.graph_path = graph_path
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

        # default to every class present in the curated mapping
        self.admissible_classes = admissible_classes if admissible_classes is not None \
            else sorted(set(self.CLASS_MAP.values()))
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

        # Run through database, applying our filters
        all_binding_pockets = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in tqdm(dataset):
            if resolution_ok(rna):
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
                    ligand_class = self.CLASS_MAP.get(ligand_code)
                    if ligand_class is not None and (ligand_class in self.admissible_classes or self.debug):
                        pocket.graph[self.target_var] = ligand_class
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
        """The task-specific post processing steps to remove redundancy and compute distances which will be used by the splitters.

        This mirrors the default Task.post_process, removing redundancy on sequence (CD-Hit) then on structure
        (US-align). The only difference is that PrepareDataset, rather than RedundancyRemover, is used: it additionally
        drops a structural-similarity cluster whose pockets do not all bind ligands from the same class, since
        structure cannot determine the label there and keeping any representative would assert an arbitrary one.

        The US-align step's mixed-label ambiguity check uses a stricter threshold (0.95, "plausibly the same site")
        than its redundancy-collapse threshold (0.8, "similar enough to be a duplicate"): PrepareDataset's
        connected-components clustering is transitive, so a single shared 0.8 threshold let a loosely-related third
        pocket bridge two structurally-similar-but-differently-labelled sites into one neighborhood and drop the
        whole group -- including members whose own label was never actually ambiguous. See PrepareDataset's
        docstring for the two-pass mechanics this avoids.

        The US-align step's collapse-to-one-representative behavior is controlled separately from the mixed-label
        drop via ``self.collapse_structural_duplicates`` (see __init__): turning it off keeps every member of a
        label-consistent structural cluster instead of just the best-resolution one, for classes that don't have
        enough post-redundancy-removal samples to learn from.

        Finally, ``self.max_per_class`` caps any class that still dominates the dataset after redundancy removal
        (see __init__ and the class docstring for why PAR specifically needs this) down to its highest-resolution
        members, so no single class can reproduce the majority-class problem this task's redesign exists to avoid.

        With only 7 admissible classes, a debug-mode build (a fixed ~50-structure sample) can easily end up with
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
        cd_hit_rr = PrepareDataset(distance_name="cd_hit", threshold=0.9, target_name=self.target_var)
        self.dataset = cd_hit_computer(self.dataset)

        us_align_computer = StructureDistanceComputer(name="USalign")
        us_align_rr = PrepareDataset(distance_name="USalign", threshold=0.8, ambiguity_threshold=0.95,
                                      target_name=self.target_var,
                                      collapse_to_representative=self.collapse_structural_duplicates)
        self.dataset = us_align_computer(self.dataset)
        if self.redundancy_removal:
            self.dataset = us_align_rr(self.dataset)

        if self.max_per_class is not None:
            self.dataset = self._cap_class_sizes(self.dataset, self.max_per_class)

        if not self.in_memory:
            # PATCH: delete graphs from dataset/ lost during redundancy removal
            for f in os.listdir(self.dataset.dataset_path):
                if Path(f).stem not in self.dataset.all_rnas:
                    os.remove(Path(self.dataset.dataset_path) / f)
            self.dataset.save_distances()

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
