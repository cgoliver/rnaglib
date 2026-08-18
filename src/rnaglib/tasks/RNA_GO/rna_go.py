import os
from collections import Counter
from pathlib import Path

import numpy as np

from rnaglib.dataset import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.encoders import MultiLabelOneHotEncoder
from rnaglib.transforms import FeaturesComputer
from rnaglib.dataset_transforms import ClusterSplitter, CDHitComputer
from rnaglib.utils.rfam_utils import pdb_sel_to_rfam, get_rfam_to_go_propagated, filter_and_dedup_go_terms, get_go_dag


class RNAGo(RNAClassificationTask):
    """Predict Gene Ontology (GO) terms for a given RNA chain from its 3D structure, following the
    methodology introduced for proteins by DeepFRI (Gligorijevic et al. 2021) and evaluated as in
    GearNet (Zhang et al. 2023).

    Labels come from Rfam's own family-to-GO curation (``rfam2go``), joined onto PDB chains via the
    same PDB-Rfam mapping used throughout rnaglib (``Rfam.pdb``). As in DeepFRI, GO terms are handled
    separately per ontology aspect -- molecular function (MF), biological process (BP), or cellular
    component (CC), selected via the ``ontology`` constructor argument -- since a term from one
    aspect is not comparable to a term from another.

    Unlike DeepFRI's protein annotations (SIFTS/UniProt, where a chain's GO terms reflect a whole,
    often multi-domain protein with combinatorial function), a raw Rfam-to-GO annotation is assigned
    *per family*, i.e. it is essentially a family fingerprint: predicting one specific GO term from a
    structure is equivalent to predicting Rfam family membership, which is solvable from sequence
    homology alone and does not probe structural generalization. To fix this without abandoning the
    real GO vocabulary (as the previous version of this task did, collapsing everything into 5
    hand-curated multi-family classes), GO annotations are propagated up the ontology DAG under the
    true-path rule (:func:`~rnaglib.utils.rfam_utils.get_rfam_to_go_propagated`): a family annotated
    with a specific/leaf term also carries every ancestor of that term. This turns a single
    family-specific label into a hierarchical multi-label signal that is genuinely shared *across*
    families -- e.g. distinct spliceosomal snRNA families (U1, U2, U4, U5, U6) each keep their own
    leaf term but all inherit shared ancestors like "spliceosome"/"RNA splicing" -- so the model has
    to generalize across real fold variation to pick up the shared, coarser labels, while the leaf
    terms remain as a harder, near-family-identity tier. An empirical audit of this propagated label
    space (see the task's design notes) confirmed most surviving terms span >=2 distinct Rfam
    families rather than just one.

    Following DeepFRI's frequency cutoff (`>50 non-redundant chains`, scaled down here to match the
    PDB's much smaller RNA structural coverage) and this codebase's own prior ad-hoc upper bound for
    ribosome/tRNA overrepresentation, surviving terms are further frequency-filtered on the actually
    built candidate set with a two-sided count filter (``min_count``/``max_frequency``): terms with
    too few examples aren't learnable or evaluable, terms present in almost every example (chiefly
    generic ancestors surfaced by propagation) are uninformative and would otherwise dominate any
    macro-averaged loss.

    Raw fragment count alone is not sufficient, though: a term can clear ``min_count`` while still
    being backed by a *single* Rfam family deposited many times over in the PDB (the exact
    family-fingerprint failure mode propagation is meant to fix), and single-family terms are also
    the ones most likely to land entirely inside one connected component under the similarity-based
    default splitter -- making the label absent from an entire split. ``min_families`` requires each
    surviving term to be backed by at least that many distinct Rfam families in the built dataset, on
    top of the count filter. Near-duplicate terms produced by propagation chains (a term and an
    ancestor that are, in this dataset, always co-annotated) are then collapsed via correlation-based
    de-duplication (``corr_threshold``), reusing the approach this codebase already used for the flat
    (non-propagated) GO term list. Each surviving term's DeepFRI-style information content
    (``ic = -log2(count / n_items)``, i.e. rarer/more specific terms score higher) is recorded in
    ``self.metadata["go_term_stats"]``.

    None of the filters above (count, frequency, family diversity) reliably predict whether a term
    will actually end up with usable support in every split once the cluster-based default splitter
    runs: which RNAs land in which split depends on cluster topology, not term statistics -- with the
    structural (USalign) splitter previously used here, two terms with >=7 distinct structural
    clusters of support still ended up entirely absent from a split while another with only 3
    clusters split fine, and there is no reason to assume a sequence-based splitter is immune to the
    same kind of accident even if it should no longer be *correlated with the label itself* the way
    structural clustering is (see ``default_splitter``). So as a final step, once the split is
    computed (:meth:`split`), any GO term with fewer than ``min_split_support`` positive examples in
    train, val, *or* test is dropped -- checking the actual outcome directly instead of relying on a
    pre-split proxy -- and any RNA left with no surviving term is removed from its split. This never
    changes which RNAs are assigned to which split (dropping a term never changes cluster membership
    for the RNAs that keep at least one other term), so no re-splitting is needed.

    Task type: multi-label classification
    Task level: RNA-level

    :param str ontology: which GO aspect to predict -- one of "molecular_function",
        "biological_process" (default), "cellular_component"
    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset (default (15, 500))
    :param int min_count: minimum number of built RNA fragments a GO term must annotate to be kept
        (default 30). DeepFRI itself requires >50 non-redundant chains, but that number is calibrated
        to a protein corpus several orders of magnitude larger than the PDB's RNA coverage; 30 is
        chosen to stay in the same spirit (a term must be well-supported, not just barely present)
        while remaining reachable at all given this corpus's actual scale -- keep in mind this counts
        *fragments* in the actually built, quality-filtered dataset, which is typically much smaller
        than Rfam's raw PDB-family coverage
    :param float max_frequency: maximum fraction of built RNA fragments a GO term may annotate to be
        kept, filtering out near-universal/generic terms (default 0.8)
    :param int min_families: minimum number of distinct Rfam families a GO term must be backed by to
        be kept, filtering out terms that are really just a single family's fingerprint despite
        clearing ``min_count`` (default 2, i.e. the minimum for the label to be cross-family at all)
    :param float corr_threshold: Pearson correlation above which two surviving GO terms are treated
        as near-duplicates and collapsed to one (default 0.9)
    :param int min_split_support: minimum number of positive examples a GO term must have in each of
        train/val/test *after* the actual split is computed to be kept (default 10, up from an
        initial 3 which let through terms like a 386/6/3 train/val/test split -- technically present
        everywhere, but not meaningfully learnable or evaluable); terms that fall short in any split
        are dropped post hoc, since this can't be reliably predicted beforehand (see class docstring)
    """

    ONTOLOGY_SUFFIXES = {
        "molecular_function": "mf",
        "biological_process": "bp",
        "cellular_component": "cc",
    }

    input_var = "nt_code"  # node level attribute
    target_var = "go_terms"  # graph level attribute
    version = "2.0.2"
    default_metric = "fmax"

    def __init__(
        self,
        ontology="biological_process",
        size_thresholds=(15, 500),
        graph_path=None,
        min_count=30,
        max_frequency=0.8,
        min_families=2,
        corr_threshold=0.9,
        min_split_support=10,
        **kwargs,
    ):
        if ontology not in self.ONTOLOGY_SUFFIXES:
            raise ValueError(f"ontology must be one of {sorted(self.ONTOLOGY_SUFFIXES)}, got {ontology!r}")
        if kwargs.get("debug", False):
            # debug=True samples a tiny fraction of the data (see RNADataset(debug=...)), at
            # which scale the production thresholds above are unreachable by construction (no
            # term can annotate 30 fragments out of a ~14-RNA sample) -- relax them enough to
            # still exercise the same filtering code paths on a smoke-test-sized dataset,
            # rather than always yielding an empty label set.
            min_count = min(min_count, 2)
            max_frequency = max(max_frequency, 0.95)
            min_families = min(min_families, 1)
            min_split_support = min(min_split_support, 1)
        self.ontology = ontology
        self.min_count = min_count
        self.max_frequency = max_frequency
        self.min_families = min_families
        self.corr_threshold = corr_threshold
        self.min_split_support = min_split_support
        # Instance-level name (distinct per aspect, and distinct from the legacy 5-class "rna_go"
        # task) so the base Task never mistakes this for the old, differently-labeled Zenodo
        # artifact registered under "rna_go" and tries to download it instead of building fresh.
        self.name = f"rna_go_{self.ONTOLOGY_SUFFIXES[ontology]}"
        meta = {"multi_label": True, "ontology": ontology}
        self.graph_path = graph_path
        super().__init__(additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task: cluster-based splitting on
        CD-Hit sequence similarity, matching DeepFRI's own practice (they split GO-term prediction data
        by sequence non-redundancy, never by structure).

        Structural (USalign) similarity was used here in an earlier version of this task and was
        deliberately dropped: for ncRNA, Rfam family membership is tightly coupled to both fold and
        (via propagation) the predicted GO term itself, so splitting by structural similarity risks
        splitting along label boundaries -- clusters of structurally similar RNAs tend to also share
        GO terms, so whole clusters (and the labels concentrated in them) can end up assigned
        disproportionately to one split. That is a very plausible explanation for the severe per-split
        label imbalance observed empirically with the USalign-based splitter (e.g. one dominant term at
        386/6/3 across train/val/test), though this specific dataset has not itself been rebuilt and
        re-checked against this CD-Hit splitter yet -- that comparison is what actually determines
        whether the hypothesis holds and should be inspected on the next rebuild. Sequence-based
        splitting also better matches the actual point of a structure-aware model: it still allows
        sequence-divergent but fold-conserved examples to appear across splits, which is exactly the
        generalization case a 3D-structure method is supposed to exploit -- splitting by structure too
        would remove precisely those informative test cases.

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        return ClusterSplitter(similarity_threshold=0.6, distance_name="cd_hit")

    def split(self, dataset: RNADataset):
        """Computes the train/val/test split, then drops any GO term without at least
        ``min_split_support`` positive examples in every one of the three splits (see class
        docstring), removing any RNA left with no surviving term from its split.

        :param dataset: dataset to split
        """
        splits = super().split(dataset)
        self._prune_labels_without_split_support()
        return self.train_ind, self.val_ind, self.test_ind

    def _prune_labels_without_split_support(self):
        split_names = ("train", "val", "test")
        split_inds = (self.train_ind, self.val_ind, self.test_ind)

        per_split_counts = {term: Counter() for term in self.metadata["label_mapping"]}
        for split_name, inds in zip(split_names, split_inds):
            for i in inds:
                for term in self.dataset[i]["rna"].graph["go_terms"]:
                    per_split_counts[term][split_name] += 1

        dropped_terms = {
            term
            for term, counts in per_split_counts.items()
            if any(counts[split_name] < self.min_split_support for split_name in split_names)
        }
        if not dropped_terms:
            return

        print(
            f">>> Dropping {len(dropped_terms)} GO term(s) with <{self.min_split_support} examples "
            f"in some split (train/val/test counts): "
            + ", ".join(f"{t}={dict(per_split_counts[t])}" for t in sorted(dropped_terms))
        )

        name_to_split = {}
        for split_name, inds in zip(split_names, split_inds):
            for i in inds:
                name_to_split[self.dataset[i]["rna"].name] = split_name

        kept_by_split = {"train": [], "val": [], "test": []}
        kept_names_by_split = {"train": [], "val": [], "test": []}
        dropped_rna_names = []
        for i in range(len(self.dataset)):
            rna = self.dataset[i]["rna"]
            surviving = sorted(set(rna.graph["go_terms"]) - dropped_terms)
            if not surviving:
                dropped_rna_names.append(rna.name)
                continue
            rna.graph["go_terms"] = surviving
            split_name = name_to_split[rna.name]
            self.add_rna_to_building_list(all_rnas=kept_by_split[split_name], rna=rna)
            kept_names_by_split[split_name].append(rna.name)
        if dropped_rna_names:
            print(f">>> Also dropping {len(dropped_rna_names)} RNA(s) left with no surviving GO term")

        # Capture the pre-pruning distance matrices and name->index mapping before dataset
        # reassignment below invalidates them: pruning only ever removes RNAs, so the split's
        # distances stay valid restricted to the surviving rows/columns, and reslicing them
        # is far cheaper than recomputing CD-Hit/USalign from scratch on the pruned set.
        old_distances = self.dataset.distances
        old_all_rnas = self.dataset.all_rnas

        all_kept = kept_by_split["train"] + kept_by_split["val"] + kept_by_split["test"]
        self.dataset = self.create_dataset_from_list(all_kept)

        if old_distances:
            new_all_rnas = self.dataset.all_rnas
            order = [None] * len(self.dataset)
            for name, new_idx in new_all_rnas.items():
                order[new_idx] = old_all_rnas[name]
            for distance_name, matrix in old_distances.items():
                self.dataset.add_distance(distance_name, matrix[np.ix_(order, order)])

        remaining_terms = sorted(t for t in self.metadata["label_mapping"] if t not in dropped_terms)
        self.metadata["label_mapping"] = {t: i for i, t in enumerate(remaining_terms)}
        self.metadata["go_term_names"] = {
            t: v for t, v in self.metadata["go_term_names"].items() if t in remaining_terms
        }
        self.metadata["go_term_stats"] = {
            t: v for t, v in self.metadata["go_term_stats"].items() if t in remaining_terms
        }
        self.dataset.features_computer = self.get_task_vars()

        # Recompute size/class-distribution metadata directly (describe() would otherwise just
        # return the pre-pruning values already cached in self.metadata instead of recomputing).
        label_mapping = self.metadata["label_mapping"]
        class_distribution = Counter()
        for i in range(len(self.dataset)):
            for term in self.dataset[i]["rna"].graph["go_terms"]:
                class_distribution[label_mapping[term]] += 1
        self.metadata["num_classes"] = len(label_mapping)
        self.metadata["dataset_size"] = len(self.dataset)
        self.metadata["class_distribution"] = dict(class_distribution)

        # RNA identities changed (some dropped) but not which split each retained RNA belongs to,
        # so re-derive index lists by name lookup instead of re-splitting.
        all_rnas = self.dataset.all_rnas
        self.train_ind = [all_rnas[n] for n in kept_names_by_split["train"]]
        self.val_ind = [all_rnas[n] for n in kept_names_by_split["val"]]
        self.test_ind = [all_rnas[n] for n in kept_names_by_split["test"]]

        if not self.in_memory:
            for f in os.listdir(self.dataset.dataset_path):
                if Path(f).stem not in self.dataset.all_rnas:
                    os.remove(Path(self.dataset.dataset_path) / f)
            # The cleanup above also removes the now-stale (wrong-shaped) distances.npz left
            # over from before pruning -- write the reattached, correctly-resliced one back out.
            self.dataset.save_distances()

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
        rfam_to_go = get_rfam_to_go_propagated(self.ontology)

        # Only pull pdb selections whose Rfam family carries >=1 propagated GO term in this aspect.
        df = pdb_sel_to_rfam()
        df = df[df['rfam_acc'].isin(rfam_to_go)]

        dataset = RNADataset(dataset_path=self.graph_path, redundancy='nr', debug=self.debug, in_memory=self.in_memory, rna_id_subset=df['pdb_id'].unique(), version=self.version)

        # First pass: build every candidate fragment along with its (pre-frequency-filter)
        # propagated GO term set. The final label vocabulary is only decided afterwards, from
        # the set of fragments that actually survive quality filtering (not from the raw
        # Rfam.pdb table), so it reflects the dataset that will actually be built.
        candidates = []  # list of (pdbsel, subgraph, go_terms, rfam_families)
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

                # Get the propagated GO term(s) for this RFAM selection. A pdbsel could in
                # principle map to more than one rfam_acc, so we union their propagated terms.
                rfams_pdbsel = lines.loc[lines['pdbsel'] == pdbsel]['rfam_acc'].values
                go_terms = set()
                for rfam_id in rfams_pdbsel:
                    go_terms |= rfam_to_go.get(rfam_id, set())
                if not go_terms:
                    continue

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

                candidates.append((pdbsel, subgraph, go_terms, set(rfams_pdbsel)))

        # Second pass: frequency-filter and de-duplicate the GO term vocabulary over the
        # candidate fragments that actually survived quality filtering, then drop any
        # fragment left with an empty label set once its terms are restricted to the
        # surviving vocabulary.
        pdbsel_to_terms = {pdbsel: go_terms for pdbsel, _, go_terms, _ in candidates}
        pdbsel_to_families = {pdbsel: families for pdbsel, _, _, families in candidates}
        filtered_terms, term_stats = filter_and_dedup_go_terms(
            pdbsel_to_terms,
            item_to_families=pdbsel_to_families,
            min_count=self.min_count,
            max_frequency=self.max_frequency,
            min_families=self.min_families,
            corr_threshold=self.corr_threshold,
        )

        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for pdbsel, subgraph, _, _ in candidates:
            final_terms = filtered_terms[pdbsel]
            if not final_terms:
                continue
            subgraph.graph['go_terms'] = sorted(final_terms)
            self.add_rna_to_building_list(all_rnas=all_rnas, rna=subgraph)
        dataset = self.create_dataset_from_list(all_rnas)

        # The label vocabulary is derived from the actually-built, frequency-filtered,
        # de-duplicated term set above -- not a fixed a priori list -- so it will vary with
        # min_count/max_frequency/corr_threshold and with debug/full runs.
        go_names = get_go_dag()
        label_mapping = {term: i for i, term in enumerate(sorted(term_stats))}
        self.metadata["label_mapping"] = label_mapping
        self.metadata["go_term_names"] = {term: go_names.get(term, {}).get("name", "?") for term in label_mapping}
        self.metadata["go_term_stats"] = term_stats
        return dataset

    def post_process(self):
        """
        Computes sequence similarity between all RNA chains using CD-Hit (needed by the
        CD-Hit-based ClusterSplitter used as this task's default splitter -- see
        ``default_splitter`` for why sequence, not structure, is used for splitting).

        Structure-based (USalign) distances are intentionally *not* computed here: they were
        used by an earlier version of this task's default splitter and are no longer needed by
        default, and USalign is by far the most expensive and failure-prone step in this
        pipeline (pairwise structural alignment, known to stall on RNAs >200nt). A subclass or
        caller that wants a structure-based splitter as an explicit ablation can still add
        ``StructureDistanceComputer(name="USalign")`` to the dataset before splitting.
        """
        cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
        self.dataset = cd_hit_computer(self.dataset)

        if not self.in_memory:
            # CDHitComputer can drop RNAs that fail to process, reassigning self.dataset to a
            # smaller subset. Without this, the dataset_path directory keeps every original
            # *.json (including now-orphaned ones) while the computed distances are only ever
            # held in memory -- so they're silently lost once this process exits, and a later
            # RNADataset load sees a *.json count that no longer matches any saved
            # distances.npz. Mirrors the base Task.post_process()'s "PATCH: delete graphs from
            # dataset/ lost during redundancy removal" behavior, which this override otherwise
            # bypasses entirely.
            for f in os.listdir(self.dataset.dataset_path):
                if Path(f).stem not in self.dataset.all_rnas:
                    os.remove(Path(self.dataset.dataset_path) / f)
            self.dataset.save_distances()
