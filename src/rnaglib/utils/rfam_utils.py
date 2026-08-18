import json
import math
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

GO_ASPECTS = {"molecular_function", "biological_process", "cellular_component"}

# The three GO roots. Every annotated RNA inherits all three through propagation,
# so they carry no discriminative signal and are always excluded.
GO_ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}


def pdb_sel_to_rfam():
    """
    Using the PDB-RFAM mapping hosted there: https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.pdb.gz
    return a df holding RFAM annotations along their PDB occurences
    :return:
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "Rfam.pdb")
    df = pd.read_csv(file_name, sep='\t')
    df['pdbsel'] = df.apply(lambda row: f"{row['pdb_id']}_{row['chain']}_{row['pdb_start']}_{row['pdb_end']}",
        axis=1)
    df = df[['pdbsel', 'pdb_id', 'rfam_acc']]
    return df


def get_rfam_to_go():
    """
    Using the RFAM-GO mapping hosted there : https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/rfam2go/rfam2go
    :return: a dict mapping RFAM ids to GO terms (without the "GO:" prefix, e.g. "0003735")
    """
    rfam_to_go = defaultdict(list)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "rfam2go")
    with open(file_name) as f:
        l = f.readlines()

    l = [x.split('GO:00') for x in l]
    for line in l:
        if len(line) == 2:
            rfam_id = line[0][5:12]
            go_term = '00' + line[1][:5]
            rfam_to_go[rfam_id].append(go_term)
    return rfam_to_go


def get_go_dag():
    """Load the cached Rfam-derived GO term lookup: GO id -> {aspect, name, ancestors}.

    ``ancestors`` is the self-inclusive, transitively-closed is_a/part_of ancestor set
    (i.e. the true-path-rule propagation target set), as resolved from the QuickGO REST
    API for every GO id appearing in ``rfam2go`` and their ancestors. Since GO ids are
    curated by Rfam per-family (not per-PDB-structure), this covers every family Rfam has
    ever annotated with a GO term, not just those currently observed in the PDB -- so newly
    deposited PDB structures for an already-GO-annotated family don't require regenerating
    this cache. A handful of GO ids referenced by rfam2go (obsolete/merged terms) are absent;
    callers should treat missing ids as unresolvable rather than erroring.

    :return: dict of GO id -> {"aspect": str, "name": str, "ancestors": list[str]}
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "rfam_go_dag.json")) as f:
        return json.load(f)


def get_rfam_to_go_propagated(aspect):
    """Rfam family -> GO terms, restricted to one GO aspect and propagated up the ontology
    DAG (true-path rule): a family directly annotated with a specific/leaf GO term also
    carries every ancestor of that term in the same aspect.

    This is the key transformation that makes per-GO-term prediction meaningful for ncRNA:
    a raw Rfam-to-GO annotation is (by construction, since Rfam curators assign GO terms per
    family) essentially a family fingerprint -- predicting it from sequence/structure is
    equivalent to predicting family membership. Propagating to ancestors turns that into a
    hierarchical multi-label signal shared *across* families that fall under a common
    ancestor (e.g. distinct spliceosomal snRNA families all inherit "spliceosome"/"RNA
    splicing" even though their direct/leaf terms differ), which is what actually lets a
    structural-generalization task be built on top of it. GO roots are excluded (see
    ``GO_ROOTS``) since every annotated family inherits them and they carry no signal.

    :param aspect: one of "molecular_function", "biological_process", "cellular_component"
    :return: dict of rfam_acc -> set of propagated GO ids (e.g. {"GO:0000244", ...})
    """
    if aspect not in GO_ASPECTS:
        raise ValueError(f"aspect must be one of {sorted(GO_ASPECTS)}, got {aspect!r}")

    rfam_to_go = get_rfam_to_go()
    dag = get_go_dag()

    out = defaultdict(set)
    for rfam, go_nums in rfam_to_go.items():
        for num in go_nums:
            go_id = f"GO:{num}"
            info = dag.get(go_id)
            if info is None:
                continue
            for ancestor in info["ancestors"]:
                if ancestor in GO_ROOTS:
                    continue
                ancestor_info = dag.get(ancestor)
                # An ancestor should always resolve (it came from this same DAG), but fall
                # back to the leaf term's own aspect defensively rather than dropping it.
                ancestor_aspect = ancestor_info["aspect"] if ancestor_info else info["aspect"]
                if ancestor_aspect == aspect:
                    out[rfam].add(ancestor)
    return dict(out)


def filter_and_dedup_go_terms(
    item_to_terms, item_to_families=None, min_count=5, max_frequency=0.8, min_families=1, corr_threshold=0.9
):
    """Frequency-filter and de-duplicate a per-item multi-label GO term annotation.

    Two-sided frequency filter: a term must annotate at least ``min_count`` items to be
    learnable/evaluable at all, and at most ``max_frequency`` of all items, since terms that
    are near-universal (typically shallow/generic ancestors surviving after propagation, e.g.
    "binding" or "metabolic process") are uninformative and dominate any macro-averaged loss.
    This mirrors DeepFRI's own frequency cutoff (`>50 non-redundant chains`), scaled down to
    match the PDB's much smaller RNA structural coverage and generalized to a two-sided range
    (an upper bound was already present in this codebase's original DeepFRI-inspired filter,
    ad-hoc-restricted to ribosome/tRNA terms -- this generalizes it to any aspect/term).

    Raw item count alone cannot tell a term that is genuinely shared across families from one
    that is a single family deposited many times over (e.g. one Rfam family solved repeatedly
    in the PDB can rack up a high fragment count on its own) -- exactly the family-fingerprint
    failure mode propagation is meant to fix. If ``item_to_families`` is given (item id -> set
    of Rfam accessions backing it), a second, independent filter requires a term to be carried
    by at least ``min_families`` distinct Rfam families among the items that survive the count
    filter; this also matters for downstream similarity-based cluster splitting, since a
    single-family term's few instances are structurally similar enough to likely land in one
    connected component, leaving the label absent from some split entirely.

    Terms surviving that filter are then de-duplicated: since propagation can create chains
    of near-perfectly-correlated ancestor/descendant terms (when a term's only observed
    parent in this dataset has no other children), pairs of terms with Pearson
    correlation >= ``corr_threshold`` are collapsed, keeping one representative -- same idea
    this codebase used previously for the flat (non-propagated) GO term list, restricted here
    to *positive* correlation (co-occurring/synonymous terms). A strong *negative*
    correlation means two terms are close to mutually exclusive, which is a real, informative
    split between categories (e.g. ribosome-associated vs. RNA-binding-but-non-ribosomal MF
    terms) -- not redundancy -- so it must not trigger de-duplication.

    :param item_to_terms: dict of item id -> set of GO ids annotating it
    :param item_to_families: optional dict of item id -> set of Rfam accessions backing it,
        used only to enforce ``min_families``; if omitted, family diversity is not checked
    :param min_count: minimum number of items a term must annotate to be kept
    :param max_frequency: maximum fraction of items a term may annotate to be kept
    :param min_families: minimum number of distinct Rfam families a term must be carried by
        (requires ``item_to_families``; ignored otherwise)
    :param corr_threshold: Pearson correlation above which two surviving terms are considered
        near-duplicates (the later one, in dict-iteration order, is dropped)
    :return: (filtered_item_to_terms, term_stats) where filtered_item_to_terms maps each
        input item id to its surviving term subset, and term_stats maps each surviving GO id
        to {"count": int, "ic": float} (``ic`` is the Shannon information content
        -log2(count / n_items), as defined in the DeepFRI paper: rarer/more specific terms
        get a higher score).
    """
    n_items = len(item_to_terms)
    counter = Counter(term for terms in item_to_terms.values() for term in terms)
    candidate_terms = sorted(
        term for term, count in counter.items() if min_count <= count <= max_frequency * n_items
    )

    if item_to_families is not None and min_families > 1:
        term_families = defaultdict(set)
        for item, terms in item_to_terms.items():
            families = item_to_families.get(item, set())
            for term in terms:
                term_families[term] |= families
        candidate_terms = [term for term in candidate_terms if len(term_families[term]) >= min_families]

    if not candidate_terms:
        return {item: set() for item in item_to_terms}, {}

    items = list(item_to_terms.keys())
    term_index = {term: i for i, term in enumerate(candidate_terms)}
    mat = np.zeros((len(items), len(candidate_terms)))
    for i, item in enumerate(items):
        for term in item_to_terms[item]:
            j = term_index.get(term)
            if j is not None:
                mat[i, j] = 1.0

    keep = list(range(len(candidate_terms)))
    if mat.shape[1] > 1:
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(mat, rowvar=False)
        dropped = set()
        for i in range(len(candidate_terms)):
            if i in dropped:
                continue
            for j in range(i + 1, len(candidate_terms)):
                if j in dropped:
                    continue
                if corr[i, j] >= corr_threshold:
                    dropped.add(j)
        keep = [i for i in keep if i not in dropped]

    kept_terms = {candidate_terms[i] for i in keep}
    filtered = {item: (item_to_terms[item] & kept_terms) for item in items}
    term_stats = {
        term: {"count": counter[term], "ic": -math.log2(counter[term] / n_items)}
        for term in kept_terms
    }
    return filtered, term_stats
