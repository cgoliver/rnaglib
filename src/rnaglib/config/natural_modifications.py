"""Curated set of natural, enzymatically installed RNA modifications.

The ``is_modified`` annotation produced at database build time flags any residue
whose PDB chemical-component code is not a single character (see
``prepare_data/fr3d_2_graphs.py``). That heuristic conflates genuine biological
modifications (pseudouridine, dihydrouridine, methylations...) with synthetic
species that share the "non-standard residue" property but are not biological
marks: crystallographic heavy-atom tags (5-bromo/5-iodo bases, selenium),
oligo-engineering chemistries (2'-F, LNA, phosphorothioate, MOE), DNA residues,
bound nucleotide ligands (GDP), and synthetic base analogs (nebularine,
2-aminopurine, isoG/isoC, glycol nucleic acids).

``NATURAL_RNA_MODIFICATIONS`` lists the codes that the cell writes onto its RNA
through dedicated modification enzymes rather than the template. This is the
property that separates them both from template-encoded residues (canonical
A/C/G/U, DNA) and from analogs installed in vitro; it is independent of whether
the enzyme acts co- or post-transcriptionally. Codes were curated by
cross-referencing each PDB component name against known MODOMICS entries. The set
is a reviewable, extensible resource: add codes as new natural modifications
appear in the PDB.
"""

# Grouped by modification family for reviewability. Codes are PDB CCD ids.
NATURAL_RNA_MODIFICATIONS = frozenset({
    # Base methylations
    "1MA",    # 1-methyladenosine (m1A)
    "2MA",    # 2-methyladenosine (m2A)
    "6MZ",    # N6-methyladenosine (m6A)
    "MA6",    # N6,N6-dimethyladenosine (m6,6A)
    "1MG",    # 1-methylguanosine (m1G)
    "2MG",    # N2-methylguanosine (m2G)
    "M2G",    # N2,N2-dimethylguanosine (m2,2G)
    "7MG",    # 7-methylguanosine (m7G)
    "G7M",    # 7-methylguanosine (m7G), alternate code
    "5MC",    # 5-methylcytidine (m5C)
    "B8T",    # N4-methylcytidine (m4C)
    "5MU",    # 5-methyluridine / ribothymidine (m5U)
    "UR3",    # 3-methyluridine (m3U)
    "A1IEA",  # 3-methyladenosine (m3A)
    # 2'-O-methylation (ribose)
    "A2M",    # 2'-O-methyladenosine (Am)
    "OMC",    # 2'-O-methylcytidine (Cm)
    "OMG",    # 2'-O-methylguanosine (Gm)
    "OMU",    # 2'-O-methyluridine (Um)
    "4OC",    # N4,2'-O-dimethylcytidine (m4Cm)
    "2MU",    # 5,2'-O-dimethyluridine (m5Um)
    # Pseudouridine and dihydrouridine
    "PSU",    # pseudouridine (Y)
    "H2U",    # dihydrouridine (D)
    # Thiolation
    "4SU",    # 4-thiouridine (s4U)
    "SUR",    # 2-thiouridine (s2U)
    "70U",    # 5-(methoxycarbonylmethyl)-2-thiouridine (mcm5s2U)
    # Acyl / carboxy modifications
    "4AC",    # N4-acetylcytidine (ac4C)
    "RSQ",    # 5-formylcytidine (f5C)
    "CM0",    # 5-carboxymethoxyuridine (cmo5U)
    # Hypermodified adenosines (tRNA)
    "6IA",    # N6-isopentenyladenosine (i6A)
    "MIA",    # 2-methylthio-N6-isopentenyladenosine (ms2i6A)
    "T6A",    # N6-threonylcarbamoyladenosine (t6A)
    "AET",    # N6-methyl-N6-threonylcarbamoyladenosine (m6t6A)
    "12A",    # 2-methylthio-N6-threonylcarbamoyladenosine (ms2t6A)
    # Queuosine / wyosine
    "QUO",    # queuosine (Q)
    "YYG",    # wybutosine (yW)
    # 2'-O-ribosylation
    "RIA",    # 2'-O-ribosyladenosine phosphate (Ar(p))
})


def is_biological_modification(nt_full: str) -> bool:
    """Whether a residue code denotes a natural, enzymatically installed RNA modification.

    :param str nt_full: the raw PDB chemical-component code of the residue
        (the ``nt_full`` node attribute), e.g. ``"PSU"``, ``"5BU"``, ``"A"``.
    :return: True only for curated natural modifications; False for canonical
        residues, synthetic analogs, DNA residues and ligands.
    """
    return nt_full in NATURAL_RNA_MODIFICATIONS
