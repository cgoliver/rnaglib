``rnaglib.tasks``
=========================

Task objects hold everything you need to feed a prediction model and evaluate its performance on a variety of tasks, as well as to easily implement your own tasks.


.. automodule:: rnaglib.tasks


Abstract classes
------------------

Subclass these to create your own tasks.

.. autosummary::
   :toctree: generated/

    Task
    ResidueClassificationTask
    RNAClassificationTask
    

RNA-level Classification
----------------------------

These tasks take as input an RNA and predict a property of the whole molecule.

.. autosummary::
    :toctree: generated/

    RNAFamily
    BindingSiteDetection


Residue-level Classification
----------------------------------


These tasks take as input an RNA and predict a property of each residue of the molecule.

.. autosummary::
    :toctree: generated/

    ChemicalModification
    ProteinBindingSiteDetection
    InverseFolding
    gRNAde


Substructure-level Classification
-------------------------------------

Predict properties of substructures of a whole molecule (e.g. binding sites)

.. autosummary::
   :toctree: generated/

    LigandIdentification



