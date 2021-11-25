rnaglib.data
====================
.. include:: ../../rnaglib/prepare_data/README.md
   :parser: myst_parser.sphinx_


Download
--------

The instructions to produce the data for all RNA are presented in :doc:`rnaglib.prepare_data`.
However since this construction is computationally expensive at database scale, we offer pre-built databases.
We however offer three possibilities to directly access pre-built databases :

-  A download script ships with the install, run : ``$ rnaglib_download -h``
-  Direct download at the address :
   http://rnaglib.cs.mcgill.ca/static/datasets/iguana.tar.gz
-  Dynamic download : if one instantiates a dataloader and the data
   cannot be found, the corresponding data will be automatically downloaded and cached

Because of this last option, after installing our tool with pip, one can start learning on RNA data without extra steps.


.. automodule:: rnaglib.data
   :members:
   :undoc-members:
   :show-inheritance:
