Installation
------------

The package can be cloned and the source code used directly.
We also deploy it as a pip package and recommend using this install in conda environments.

If one wants to use GPU support, one should install `Pytorch <https://pytorch.org/get-started/locally/>`__
and `DGL <https://www.dgl.ai/pages/start.html>`__ with the appropriate options.
Otherwise you can just skip this step and the pip installs of Pytorch and DGL will be used.

Then, one just needs to run :

::

    pip install rnaglib

and can start using the packages functionalities by importing them in one's python script.

To have an idea on how to use the main functions of RNAGlib, please visit :doc:`quickstart`