RNADataset
##########

``RNADataset`` inherits the Pytorch Dataset object, and is made to hold several RNAs.
A key feature is a bidict that holds a mapping between RNA names and their index in the dataset.
This allows for constant time access to an RNA with a given name.

The RNAs contained in an RNADataset can either live in the memory, or be specified as files.
In the latter case, an RNADataset can be seen as an ordered list of file in a given directory.

.. raw:: html
    :file: ../images/fig_dataset.svg


Creating, subsetting and saving a dataset:
******************************************

If no ``dataset_path`` argument is provided, our object will use the data downloaded by rnaglib.
One can specifiy a version number and a redundancy; the following line will instantiate a dataset with
version version="2.0.2" corresponding to the non-redundant BGSU dataset (corresponding to default values) : ::

    >>> from rnaglib.dataset import RNADataset
    >>> dataset = RNADataset(in_memory=False, version="2.0.2", redundancy="nr")
    >>> dataset.all_rnas
    bidict({'1a9n': 0, '1av6': 1, '1b23': 2, ...})

One can also specify a list of names to use.
A similar result can be achieved by subsetting the initial dataset using names or ids. ::

    >>> rna_names = ['1a9n', '1av6', '1b23']
    >>> rna_ids = [0, 1, 2]
    >>> subset1 = RNADataset(in_memory=False, version="2.0.2", redundancy="nr", rna_id_subset=rna_names)
    >>> subset2 = dataset.subset(list_of_names=rna_names)
    >>> subset3 = dataset.subset(list_of_ids=rna_ids)
    >>> len(subset1) == len(subset2) == len(subset3)
    True

We can then dump and load the resulting datasets in a place of our choice: ::

    >>> subset1.save('test_subset')
    >>> subset_load = RNADataset(dataset_path='test_subset')
    >>> len(subset_load)
    3

Using a dataset:
****************

The dataset defines a __getitem__ method, allowing for index access of the dataset.
Accessing a datapoint by name is constant time thanks to the bidict, using the ``get_by_name`` method: ::

    >>> rna = subset_load[0]
    >>> rna2 = subset_load.get_by_name('1a9n')
    >>> rna == rna2
    True

In the __getitem__(), the following steps happen:

#. The rna name is retrieved, and used to build the path to the graph and to the corresponding structure
#. The result is stored in a dictionnary ``rna_dict`` with keys: ['rna','graph_path', 'cif_path']
#. [optional] Transforms present in dataset.transforms are applied to this ``rna_dict``
#. Features are computed by dataset.features_computer
#. Representations in dataset.representations are computed, adding repr.name keys in ``rna_dict``
#. ``rna_dict`` is returned

