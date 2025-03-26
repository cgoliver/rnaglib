Build your own prediction task
-----------------------------------

An ``rnaglib.Task`` object packages everything you need to train a model for a particular biological problem.
This means you can also create your own tasks so that others can work on them
and share their best models.


If you would like to propose a new prediction task for the machine learning community. You just have to implement a few methos in a subclass of the``Task`` class.

An instance of the ``Task`` class packages the following attributes:

- ``dataset``: full collection of RNAs to use in the task.
- ``splitter``: method for partitioning the dataset into train, validation, and test subsets.
- ``target_vars``: method for setting and encoding input and target variables.
- ``evaluate``: method which accepts a model and returns performance metrics.
- ``metadata``: this is a simple (optional) dictionary that holds useful info about the task (e.g. task type, number of classes, etc.)

Once the task processing is complete, all task data is dumped into ``root`` which is a path passed to the task init method.


Here is a minimal template for a custom task::

    from rnaglib.tasks import Task
    from rnaglib.dataset import RNADataset
    from rnaglib.dataset_transforms import Splitter 
    from rnaglib.transforms import FeaturesComputer

    class MyTask(Task):

        def __init__(self, root):
            super().__init__(root)

        def process(self) -> RNADataset:
            # build the task's dataset
            # ...
            pass

        @property
        def default_splitter() -> Splitter:
            # return a splitter object to build train/val/test splits
            # ...
            pass
            
        def get_task_vars() -> FeaturesComputer:
            # computes the task's default input and target variables
            # managed by creating a FeaturesComputer object
            # ...
            pass

        def init_metadata(self):
            return {'task_name': 'my task'}


In this tutorial we will walk through the steps to create a task with the aim of predicting for each residue, whether or not it will be chemically modified.

Types of tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

Tasks can operate at the residue, edge, and whole RNA level. 
Boilerplate code for model evaluation and loading would be affected depending on the choice of level.
For that reason we create sub-classes of the ``Task`` clas which you can use to avoid re-coding such things.


Since chemical modifications are applied to residues, Let's build a residue-level binary classification task.::

    from rnaglib.tasks import ResidueClassificationTask

    class ChemicalModification(ResidueClasificationTask):
        ....




1. Create the task's dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

Each task needs to define which RNAs among all RNAs in the PDB to use. Typically this involves filtering a whole dataset of available RNAs by certain attributes to retain only the ones that contain certain annotations or pass certain criteria (e.g. size, origin, resolution, etc.).

You are free to do this in any way you like as long as after ``Task.process()`` is called, an ``RNADataset`` object with the relevant RNAs is returned.

To make things easier you can take advantage of the ``rnaglib.Tranforms`` library which provides funcionality for manipulating datasets of RNAs.

Let's define a ``Task.process()`` method which builds a dataset with a single criterion:

* Only keep RNAs that contain at least one chemically modified residue

The ``Transforms`` library provides a filter which checks that an RNA's residues are of a desired value. ::

    from rnaglib.dataset import RNADataset
    from rnaglib.tasks import ResidueClassificationTask
    from rnaglib.transforms import ResidueAttributeFilter
    from rnaglib.transforms import PDBIDNameTransform

    class ChemicalModification(ResidueClassificationTask):
        def process(self) -> RNADataset:
            # grab a full set of available RNAs
            rnas = RNADataset()

            filter = ResidueAttributeFilter(attribute='is_modified',
                                            val_checker=lambda val: val == True
                                            )

            rnas = filter(rnas)

            rnas = PDBIDNameTransform()(rnas)
            dataset = RNADataset(rnas=[r["rna"] for r in rnas])
            return dataset

            pass


Applying the filter gives us a new list containing only the RNAs that passed the filter. The last thing we need to do is assign a ``name`` value to each RNA so that they can be properly managed by the ``RNADataset``. We assign the PDBID as the name of each item in our dataset using the ``PDBIDNameTransform``.

Now we just create a new ``RNADataset`` object using the reduced list. The dataset object requires a list and not a generator so we just unroll before passing it.

That's it now you just return the new ``RNADataset`` object.

2. Set the task's variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

Apart from the RNAs themselves, the task needs to know which variables are relevant. In particular we need to set the prediction target. Additionally we can set some default input features, which are always provided. The user can always add more input features once a Task is intantiated if he/she desires by manipulating ``task.dataset.features_computer`` but at the minimum we need to define target variables.::

    from rnaglib.dataset import RNADataset
    from rnaglib.tasks import ResidueClassificationTask
    from rnaglib.transforms import ResidueAttributeFilter
    from rnaglib.transforms import PDBIDNameTransform
    from rnaglib.transforms import FeaturesComputer

    class ChemicalModification(ResidueClassificationTask):
        def process(self) -> RNADataset:
            ...
            pass

        def get_task_vars(self) -> FeaturesComputer:
            return FeaturesComputer(nt_features=['nt_code'], nt_targets=['is_modified'])


Here we simply have a nucleotide level target so we pass the ``'is_modified'`` attribute to the ``FeaturesComputer`` object. This will take care of selecting the residue when encoding the RNA into tensor form. In addition we provide the nucleotide identity (``'nt_code'``) as a default input feature.


3. Train/val/test splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

The last necessary step is to define the train, validation and test subsets of the whole dataset. Once these are set, the task's boilerplate will take care of generating the appropriate loaders.

To set the splits, you implement the ``default_splitter()`` method which returns a ``Splitter`` object. A ``Splitter`` object is simply a callable which accepts a dataset and returns three lists of indices representing the train, validation and test subsets.

You can select from the library of implemented splitters of implement your own.

For this example, we will split the RNAs by structural similarity using US-align.::

    from rnaglib.dataset import RNADataset
    from rnaglib.tasks import ResidueClassificationTask

    from rnaglib.transforms import ResidueAttributeFilter
    from rnaglib.transforms import PDBIDNameTransform
    from rnaglib.transforms import FeaturesComputer

    from rnaglib.dataset_transforms import Splitter, ClusterSplitter, StructureDistanceComputer

    class ChemicalModification(ResidueClassificationTask):
        def process(self) -> RNADataset:
            ...
            pass

        def get_task_vars(self) -> FeaturesComputer:
            return FeaturesComputer(nt_features=['nt_code'], nt_targets=['is_modified'])

        @property
        def default_splitter(self) -> Splitter:
            return ClusterSplitter(distance_name="USalign", similarity_threshold=0.6)


Now our splits will guarantee a maximum structural similarity of 0.6 between them according to USAlign metrics.

Check out the Splitter class for a quick guide on how to create your own splitters.

Note that this is only setting the default method to use for splitting the dataset. If a user wants to try a different splitter it can be pased to the task's init.

That's it! Your task is now fully defined and can be used in model training and evaluation.

Here is the ful task implementation::


    from rnaglib.dataset import RNADataset
    from rnaglib.tasks import ResidueClassificationTask
    from rnaglib.transforms import FeaturesComputer
    from rnaglib.transforms import ResidueAttributeFilter
    from rnaglib.transforms import PDBIDNameTransform
    from rnaglib.dataset_transforms import Splitter, ClusterSplitter


    class ChemicalModification(ResidueClassificationTask):
        """Residue-level binary classification task to predict whether or not a given
        residue is chemically modified.
        """

        target_var = "is_modified"

        def __init__(self, root, splitter=None, **kwargs):
            super().__init__(root=root, splitter=splitter, **kwargs)

        def get_task_vars(self):
            return FeaturesComputer(nt_targets=self.target_var)

        def process(self):
            rnas = ResidueAttributeFilter(
                attribute=self.target_var, value_checker=lambda val: val == True
            )(RNADataset(debug=self.debug))
            rnas = PDBIDNameTransform()(rnas)
            dataset = RNADataset(rnas=[r["rna"] for r in rnas])
            return dataset

        def default_splitter(self) -> Splitter:
            return ClusterSplitter(distance_name="USalign", similarity_threshold=0.6)


Metadata
~~~~~~~~~~~~~~~

Each task holds a ``metadata`` attribute which is a simple dictionary holding useful information about the task (e.g. number of classes, task type, name, description). You can modify this during task setup and it is saved to disk once the task is built.

Task saving and loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the task is completely built (dataset and splits), the task class automatically calls its ``write()`` method which dumps to the ``root`` directory all the information necessary to skip processing if the task is re-loaded.

Your ``root`` directory will look something like::

        my_root/
            train_idx.txt
            val_idx.txt
            test_idx.txt
            task_id.txt
            metadata.json
            dataset/
                1abc.json
                2xzy.json
                ...

The task folder contains 3 ``.txt`` files with the indices for each split. The ``metadata.json`` file stores any additional information relevant to the task, the ``task_id.txt`` file holds a unique identifier for the task which is built by hashing all the RNAs and splits so that if anything about the task changes the ID will be different, and bfinally the ``dataset/`` folder holds ``.json`` files which can be loaded into RNA dicts and used to re-instantiate the task.



Customize Splitting
~~~~~~~~~~~~~~~~~~~~~~

We provide some pre-defined splitters for sequence and structure-based splitting. If you have other criteria for splitting you can subclass the ``Splitter`` class. All you have to do is implement the ``__call__()`` method which takes a dataset and returns three lists of indices::

    class Splitter:
        def __init__(self, split_train=0.7, split_valid=0.15, split_test=0.15):
            assert sum([split_train, split_valid, split_test]) == 1, "Splits don't sum to 1."
            self.split_train = split_train
            self.split_valid = split_valid
            self.split_test = split_test
            pass

        def __call__(self, dataset):
            return None, None, None


The ``__call__(self, dataset)`` method returns three lists of indices from the given ``dataset`` object.

The splitter can be initiated with the desired proportions of the dataset for each subset.
