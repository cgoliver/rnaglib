Anatomy of a Task
-------------------------------------

If you would like to propose a new prediction task for the machine learning community. We provide the customizable ``Task`` class.

An instance of the ``Task`` class packages the following attributes:

- ``dataset``: full collection of RNAs to use in the task.
- ``splitter``: method for partitioning the dataset into train, validation, and test subsets.
- ``features_computer``: method for setting and encoding input and target variables.
- ``evaluate``: method which accepts a model and returns performance metrics.


Here is a template for a custom task::

    from rnaglib.tasks import Task
    from rnaglib.data_loading import RNADataset
    from rnaglib.splitters import Splitter 

    class MyTask(Task):

        def build_dataset(self) -> RNADataset:
            # build the task's dataset
            pass

        def default_splitter() -> Splitter:
            # return a splitter object to build train/val/test splits
            pass

        def features_computer() -> FeaturesComputer:
            # computes the task's default input and target variables
            pass
                
