RNA Tasks
#########

A "task" is a machine learning term refering to a learning problem formulated in a precise way with a precise measure of performance.
Tasks are mostly used to benchmark modeling approaches in a fair and comparable way.
While tasks are not necessarily directly transferable to an actual application, the closer they get to relevance, the higher
the chance that the best model on a benchmark is also the best for applications.

In *RNAGlib*, we propose a set of benchmark tasks relevant to the modeling of RNA structures.
A detailed description of the proposed tasks can be found :doc:`here<../data_reference/available_tasks>`.

the ``Task`` object uses all objects previously described in this tutorial section (RNA, RNATransforms, RNADataset, DSTransforms).
As sketched below, it contains:

#. an annotated dataset
#. train, validation and test splits
#. precise metrics computation


.. raw:: html
    :file: ../images/fig_task.svg


Creating a Task
***************

For complete examples on how to build a task, see :doc:`this detailed tutorial<../tutorials/tuto_custom_task>` or source code of of the provided tasks.

Briefly, the idea is to implement a subclass of Task, with process() and post_process() methods.
The process() function defines the steps to select and annotate the data (using RNATransforms).
The post_process() ones compute distance matrices between points and optionnally remove redundancy (using DSTransforms).

In addition, one should implement a get_task_vars() which creates a FeaturesComputer with the features to use and predict,
and a default_splitter() that returns the splitter one wants to use.
Those steps are summarized below in a simplified version of the RNA_CM task: ::

    from rnaglib.tasks import ResidueClassificationTask
    from rnaglib.dataset import RNADataset
    from rnaglib.transforms import FeaturesComputer
    from rnaglib.dataset_transforms import RandomSplitter


    class DummyTask(ResidueClassificationTask):
        input_var = "nt_code"
        target_var = "is_modified"
        name = "rna_dummy"

        def __init__(self, **kwargs):
            super().__init__(additional_metadata={"multi_label": False}, **kwargs)

        def process(self) -> RNADataset:
            return RNADataset(in_memory=self.in_memory, debug=True)

        def post_process(self):
            pass

        def get_task_vars(self) -> FeaturesComputer:
            return FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)

        @property
        def default_splitter(self):
            return RandomSplitter()


Loading a Task
**************

Once created, tasks can be saved and loaded seamlessly, such that we the following output: ::

    >>> task = DummyTask(root="test_task")
    >>> task = DummyTask(root="test_task")
    Task was found and not overwritten

Observe that on second call, the computations were not run a second time.
Moreover, for the set of proposed tasks (available in the ``rnaglib.task.TASKS`` variable), we host precomputed tasks on Zenodo. ::

    >>> from rnaglib.tasks import BindingSite
    >>> task_site = BindingSite(root='my_root')
    Downloading task dataset from Zenodo...
    Done

Users can access our tasks by id, using ``rnaglib.task.get_task`` small function.


Using a Task
************

Once equiped with a Task, users can get datasets or dataloaders from it and train machine learning models.

    >>> from rnaglib.transforms import GraphRepresentation
    >>> task_site.add_representation(GraphRepresentation(framework='pyg'))
    >>> train_loader, _, test_loader = task_site.get_split_loaders()
    >>> for batch in train_loader:
    >>>     print(batch['graph']) # train
    >>>     break
    DataBatch(x=[20, 4], edge_index=[2, 54], edge_attr=[54], y=[20], batch=[20], ptr=[2])

Importantly, the task is equipped with a principled way to compute metrics over predictions.  ::

    >>> import torch
    >>> task_site.set_loaders(batch_size=1, recompute=False)
    >>> all_preds, all_probs, all_labels = [], [], []
    >>> for batch in task_site.train_dataloader:
    >>>     graph = batch['graph']
    >>>     label = graph.y
    >>>     probs = torch.rand(label.shape)  # insert predictions here
    >>>     all_labels.append(label)
    >>>     all_probs.append(probs)
    >>>     all_preds.append(torch.round(probs))
    >>>
    >>> task_site.compute_metrics(all_preds, all_probs, all_labels)
    {'accuracy': 0.4943, 'auc': 0.4765, 'balanced_accuracy': 0.4794, ... 'global_auc': 0.4846}

