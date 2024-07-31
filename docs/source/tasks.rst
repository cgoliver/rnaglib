``rnaglib``'s Task Module
=======================

The new tasks module allows the use and creation of a variety of machine learning tasks on RNA structure. The list and description of the task is found below, followed by tutorials on using existing tasks, as well as on developing new tasks.

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Task Name
     - Description
     - Class
     - Source
   * - RNA_CM
     - Prediction of chemical modifications at the residue level.
     - ``ChemicalModification``
     - No published instance
   * - RNA_IF
     - Prediction of nucleotide identity (sequence) at each residue.
     - ``InverseFolding``
     - Source Y
   * - 
     - Prediction of nucleotide identiy at each residue using data and splits from ``gRNAde``.
     - ``gRNAde``
     - Source Y
   * - RNA_VS
     - Scoring of candidates in virtual screening scenario based on ``RNAmigos 2.0``.
     - ``VSTask``
     - Source Z
   * - RNA_Site
     - Prediction of whether a residue is part of a binding site.
     - ``BindingSiteDetection``
     - Source W
   * - 
     - Prediction of whether a residue is part of a binding site using data and splits from ``RNASite``
     - ``BenchmarkLigandBindingSiteDetection``
     - Source W
   * - RNA_Ligand
     - Prediction of ligand identity given a binding pocket (RNA structure subgraph) using data and splits from ``GMSM``.
     - ``GMSM``
     - Source V
   * - RBP_Graph
     - Prediction of protein binding at the RNA level.
     - ``ProteinBindingDetection``
     - No published instance
   * - RBP_Node
     - Prediction of whether a residue is part of a protein binding site.
     - ``ProteinBindingSiteDetection``
     - Source T

We provide a short tutorial on (1) using an existing tasks to assess model perfomance and (2) building custom tasks using modular `rnaglib` functionality.

Code to reproduce the results included in the correspoding submission can be found in the `experiments/` directory.

Tutorial 1: Using an existing task for model evaluation
-------

`rnaglib`'s task module provides you with readymade dataset splits for your model evaluation in just a few lines of code.

0.) Generate necessary index files::

$ rnaglib_index


1.) Choose the task appropriate to your model. Here, we chose *RNA-Site*, a task instance called `LigandBindindSite` for illustration.

When instantiating the task, custom splitters or other arguments can be passed if needed.::

	from rnaglib.tasks import BindingSiteDetection
	from rnaglib.representations import GraphRepresentation

::

	task = BindingSiteDetection(root='tutorial') 
	#You can pass arguments to use a custom splitter or dataset etc. if desired.

2.) Add the representation used by your model to the task object. Voxel grid or point cloud are also possible representations; here we use a graph representation in the `pytorch-geometric` framework.::

	representation = GraphRepresentation('pyg')

	task.dataset.add_representation(representation)

3.) Lastly, split your task dataset.::

	train_ind, val_ind, test_ind = task.split()

	train_set = task.dataset.subset(train_ind)
	
	val_set = task.dataset.subset(val_ind)
	
	test_set = task.dataset.subset(test_ind)

Here you go, these splits are now ready to be used by your model of choice and can for example be passed to a `DataLoader`. For an example of a model trainig on these splits have a look at this  `simple model <https://github.com/cgoliver/rnaglib/blob/master/rnaglib/tasks/models/binding_site_model.py>`_.

Tutorial 2: Creating a new task
-------

The task module provides the logic to develop new tasks from scratch with little effort. 

1.) Start with the task type you would like to implement. In this case, we will build a residue classification task and can inherit from that class type. You can inherit directly from the `Task` class if preferred.::
	
	from rnaglib.tasks import ResidueClassificationTask

	class TutorialTask(ResidueClassificationTask):

2.) Specify your input and target variables, which in the case of a residue classification task should be node attributes.::

	 target_var = 'binding_ion'  # for example
	
	 input_var = "nt_code" # if sequence information should be used. 

3.) Next, you can define a splitter you want to use for your task. This can always be overwritten at instantiation. You can chose any available splitter object, write your own splitter object and call it here, or simply have the default_splitter return three lists of indices.::

	from rnaglib.splitters import DasSplitter

	def default_splitter(self):

		return DasSplitter()


4.) It is not mandatory but we recommend you include a static `evaluate` method with your task which you can call when training your model. In this example we will use Matthew's correlation coefficient.::

	from sklearn.metrics import matthews_corrcoef

	@staticmethod

	def evaluate(data, predictions):

		mcc = matthews\_corrcoef(data, predictions)

		return mcc

5.) In the simplest case, you just need to include the code to create the dataset and your new task is ready to go.::
	from rnaglib.data_loading import RNADataset

	def build_dataset(self, root)
	
		dataset = RNADataset(nt_targets=[self.target_var],
							nt_features=[self.input_var]
							)
		return dataset

6.) However, you may want your dataset to contain only a selection of RNA structures or you may want to use a node label not available in the base dataset or you may want to include only certain nucleotides with specific properties. In this case ``rna_filter`` andor ``annotator`` andor ``nt_filter``  can be passed to ``RNADataset``.

For example:

* ``rna_filter=lambda x: x.graph['pdbid'][0] in rnas_keep`` where rnas_keep is a list of pdbids that you want your dataset to contain.
* ``annotator=self._annotator``

A simple annotator could add a dummy variable to each node:::

	from networkx import set_node_attributes
	
	def _annotator(self, x):
		dummy = {
			node: 1
			for node, nodedata in x.nodes.items()
		}
	
		set_node_attributes(x, dummy, 'dummy')
		return x

7.) Here an example of a complete task definition (including init method). You are done now and ready to go!::
	
	from rnaglib.tasks import ResidueClassificationTask
	from rnaglib.data_loading import RNADataset
	from rnaglib.splitters import DasSplitter
	from rnaglib.utils import load_index
	from networkx import set_node_attributes
	
	class TutorialTask(ResidueClassificationTask):
	    target_var = 'binding_ion'
	    input_var = 'nt_code'
	
	    def __init__(self, root, splitter=None, **kwargs):
	        super().__init__(root=root, splitter=splitter, **kwargs)
	        pass
	    pass
	
	    def default_splitter(self):
	        return DasSplitter()
	
	    def _annotator(self, x):
	        dummy = {
	                node: 1
	                for node, nodedata in x.nodes.items()
	        }
	
	        set_node_attributes(x, dummy, 'dummy')
	        return x
	
	    def build_dataset(self, root):
	        graph_index = load_index()
	        rnas_keep = []
	
	        for graph, graph_attrs in graph_index.items():
	                if "node_" + self.target_var in graph_attrs:
	                        rnas_keep.append(graph.split(".")[0])
	
	        dataset = RNADataset(nt_targets=[self.target_var],
	                                                    nt_features=[self.input_var],
	                                                    rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep,
	                                                    annotator=self._annotator
	                                                    )
	        return dataset

8.) Don't forget to add your task name to the ``__init__.py`` file. (And if you feel like it, submit a pull request ;) )

