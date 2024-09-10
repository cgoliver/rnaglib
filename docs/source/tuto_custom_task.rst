Creating a custom task
-------------------------------------

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


