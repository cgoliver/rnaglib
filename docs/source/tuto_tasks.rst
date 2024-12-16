Using an existing task for model evaluation
-----------------------------------------------------------

`rnaglib`'s task module provides you with readymade dataset splits for your model evaluation in just a few lines of code.

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


