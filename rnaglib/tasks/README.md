
# `rnaglib`'s task module

The new tasks module allows the use and creation of a variety of machine learning tasks on RNA structure. 
We provide a short tutorial on (1) using an existing tasks to assess model perfomance and (2) building custom tasks using modular `rnaglib` functionality.

Code to reproduce the results included in the correspoding submission can be found in the `experiments/` directory.



## Tutorial 1: Using an existing task for model evaluation
`rnaglib`'s task module provides you with readymade dataset splits for your model evaluation in just a few lines of code.

1.) Choose the task appropriate to your model. Here, we chose _RNA-Site_, a task instance called `LigandBindindSite` for illustration.
When instantiating the task, custom splitters or other arguments can be passed if needed.
 ```
from rnaglib.tasks import LigandBindingSite
from rnaglib.representations import GraphRepresentation
```

```
task = BindingSiteDetection(root='tutorial) # You can pass arguments to use a custom splitter or dataset etc. if desired.
```

2.) Add the representation used by your model to the task object. Voxel grid or point cloud are also possible representations; here we use a graph representation in the `pytorch-geometric` framework.

```
representation = GraphRepresentation('pyg')
task.dataset.add_representation(representation)
```

3.) Lastly, split your task dataset.
```
train_ind, val_ind, test_ind = task.split()
train_set = task.dataset.subset(train_ind)
val_set = task.dataset.subset(val_ind)
test_set = task.dataset.subset(test_ind)
```

Here you go, these splits are now ready to be used by your model of choice and can for example be passed to a `DataLoader`.

## Tutorial 2: Creating a new task

todo
