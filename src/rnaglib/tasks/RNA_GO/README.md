# RNA-GO

In this directory you can find the implementation of the `RNA-GO` task, that is a close equivalent to the Go-terms
task for proteins as introduced by DeepFRI.

It provides a dataset of RNA along some GO annotations.
These annotations were obtained from the files produced by RFAM.
GO annotations with less than 50 or more than 1000 examples were discarded.

Please note that some RNA have more than one annotation.