Command Line Utilities
-------------------------


We provide several command line utilities which you can use to set up
the rnaglib environment.


Database building
~~~~~~~~~~~~~~~~~~~~~~~~

To build or update a local database of RNA structures along with their annotations,
you can use the ``rnaglib_prepare_data`` command line utility.


::

    $ rnaglib_prepare_data -s structures/ --tag first_build -o builds/ -d

Database Indexing
~~~~~~~~~~~~~~~~~~~

Indexing a database collects information about annotations present in a
database to enable rapid access of particular RNAs given some desired
properties.::

    $ rnaglib_index


