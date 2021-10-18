# How to build the docs

## Requirements

* sphinx
* spinx_rtd_theme
* myst_parser

## Docstring format

* Use [this guide](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) for formatting docstrings.


## Build the docs

```
cd docs
make html
```

## View the docs

```
open build/html/index.html
```

## Push the docs

```
scp -r build flask@rnaglib.cs.mcgill.ca:/home/www/rnabayespairing/static
```

