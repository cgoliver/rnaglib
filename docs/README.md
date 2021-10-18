# How to build the docs

## Requirements

* sphinx
* spinx_rtd_theme
* myst_parser


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

