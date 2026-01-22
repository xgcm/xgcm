# Installation

## Requirements

xgcm is compatible with python 3 (>= version 3.9). It requires [xarray]
(>= version 0.20.0) [numpy] and [dask].

## Installation from Conda Forge

The easiest way to install xgcm along with its dependencies is via conda
forge:

```
conda install -c conda-forge xgcm
```


## Installation from Pip

An alternative is to use pip:

```
pip install xgcm
```

This will install the latest release from
[pypi](https://pypi.python.org/pypi).

## Installation from GitHub

xgcm is under active development. To obtain the latest development version,
you may clone the [source repository](https://github.com/xgcm/xgcm)
and install it using pip:

```
pip install git+https://github.com/xgcm/xgcm.git
```

More comprehensive instructions for installing a development environment can be found in the [Contributor Guide](https://xgcm.readthedocs.io/en/latest/contributor_guide.html).

Users are encouraged to [fork](https://help.github.com/articles/fork-a-repo/)
xgcm and submit [issues] and [pull requests].


[dask]: http://dask.pydata.org
[numpy]: https://numpy.org
[xarray]: http://xarray.pydata.org
[issues]: https://github.com/xgcm/xgcm/issues
[pull requests]: https://github.com/xgcm/xgcm/pulls
