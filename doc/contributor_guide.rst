.. _contributor_guide:

Contributor Guide
-----------------

**xgcm** is meant to be a community driven package and we welcome feedback and
contributions.

Did you notice a bug? Are you missing a feature? A good first starting place is to
open an issue in the `github issues page <https://github.com/xgcm/xgcm/issues>`_.


In order to contribute to xgcm, please fork the repository and submit a pull request.
A good step by step tutorial for this can be found in the
`xarray contributor guide <https://xarray.pydata.org/en/stable/contributing.html#working-with-the-code>`_.


Environments
^^^^^^^^^^^^
The easiest way to start developing xgcm pull requests,
is to install one of the conda environments provided in the `ci folder <https://github.com/xgcm/xgcm/tree/master/ci>`_::

    conda env create -f ci/environment-py36.yml

We use `black <https://github.com/python/black>`_ as code formatter and pull request will
fail in the CI if not properly formatted.

All conda environments contain black and you can reformat code using::

    black xgcm

`pre-commit <https://pre-commit.com/>`_ provides an automated way to reformat your code
prior to each commit. Simply install pre-commit::

    pip install pre-commit

and install it in the xgcm root directory with::

    pre-commit install

and your code will be properly formatted before each commit.
