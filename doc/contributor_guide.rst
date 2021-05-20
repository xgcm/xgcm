.. _contributor_guide:

Contributor Guide
-----------------

**xgcm** is meant to be a community driven package and we welcome feedback and
contributions.

Did you notice a bug? Are you missing a feature? A good first starting place is to
open an issue in the `github issues page <https://github.com/xgcm/xgcm/issues>`_.

Want to show off a cool example using xgcm? Please consider contributing to [xgcm-examples](https://github.com/xgcm/xgcm-examples). Notebooks from there will be rendered in [pangeo-gallery](https://gallery.pangeo.io/repos/xgcm/xgcm-examples/).


In order to contribute to xgcm, please fork the repository and submit a pull request.
A good step by step tutorial for this can be found in the
`xarray contributor guide <https://xarray.pydata.org/en/stable/contributing.html#working-with-the-code>`_.


Environments
^^^^^^^^^^^^
The easiest way to start developing xgcm pull requests,
is to install one of the conda environments provided in the `ci folder <https://github.com/xgcm/xgcm/tree/master/ci>`_::

    conda env create -f ci/environment-py3.8.yml

Activate the environment with::

    conda activate test_env_xgcm

Finally install xgcm itself in the now activated environment::

    pip install -e .

A good first step is to check if all the tests pass locally::

    pytest -v

And now you can develop away...

Code Formatting
^^^^^^^^^^^^^^^

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

How to release a new version of xgcm (for maintainers only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The process of releasing at this point is very easy.

We need only two things: A PR to update the documentation and and making a release on github.

1. Make sure that all the new features/bugfixes etc are appropriately documented in `doc/whats-new.rst`, add the date to the current release and make an empty (unreleased) entry for the next minor release as a PR.
2. Navigate to the 'tags' symbol on the repos main page, click on 'Releases' and on 'Draft new release' on the right. Add the version number and a short description and save the release.

From here the github actions take over and package things for `Pypi <https://pypi.org/project/xgcm/>`_.
The conda-forge package will be triggered by the Pypi release and you will have to approve a PR in `xgcm-feedstock <https://github.com/conda-forge/xgcm-feedstock>`_. This takes a while, usually a few hours to a day.

Thats it!

How to synchronize examples from xgcm-examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Most of the example notebooks in this documentation are located in the seperate repo `xgcm-examples <https://github.com/xgcm/xgcm-examples>`_, which is automatically linked to `pangeo gallery <https://gallery.pangeo.io>`_. These examples are synced into this documentation using git submodules.
Currently updates in the example repo need to be manually synced to this repo with the following steps:

From the xgcm root directory do::

    cd doc/xgcm-examples

You are now in a seperate git repository and can pull all updates::

    git pull

Now navigate back to the xgcm repo::

    cd -

And commit, push like usual to create a pull request.::
