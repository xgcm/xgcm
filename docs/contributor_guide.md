# Contributor Guide {#contributor-guide}

**xgcm** is meant to be a community driven package and we welcome feedback and
contributions.

Did you notice a bug? Are you missing a feature? A good first starting place is to
open an issue in the [github issues page](https://github.com/xgcm/xgcm/issues).

Want to show off a cool example using xgcm? Please consider contributing to [xgcm-examples](https://github.com/xgcm/xgcm-examples). Notebooks from there will be rendered in [pangeo-gallery](https://gallery.pangeo.io/repos/xgcm/xgcm-examples/).


In order to contribute to xgcm, please fork the repository and submit a pull request.
A good step by step tutorial for this can be found in the
[xarray contributor guide](https://xarray.pydata.org/en/stable/contributing.html#working-with-the-code).


## Development Setup

We use [Pixi](https://pixi.sh) for development. After forking and cloning the repository:

```
pixi install
```

This installs xgcm in editable mode with all development dependencies.

## Running Tests

```
pixi run tests
```

## Code Formatting

We use [pre-commit](https://pre-commit.com/) for code formatting. To run linting:

```
pixi run lint
```

To auto-format before each commit, install the pre-commit hooks:

```
pre-commit install
```

## Building the Documentation

```
pixi run -e docs docs
```


## How to release a new version of xgcm (for maintainers only)

The process of releasing at this point is very easy.

We need only two things: A PR to update the documentation and a release on github.

1. Make sure that all the new features/bugfixes etc are appropriately documented in `docs/whats-new.md`, add the date to the current release and make an empty (unreleased) entry for the next minor release as a PR.
2. Navigate to the 'tags' symbol on the repos main page, click on 'Releases' and on 'Draft new release' on the right. Add the version number and a short description and save the release.

From here the github actions take over and package things for [Pypi](https://pypi.org/project/xgcm/).
The conda-forge package will be triggered by the Pypi release and you will have to approve a PR in [xgcm-feedstock](https://github.com/conda-forge/xgcm-feedstock). This takes a while, usually a few hours to a day.

Thats it!

## How to synchronize examples from xgcm-examples

Most of the example notebooks in this documentation are located in the seperate repo [xgcm-examples](https://github.com/xgcm/xgcm-examples), which is automatically linked to [pangeo gallery](https://gallery.pangeo.io). These examples are synced into this documentation using git submodules.
Currently updates in the example repo need to be manually synced to this repo with the following steps:

From the xgcm root directory do:

```
cd docs/xgcm-examples
```

If this directory is empty, it means your original install did not pull the submodule; to configure the submodule, do:

```
git submodule update --init
```

You are now in a seperate git repository and can pull all updates:

```
git pull
```

Now navigate back to the xgcm repo:

```
cd -
```

And commit, push like usual to create a pull request.
