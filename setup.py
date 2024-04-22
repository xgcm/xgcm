import os

from setuptools import find_packages, setup

here = os.path.dirname(__file__)
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "xarray>=0.20.0",
    "dask",
    "numpy",
    "future",
]

dev_require = install_requires + [
    "pytest",
    "pytest-cov",
    "scipy",
    "flake8",
    "black",
    "codecov",
    "pandoc",
    "numba",
    "intake-esm",
    "requests",
    "gcsfs",
    "cf_xarray",
    "sphinx==4.5.0",
    "sphinx_rtd_theme",
    "netcdf4",
    "cartopy",
    "matplotlib",
    "numpydoc",
    "sphinxcontrib-applehelp<1.0.5",
    "sphinxcontrib-devhelp<1.0.6",
    "sphinxcontrib-htmlhelp<2.0.5",
    "sphinxcontrib-serializinghtml<1.1.10",
    "sphinxcontrib-qthelp<1.0.7",
    "pangeo-sphinx-book-theme",
    "sphinx-copybutton",
    "sphinx-panels",
    "sphinxcontrib-srclinks",
    "sphinx_rtd_theme",
    "sphinx-pangeo-theme",
    "ipykernel",
    "ipython>=8.5.0",
    "nbsphinx",
    "jupyter_client",
    "pickleshare",
    "pre-commit",
]

setup(
    name="xgcm",
    description="General Circulation Model Postprocessing with xarray",
    url="https://github.com/xgcm/xgcm",
    author="xgcm Developers",
    author_email="rpa@ldeo.columbia.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=install_requires,
    extras_require={"dev": dev_require},
    python_requires=">=3.9",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    setup_requires="setuptools_scm",
    use_scm_version={
        "write_to": "xgcm/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
