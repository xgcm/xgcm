import os
import versioneer
from setuptools import find_packages, setup

here = os.path.dirname(__file__)
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = ["xarray>=0.14.1", "dask", "numpy", "future", "docrep<=0.2.7"]
doc_requires = [
    "sphinx",
    "sphinxcontrib-srclinks",
    "sphinx-pangeo-theme",
    "numpydoc",
    "IPython",
    "nbsphinx",
]

extras_require = {
    "complete": install_requires,
    "docs": doc_requires,
}
extras_require["dev"] = extras_require["complete"] + [
    "pytest",
    "pytest-cov",
    "flake8",
    "black",
    "codecov",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=long_description,
    long_description_content_type="text/x-rst",
)
