 #!/usr/bin/env python
import os
import re
import sys
import warnings

from setuptools import setup, find_packages

ISRELEASED = False
VERSION = '0.0.0'

DISTNAME = 'xgcm'
LICENSE = 'MIT'
AUTHOR = 'xgcm Developers'
AUTHOR_EMAIL = 'rpa@ldeo.columbia.edu'
URL = 'https://github.com/xgcm/xgcm'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['xarray', 'dask', 'numpy', 'future', 'docrep']
SETUP_REQUIRES = ['pytest-runner']
TESTS_REQUIRE = ['pytest >= 2.8', 'coverage']

DESCRIPTION = "General circulation model analysis in python with xarray"
LONG_DESCRIPTION = """To be written.
"""

setup(name=DISTNAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      install_requires=INSTALL_REQUIRES,
      setup_requires=SETUP_REQUIRES,
      tests_require=TESTS_REQUIRE,
      url=URL,
      packages=find_packages())
