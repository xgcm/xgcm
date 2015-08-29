 #!/usr/bin/env python
import os
import re
import sys
import warnings

from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''


DISTNAME = 'xgcm'
LICENSE = 'Apache'
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
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['xray']
TESTS_REQUIRE = ['nose >= 1.0']

if sys.version_info[:2] < (2, 7):
    TESTS_REQUIRE += ["unittest2 == 0.5.1"]

DESCRIPTION = "General circulation model analysis in python with xray"
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
      tests_require=TESTS_REQUIRE,
      url=URL,
      test_suite='nose.collector',
      packages=find_packages())
