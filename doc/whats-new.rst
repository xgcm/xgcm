.. currentmodule:: xarray

What's New
==========

.. _whats-new.0.3.0:

v0.3.0 (31 January 2020)
-----------------
This release adds support for `model grid metrics <file:///Users/juliusbusecke/code/xgcm/doc/_build/html/grid_metrics.html#Grid-Metrics>`_ , bug fixes and extended documentation.

Breaking changes
~~~~~~~~~~~~~~~~

New Features
~~~~~~~~~~~~
- Support for 'grid-aware' average and cumsum using :py:class:`~xgcm.Grid.average` and :py:class:`~xgcm.Grid.cumsum` (:issue:`162`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Support for 'grid-aware' integration using :py:class:`~xgcm.Grid.integrate` (:issue:`130`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

Bug fixes
~~~~~~~~~
- Fix for broken stale build (:issue:`155`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Fixed bug in handling of grid metrics. (:issue:`136`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Fixed bug in
  :py:class:`~xgcm.Grid.derivative` (:issue:`132`).
  By `Timothy Smith <https://github.com/timothyas>`_.

Documentation
~~~~~~~~~~~~~
- Added docs for :py:class:`~xgcm.Grid.derivative` (:issue:`163`)
  By `Timothy Smith <https://github.com/timothyas>`_.

- Add binderized examples (:issue:`141`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Simplify example notebooks (:issue:`140`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Execute example notebook during doc build (:issue:`138`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Added contributor guide to docs (:issue:`137`).
  By `Julius Busecke <https://github.com/jbusecke>`_.


Internal Changes
~~~~~~~~~~~~~~~~
- Added GitHub Action to publish xgcm to PyPI on release (:issue:`170`).
  By `Anderson Banihirwe <https://github.com/andersy005>`_.

- Reorganized environment names for CI (:issue:`139`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Added automatic code formatting via `black <https://black.readthedocs.io/en/stable/>`_ (:issue:`131`).
  By `Julius Busecke <https://github.com/jbusecke>`_.


v0.2.0 (21 March 2019)
-----------------
Changes not documented for this release

v0.1.0 (13 July 2014)
-----------------
Changes not documented for this release

Initial release.
