.. currentmodule:: xgcm

What's New
===========

v0.6.0 (unreleased)
-------------------

.. _whats-new.0.6.0:


v0.5.2 (2021/5/27)
-------------------

.. _whats-new.0.5.2:

Bug fixes
~~~~~~~~~
- Raise more useful errors when datasets are provided as arguments to grid.transform (:pull:`329`, :issue:`328`). By `Julius Busecke <https://github.com/jbusecke>`_.


Documentation
~~~~~~~~~~~~~
- Updated Realistic Data examples in `Transforming Vertical Coordinates <https://xgcm.readthedocs.io/en/latest/transform.html>`_ (:pull:`322`)
  By `Dianne Deauna <https://github.com/jdldeauna>`_.

- Migrated model example notebooks to `xgcm-examples <https://github.com/xgcm/xgcm-examples>`_ which integrates with `pangeo gallery <https://gallery.pangeo.io/repos/xgcm/xgcm-examples/>`_ (:pull:`294`)
  By `Julius Busecke <https://github.com/jbusecke>`_.

v0.5.1 (2020/10/16)
-------------------

.. _whats-new.0.5.1:

Bug fixes
~~~~~~~~~
- Add support for older numba versions (<0.49) (:pull:`263`, :issue:`262`). By `Navid Constantinou <https://github.com/navidcy>`_.



v0.5.0 (2020/9/28)
------------------
.. _whats-new.0.5.0:

New Features
~~~~~~~~~~~~
- :py:meth:`~xgcm.grid.Grid.transform` and :py:meth:`~xgcm.grid.Axis.transform` now enable 1-dimensional coordinate transformation (:pull:`205`, :issue:`222`).
  By `Ryan Abernathey <https://github.com/rabernat>`_ and `Julius Busecke <https://github.com/jbusecke>`_.

Bug fixes
~~~~~~~~~
- More reliable handling of missing values in :py:meth:`Grid.average`. Missing values between data and metrics do not have to be aligned by the user anymore. (:pull:`259`). By `Julius Busecke <https://github.com/jbusecke>`_.

- Remove outdated `example_notebooks` folder (:pull:`244`, :issue:`243`). By `Nikolay Koldunov <https://github.com/koldunovn>`_ and `Julius Busecke <https://github.com/jbusecke>`_.
.. _whats-new.0.5.0:


v0.4.0 (2020/9/2)
-------------------------
New Features
~~~~~~~~~~~~
- Support for keeping compatible coordinates in most Grid operations (:issue:`186`).
  By `Aurélien Ponte <https://github.com/apatlpo>`_.

- Support for specifying default ``boundary`` and ``fill_value`` in the :py:class:`Grid` constructor.
  Default values can be overridden in individual method calls (e.g. :py:meth:`Grid.interp`) as usual.
  By `Deepak Cherian <https://github.com/dcherian>`_.

  Bug fixes
~~~~~~~~~
- Fix for parsing fill_values as dictionary (:issue:`218`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Complete refactor of the CI to github actions (:issue:`214`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

.. _whats-new.0.4.0:

v0.3.0 (31 January 2020)
-------------------------
This release adds support for `model grid metrics <https://xgcm.readthedocs.io/en/latest/grid_metrics.html>`_ , bug fixes and extended documentation.

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
----------------------
Changes not documented for this release

v0.1.0 (13 July 2014)
----------------------
Changes not documented for this release

Initial release.
