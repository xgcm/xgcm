.. currentmodule:: xgcm

What's New
===========

v0.9.0 (unreleased)
-------------------

.. _whats-new.0.9.0:

New Features
~~~~~~~~~~~~
- Methods for autoparsing of dataset metadata to construct a :py:class:`xgcm.Grid` class have been added.
  Currently these include restructred functionality for the COMODO conventions (already in xgcm) and the
  addition of SGRID conventions (:issue:`109`, :pull:`559`).
  By `Jack Atkinson <https://github.com/jatkinson1000>`_.


Breaking Changes
~~~~~~~~~~~~~~~~
- All computation methods on the :py:class:`xgcm.Axis` class have been removed, in favour of using the corresponding
  methods on the :py:class:`xgcm.Grid` object. The :py:class:`xgcm.Axis` class has also been removed from public API.
  (:issue:`405`, :pull:`557`).
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.

- All functionality for generating c-grid dimensions on incomplete datasets via :py:meth:`xgcm.Grid.autogenerate`,  was removed (:pull:`557`).
   By `Julius Busecke <https://github.com/jbusecke>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Switch CI environment setup to micromamba (:issue:`576`, :pull:`577`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- pre-commit autoupdate frequency reduced (:pull:`563`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

Documentation
~~~~~~~~~~~~~

Bugfixes
~~~~~~~~
- Fix bug in :py:meth:`xgcm.padding._maybe_rename_grid_positions` where dimensions were assumed to have coordinate
  values leading to errors with ECCO data. (:issue:`531`, :issue:`595`, :pull:`597`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Remove remaining mentions of `extrapolate` as boundary option (:pull:`602`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Fix broken docs build due to broken backwards compatibility in sphinx extensions (:pull:`631`)
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Fix bug that did not allow to create grids with faceconnections if the face dimension was coordinate-less. (:issue:`616`, :pull:`616`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Fix bug in :py:meth:`xgcm.padding._maybe_rename_grid_positions` where dimensions were assumed to have coordinate values leading to errors with ECCO data. (:issue:`531`, :issue:`595`, :pull:`597`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

v0.8.1 (2022/11/22)
-------------------

.. _whats-new.0.8.1:

New Features
~~~~~~~~~~~~

Breaking Changes
~~~~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

- Rewrote cumsum to use a different code path from :py:func:`~xgcm.apply_as_grid_ufunc` internally,
  which makes it less susceptible to subtle bugs like the one reported in :issue:`507`. (:pull:`558`).
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.

Documentation
~~~~~~~~~~~~~

- Improved error message to suggest rechunking to a single chunk when trying to perform disallowed operations
  along chunked core dims.
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.

Bugfixes
~~~~~~~~

- Fix bug where chunked core dims of only a single chunk triggered errors. (:pull:`558`, :issue:`518`, :issue:`522`)
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.


v0.8.0 (2022/06/14)
-------------------

.. _whats-new.0.8.0:

New Features
~~~~~~~~~~~~

- Addition of logarithmic interpolation to transform (:pull:`483`).
  By `Jonathan Thielen <https://github.com/jthielen>`_.

Breaking Changes
~~~~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

- Switching code linting to the pre-commit.ci service (:pull:`490`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

Documentation
~~~~~~~~~~~~~

- Fix 'suggest edits' button in docs (:pull:`512`, :issue:`503`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

Bugfixes
~~~~~~~~

- Fix formatting of the CITATION.cff file (:pull:`500`).
  By `Julius Busecke <https://github.com/jbusecke>`_.
- Fix bug with cumsum when data chunked with dask. (:pull:`415`, :issue:`507`)
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.

v0.7.0 (2022/4/20)
-------------------

.. _whats-new.0.7.0:

New Features
~~~~~~~~~~~~

- Turn numpy-style ufuncs into grid-aware "grid-ufuncs" via new functions :py:meth:`~xgcm.grid_ufunc.apply_as_grid_ufunc`
  and :py:meth:`~xgcm.grid_ufunc.as_grid_ufunc`. (:pull:`362`, :issue:`344`)
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.

- Padding of vector fields for complex topologies via a dictionary-like syntax has been added (:pull:`459`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

Breaking Changes
~~~~~~~~~~~~~~~~

- Removed the ``extrapolate`` boundary option (:pull:`470`).
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- All computation methods on the `Grid` object are now re-routed through :py:meth:`~xgcm.grid_ufunc.apply_as_grid_ufunc`.
  By `Thomas Nicholas <https://github.com/tomnicholas>`_.

Documentation
~~~~~~~~~~~~~

- Switch to pangeo-book-scheme (:pull:`482`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Add CITATION.cff file (:pull:`450`).
  By `Julius Busecke <https://github.com/jbusecke>`_.


v0.6.1 (2022/02/15)
-------------------

.. _whats-new.0.6.1:


Documentation
~~~~~~~~~~~~~
- Switch RTD build to use mamba for increased speed and reduced memory useage (:pull:`401`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Switch CI to use mamba (:pull:`412`, :issue:`398`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

- Add deprecation warnings for future changes in the API (:issue:`409`,:pull:`411`).
  By `Julius Busecke <https://github.com/jbusecke>`_.


v0.6.0 (2021/11/03)
-------------------

.. _whats-new.0.6.0:

New Features
~~~~~~~~~~~~
- :py:meth:`~xgcm.grid.Grid.set_metrics` now enables adding metrics to a grid object (:pull:`336`, :issue:`199`).
  By `Dianne Deauna <https://github.com/jdldeauna>`_ under the `SIParCS internship <https://www2.cisl.ucar.edu/siparcs-2021-projects#8>`_.

- :py:meth:`~xgcm.grid.Grid.get_metric` refactored, and now incorporates :py:meth:`~xgcm.grid.Grid.interp_like` to allow for automatic interpolation of missing metrics from available values on surrounding positions (:pull:`345`, :pull:`354`).
  By `Dianne Deauna <https://github.com/jdldeauna>`_.[*]_

- :py:meth:`~xgcm.grid.Grid.set_metrics` enables overwriting of previously assigned metrics to a grid object, and allows for multiple metrics on the same axes (must be different dimensions) (:pull:`351`, :issue:`199`).
  By `Dianne Deauna <https://github.com/jdldeauna>`_.[*]_

- :py:meth:`~xgcm.grid.Grid.interp_like` enables users to interpolate arrays onto the grid positions of another array, and can specify boundary conditions and fill values (:issue:`234` , :issue:`343`, :pull:`350`).
  By `Dianne Deauna <https://github.com/jdldeauna>`_.[*]_

- Better input checking when creating a grid object avoids creating grid positions on dataset coordinates which are not 1D (:issue:`208`, :pull:`358`).
  By `Julius Busecke <https://github.com/jbusecke>`_.

.. [*] under the `SIParCS internship <https://www2.cisl.ucar.edu/siparcs-2021-projects#8>`

Breaking Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.6 (:issue:`360`, :pull:`361`). By `Julius Busecke <https://github.com/jbusecke>`_.

Documentation
~~~~~~~~~~~~~
- Added documentation on boundary conditions (:issue:`273`, :pull: `325`)
  By `Romain Caneill <https://github.com/rcaneill>`_.
- Updated metrics documentation for new methods in `Grid Metrics <https://xgcm.readthedocs.io/en/latest/grid_metrics.html>`_.
  By `Dianne Deauna <https://github.com/jdldeauna>`_.[*]_

Internal Changes
~~~~~~~~~~~~~~~~

- Fixed metrics tests so some tests that previously did not run now do run, and refactored the metrics tests.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.[*]_
- Enabled type checking on the repository with mypy.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.[*]_

- Removed dependency on docrep, which as docrep 2.7 used a GPL licence, implicitly changed the license of xGCM.
  Therefore xGCM now has a valid MIT license, instead of accidentally being a GPL licence as it was before.
  (:issue:`308`, :pull:`384`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.[*]_

Deprecations
~~~~~~~~~~~~~

- The `keep_coords` kwarg is now deprecated, and will be removed in the next version. (:issue:`382`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.[*]_



v0.5.2 (2021/5/27)
-------------------

.. _whats-new.0.5.2:

Bug fixes
~~~~~~~~~
- Raise more useful errors when datasets are provided as arguments to grid.transform (:pull:`329`, :issue:`328`). By `Julius Busecke <https://github.com/jbusecke>`_.


Documentation
~~~~~~~~~~~~~
- Updated Realistic Data examples in `Transforming Vertical Coordinates <https://xgcm.readthedocs.io/en/latest/transform.html>`_ (:pull:`322`)
  By `Dianne Deauna <https://github.com/jdldeauna>`_.[*]_

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
