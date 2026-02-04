# What's New

## v0.9.0 (unreleased) {#whats-new-0-9-0}


### New Features

- Methods for autoparsing of dataset metadata to construct a `xgcm.Grid` class have been added.
  Currently these include restructred functionality for the COMODO conventions (already in xgcm) and the
  addition of SGRID conventions ([#109](https://github.com/xgcm/xgcm/issues/109), [#559](https://github.com/xgcm/xgcm/pull/559)).
  By [Jack Atkinson](https://github.com/jatkinson1000).

- Vertical coordinate transformations are now also supported for multi-dimensional targets, for example a
  terrain-following (spatially varying) vertical coordinate. This feature currently only works with the linear
  interpolation method ([#614](https://github.com/xgcm/xgcm/issues/614), [#642](https://github.com/xgcm/xgcm/pull/642)).
  By [Nora Loose](https://github.com/noraloose).

### Breaking Changes

- All computation methods on the `xgcm.Axis` class have been removed, in favour of using the corresponding
  methods on the `xgcm.Grid` object. The `xgcm.Axis` class has also been removed from public API.
  ([#405](https://github.com/xgcm/xgcm/issues/405), [#557](https://github.com/xgcm/xgcm/pull/557)).
  By [Thomas Nicholas](https://github.com/tomnicholas).

- All functionality for generating c-grid dimensions on incomplete datasets via `Grid.autogenerate`,  was removed ([#557](https://github.com/xgcm/xgcm/pull/557)).
   By [Julius Busecke](https://github.com/jbusecke).

### Internal Changes

- Switch CI environment setup to micromamba ([#576](https://github.com/xgcm/xgcm/issues/576), [#577](https://github.com/xgcm/xgcm/pull/577)).
  By [Julius Busecke](https://github.com/jbusecke).

- pre-commit autoupdate frequency reduced ([#563](https://github.com/xgcm/xgcm/pull/563)).
  By [Julius Busecke](https://github.com/jbusecke).

### Documentation

### Bugfixes

- Fix bug in `xgcm.transform.transform` that violated tracer conservation when using conservative interpolation in the presence of nans. ([#635](https://github.com/xgcm/xgcm/pull/635))
  By [Julius Busecke](https://github.com/jbusecke).
- Fix bug in `xgcm.padding._maybe_rename_grid_positions` where dimensions were assumed to have coordinate
  values leading to errors with ECCO data. ([#531](https://github.com/xgcm/xgcm/issues/531), [#595](https://github.com/xgcm/xgcm/issues/595), [#597](https://github.com/xgcm/xgcm/pull/597)).
  By [Julius Busecke](https://github.com/jbusecke).

- Remove remaining mentions of `extrapolate` as boundary option ([#602](https://github.com/xgcm/xgcm/pull/602)).
  By [Julius Busecke](https://github.com/jbusecke).

- Fix broken docs build due to broken backwards compatibility in sphinx extensions ([#631](https://github.com/xgcm/xgcm/pull/631))
  By [Julius Busecke](https://github.com/jbusecke).

- Fix bug that did not allow to create grids with faceconnections if the face dimension was coordinate-less. ([#616](https://github.com/xgcm/xgcm/issues/616), [#616](https://github.com/xgcm/xgcm/pull/616)).
  By [Julius Busecke](https://github.com/jbusecke).

- Fix bug in `xgcm.padding._maybe_rename_grid_positions` where dimensions were assumed to have coordinate values leading to errors with ECCO data. ([#531](https://github.com/xgcm/xgcm/issues/531), [#595](https://github.com/xgcm/xgcm/issues/595), [#597](https://github.com/xgcm/xgcm/pull/597)).
  By [Julius Busecke](https://github.com/jbusecke).

## v0.8.1 (2022/11/22) {#whats-new-0-8-1}


### New Features

### Breaking Changes

### Internal Changes

- Rewrote cumsum to use a different code path from `apply_as_grid_ufunc` internally,
  which makes it less susceptible to subtle bugs like the one reported in [#507](https://github.com/xgcm/xgcm/issues/507). ([#558](https://github.com/xgcm/xgcm/pull/558)).
  By [Thomas Nicholas](https://github.com/tomnicholas).

### Documentation

- Improved error message to suggest rechunking to a single chunk when trying to perform disallowed operations
  along chunked core dims.
  By [Thomas Nicholas](https://github.com/tomnicholas).

### Bugfixes

- Fix bug where chunked core dims of only a single chunk triggered errors. ([#558](https://github.com/xgcm/xgcm/pull/558), [#518](https://github.com/xgcm/xgcm/issues/518), [#522](https://github.com/xgcm/xgcm/issues/522))
  By [Thomas Nicholas](https://github.com/tomnicholas).


## v0.8.0 (2022/06/14) {#whats-new-0-8-0}


### New Features

- Addition of logarithmic interpolation to transform ([#483](https://github.com/xgcm/xgcm/pull/483)).
  By [Jonathan Thielen](https://github.com/jthielen).

### Breaking Changes

### Internal Changes

- Switching code linting to the pre-commit.ci service ([#490](https://github.com/xgcm/xgcm/pull/490)).
  By [Julius Busecke](https://github.com/jbusecke).

### Documentation

- Fix 'suggest edits' button in docs ([#512](https://github.com/xgcm/xgcm/pull/512), [#503](https://github.com/xgcm/xgcm/issues/503)).
  By [Julius Busecke](https://github.com/jbusecke).

### Bugfixes

- Fix formatting of the CITATION.cff file ([#500](https://github.com/xgcm/xgcm/pull/500)).
  By [Julius Busecke](https://github.com/jbusecke).
- Fix bug with cumsum when data chunked with dask. ([#415](https://github.com/xgcm/xgcm/pull/415), [#507](https://github.com/xgcm/xgcm/issues/507))
  By [Thomas Nicholas](https://github.com/tomnicholas).

## v0.7.0 (2022/4/20) {#whats-new-0-7-0}


### New Features

- Turn numpy-style ufuncs into grid-aware "grid-ufuncs" via new functions `apply_as_grid_ufunc`
  and `as_grid_ufunc`. ([#362](https://github.com/xgcm/xgcm/pull/362), [#344](https://github.com/xgcm/xgcm/issues/344))
  By [Thomas Nicholas](https://github.com/tomnicholas).

- Padding of vector fields for complex topologies via a dictionary-like syntax has been added ([#459](https://github.com/xgcm/xgcm/pull/459)).
  By [Julius Busecke](https://github.com/jbusecke).

### Breaking Changes

- Removed the `extrapolate` boundary option ([#470](https://github.com/xgcm/xgcm/pull/470)).
  By [Thomas Nicholas](https://github.com/tomnicholas).

### Internal Changes

- All computation methods on the `Grid` object are now re-routed through `apply_as_grid_ufunc`.
  By [Thomas Nicholas](https://github.com/tomnicholas).

### Documentation

- Switch to pangeo-book-scheme ([#482](https://github.com/xgcm/xgcm/pull/482)).
  By [Julius Busecke](https://github.com/jbusecke).

- Add CITATION.cff file ([#450](https://github.com/xgcm/xgcm/pull/450)).
  By [Julius Busecke](https://github.com/jbusecke).


## v0.6.1 (2022/02/15)



### Documentation {#whats-new-0-6-1}

- Switch RTD build to use mamba for increased speed and reduced memory useage ([#401](https://github.com/xgcm/xgcm/pull/401)).
  By [Julius Busecke](https://github.com/jbusecke).

### Internal Changes

- Switch CI to use mamba ([#412](https://github.com/xgcm/xgcm/pull/412), [#398](https://github.com/xgcm/xgcm/issues/398)).
  By [Julius Busecke](https://github.com/jbusecke).

- Add deprecation warnings for future changes in the API ([#409](https://github.com/xgcm/xgcm/issues/409),[#411](https://github.com/xgcm/xgcm/pull/411)).
  By [Julius Busecke](https://github.com/jbusecke).


## v0.6.0 (2021/11/03) {#whats-new-0-6-0}


### New Features

- `Grid.set_metrics` now enables adding metrics to a grid object ([#336](https://github.com/xgcm/xgcm/pull/336), [#199](https://github.com/xgcm/xgcm/issues/199)).
  By [Dianne Deauna](https://github.com/jdldeauna) under the [SIParCS internship](https://www2.cisl.ucar.edu/siparcs-2021-projects#8).

- `Grid.get_metric` refactored, and now incorporates `Grid.interp_like` to allow for automatic interpolation of missing metrics from available values on surrounding positions ([#345](https://github.com/xgcm/xgcm/pull/345), [#354](https://github.com/xgcm/xgcm/pull/354)).
  By [Dianne Deauna](https://github.com/jdldeauna).[^siparcs]

- `Grid.set_metrics` enables overwriting of previously assigned metrics to a grid object, and allows for multiple metrics on the same axes (must be different dimensions) ([#351](https://github.com/xgcm/xgcm/pull/351), [#199](https://github.com/xgcm/xgcm/issues/199)).
  By [Dianne Deauna](https://github.com/jdldeauna).[^siparcs]

- `Grid.interp_like` enables users to interpolate arrays onto the grid positions of another array, and can specify boundary conditions and fill values ([#234](https://github.com/xgcm/xgcm/issues/234) , [#343](https://github.com/xgcm/xgcm/issues/343), [#350](https://github.com/xgcm/xgcm/pull/350)).
  By [Dianne Deauna](https://github.com/jdldeauna).[^siparcs]

- Better input checking when creating a grid object avoids creating grid positions on dataset coordinates which are not 1D ([#208](https://github.com/xgcm/xgcm/issues/208), [#358](https://github.com/xgcm/xgcm/pull/358)).
  By [Julius Busecke](https://github.com/jbusecke).

[^siparcs]: under the [SIParCS internship](https://www2.cisl.ucar.edu/siparcs-2021-projects#8)

### Breaking Changes

- Drop support for Python 3.6 ([#360](https://github.com/xgcm/xgcm/issues/360), [#361](https://github.com/xgcm/xgcm/pull/361)). By [Julius Busecke](https://github.com/jbusecke).

### Documentation

- Added documentation on boundary conditions ([#273](https://github.com/xgcm/xgcm/issues/273), [#325](https://github.com/xgcm/xgcm/pull/325))
  By [Romain Caneill](https://github.com/rcaneill).
- Updated metrics documentation for new methods in [Grid Metrics](https://xgcm.readthedocs.io/en/latest/grid_metrics.html).
  By [Dianne Deauna](https://github.com/jdldeauna).[^siparcs]

### Internal Changes

- Fixed metrics tests so some tests that previously did not run now do run, and refactored the metrics tests.
  By [Tom Nicholas](https://github.com/TomNicholas).[^siparcs]
- Enabled type checking on the repository with mypy.
  By [Tom Nicholas](https://github.com/TomNicholas).[^siparcs]

- Removed dependency on docrep, which as docrep 2.7 used a GPL licence, implicitly changed the license of xGCM.
  Therefore xGCM now has a valid MIT license, instead of accidentally being a GPL licence as it was before.
  ([#308](https://github.com/xgcm/xgcm/issues/308), [#384](https://github.com/xgcm/xgcm/pull/384))
  By [Tom Nicholas](https://github.com/TomNicholas).[^siparcs]

### Deprecations

- The `keep_coords` kwarg is now deprecated, and will be removed in the next version. ([#382](https://github.com/xgcm/xgcm/issues/382))
  By [Tom Nicholas](https://github.com/TomNicholas).[^siparcs]



## v0.5.2 (2021/5/27)


### Bug fixes {#whats-new-0-5-2}

- Raise more useful errors when datasets are provided as arguments to grid.transform ([#329](https://github.com/xgcm/xgcm/pull/329), [#328](https://github.com/xgcm/xgcm/issues/328)). By [Julius Busecke](https://github.com/jbusecke).


### Documentation

- Updated Realistic Data examples in [Transforming Vertical Coordinates](https://xgcm.readthedocs.io/en/latest/transform.html) ([#322](https://github.com/xgcm/xgcm/pull/322))
  By [Dianne Deauna](https://github.com/jdldeauna).[^siparcs]

- Migrated model example notebooks to [xgcm-examples](https://github.com/xgcm/xgcm-examples) which integrates with [pangeo gallery](https://gallery.pangeo.io/repos/xgcm/xgcm-examples/) ([#294](https://github.com/xgcm/xgcm/pull/294))
  By [Julius Busecke](https://github.com/jbusecke).

## v0.5.1 (2020/10/16)


### Bug fixes {#whats-new-0-5-1}

- Add support for older numba versions (<0.49) ([#263](https://github.com/xgcm/xgcm/pull/263), [#262](https://github.com/xgcm/xgcm/issues/262)). By [Navid Constantinou](https://github.com/navidcy).



## v0.5.0 (2020/9/28) {#whats-new-0-5-0}


### New Features

- `Grid.transform` and `Axis.transform` now enable 1-dimensional coordinate transformation ([#205](https://github.com/xgcm/xgcm/pull/205), [#222](https://github.com/xgcm/xgcm/issues/222)).
  By [Ryan Abernathey](https://github.com/rabernat) and [Julius Busecke](https://github.com/jbusecke).

### Bug fixes

- More reliable handling of missing values in `Grid.average`. Missing values between data and metrics do not have to be aligned by the user anymore. ([#259](https://github.com/xgcm/xgcm/pull/259)). By [Julius Busecke](https://github.com/jbusecke).

- Remove outdated `example_notebooks` folder ([#244](https://github.com/xgcm/xgcm/pull/244), [#243](https://github.com/xgcm/xgcm/issues/243)). By [Nikolay Koldunov](https://github.com/koldunovn) and [Julius Busecke](https://github.com/jbusecke).


## v0.4.0 (2020/9/2)

### New Features

- Support for keeping compatible coordinates in most Grid operations ([#186](https://github.com/xgcm/xgcm/issues/186)).
  By [Aurélien Ponte](https://github.com/apatlpo).

- Support for specifying default `boundary` and `fill_value` in the `xgcm.Grid` constructor.
  Default values can be overridden in individual method calls (e.g. `Grid.interp`) as usual.
  By [Deepak Cherian](https://github.com/dcherian).

### Bug fixes

- Fix for parsing fill_values as dictionary ([#218](https://github.com/xgcm/xgcm/issues/218)).
  By [Julius Busecke](https://github.com/jbusecke).

### Internal Changes

- Complete refactor of the CI to github actions ([#214](https://github.com/xgcm/xgcm/issues/214)).
  By [Julius Busecke](https://github.com/jbusecke).


## v0.3.0 (31 January 2020) {#whats-new-0-4-0}

This release adds support for [model grid metrics](https://xgcm.readthedocs.io/en/latest/grid_metrics.html) , bug fixes and extended documentation.

### Breaking changes

### New Features

- Support for 'grid-aware' average and cumsum using `Grid.average` and `Grid.cumsum` ([#162](https://github.com/xgcm/xgcm/issues/162)).
  By [Julius Busecke](https://github.com/jbusecke).

- Support for 'grid-aware' integration using `Grid.integrate` ([#130](https://github.com/xgcm/xgcm/issues/130)).
  By [Julius Busecke](https://github.com/jbusecke).

### Bug fixes

- Fix for broken stale build ([#155](https://github.com/xgcm/xgcm/issues/155)).
  By [Julius Busecke](https://github.com/jbusecke).

- Fixed bug in handling of grid metrics. ([#136](https://github.com/xgcm/xgcm/issues/136)).
  By [Ryan Abernathey](https://github.com/rabernat).

- Fixed bug in
  `Grid.derivative` ([#132](https://github.com/xgcm/xgcm/issues/132)).
  By [Timothy Smith](https://github.com/timothyas).

### Documentation

- Added docs for `Grid.derivative` ([#163](https://github.com/xgcm/xgcm/issues/163))
  By [Timothy Smith](https://github.com/timothyas).

- Add binderized examples ([#141](https://github.com/xgcm/xgcm/issues/141)).
  By [Ryan Abernathey](https://github.com/rabernat).

- Simplify example notebooks ([#140](https://github.com/xgcm/xgcm/issues/140)).
  By [Ryan Abernathey](https://github.com/rabernat).

- Execute example notebook during doc build ([#138](https://github.com/xgcm/xgcm/issues/138)).
  By [Ryan Abernathey](https://github.com/rabernat).

- Added contributor guide to docs ([#137](https://github.com/xgcm/xgcm/issues/137)).
  By [Julius Busecke](https://github.com/jbusecke).


### Internal Changes

- Added GitHub Action to publish xgcm to PyPI on release ([#170](https://github.com/xgcm/xgcm/issues/170)).
  By [Anderson Banihirwe](https://github.com/andersy005).

- Reorganized environment names for CI ([#139](https://github.com/xgcm/xgcm/issues/139)).
  By [Julius Busecke](https://github.com/jbusecke).

- Added automatic code formatting via [black](https://black.readthedocs.io/en/stable/) ([#131](https://github.com/xgcm/xgcm/issues/131)).
  By [Julius Busecke](https://github.com/jbusecke).


## v0.2.0 (21 March 2019)

Changes not documented for this release

## v0.1.0 (13 July 2014)

Changes not documented for this release

Initial release.
