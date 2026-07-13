# What's New

## v1.0.0 (unreleased) {#whats-new-1-0-0}


### New Features

### Breaking Changes

### Internal Changes

### Documentation

### Bugfixes

- `Grid.transform` now automatically rechunks the transform axis of both `da` and
  `target_data` to a single dask chunk, instead of raising a `ValueError` when the
  input is chunked along that axis. Previously only `target_data` was rechunked (and
  only for the conservative method), so transforming a dask array chunked along the
  vertical (e.g. CMIP6 ocean output) required a manual `.chunk({<axis>: -1})` at every
  call site. Chunking along other dimensions is preserved, and numpy-backed inputs are
  unaffected. A `dask` `PerformanceWarning` is emitted only when collapsing the axis
  produces a chunk larger than dask's `array.chunk-size` guideline
  ([#753](https://github.com/xgcm/xgcm/issues/753),
  [#754](https://github.com/xgcm/xgcm/pull/754)).
  By [Henri Drake](https://github.com/hdrake).

## v0.10.0 (2026/07/12) {#whats-new-0-10-0}


### New Features

- Add a `reverse` option to `Grid.cumsum` and `Grid.cumint` to accumulate from the end of
  an axis toward the start instead of from the start. `reverse` accepts a bool (applied to
  every axis being accumulated) or a per-axis `dict`; passing it for an axis that is not
  being cumulatively summed raises an informative `ValueError`
  ([#729](https://github.com/xgcm/xgcm/pull/729)).
  By [Henri Drake](https://github.com/hdrake).

### Breaking Changes

- The `boundary` argument (and `boundary_width`) has been renamed to `padding` (and `padding_width`)
  throughout the public API, to better reflect the process of array padding and avoid confusion with
  physical boundary conditions (e.g. an ocean-land boundary). The old names now raise an informative
  error rather than going through a deprecation cycle
  ([#696](https://github.com/xgcm/xgcm/pull/696), [#761](https://github.com/xgcm/xgcm/pull/761),
  [#678](https://github.com/xgcm/xgcm/issues/678)).
  By [Nick Hodgskin](https://github.com/VeckoTheGecko).

- Removed the `periodic` argument of `xgcm.Grid`. Boundary behavior is now
  controlled exclusively by the `padding` argument (see the `boundary` → `padding`
  rename above). Migrate as follows:
  `periodic=True` → `padding="periodic"`; `periodic=False` → `padding="fill"`
  (the previous implicit mapping); and the per-axis list form
  `periodic=["X"]` → a per-axis dict, e.g.
  `padding={"X": "periodic", "Y": "fill"}`. Passing `periodic=` now raises an
  informative `ValueError` naming the replacement.

  The default boundary semantics have also changed: previously a `Grid`
  constructed without any padding specification defaulted to *periodic* along
  every axis (silently wrapping), which was the root of several boundary-condition
  bugs. Now, an axis with no padding specified applies *no* boundary condition:
  operations that require padding along such an axis raise an informative error
  instead of silently wrapping. Pass `padding="periodic"` to recover the old
  wrap-around behavior. This makes the default explicitly non-periodic and fixes
  cases where a declared non-periodic axis could still wrap.
  ([#746](https://github.com/xgcm/xgcm/pull/746);
  closes [#195](https://github.com/xgcm/xgcm/issues/195),
  [#509](https://github.com/xgcm/xgcm/issues/509),
  [#604](https://github.com/xgcm/xgcm/issues/604),
  [#624](https://github.com/xgcm/xgcm/issues/624),
  [#625](https://github.com/xgcm/xgcm/issues/625))
  By [Henri Drake](https://github.com/hdrake).
  Supersedes earlier work by [Julius Busecke](https://github.com/jbusecke) in
  [#626](https://github.com/xgcm/xgcm/pull/626).

- Removed the deprecated `keep_coords` keyword argument from grid operations
  (`Grid.interp`, `Grid.diff`, `Grid.min`, `Grid.max`, `Grid.cumsum`, etc.) and from
  `apply_as_grid_ufunc`. The behavior is now always that formerly given by
  `keep_coords=True`: coordinates compatible with the output (including non-dimension
  coordinates) are preserved. Note that this silently changes the **default** output of
  `Grid.interp`, `Grid.diff`, `Grid.min`, `Grid.max`, `Grid.cumsum`, `Grid.derivative`,
  and `Grid.cumint`, which previously dropped non-dimension coordinates from the result
  and now retains them. Passing `keep_coords=` now raises a `ValueError`
  ([#745](https://github.com/xgcm/xgcm/pull/745), [#382](https://github.com/xgcm/xgcm/issues/382)).
  By [Henri Drake](https://github.com/hdrake).

- `Axis` is no longer importable from the top-level `xgcm` namespace, making effective the
  removal announced in v0.9.0; internal use continues via `xgcm.axis.Axis`
  ([#557](https://github.com/xgcm/xgcm/pull/557), [#743](https://github.com/xgcm/xgcm/pull/743),
  [#405](https://github.com/xgcm/xgcm/issues/405)).
  By [Henri Drake](https://github.com/hdrake).

### Internal Changes

- Advertise Python 3.12 and 3.13 support by adding their `Programming Language :: Python` trove
  classifiers, and drop the unused `future` dependency (the package was never imported; only the
  stdlib `from __future__` is used) ([#744](https://github.com/xgcm/xgcm/pull/744)).
  By [Henri Drake](https://github.com/hdrake).

- Migrate development workflow to Pixi ([#691](https://github.com/xgcm/xgcm/pull/691))
  By [Nick Hodgskin](https://github.com/VeckoTheGecko).

- Improve xgcm import speed by lazy-loading the transform module, reducing import time from 3.4s to 0.8s ([#697](https://github.com/xgcm/xgcm/pull/697))
  By [Nick Hodgskin](https://github.com/VeckoTheGecko).

### Documentation

- Reword the "Metrics" note in `grid_ufuncs.md` to a stable, non-promissory statement: metrics are
  not automatically supplied to grid ufuncs, so pass any needed metric explicitly as an input
  ([#744](https://github.com/xgcm/xgcm/pull/744)).
  By [Henri Drake](https://github.com/hdrake).

- Migrate documentation to mkdocs ([#691](https://github.com/xgcm/xgcm/pull/691))
  By [Nick Hodgskin](https://github.com/VeckoTheGecko).

- Document which environment runs the documentation notebooks (`transform.ipynb`, `grid_metrics.ipynb`).
  The existing `docs` pixi environment now bundles Jupyter Lab and can be launched with `pixi run notebooks`,
  and the notebooks and contributor guide note the required dependencies
  ([#750](https://github.com/xgcm/xgcm/pull/750), [#667](https://github.com/xgcm/xgcm/issues/667)).

- xgcm now follows [Intended Effort Versioning (EffVer)](https://jacobtomlinson.dev/effver/); the policy
  is documented in the contributor guide and advertised by a README badge
  ([#680](https://github.com/xgcm/xgcm/pull/680), [#742](https://github.com/xgcm/xgcm/pull/742), [#679](https://github.com/xgcm/xgcm/issues/679)).
  By [Nick Hodgskin](https://github.com/VeckoTheGecko) and [Henri Drake](https://github.com/hdrake).

- Refresh the `grid_metrics.ipynb` and `transform.ipynb` documentation notebooks and fetch the
  `grid_metrics` example data from Zenodo (the previous THREDDS source is no longer available)
  ([#756](https://github.com/xgcm/xgcm/pull/756)).
  By [Henri Drake](https://github.com/hdrake).

- Document why `Grid.get_metric` prefers interpolating an exact-axes metric over composing one
  from sub-axis metrics ([#760](https://github.com/xgcm/xgcm/pull/760)).
  By [Henri Drake](https://github.com/hdrake).

### Bugfixes

- `Grid.get_metric` (and the operations built on it, e.g. `Grid.integrate` and
  `Grid.average`) no longer emits spurious "Metric ... being interpolated ..."
  `UserWarning`s when an exact-position metric combination exists but is not the
  first candidate tried. The search now looks for an exact-position match across
  all candidate combinations before falling back to interpolation, and warns at
  most once. The returned metric is unchanged
  ([#758](https://github.com/xgcm/xgcm/pull/758)).
  By [Henri Drake](https://github.com/hdrake).

- `Axis` now raises a `ValueError` immediately if the same dimension name is
  assigned to more than one position (e.g. `{'center': 'x', 'outer': 'x'}`),
  rather than silently accepting the invalid configuration
  ([#752](https://github.com/xgcm/xgcm/pull/752), [#634](https://github.com/xgcm/xgcm/issues/634)).
  By [Mike German](https://github.com/steps-re).

- Respect the `fill_value` bound on the `@as_grid_ufunc` decorator (or passed to `GridUFunc`). It was
  silently dropped in `GridUFunc.__call__` — only a call-time `fill_value` took effect, so a bound value
  fell through to the `apply_as_grid_ufunc` default of `0`. The bound value is now forwarded (and still
  overridable at call time), mirroring the other bound boundary kwargs
  ([#710](https://github.com/xgcm/xgcm/pull/710), [#652](https://github.com/xgcm/xgcm/issues/652)).
  By [Vincent Gao](https://github.com/gaoflow).

- Fix `xgcm.padding.pad(..., other_component=...)` (and hence vector `Grid.diff`/`Grid.interp`)
  silently ignoring the vector rotation when the component is passed as a bare `DataArray`
  rather than a `{axis_name: DataArray}` dict. On a `face_connections` grid the bare form padded
  the component scalar-style, so the halo across a rotated (axis-swapping) or reversed seam was
  wrong and no error was raised — e.g. on ECCO LLC90 the global convergence sum of the advective
  heat flux was `-1.6e9` (bare) versus `0` (dict). The bare form now recovers the component's axis
  from its own staggering and runs the same rotation/sign-flip logic as the dict form, giving
  identical output, and raises a clear error when the axis cannot be inferred (e.g. a cell-centre
  field) ([#749](https://github.com/xgcm/xgcm/pull/749), [#748](https://github.com/xgcm/xgcm/issues/748)).
  By [Henri Drake](https://github.com/hdrake).

- Fix `Grid.transform(..., method="conservative")` falsely raising `NotImplementedError`
  ("not yet supported for multi-dimensional targets") when a 1-dimensional target was combined
  with an explicit `target_dim` longer than one character: the guard tested the length of the
  dimension *name* instead of the number of target dimensions
  ([#741](https://github.com/xgcm/xgcm/pull/741)).
  By [Henri Drake](https://github.com/hdrake).

- Preserve the input DataArray's dimension order in the output of `apply_as_grid_ufunc`
  (and the `Grid.apply_as_grid_ufunc` method). Previously `xarray.apply_ufunc` moved the
  operated-on core dimension to the end and never moved it back, so an input with dims
  `('tile', 'j', 'i')` came back as `('tile', 'i', 'j')`. The output now follows the input
  ordering (with the core dim renamed in-place if it changes grid position)
  ([#722](https://github.com/xgcm/xgcm/pull/722), [#533](https://github.com/xgcm/xgcm/issues/533)).

- Grid operations (e.g. `Grid.interp`, `Grid.diff`, `Grid.cumsum`) no longer drop or clobber non-core
  coordinates carried on the input `DataArray`. Padding strips all coordinates and they were only restored
  from the grid's own dataset, so a coordinate that lived on the input but not on the grid (e.g. a `time`
  coordinate) was lost, and a coordinate present on both was overwritten with the grid's (possibly stale)
  copy. Coordinates on non-core dimensions are now preserved from the input array (first input wins for
  repeated names), while the newly position-shifted core-dim coordinate still comes from the grid
  ([#721](https://github.com/xgcm/xgcm/pull/721), [#496](https://github.com/xgcm/xgcm/issues/496), [#575](https://github.com/xgcm/xgcm/issues/575)).

- Fix `TypeError: dict.copy() takes no keyword arguments` when applying vector grid ufuncs (e.g.
  `diff_2d_vector`, `interp_2d_vector`) on grids *without* face connections. A vector component supplied
  as a `{axis_name: DataArray}` dict was forwarded unchanged by `xgcm.padding.pad` to the basic padding
  routine `_pad_basic` (which expects a `DataArray`); `pad` now unpacks the inner `DataArray` on the
  non-face-connection path, mirroring the existing face-connection path
  ([#720](https://github.com/xgcm/xgcm/pull/720), [#581](https://github.com/xgcm/xgcm/issues/581)).
  By [Henri Drake](https://github.com/hdrake).

- Fix vector-component `Grid.diff`/`Grid.interp` (and `diff_2d_vector`/`interp_2d_vector`) crashing with
  `AttributeError: 'dict' object has no attribute 'variable'` on grids with face connections when a
  vector component was supplied as a `{axis_name: DataArray}` dict *and* backed by dask. The chunked
  `map_overlap` path still assumed the component was a bare `DataArray` when setting up the dask overlap
  and when reading padded chunk sizes; it now unwraps the inner `DataArray` first, so chunked vector
  diff/interp follows the correct face-connection logic
  ([#705](https://github.com/xgcm/xgcm/pull/705), [#704](https://github.com/xgcm/xgcm/issues/704)).
  By [Anthony Meza](https://github.com/anthony-meza).

- Fix `diff_2d_vector`/`interp_2d_vector` (and the equivalent vector-component `Grid.diff`/`Grid.interp`)
  on grids with face connections when a vector component is a dask array chunked into more than one chunk
  along its core dimension. The `map_overlap` path now derives the output chunk spec from the padded,
  rechunked array, so face-connection padding that rechunks non-core dimensions no longer raises
  `ValueError: Dimension 0 has 2 blocks, adjust_chunks specified with 1 blocks`
  ([#709](https://github.com/xgcm/xgcm/pull/709), [#708](https://github.com/xgcm/xgcm/issues/708)).
  By [Henri Drake](https://github.com/hdrake).

- Fix non-deterministic, hash-seed-dependent halo values in face-connection padding that could yield
  incorrect results across a face-connection seam between runs ([#713](https://github.com/xgcm/xgcm/pull/713)).
  By [Henri Drake](https://github.com/hdrake).

## v0.9.0 (2025/08/20) {#whats-new-0-9-0}


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
- Updated metrics documentation for new methods in [Grid Metrics](https://xgcm.readthedocs.io/en/latest/grid_metrics/).
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

- Updated Realistic Data examples in [Transforming Vertical Coordinates](https://xgcm.readthedocs.io/en/latest/transform/) ([#322](https://github.com/xgcm/xgcm/pull/322))
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

This release adds support for [model grid metrics](https://xgcm.readthedocs.io/en/latest/grid_metrics/) , bug fixes and extended documentation.

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
