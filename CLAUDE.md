# CLAUDE.md

This file orients an AI assistant (Claude Code and others) working in the xgcm
repository: how the code is structured, and the norms the project holds so that
generated changes fit in. It is **complementary to** the human-facing
[contributor guide](docs/contributor_guide.md), which owns the *mechanics* —
setting up a dev environment, opening a PR, cutting a release. This file owns
*how the code is shaped and how we think about changing it*. Keep it durable:
prefer pointers to the real source of truth over specifics (exact versions, env
lists, tool configs) that drift.

## What xgcm is

xgcm analyzes gridded General Circulation Model (GCM) output. It teaches
`xarray` about finite-volume **Arakawa grids** — where different variables live
at different cell positions (center, left, right, inner, outer) — and adds
differential/integral operators (`interp`, `diff`, `derivative`, `integrate`,
`cumsum`, `cumint`) plus vertical/coordinate remapping (`transform`). It consumes
and produces `xarray` objects and is dask-aware for out-of-core/parallel work.

## Architecture

The public API is the `Grid` class plus two decorators (`apply_as_grid_ufunc`,
`as_grid_ufunc`), exported from `xgcm/__init__.py`. (`Axis` is internal —
`xgcm.axis.Axis` — no longer part of the public namespace.)

- **`Grid` (`grid.py`, the large central module)** — the user-facing object.
  Holds the dataset, a dict of `Axis` objects, optional `metrics` (cell
  distances/areas/volumes, used by `derivative`/`integrate`/`average`), and
  optional `face_connections` (topology linking multiple faces/tiles, e.g. a
  cubed sphere). All grid operations are methods here, each **dispatched to a
  "grid ufunc" rather than implemented inline**.
- **`Axis` (`axis.py`)** — one logical direction (e.g. "X"), mapping cell
  **positions** (`center`/`left`/`right`/`inner`/`outer`, validated against
  `VALID_POSITION_NAMES` in `axis.py`) to dataset dimension names. Mostly a data
  holder consumed by `Grid` and the grid-ufunc machinery.
- **Grid ufuncs (`grid_ufunc.py` + `gridops.py`)** — the core compute
  abstraction: like numpy gufuncs but with a position-aware signature such as
  `"(X:center)->(X:left)"`. `grid_ufunc.py` parses signatures and handles
  padding, broadcasting, and dask core-dim chunking; `gridops.py` is a registry
  of concrete implementations (e.g. `diff_center_to_left`). **Dispatch:**
  `Grid.diff()` etc. call `xgcm.grid._select_grid_ufunc`, which finds functions
  in `gridops.py` whose name starts with the method name (`diff_*`) and whose
  signature matches the requested position shift. To add an operation variant,
  add an `@as_grid_ufunc`-decorated function to `gridops.py` named
  `<method>_<description>`.
- **`padding.py`** — applies edge conditions (`periodic`→wrap, `fill`→constant,
  `extend`→edge) before positions are shifted.
- **`transform.py`** — an independent subsystem for 1D coordinate transformation
  (e.g. depth→density remapping). Its low-level kernels use **numba**
  `guvectorize`, so numba is required only here (see norm 3).
- **Metadata autoparsing (`metadata_parsers.py`, `comodo.py`, `sgrid.py`)** —
  with `Grid(..., autoparse_metadata=True)`, xgcm infers axes/positions from
  dataset attributes following the **COMODO** and **SGRID** conventions (kept as
  separate parsers).

## Engineering norms

Follow these when changing xgcm's code. (Contribution *mechanics* — PR
conventions and the changelog — are under *Pull request guidelines* in the
[contributor guide](docs/contributor_guide.md); the person running the AI is
responsible for every change under its *AI Usage Policy*, which requires
disclosing AI assistance and being able to explain the diff.)

1. **Deprecate by removing, not warning.** When you rename or remove public API,
   do not keep the old name working behind a `DeprecationWarning` across
   releases. Remove it and make the old name fail immediately. Pattern: give the
   function a `**kwargs` sink, check for the old name first, and
   `raise ValueError("Argument 'old' has been renamed to 'new'.")`; for a renamed
   attribute, add a property that raises `AttributeError` with the same message.
   Tag each guard with `# TODO - remove deprecation handling`. (Rationale: xgcm
   uses EffVer and accepts breaking changes — see norm 5 — so multi-release
   deprecation cycles are avoidable maintenance cost.)
2. **Raise on bad input; never return a wrong answer.** If input is ambiguous,
   invalid, or an unsupported combination, raise a clear, specific error instead
   of guessing or silently producing an incorrect array. In scientific software a
   silently wrong number is worse than an exception. Prefer raising your own
   message over letting a confusing error leak from `xarray`/`numpy` internals.
3. **Do not grow the core dependencies.** Core runtime dependencies are `xarray`,
   `dask`, and `numpy` only. `numba` is required solely by `transform`: keep it
   an optional extra and import it lazily. Before adding any import to a core
   module, confirm it is not pulling a heavy or optional dependency into the core
   path — the `test-core` CI environment runs without `numba` and must stay green.
4. **Every code change ships with tests.** For a bug fix, first write a test that
   fails on `master`, then make it pass. For grid operations, exercise all four
   standard topologies: non-periodic, periodic, face-connections, and the bipolar
   north-fold (experimental — coverage is being standardized in
   [#711](https://github.com/xgcm/xgcm/issues/711)). Build test data by reusing
   or extending the parametrized synthetic grids in `xgcm/test/datasets.py`; do
   not hand-roll datasets.
5. **Version by effort (EffVer); breaking changes are allowed.** Version numbers
   communicate expected upgrade effort, not an API-compatibility guarantee, so a
   clean break (norm 1) is preferred over a compatibility shim. Do not build
   multi-release deprecation machinery. (Scheme and release steps: *Versioning
   policy* in the [contributor guide](docs/contributor_guide.md).)

## Running things

Development uses **Pixi**. The two commands you'll use most:

```bash
pixi run tests    # full suite (pytest -n auto, parallel)
pixi run lint     # pre-commit: ruff, mypy, blackdoc, zizmor
```

Run one test with `pixi run pytest xgcm/test/test_grid.py::test_name` (or
`-k "interp and periodic"`). Note `print` is banned by lint (`T201`). Full
environment setup, the CI test matrix, and the docs workflow are in the
[contributor guide](docs/contributor_guide.md).
