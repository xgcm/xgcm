# Contributor Guide {#contributor-guide}

**xgcm** is meant to be a community driven package and we welcome feedback and
contributions.

Did you notice a bug? Are you missing a feature? A good first starting place is to
open an issue in the [github issues page](https://github.com/xgcm/xgcm/issues).

Want to show off a cool example using xgcm? Please consider contributing to [xgcm-examples](https://github.com/xgcm/xgcm-examples).


In order to contribute to xgcm, please fork the repository and submit a pull request.
A good step by step tutorial for this can be found in the
[xarray contributor guide](https://xarray.pydata.org/en/stable/contributing.html#working-with-the-code).


## Development Setup

We use [Pixi](https://pixi.sh) for development. After forking and cloning the repository:

```
pixi install
```

This installs xgcm in editable mode with all development dependencies.

## Running Tests

```
pixi run tests
```

## Code Formatting

We use [pre-commit](https://pre-commit.com/) for code formatting. To run linting:

```
pixi run lint
```

To auto-format before each commit, install the pre-commit hooks:

```
pre-commit install
```

## Building the Documentation

```
pixi run docs-serve
```
This will start a live preview running on `http://127.0.0.1:8000/`

## Running the documentation notebooks

The documentation notebooks that live directly in `docs/` (`transform.ipynb` and
`grid_metrics.ipynb`) run in the `docs` pixi environment, which bundles everything
they need (including `numba` for `Grid.transform` and `matplotlib`). To open them
interactively in Jupyter Lab:

```
pixi run notebooks
```

This launches Jupyter Lab rooted at the `docs` folder.

## Documentation {#documentation}

The docs are the primary way new users learn about xgcm, so keep them as simple and
clear as possible, and build every example on a publicly available dataset so that
anyone can reproduce its output locally. The more advanced, comprehensive, or
big-data examples — illustrating real use cases of xgcm's features on the
ocean-model configurations and datasets commonly used in the community — live in
[xgcm-examples](https://github.com/xgcm/xgcm-examples) rather than in the core docs.

## Pull request guidelines {#pull-request-guidelines}

A few conventions keep changes easy to review and release:

- **One focused change per PR.** Keep diffs small and single-purpose. PRs are
  squash-merged, so each lands as a single commit whose title ends with the PR
  number, e.g. `Add reverse option to Grid.cumsum (#729)`. If a change genuinely
  cannot be split, structure the commits to aid review and say why in the PR body.
- **Document user-visible changes in `docs/whats-new.md`.** Anything a user would
  notice — a new feature, bug fix, or breaking change — gets an entry in the
  matching section of the unreleased release, ending with an attribution line
  `By [Your Name](https://github.com/yourhandle).` New public functions or
  methods also go in `docs/api.md`. This is the single most common thing
  reviewers ask contributors to add.
- **Keep the built docs in sync.** A breaking change, or any change that alters the
  rendered output of the built `.md` docs or the executed `.ipynb` notebooks
  (`docs/transform.ipynb`, `docs/grid_metrics.ipynb`), must update those files in
  the same PR so the published [documentation](#documentation) still reproduces.
- **Follow the [AI Usage Policy](#ai-usage-policy).** AI-assisted contributions
  are welcome, but you must be able to explain every change and disclose the
  assistance openly — a `Co-Authored-By:` trailer on commits, and a note on any
  PR or review comment that was drafted with AI.

## AI Usage Policy {#ai-usage-policy}

Some xgcm contributors use AI tools (Claude Code and others) as part of their
workflow, and this is welcome. This policy applies to every change regardless of
whether it was written by hand, with AI assistance, or generated entirely by an AI
tool. It aligns closely with, and borrows heavily from,
[xarray's AI usage policy](https://docs.xarray.dev/en/stable/contribute/ai-policy.html),
adapted for xgcm.

xgcm is maintained by a small group of volunteers who fit it around other
responsibilities. The aim of this policy is to keep the door open to AI-assisted
work while ensuring that:

- reviewers are not overburdened,
- contributions can be understood and maintained over time, and
- the submitter can vouch for and explain every change.

### You are responsible for your changes

If you open a pull request, you are responsible for having fully reviewed and
understood it. You must be able to explain why each change is correct and how it
fits into the project — the same bar as a hand-written PR. Keep diffs small and
single-purpose (see [Pull request guidelines](#pull-request-guidelines)) to ease
the burden on reviewers, and leave out unnecessary or loosely related changes. If
you are unsure of the best approach, open a draft PR or an issue to discuss it
first.

### Communicate in your own words

PR descriptions, issue comments, and review responses must be your own words; the
substance and reasoning must come from you. Do not paste AI-generated text as a
comment or a review response, and please be concise. Using AI to polish your own
writing (grammar, phrasing, spelling) is fine, as long as it does not introduce
inaccuracies.

Disclose AI assistance openly: add a `Co-Authored-By:` trailer to commits and a
note on any PR or comment that was drafted with an AI tool. (The PR that added this
policy is itself an example of that disclosure.)

### Review every line of code and tests

You must have personally read and understood all changes before submitting them. If
you used an AI tool to generate code, you are expected to have read it critically
and tested it, and the PR description should explain the approach and the reasoning
behind it. Do not leave it to reviewers to work out what the code does and why.

Not acceptable:

> I pointed an agent at the issue and here are the changes.

> This is what Claude came up with. 🤷

Acceptable:

> I iterated with an agent to produce this; it wrote the code at my direction and I
> have fully read, tested, and validated the changes.

> An agent generated a first draft from the issue. I reviewed it thoroughly and
> understand the implementation.

### Discuss large AI-assisted contributions first

Generating a large diff with an agent is fast; reviewing one is not. A large PR
shifts the burden from the contributor onto our small pool of maintainers. If you
are planning a substantial AI-assisted change — a significant refactor, a new
subsystem, a framework migration — **open an issue first** to agree on the scope
and approach before writing the code.

Maintainers may ask for large changes to be broken into smaller, reviewable pieces,
and reserve the right to close PRs where the scope makes meaningful review
impractical or where this policy has not been followed, and to hide or delete
comments that violate it.

## Versioning policy

xgcm uses [Intended Effort Versioning (EffVer)](https://jacobtomlinson.dev/effver/): version numbers are `MACRO.MESO.MICRO`. A **MACRO** bump signals that adopting the release may require a large effort from users; a **MESO** bump signals that some effort may be required; a **MICRO** bump signals that little to no effort is expected. Version numbers communicate the *intended* upgrade effort, not a strict API-compatibility guarantee. (For releases below 1.0.0, the segments shift one position per EffVer's zero-version guidance: `0.MACRO.MESO`.)

## How to release a new version of xgcm (for maintainers only)

The process of releasing at this point is very easy.

We need only two things: A PR to update the documentation and a release on github.

1. Make sure that all the new features/bugfixes etc are appropriately documented in `docs/whats-new.md`, add the date to the current release and make an empty (unreleased) entry for the next release as a PR. Choose the next version number according to the [versioning policy](#versioning-policy) above. When citing changes in `docs/whats-new.md`, cite the PR (always) and the issues it resolves (optionally), listing the PR first.
2. Navigate to the 'tags' symbol on the repos main page, click on 'Releases' and on 'Draft new release' on the right. Add the version number and a short description and save the release.

From here the github actions take over and package things for [Pypi](https://pypi.org/project/xgcm/).
The conda-forge package will be triggered by the Pypi release and you will have to approve a PR in [xgcm-feedstock](https://github.com/conda-forge/xgcm-feedstock). This takes a while, usually a few hours to a day.

Thats it!

## How to synchronize examples from xgcm-examples

Most of the example notebooks in this documentation are located in the seperate repo [xgcm-examples](https://github.com/xgcm/xgcm-examples), which is automatically linked to [pangeo gallery](https://gallery.pangeo.io). These examples are synced into this documentation using git submodules.
Currently updates in the example repo need to be manually synced to this repo with the following steps:

From the xgcm root directory do:

```
cd docs/xgcm-examples
```

If this directory is empty, it means your original install did not pull the submodule; to configure the submodule, do:

```
git submodule update --init
```

You are now in a seperate git repository and can pull all updates:

```
git pull
```

Now navigate back to the xgcm repo:

```
cd -
```

And commit, push like usual to create a pull request.
