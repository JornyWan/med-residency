# Contributing

Contributions are welcome: a model, a task, a method, a reward, a benchmark row, a bug fix, a clearer sentence in
the documentation. This page says how to set up, what a pull request should contain, and the two rules that are not
negotiable.

## Two rules

**No data in the repository, in issues or in pull requests.** That covers images, reports, prompts, model outputs,
per-item scores, raw logs and model weights. Several tasks are built on MIMIC-CXR, and its data-use agreement extends
to what models write about its images ([docs/data.md](docs/data.md#credentialed-data)). Results are contributed as
corpus-level numbers. `scripts/check_release.py` enforces the part a script can check.

**The protocol of a built-in task does not change silently.** The benchmark is only useful if a row means the same
thing next year. A change that alters the ms-swift arguments of an existing task × method fails
`tests/test_methods.py`. If the change is intended, open an issue first; it needs an entry in
[CHANGELOG.md](CHANGELOG.md) and a note on the benchmark page saying which rows it affects.

## Setting up

```bash
git clone https://github.com/JornyWan/med-residency.git && cd med-residency
pip install -e ".[dev]"
pytest
```

This is all the development of configuration, methods, builders and statistics needs: the tests run on a CPU and
use neither models nor data. The training and metrics environments
([docs/installation.md](docs/installation.md)) are only needed to run something on a GPU.

## Before you open a pull request

```bash
ruff format .
ruff check .
pytest
python scripts/check_release.py
```

Continuous integration runs the same checks.

Code that runs on a GPU (generation, the gap probe, the ms-swift plugins, the reward server) is not covered by the
tests. If you touch it, run it and say in the pull request what you ran. A few steps are usually enough to see that
a training path still works:

```bash
medres train --task med_vqa --method adaopd --name smoke_adaopd train.steps=4 train.save_steps=2
```

## What a contribution consists of

[docs/extending.md](docs/extending.md) walks through each of these.

| Contribution | Files |
|:--|:--|
| a model | one line in `configs/models.yaml` |
| a task | `configs/tasks/<task>.yaml`, a builder in `src/medresidency/data/` if the data need one, a section in `docs/tasks.md` and `docs/data.md`, new entries in `tests/golden_args.json` |
| a method | `configs/methods/<method>.yaml`, a builder in `src/medresidency/methods.py`, a section in `docs/methods.md` with the reference, new entries in `tests/golden_args.json` |
| a reward | a class in `src/medresidency/plugins/reward_plugin.py`, its name in `REWARDS`, a test in `tests/test_rewards.py` |
| a benchmark row | the row in `benchmarks/README.md` with its commands ([what to include](benchmarks/README.md#contributing-results)) |

A new task should come with its reference rows: the untrained student, the teacher, and at least `opd` and
`offline_kd`, so that the next method has something to be compared with.

## Style

- `ruff format` decides the formatting; lines are at most 120 characters, in Markdown as well (tables excepted).
- Prefer a plain function to a class, and a clear error to a silent default. An error message names the key or the
  variable the user has to change.
- Tools print counts and aggregate numbers, never the text of an example.
- Every command in the documentation has to work. `tests/test_docs.py` resolves the `medres train` commands of all
  Markdown files and checks the options of the other tools against their parsers, so an example that drifts from
  the code fails the tests.
- Document what was measured. Numbers in the documentation come from a file someone can regenerate, and a claim
  that was not checked is written as such or left out.

## Commits and pull requests

One change per pull request. Say what the change does and how you checked it; for a result, give the commands, the
seeds and the hardware. Keep generated files, notebooks with outputs, and editor settings out of the commit.

## Reporting bugs and asking questions

Use the issue templates. Attach the files of `<run>/medres/` and the error message, and never text from credentialed
data ([docs/troubleshooting.md](docs/troubleshooting.md#reporting-a-problem)).

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0, as the rest of the
repository is.
