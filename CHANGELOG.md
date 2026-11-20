# Changelog

Changes that a user of the framework or a reader of the benchmark needs to know about. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Changes of the **protocol** get their own heading. A protocol change is anything that alters the training or
evaluation of a built-in task, and with it the meaning of a benchmark row: a budget, a learning rate, a prompt, a
test set, a metric. Such a change raises the minor version, and the benchmark page says which rows were produced
before it.

## 0.1.0 (unreleased)

First version.

### Added

- `medres train`: one launcher for nine methods (`sft`, `offline_kd`, `opd`, `gkd`, `two_stage`, `chord`,
  `chord_phi`, `adaopd`, `grpo`) on top of ms-swift 4.6.0 and vLLM 0.23.0, with teacher servers, an optional
  report-metric reward server, resumable runs, and a record of configuration, commands and versions in every run
  directory.
- Four tasks: `cxr_report`, `med_vqa`, `cxr_report_vqa`, `modality_experts`, and the recipes of the remaining
  benchmark settings in `configs/recipes/adaopd/`.
- `medres data`: builders for the task files and for teacher outputs.
- `medres eval`: generation with vLLM, report metrics (RadEval, with a length cap for GREEN), VQA accuracy, and
  `medres eval run` that chains them.
- `medres probe gap`: the language gap and the visual evidence of a teacher–student pair before training.
- `medres stats compare`: paired bootstrap comparison of two runs, with resampling by patient.
- The benchmark page with the results of the AdaOPD paper and the command behind every row.
- Install scripts for the training and the metrics environment, a download script, and SLURM templates.

### Protocol

- Version 1 of the protocol of the four tasks, as listed on the benchmark page.
