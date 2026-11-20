# Installation

MedResidency has three parts with different requirements. Install only what you need.

| Part | Needed for | Install |
|:--|:--|:--|
| Core | `medres list`, `--dry-run`, `--print-config`, `medres eval vqa`, `medres stats`, the tests | `pip install -e .` |
| Training environment | `medres train`, `medres data`, `medres eval generate`, `medres probe gap` | `scripts/install_train_env.sh` |
| Metrics environment | report metrics and the report reward of GRPO | `scripts/install_metrics_env.sh` |

Training and report scoring cannot share an environment: RadEval needs a newer `transformers` than ms-swift accepts.
The tools of the training environment start the scoring process with the Python of the metrics environment, so you
only ever activate one of the two.

## Core

```bash
pip install -e .            # from the repository root; add ".[dev]" for pytest and ruff
medres list
```

The core depends on `pyyaml` and `numpy` only. It is enough to read and resolve configurations, to print the
commands of a run without executing them, and to run the test suite on a laptop. The data builders additionally
need Pillow (and pandas with pyarrow for MMMU), which the training environment has.

The install has to be editable (`-e`), or `MEDRES_CONFIGS` has to point to the `configs/` directory: the
configuration files are read from the clone, not copied into the package.

## Training environment

```bash
bash scripts/install_train_env.sh /path/to/envs/medres
conda activate /path/to/envs/medres
```

The script creates a conda environment with Python 3.12, installs the packages of `requirements-train.txt`,
installs MedResidency in editable mode, and prints the versions it ended up with. The requirements file pins the
versions the benchmark was produced with:

| Package | Version |
|:--|:--|
| ms-swift | 4.6.0 |
| vLLM | 0.23.0 |
| torch | 2.11.0 (CUDA 13.0 build; pinned by vLLM) |
| transformers | 4.57.6 |
| trl | 0.29.1 |
| peft | 0.20.0 |
| accelerate | 1.15.0 |
| datasets | 4.8.4 |
| deepspeed | 0.19.8 |
| qwen-vl-utils | 0.0.14 |

ms-swift 4.6.0 itself accepts newer releases of `transformers` and `trl` than these. They have not been tried.

The script also applies two small patches to the installed ms-swift. Both are idempotent and marked with
`medresidency patch` in the patched file:

1. `swift/pipelines/infer/deploy.py`. On-policy distillation asks the teacher server to score the student's
   response, which is sent as raw token ids. Without the patch the server's media pre-processing expects every
   content item to be a dictionary and answers HTTP 500.
2. `swift/infer_engine/vllm_engine.py`. At temperature 1 a student occasionally samples a vision placeholder token
   such as `<|image_pad|>`, and the next forward pass fails because there are more image placeholders than image
   features. The patch passes the tokens of `train.banned_tokens` to vLLM as `bad_words`.

If you already have an environment with ms-swift 4.6.0, activate it and run
`bash scripts/install_train_env.sh --patch-only`, then `pip install -e .`. With another ms-swift version the script
stops instead of patching the wrong place.

## Metrics environment

```bash
bash scripts/install_metrics_env.sh /path/to/envs/medres_metrics
export MEDRES_METRICS_PYTHON=/path/to/envs/medres_metrics/bin/python
```

This installs RadEval 2.2.1, which provides RadCliQ-v1, RadGraph-F1, CheXbert, RaTEScore and GREEN, and fixes two
problems of that release (an import that fails on case-sensitive file systems, and a CheXbert line that fails with
scikit-learn 1.8 or newer). Put the `export` line into your shell profile or job script: `medres eval run` and
`medres eval report-score` on report tasks, and `--method grpo` with a report reward, read it.

The metric models are downloaded to the Hugging Face cache on first use. GREEN uses a 7B judge
(`bash scripts/download_models.sh green`). The VQA tasks need none of this.

## Models and source data

```bash
bash scripts/download_models.sh med_vqa             # or: cxr_report, modality_experts, green
bash scripts/download_models.sh model Qwen/Qwen3-VL-4B-Instruct
```

Models go to `$MEDRES_MODELS` and source datasets to `$MEDRES_RAW`. The script uses the `hf` command of
`huggingface_hub`, which the training environment provides, and `git` for the splits of the `modality_experts`
group. MIMIC-CXR is credentialed and is not downloaded; [data.md](data.md) explains how to get it and how the
task files are built from the sources.

## Environment variables

| Variable | Default | Meaning |
|:--|:--|:--|
| `MEDRES_MODELS` | `./models` | directory of the model weights |
| `MEDRES_DATA` | `./data` | directory of the task files |
| `MEDRES_RAW` | `./raw` | where `download_models.sh` puts source datasets |
| `MEDRES_METRICS_PYTHON` | unset | the Python of the metrics environment |
| `MEDRES_CONFIGS` | the clone's `configs/` | another configuration directory |
| `MEDRES_ENV` | unset | directory of the training environment, read by the SLURM templates |

With the defaults everything lives under the directory you start `medres` from, usually the repository root.
`.gitignore` keeps `data/`, `models/`, `raw/`, `runs/` and `evals/` out of version control.

## Checking the installation

```bash
medres train --task med_vqa --method adaopd --dry-run     # core: prints the commands of a run
python -m pytest -q                                       # core with ".[dev]": the test suite, CPU only
pip list 2>/dev/null | grep -i -E '^(ms-swift|vllm|torch|transformers) '          # training environment
```

## Hardware

One run uses the GPUs of one node. The default layout is four cards: the student trains on two, and the teacher
servers use the other two. This is what the benchmark ran on, with 48 GB cards (A40) for the 2B students and 80 GB
cards for the 4B student. `hardware.gpus` and `hardware.teacher_gpus` change the layout
([concepts.md](concepts.md#gpu-layout)); only the default layout has been run at scale.

Evaluation and the gap probe take one GPU. Scores of the same checkpoint differ slightly between GPU types, so
evaluate the checkpoints you want to compare on the same type.

## SLURM

`scripts/slurm/` has three templates. They take the partition on the command line and pass everything after the
script name to `medres`:

```bash
mkdir -p logs
export MEDRES_ENV=/path/to/envs/medres

sbatch -p <partition> scripts/slurm/train.sbatch --task cxr_report --method adaopd --seed 42
sbatch -p <partition> scripts/slurm/eval.sbatch --task cxr_report --model runs/cxr_report_adaopd_s42/checkpoint-300
```

`teacher_outputs.sbatch` is an array job that writes a teacher's answers to the training prompts, one shard per
GPU; its header shows how to merge the shards. Adjust the `#SBATCH` lines for memory and time to your cluster. A run
that hit the time limit or was preempted continues from its last checkpoint when you submit the same command again
with `run.resume=true`.
