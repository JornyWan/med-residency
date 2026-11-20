# Concepts

## A run is a task, a method and a seed

```bash
medres train --task cxr_report --method adaopd --seed 42
```

- A **task** fixes what is learned and how it is measured: the student, the teachers, the training files, the budget
  (steps, sequences per step, learning rate, response limit) and the test sets. Tasks are the files in
  `configs/tasks/`.
- A **method** is the loss. It may train in one stage or two, with or without teacher servers. Methods are the
  files in `configs/methods/` together with their builders in `src/medresidency/methods.py`.
- The **seed** sets both the model seed and the order of the data.

Two runs of the same task that differ only in `--method` are a fair comparison by construction. That is the point of
the framework.

```bash
medres list                  # tasks, methods, models and rewards
medres list methods          # one of them
```

## Configuration

A run is one nested dictionary, built from YAML layers. Later layers win:

```
configs/defaults.yaml            every key, with its default and a comment
  <- configs/tasks/<task>.yaml       the task's student, teachers, data and budget
  <- configs/methods/<method>.yaml   the method's own keys, under `method:`
  <- the task's `overrides.<method>` block, for what one method needs on this task
  <- a recipe file (optional)
  <- key=value arguments
```

`--print-config` shows the result without starting anything:

```bash
medres train --task cxr_report --method adaopd --print-config
```

Any key can be set on the command line with its dotted path:

```bash
medres train --task cxr_report --method adaopd method.tau=0.5 train.lr=1e-6 --name cxr_report_adaopd_tau05_s42
```

Three rules keep configurations honest:

- **Unknown keys are errors.** `train.learning_rate=1e-6` stops with a message instead of training at the default
  rate. The valid keys are those of `configs/defaults.yaml` and of the method's file.
- **`teachers` is replaced, not merged.** A layer that sets `teachers` defines the complete set.
- **ms-swift arguments pass through `train.swift_args`.** MedResidency does not wrap every option of the trainer.
  `train.swift_args.beta=0.04` adds `--beta 0.04` to the training command and wins over what the method set.

The sections of a configuration:

| Section | Holds |
|:--|:--|
| `run` | name, seed, output directory, whether to resume |
| `student` | the model to train (`model`, a name of `configs/models.yaml`) and optionally a checkpoint to start from (`init`) |
| `teachers` | routing tag → model name |
| `data` | the training files: `prompts`, `labels`, `teacher_outputs` |
| `train` | the budget and the sampling settings shared by all methods |
| `teacher_server` | memory and batching of the teacher servers |
| `hardware` | GPUs of the run, and how many of them serve teachers |
| `method` | the method's name and its own keys |

## Recipes

A recipe is a YAML file with the same keys, for a run that is more than a task and a method. The recipes of the
benchmark are in `configs/recipes/`:

```bash
medres train configs/recipes/adaopd/cxr_report_4b_adaopd.yaml --seed 42
medres train configs/recipes/adaopd/cxr_report_4b.yaml --method gkd --seed 42
```

The first recipe names its task and method. The second names only the task (the 4B student on the report task) and
takes the method from the command line.

## Run names and output

A run is called `<task>_<method>_s<seed>` unless `--name` is given. With a recipe the name is the recipe's file name
plus the seed, with the method in between when the recipe leaves it open: the two commands above produce
`cxr_report_4b_adaopd_s42` and `cxr_report_4b_gkd_s42`.

Give a run with overrides its own `--name`. The launcher refuses to start in a directory that already holds
checkpoints, unless the run is being resumed.

```
runs/cxr_report_adaopd_s42/
    checkpoint-50/ ... checkpoint-300/     model weights; the newest one also holds the optimizer state
    logging.jsonl                          one line per logged step (loss, lang_gap, adaopd_mu, ...)
    medres/
        config.yaml                        the resolved configuration
        commands.sh                        the environment and every command that was run
        launches.jsonl                     per launch: time, host, GPUs, job id, code commit, package versions
        teacher_<tag>.log                  output of each teacher server
```

`commands.sh` is plain shell. If a number in a paper is questioned a year later, this file and `config.yaml` say
exactly how the checkpoint was produced.

## What the launcher does

`medres train` asks the method for its stages and runs them in order. For every stage it

1. starts one server per teacher (`swift deploy` with the vLLM backend, bound to 127.0.0.1) and waits until each
   one answers and reports the model it is supposed to serve;
2. starts the report-metric server if the reward needs one;
3. runs the ms-swift training command and keeps checking that the servers are alive;
4. stops every process it started, whether the stage finished, failed or was cancelled.

`--dry-run` prints the same commands and starts nothing.

### GPU layout

One run uses `hardware.gpus` cards of one node, the first ones of `CUDA_VISIBLE_DEVICES` when it is set.

| Stage | Student | Teachers |
|:--|:--|:--|
| without teachers (`sft`, `offline_kd`, `grpo`, stage 1 of `two_stage`) | all cards | – |
| with one teacher | the first `gpus − teacher_gpus` cards | one server across the last `teacher_gpus` cards (tensor parallel) |
| with several teachers | the first `gpus − teacher_gpus` cards | one card per server, spread over the last `teacher_gpus` cards |

With the default of four cards and two teacher cards, the two teachers of `cxr_report_vqa` get one card each, and
the eight experts of `modality_experts` share two cards, four per card. The student samples its own rollouts with a
vLLM engine on its training cards (`train.rollout_gpu_memory` of each card's memory).

`train.sequences_per_step` must be a multiple of the student cards times `train.per_device_batch`; the launcher
derives the gradient accumulation from it, so the number of sequences per update does not change with the layout.

```bash
# two cards: the student on one, the teacher on the other
medres train --task cxr_report --method opd hardware.gpus=2 hardware.teacher_gpus=1
```

The four-card layout is the one the benchmark used. Other layouts build correct commands but have not been run at
scale.

## Resuming

A checkpoint is saved with the optimizer state. A run that was interrupted continues from its last checkpoint when
the same command is run again with `run.resume=true`:

```bash
medres train --task cxr_report --method adaopd --seed 42 run.resume=true
```

Stages that already have their final checkpoint are skipped. AdaOPD restores its controller from `logging.jsonl`,
so the resumed run continues with the smoothed gap and the first-step gap of the original run.

The optimizer state is the large part of a checkpoint: about 24 GB for a 2B student, next to 5 GB of weights. A
resume needs only the newest one, so the launcher deletes the state of a checkpoint once a newer checkpoint is
complete; the weights of every checkpoint stay. A 300-step run then takes up to 80 GB while it trains and about
55 GB when it has finished. If you will not continue a finished run, delete its last optimizer state as well
(`rm -r runs/<run>/checkpoint-*/global_step*`), which leaves 30 GB. `run.keep_optimizer_states=true` turns the
deletion off; the same run then takes 175 GB.

## What the launcher sets in the environment

Variables you have set yourself are kept.

| Variable | Value | Why |
|:--|:--|:--|
| `HF_HUB_OFFLINE` | `1` | models and data are local files |
| `VLLM_NO_USAGE_STATS` | `1` | no usage reports from compute nodes |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | the sampler compiles with `nvcc` at first use, which compute nodes rarely have |
| `MEDRES_BAD_WORDS` | `train.banned_tokens` | vision placeholder tokens the student must not sample |
| `MODELSCOPE_CACHE`, `HF_DATASETS_CACHE` | a temporary directory per run | concurrent runs that share ms-swift's dataset cache crash each other |

The method adds its own variables to the training command (`MEDRES_ADAOPD_*` for AdaOPD, `MEDRES_REWARD_URL` for a
reward that is scored by the metric server). They appear in `commands.sh` and in the dry run.

`medres eval generate` starts vLLM without the launcher and sets the first three variables in the same way.
