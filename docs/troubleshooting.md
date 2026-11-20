# Troubleshooting

The messages of `medres` you are most likely to see, and the failures we ran into while producing the benchmark,
with their causes and fixes. Several of the failures are handled by the launcher or by the install scripts today.
They are listed anyway, because the same failure comes back when ms-swift or vLLM is called outside MedResidency, or
in another version.

When a run fails, start with the files of `<run>/medres/`: `commands.sh` has every command with its environment,
and `teacher_<tag>.log` has the output of each teacher server.

## Configuration

**`unknown configuration keys: train.learning_rate`**

A key that no layer defines. The key is `train.lr`; `configs/defaults.yaml` lists all of them, and `--print-config`
shows the resolved values. ms-swift arguments that MedResidency has no key for go through `train.swift_args`.

**`method sft needs data.labels, which task med_vqa does not set`**

The method reads a file the task does not have. Label SFT needs reference answers written out as training targets,
which only `cxr_report` provides. The methods with an SFT term need `data.teacher_outputs`
([data.md](data.md#teacher-outputs)).

**`train.sequences_per_step=64 is not a multiple of 3 student GPUs x per_device_batch 4`**

The number of sequences per update is fixed by the task, and the launcher derives the gradient accumulation from it.
With three student cards no accumulation gives 64. Change `hardware.gpus` or `train.per_device_batch`, not
`train.sequences_per_step`.

## Starting a run

**`<run> already holds checkpoint-150: choose another run.name, or continue this run with run.resume=true`**

The launcher never trains into a directory that has checkpoints. Give a run with different settings its own
`--name`. If the run was interrupted, repeat the command with `run.resume=true`.

**`hardware.gpus=4, but CUDA_VISIBLE_DEVICES lists 3 cards`**

The job was given fewer cards than the run needs. On our cluster a four-card request on a shared node was sometimes
granted three; requesting the node exclusively fixed it.

**`teacher unirg exited with code 1; its log: ...`, or `teacher unirg was not ready after 3600 s; its log: ...`**

Read the log named in the message. The usual causes: the model directory does not exist (check `MEDRES_MODELS`;
with `HF_HUB_OFFLINE=1` the error speaks of an unreachable hub), another process holds memory on the card, or the
share of memory in `teacher_server.gpu_memory` is too small for the model.

**`teacher unirg: http://127.0.0.1:8000 serves ['MediX-R1-8B'], not UniRG-CXR`**

Another server already answers on that port. This happened to us when runs were chained in one job and the server
of the previous run survived: `swift deploy` is a wrapper, and killing it leaves the actual server running. The
training of the next run then waited for its teacher, found the old one ready, and distilled from the wrong model
without any error. The launcher now stops whole process groups and asks every server which model it serves before
training starts. If you see this message, a server from outside the run is still alive: stop it.

**The teacher answers every scoring request with HTTP 500**

ms-swift 4.6.0 without the first patch of `scripts/install_train_env.sh`. The student's response is sent to the
teacher as raw token ids, and the server's pre-processing does not expect that. Run
`bash scripts/install_train_env.sh --patch-only` in the training environment.

## During training

**`ValueError: Image features and image tokens do not match: tokens: 257, features 256`**

At temperature 1 the student sampled an image placeholder token, and the training forward pass now finds more
placeholders than image features. This is rare per response, so a run can die hours into training. The second patch
of the install script passes `train.banned_tokens` to vLLM as words that are never sampled. For a model family other
than Qwen-VL, put its placeholder tokens into that list.

**`Gradient computed twice for this partition`** (DeepSpeed)

A method with an SFT term runs a second forward pass per step, and with the default (reentrant) gradient
checkpointing DeepSpeed ZeRO-2 sees the same gradient twice. The built-in methods set
`gradient_checkpointing_kwargs: {use_reentrant: false}`; a method of your own that adds a forward pass needs the
same (`Run.replay` in `src/medresidency/methods.py`).

**Runs that start at the same time die with a bus error (exit code -7) while loading the dataset**

ms-swift caches the pre-processed dataset under a name that does not depend on the run, so a run that starts
rewrites the cache another run is reading. The launcher gives every run its own `MODELSCOPE_CACHE` and
`HF_DATASETS_CACHE`. Keep these lines when you run the commands of `commands.sh` by hand.

**vLLM stops with a compiler error (`nvcc`) at the first sampled token**

vLLM's FlashInfer sampler is compiled at first use, and compute nodes rarely have the CUDA compiler. The launcher
and `medres eval generate` set `VLLM_USE_FLASHINFER_SAMPLER=0`.

**CUDA out of memory on the student's cards**

Lower `train.per_device_batch`; the launcher raises the gradient accumulation, so the update stays the same. The
rollout engine holds `train.rollout_gpu_memory` of each student card (0.3 by default). The 2B students of the
benchmark trained on 48 GB cards, the 4B student on 80 GB cards.

**The disk is full while a checkpoint is written**

A checkpoint of a 2B student takes 29 GB, and a run needs room for two of them next to the weights of its earlier
checkpoints ([concepts.md](concepts.md#resuming)). Free the space and submit the same command again with
`run.resume=true`: the run continues from the last checkpoint that was written completely.

**All responses reach the response limit** (`completions/clipped_ratio` close to 1 in `logging.jsonl`)

This is not a software failure. The student has collapsed into repetition, which is what plain OPD does when student
and teacher start far apart. Measure the pair with `medres probe gap`
([evaluation.md](evaluation.md#the-gap-probe)) and see [methods.md](methods.md#adaopd).

**The job was cut by the time limit or preempted**

Submit the same command again with `run.resume=true`. The last checkpoint holds the optimizer state, finished stages
are skipped, and AdaOPD restores its controller from `logging.jsonl`. At most `train.save_steps` steps are repeated.

## Evaluation

**`report test sets are scored in the metrics environment: set MEDRES_METRICS_PYTHON`**

Report metrics need RadEval, which lives in an environment of its own
([installation.md](installation.md#metrics-environment)). The check comes before anything is generated. VQA test
sets need no metrics environment.

**RadEval fails at import, or CheXbert fails with scikit-learn 1.8 or newer**

Two problems of RadEval 2.2.1 that `scripts/install_metrics_env.sh` fixes. If you installed RadEval yourself,
compare with what the script does after `pip install`.

**The reward server of a `grpo` run cannot load the metric models**

RadEval fetches its models from the Hugging Face hub at first use, and the launcher runs everything with
`HF_HUB_OFFLINE=1`. Score one file with `medres eval report-score` on a machine with network access before the
first such run. The models are then in the Hugging Face cache (`HF_HOME`), which must be on a disk the compute nodes
see.

**GREEN is zero for many reports, or a GREEN batch fails**

A length artifact of GREEN, not a property of the reports: one looping response in a batch leaves the others no room
for the judge's answer. `medres eval report-score` cuts over-long responses and sorts the batches by length
([evaluation.md](evaluation.md#report-metrics)). GREEN numbers computed without the cap on a model that sometimes
loops are not comparable with ours.

**Images are not found after `data/` was moved**

The task files hold absolute image paths ([data.md](data.md#file-formats)). Build them again in the new place.

## Results that look wrong

**Runs with different seeds are almost the same model**

ms-swift keeps the seed of the data order (`data_seed`) at 42 unless it is passed, so a seed alone only changes the
model seed, and supervised runs then see the same batches in the same order. `medres train --seed` sets both. Check
this first when seeds from another code base agree suspiciously well.

**A difference between two methods changes sign with the seed, or with the GPU type**

Training is noisy. In the benchmark the same recipe run on two GPU types differed by 0.048 in MIMIC-CXR 1/RadCliQ.
See [evaluation.md](evaluation.md#comparing-two-runs) for how to compare.

**Your numbers differ from the benchmark**

Compare `<run>/medres/launches.jsonl` (package versions, code version, host) and `config.yaml` with the
[protocol](../benchmarks/README.md#protocol). Check that the evaluated checkpoint is the final one, and that the
test files have the sizes given on the benchmark page.

## Reporting a problem

Open an issue with the command, the last lines of the failing log, and the files `config.yaml`, `commands.sh` and
`launches.jsonl` of `<run>/medres/`.

Do not paste text from credentialed data. The training log of a MIMIC-CXR run can contain a training example, and
`completions.jsonl` holds prompts and reference reports ([data.md](data.md#credentialed-data)). Error messages and
tracebacks are enough.
