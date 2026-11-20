# Design

What MedResidency is meant to be, what it leaves out, and where its conventions come from.

## The problem

Post-training turns a pretrained vision-language model into a model for a job: supervised fine-tuning on labels,
distillation from a stronger model, reinforcement learning on a reward. In the general domain this work happens in a
few shared code bases, so two papers that report GRPO on the same model usually ran comparable code.

For medical VLMs we did not find such a code base (November 2026). What exists falls into four groups:

| Kind | Examples | What it covers |
|:--|:--|:--|
| training pipelines for medical language models | [MedicalGPT](https://github.com/shibing624/MedicalGPT) | text only |
| code bases for building medical VLMs | [ViLMedic](https://github.com/jbdel/vilmedic), [MedM-VL](https://github.com/MSIIP/MedM-VL), [MedVLThinker](https://github.com/UCSC-VLAA/MedVLThinker) | architectures and training recipes of their own models |
| code released with one method | [MedVLM-R1](https://arxiv.org/abs/2502.19634), [Med-R1](https://arxiv.org/abs/2503.13939), [ClinKD](https://arxiv.org/abs/2502.05928), [MedGRPO](https://arxiv.org/abs/2512.06581) | that method, on its own data and budget |
| benchmarks and environments | [BenchX](https://arxiv.org/abs/2410.21969), [MedVLMBench](https://github.com/ubc-tea/MedVLMBench), [MedAgentGym](https://github.com/wshi83/MedAgentGym) | evaluation of finished models; agent tasks |

None of them puts several post-training methods under one protocol on medical image tasks. A method paper therefore
compares itself with numbers from other setups, or re-implements its baselines once more. MedResidency is our
attempt at the shared piece: the tasks, the baselines and the evaluation that a new method can be dropped into.

## Goals

1. **One protocol, many methods.** A task fixes the student, the teachers, the data, the budget and the evaluation.
   A method changes the objective and nothing else.
2. **Traceable numbers.** Every hyper-parameter is in a YAML file. A run stores its resolved configuration, its
   commands and the package versions; a benchmark row comes with its command.
3. **Small.** The trainer is ms-swift's and the rollout engine is vLLM's. MedResidency adds configuration, a
   launcher, the medical tasks, the evaluation, and the code of the methods that did not exist yet. A method is one
   short function.
4. **Evaluation that survives a second look.** A validation split for every choice made after looking at a score,
   several seeds, paired comparisons, and the truncation rate next to every report metric.
5. **Data stay under their licences.** The repository holds no data, no weights and no model outputs, and the tools
   print counts and corpus-level numbers only.

## Non-goals

- **A new trainer.** On-policy distillation with teacher servers is available in ms-swift, TRL and verl. What
  MedResidency contributes is the medical protocol around it, the benchmark, AdaOPD and the gap probe.
- **Pre-training, or a model zoo.** The models are other people's releases, used under their terms.
- **A leaderboard of foundation models.** The benchmark compares methods at a fixed student, not models.
- **Scale.** A run uses the GPUs of one node. Students of 2B and 4B parameters and teachers of up to 8B are what
  was run.

## How a run is put together

```
configs/defaults.yaml ──┐
configs/tasks/<task> ───┤
configs/methods/<m> ────┼──> resolved configuration ──> builder ──> stages ──> launcher ──> runs/<name>/
recipe (optional) ──────┤                                                        │
key=value ──────────────┘                                                        ├── teacher servers (vLLM)
                                                                                 ├── reward server (some rewards)
                                                                                 └── ms-swift trainer + plugin
```

The configuration is plain YAML merged in a fixed order ([concepts.md](concepts.md#configuration)). The builder of
the method turns it into one or more stages, each an ms-swift command with its arguments. The launcher starts what a
stage needs around the trainer, checks it, runs the trainer and cleans up. Evaluation is a separate command that
only needs a model directory.

## Decisions

**Wrap ms-swift; do not fork it.** It already has a GRPO trainer with a per-token teacher term and CHORD, a GKD
trainer, rollouts with a colocated vLLM engine, and teachers behind an OpenAI-compatible API. Writing this again
would have been the largest part of the project and the least interesting. The price is a pinned version (4.6.0)
and two small patches. `scripts/install_train_env.sh` applies them and stops if one no longer fits. AdaOPD lives in
a plugin file that ms-swift imports, not in a patch.

**A task is more than a dataset.** Budget, response limit and learning rate decide the ranking of methods as much as
the data do, so they belong to the task. The alternative, a dataset registry plus free hyper-parameters, is more
flexible and is how comparisons drift apart.

**Methods are functions that return arguments.** A builder does not train anything. It returns the ms-swift
arguments of each stage, which makes a method easy to read, lets `--dry-run` print the full commands without a GPU,
and lets a test pin the arguments of every task × method (`tests/golden_args.json`). A change that would alter a
benchmark row cannot pass unnoticed.

**Unknown keys are errors.** A misspelt override that is silently ignored produces a run with default settings and
a name that says otherwise. In a benchmark this is the worst kind of bug, so the loader accepts only keys that a
YAML file declares. `train.swift_args` is the one deliberate escape hatch.

**Two environments.** The report metrics come from RadEval, whose dependencies we keep away from the training stack.
Training and generation never import a metric package, and scoring never imports vLLM.

**Runs do not overwrite each other.** The launcher refuses a directory that already has checkpoints unless
`run.resume=true`, every run gets its own dataset cache, and a teacher server is asked which model it serves before
training starts. Each of these rules is there because its absence cost us a result
([troubleshooting.md](troubleshooting.md)).

**Seeds mean seeds.** `--seed` sets the model seed and the seed of the data order. The benchmark reports the mean
over three seeds and says how large the run-to-run spread is.

**No text on the terminal.** Builders, generation and scoring print counts and corpus-level numbers. Per-item files
of the evaluation hold ids and numbers. With MIMIC-CXR on the task list, a tool that prints an example by default
would sooner or later put credentialed text into a shared log.

## What we borrowed

Most conventions of MedResidency are taken from general-domain frameworks, on purpose: someone who has used one of
them should not have to learn new habits.

| Convention | Seen in | Here |
|:--|:--|:--|
| one sub-command per job | [TRL](https://huggingface.co/docs/trl/clis) (`trl sft`), [ms-swift](https://github.com/modelscope/ms-swift) (`swift sft`, `swift rlhf`, `swift deploy`) | `medres train`, `medres eval`, `medres data`, `medres probe`, `medres stats`, `medres list` |
| `key=value` overrides on top of YAML defaults | [verl](https://github.com/volcengine/verl) and [EasyR1](https://github.com/hiyouga/EasyR1) | the last layer of the configuration |
| one YAML file per run worth repeating | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (`llamafactory-cli train examples/...yaml`) | `configs/recipes/` |
| tasks as YAML files selected by name | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (`--tasks`) | `configs/tasks/`, `medres list tasks` |
| tuned settings stored per algorithm and environment, next to a table of results | [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) | the task files, `overrides.<method>`, `benchmarks/README.md` |
| a benchmark where every result has its command, and a page per algorithm that states what is implemented | [CleanRL](https://docs.cleanrl.dev/) | the commands under every benchmark table, [methods.md](methods.md) |
| rewards as small registered functions | ms-swift (`orms`), EasyR1 | `plugins/reward_plugin.py`, `medres list rewards` |
| teachers behind an OpenAI-compatible server | ms-swift, [TRL's distillation example](https://github.com/huggingface/trl/tree/main/examples/async_distillation_mopd) | the launcher starts and checks one server per teacher |
| generation engine separate from the learner | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), verl | vLLM for rollouts, teachers and evaluation |

Three things we did differently:

- **No experiment manager.** Hydra, as used by verl, composes configurations well, but its output directories and
  multi-run machinery are more than a single-node benchmark needs. The loader is under 200 lines of plain Python.
- **No web UI and no dataset registry file.** A task file names its data files directly.
- **Benchmark results are part of the repository**, as one Markdown page with the commands, not a separate
  leaderboard service. A result is contributed by pull request
  ([benchmarks](../benchmarks/README.md#contributing-results)).

## The name

Medical school gives a physician general knowledge; residency is the supervised practice afterwards, next to
attending physicians, that makes a specialist. Post-training plays that part for a pretrained model, and
distillation from expert teachers is the closest analogue of all. The command is `medres`.

## Limits of the current version

- Only the Qwen2-VL and Qwen3-VL families have been run, and only with teachers that share the student's tokenizer.
- One node per run. Only the default four-card layout was used for the benchmark.
- ms-swift is pinned to 4.6.0. A newer release needs the two patches checked again.
- `grpo` is implemented and tested at the level of its arguments and rewards, but the benchmark has no row for it.
- The four tasks cover chest X-ray reports and multiple-choice VQA. Grounding, segmentation-style outputs, 3D
  volumes and multi-image studies are not covered.

## Further reading

- TRL: <https://github.com/huggingface/trl>
- LLaMA-Factory: <https://github.com/hiyouga/LLaMA-Factory>
- ms-swift: <https://github.com/modelscope/ms-swift>
- verl: <https://github.com/volcengine/verl>
- EasyR1: <https://github.com/hiyouga/EasyR1>
- OpenRLHF: <https://github.com/OpenRLHF/OpenRLHF>
- lmms-eval: <https://github.com/EvolvingLMMs-Lab/lmms-eval>
- rl-baselines3-zoo: <https://github.com/DLR-RM/rl-baselines3-zoo>
- CleanRL: <https://github.com/vwxyzjn/cleanrl>
