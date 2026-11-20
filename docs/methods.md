# Methods

The task fixes the student, the prompts and the budget. A method only decides what the loss is. This page says what
each method optimizes, which files and servers it needs, and which keys it has. The keys and their defaults are in
`configs/methods/<name>.yaml`; the code that turns them into training commands is in `src/medresidency/methods.py`.

| Method | `--method` | Trains on | Teacher servers | Stages |
|:--|:--|:--|:--|--:|
| [Label SFT](#label-sft) | `sft` | `data.labels` | – | 1 |
| [Offline KD](#offline-kd) | `offline_kd` | `data.teacher_outputs` | – | 1 |
| [OPD](#opd) | `opd` | `data.prompts` | yes | 1 |
| [GKD](#gkd) | `gkd` | `data.prompts` | yes | 1 |
| [Two-stage](#two-stage) | `two_stage` | `data.teacher_outputs`, then `data.prompts` | in stage 2 | 2 |
| [CHORD](#chord) | `chord` | `data.prompts` and `data.teacher_outputs` | yes | 1 |
| [CHORD-φ](#chord-φ) | `chord_phi` | `data.prompts` and `data.teacher_outputs` | yes | 1 |
| [AdaOPD](#adaopd) | `adaopd` | `data.prompts` and `data.teacher_outputs` | yes | 1 |
| [GRPO](#grpo) | `grpo` | `data.prompts` with a `solution` field | – | 1 |

Three kinds of training file appear in the table. *Prompts* are questions with an image and no answer. *Labels* are
the same prompts with the reference answer. *Teacher outputs* are the same prompts with the answer a teacher wrote
before training ([data.md](data.md#teacher-outputs) shows how to produce them).

## What all methods share

- Full fine-tuning of the student with the vision tower frozen, bf16, DeepSpeed ZeRO-2.
- `train.steps` optimizer steps of `train.sequences_per_step` sequences, at a constant learning rate `train.lr`.
- The on-policy methods (OPD and everything below it in the table) let the student write its own responses, at
  temperature 1 and with at most `train.max_completion` tokens, from a vLLM engine on the student's GPUs.
- The methods with an SFT term read `train.replay_per_device_batch` teacher outputs per GPU in every forward pass
  in addition. With the default layout that is 64 teacher outputs per step.

Below, `x` is a prompt with its image, `y` a response, `p_S` the student and `p_T` the teacher.

## Label SFT

```bash
medres train --task cxr_report --method sft
```

Cross-entropy on the reference answers. The method needs `data.labels`, which only `cxr_report` has among the
built-in tasks; on the others the command stops with a message.

## Offline KD

```bash
medres train --task cxr_report --method offline_kd
```

Cross-entropy on the teacher's answers (sequence-level distillation; Kim and Rush, 2016). No teacher server runs
during training, since the answers were written beforehand. With several teachers the file holds each prompt with
the answer of its own teacher.

## OPD

```bash
medres train --task cxr_report --method opd
```

On-policy distillation. The student samples a response `y` to every prompt, and the teacher scores each sampled
token. The advantage of token `t` is

```
A_t = log p_T(y_t | x, y_<t) − log p_S(y_t | x, y_<t)
```

and the student is updated by policy gradient with this advantage. Position by position, this lowers the reverse KL
divergence from the student to the teacher on the student's own samples (Lu and Thinking Machines Lab, 2025). The
implementation is ms-swift's GRPO trainer with the reward switched off: `teacher_kl_coef 1`, one sample per prompt,
the clipped surrogate with ε = 0.2, and no KL term to a reference model (`beta 0`).

With several teachers, each row is scored by the teacher named in its `teacher_tag`. This is multi-teacher OPD
(Ma et al., 2026), and it needs no extra option: `medres train --task cxr_report_vqa --method opd`.

| Key | Default | Meaning |
|:--|:--|:--|
| `method.repetition_penalty` | `null` | repetition penalty of the student's sampling, a common fix for looping |
| `method.monitor_gap` | `false` | log AdaOPD's language gap at every step without acting on it |

`monitor_gap` costs one more teacher pass per step. It shows how the gap develops in a run that does nothing about
it.

## GKD

```bash
medres train --task cxr_report --method gkd
```

Generalized knowledge distillation (Agarwal et al., 2024). Where OPD uses the teacher's log-probability of the one
token the student sampled, GKD compares whole distributions: the loss is a generalized Jensen–Shannon divergence
between the student and the teacher's top-k distribution at every position of the student's samples.

| Key | Default | Meaning |
|:--|:--|:--|
| `method.topk` | `20` | teacher log-probabilities per position |
| `method.lmbda` | `1.0` | probability that a batch uses the student's own samples rather than answers from the file |
| `method.beta` | `0.5` | interpolation of the generalized JSD |

The teacher servers return the top `topk` log-probabilities for every position instead of one. The built-in prompt
files hold no answers, so `lmbda` stays at 1 on the built-in tasks.

## Two-stage

```bash
medres train --task cxr_report --method two_stage
```

Offline KD for `method.sft_steps` steps, then OPD for the remaining `train.steps − sft_steps`. The total budget is
that of the other methods. Stage 1 writes to `<run>/stage1`, stage 2 to the run directory, so the final checkpoint
is `<run>/checkpoint-<steps − sft_steps>` (`checkpoint-150` with the defaults).

| Key | Default | Meaning |
|:--|:--|:--|
| `method.sft_steps` | `150` | length of stage 1 |
| `method.stage1_checkpoint` | `null` | start stage 2 from this checkpoint and skip stage 1 |

## CHORD

```bash
medres train --task cxr_report --method chord
```

Every step mixes OPD on the student's samples with SFT on teacher outputs:

```
L = (1 − μ) · L_OPD + μ · L_SFT
```

The weight μ follows a schedule fixed in advance, a cosine from `peak` to `valley` over `decay_steps`, and stays at
`valley` afterwards. CHORD was proposed for mixing SFT on expert data with reinforcement learning (Zhang et al.,
2026); here its on-policy term is OPD. The default goes from pure SFT to pure OPD over the whole run.

| Key | Default | Meaning |
|:--|:--|:--|
| `method.peak` | `1.0` | μ at the first step |
| `method.valley` | `0.0` | μ after the decay |
| `method.decay_steps` | `null` | length of the decay; `null` means `train.steps` |

A constant weight is `method.peak=0.27 method.valley=0.27`.

## CHORD-φ

```bash
medres train --task cxr_report --method chord_phi
```

The second variant of the CHORD paper: a small constant μ, and an SFT term in which every token is weighted by
`p (1 − p)`, where `p` is the student's probability of that token. The weight is small where the student is already
sure or far off, and large in between.

| Key | Default | Meaning |
|:--|:--|:--|
| `method.mu` | `0.1` | the constant weight of the SFT term |

## AdaOPD

```bash
medres train --task cxr_report --method adaopd
```

AdaOPD uses the same mixture as CHORD, but it measures μ at every step where CHORD schedules it.

The observation behind it: a teacher's score of a student token has two parts. Scoring the same response a second
time *without the image* separates them:

```
A_t = [log p_T(y_t | x, y_<t) − log p_T(y_t | no image, y_<t)]      visual evidence
    − [log p_S(y_t | x, y_<t) − log p_T(y_t | no image, y_<t)]      language gap
```

The visual evidence says whether the image supports the token. The language gap says how differently student and
teacher write, whatever the image shows. When the gap is large, for example a general student facing a teacher that
writes in the register of radiology reports, it dominates the advantage, and OPD mostly pushes the student towards
the teacher's wording along its own, still foreign, samples. That is the regime in which OPD collapsed into
repetition in the benchmark. SFT on the teacher's outputs closes the gap directly.

AdaOPD therefore sets the weight of the SFT term from the measured gap `g_s` of step `s`:

```
μ_s = clip((EMA(g_s) − τ) / max(g_0 − τ, 1), 0, 1)
```

- `g_s` is the mean language gap over the *fresh* tokens of the step's samples. A token is fresh unless it
  completes a 4-gram that already occurred in the same response; a student that has begun to repeat itself would
  otherwise hide its gap behind tokens both models find predictable.
- `g_0` is the gap of the first step. Training starts at μ = 1 when that gap is at least one nat above τ.
- τ is the threshold, in nats. Once the smoothed gap is below it, the step is pure OPD.

A student that starts close to its teacher (`g_0 ≤ τ`) trains as plain OPD for as long as its gap stays below τ. A
student that starts far away begins with SFT on the teacher's outputs and moves to OPD as fast as its own gap
closes.

| Key | Default | Meaning |
|:--|:--|:--|
| `method.tau` | `1.0` | the threshold τ. Select it on validation data ([benchmark](../benchmarks/README.md#selecting-τ)) |
| `method.ema` | `0.9` | smoothing of the gap across steps; `0` uses the gap of the current step |
| `method.measure_tags` | `[]` | measure the gap on the rows of these teachers only; empty means all |
| `method.fresh_tokens` | `true` | `false` measures the gap on all tokens |
| `method.signal` | `lang` | `kl` measures the gap to the teacher *with* the image instead (an ablation) |
| `method.gap0` | `null` | the normalizer `g_0`; `null` means the gap of the first step |

**Several teachers.** One μ is shared by the whole step. Only teachers that are far from the student should drive
it and be replayed: `measure_tags` names them, and `data.teacher_outputs` holds their outputs only. The built-in
tasks set both in their `overrides.adaopd` block (the report teacher in `cxr_report_vqa`; CT, MRI and ultrasound in
`modality_experts`). For your own teachers, `medres probe gap` measures the initial gap of a pair on one GPU
([evaluation.md](evaluation.md#the-gap-probe)).

**Cost.** One more teacher pass per step, for the image-free score. The teacher server is the same; no second
copy of the teacher is loaded.

**Logging.** `logging.jsonl` gets `lang_gap`, `visual_evidence` and `adaopd_mu` at every step. When a run is
resumed, the smoothed gap and `g_0` are restored from that file.

The other methods are points of the same family: offline KD is μ = 1 throughout, OPD is μ = 0, two-stage switches
from 1 to 0 at a fixed step, and CHORD follows a fixed curve. The
[ablations](../benchmarks/README.md#ablations-of-adaopd) compare these schedules under one budget.

## GRPO

```bash
medres train --task med_vqa --method grpo
```

Reinforcement learning on a task reward, without a teacher (Shao et al., 2024). The student writes
`num_generations` responses per prompt, a reward function scores each of them against the row's `solution`, the
advantages are normalized within the group of one prompt, and a KL penalty with weight `beta` keeps the policy near
the initial student.

| Key | Default | Meaning |
|:--|:--|:--|
| `method.reward` | set by the task | one of `medres list rewards` |
| `method.beta` | `0.04` | weight of the KL penalty |
| `method.num_generations` | `4` | responses per prompt |

| Reward | Used by | Scores |
|:--|:--|:--|
| `report_radcliq` | `cxr_report` | −RadCliQ-v1 of the report against the reference report |
| `mcq_accuracy` | `med_vqa` | 1 if the option letter of the response is the solution's, else 0 |
| `answer_exact` | `modality_experts` | 1 if the content of `<answer></answer>` equals the solution's |
| `mixed_report_vqa` | `cxr_report_vqa` | `mcq_accuracy` or `report_radcliq`, depending on the row |

The two report rewards are computed by a metric server that the launcher starts on the student's first GPU. They
need the metrics environment (`MEDRES_METRICS_PYTHON`, see [installation.md](installation.md#metrics-environment)).

GRPO is included so that reward-based and teacher-based post-training can be compared under one protocol. The
benchmark has no GRPO rows yet.

## Options that are not methods

Some variants in the literature are one override away and have no method of their own:

```bash
# OPD with a KL penalty to the initial student
medres train --task cxr_report --method opd train.swift_args.beta=0.04 --name cxr_report_opd_kl_s42

# OPD at a lower learning rate
medres train --task cxr_report --method opd train.lr=1e-6 --name cxr_report_opd_lr1e-6_s42
```

To add a method of your own, see [extending.md](extending.md#a-method).

## References

- R. Agarwal, N. Vieillard, Y. Zhou, et al. On-policy distillation of language models: Learning from
  self-generated mistakes. ICLR, 2024. (GKD)
- Y. Kim and A. M. Rush. Sequence-level knowledge distillation. EMNLP, 2016. (offline KD)
- K. Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025.
  <https://thinkingmachines.ai/blog/on-policy-distillation/> (OPD)
- W. Ma et al. MOPD: Multi-teacher on-policy distillation for capability integration in LLM post-training.
  arXiv:2606.30406, 2026. (multi-teacher OPD)
- Z. Shao et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models.
  arXiv:2402.03300, 2024. (GRPO)
- W. Zhang, Y. Xie, Y. Sun, et al. On-policy RL meets off-policy experts: Harmonizing supervised fine-tuning and
  reinforcement learning via dynamic weighting. ICLR, 2026. (CHORD, CHORD-φ)
