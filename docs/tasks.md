# Tasks

A task fixes everything that two methods must share to be comparable: the student, the teachers, the training
prompts, the budget and the test sets. Tasks are the files in `configs/tasks/`. `medres list tasks` prints them, and
`medres train --task <task> --method <method> --print-config` shows every value a run would use.

| | `cxr_report` | `med_vqa` | `cxr_report_vqa` | `modality_experts` |
|:--|:--|:--|:--|:--|
| Student | Qwen3-VL-2B | Qwen3-VL-2B | Qwen3-VL-2B | Qwen2-VL-2B |
| Teachers | UniRG-CXR (8B) | MediX-R1 (8B) | both | eight Med-R1 experts (2B) |
| Training prompts | 20,000 MIMIC-CXR studies | 20,000 PMC-VQA questions | 10,000 of each | OmniMedVQA, Med-R1's split |
| Steps × sequences | 300 × 64 | 300 × 64 | 300 × 64 | 500 × 128 |
| Learning rate | 5e-6 | 1e-6 | 5e-6 | 5e-6 |
| Response limit (tokens) | 512 | 2,048 | 2,048 | 512 |
| Test sets | `mimic_cxr`, `iu_xray` | `pmc_vqa`, `medxpertqa_mm`, `mmmu_health` | all five | `omnimedvqa` |
| Reward of `grpo` | `report_radcliq` | `mcq_accuracy` | `mixed_report_vqa` | `answer_exact` |
| Data access | credentialed | public | credentialed | public |

All methods run on all tasks, with one exception: `sft` needs reference answers written out as training targets
(`data.labels`), which only `cxr_report` provides.

The task files of `$MEDRES_DATA` are built from the sources with `medres data` ([data.md](data.md)). The results of
every method on every task are on the [benchmark page](../benchmarks/README.md).

## `cxr_report`: chest X-ray report generation

A general 2B student learns to write the findings and the impression of a chest X-ray from UniRG-CXR, an 8B report
specialist. The specialist writes in the register of radiology reports, which the student does not: this is the
task with a large language gap (3.05 nats), and the one on which plain OPD collapses into repetition.

| File in `data/cxr_report/` | Content |
|:--|:--|
| `train.jsonl` | 20,000 studies of the official MIMIC-CXR training split, one frontal image each |
| `train_labels.jsonl` | the same studies with the reference report as the answer (`sft`) |
| `train_teacher.jsonl` | the same studies with the teacher's report (`offline_kd` and the SFT terms) |
| `val.jsonl` | 500 studies of the official validation split |
| `test_mimic.jsonl` | the MIMIC-CXR test studies of ReXrank (1,624) |
| `test_iu.jsonl` | the IU X-ray test studies of ReXrank (590); no IU X-ray study is used in training |

Images are resized once to 512 × 512, so the student, the teacher and the evaluation see the same pixels. The
prompt is UniRG-CXR's findings-and-impression template, with the indication and the comparison of the study as
context. Reports are scored on findings and impression together; the headline metric is 1/RadCliQ-v1.

```bash
medres train --task cxr_report --method adaopd --seed 42
medres eval run --task cxr_report --model runs/cxr_report_adaopd_s42/checkpoint-300 --green
```

MIMIC-CXR is credentialed. Only `test_iu.jsonl` can be built without PhysioNet access, which is enough to evaluate
a model on IU X-ray but not to train on this task.

## `med_vqa`: reasoning VQA

The same student learns from MediX-R1-8B, a reasoning teacher, on multiple-choice questions about figures of
biomedical papers. Every prompt asks for step-by-step reasoning and for the letter of the answer inside
`<answer></answer>`, so responses are long: the limit is 2,048 tokens.

Student and teacher start close to each other here (0.64 nats, below AdaOPD's threshold). This is the control
task: plain OPD works, and AdaOPD measures the gap and trains exactly as OPD.

| File in `data/med_vqa/` | Content |
|:--|:--|
| `train.jsonl` | 20,000 questions of PMC-VQA `train_2.csv` |
| `train_teacher.jsonl` | the same questions with the teacher's answer |
| `val.jsonl` | 1,000 further training questions, on figures that no training prompt uses |
| `test_pmc_vqa.jsonl` | 1,000 questions of PMC-VQA `test_2.csv` |
| `test_medxpertqa_mm.jsonl` | MedXpertQA-MM test, the questions with a single image (1,581) |
| `test_mmmu_health.jsonl` | MMMU validation, the five health subjects, single-image multiple choice (136) |

The metric is accuracy. A response in which no option letter can be found counts as wrong.

```bash
medres train --task med_vqa --method opd --seed 42
medres eval run --task med_vqa --model runs/med_vqa_opd_s42/checkpoint-300
```

Everything in this task is public, which makes it the one to start with.

## `cxr_report_vqa`: two teachers, one student

Report generation and reasoning VQA in one run: 10,000 prompts of each of the two tasks above, shuffled together.
Every row carries the name of its teacher in `teacher_tag`, and the launcher starts one server per teacher. One
student has to learn both skills at once, from two teachers that are at very different distances from it.

| File in `data/cxr_report_vqa/` | Content |
|:--|:--|
| `train.jsonl` | the 20,000 mixed prompts |
| `train_teacher.jsonl` | the same prompts, each with the answer of its own teacher |

The task has no test files of its own. It is evaluated on the five test sets of the two single-teacher tasks, each
generated with the settings of its own task (`eval.test_from` in the task file).

Two methods get their own settings on this task, in the `overrides` block of the task file:

- `adaopd` measures the gap on the report teacher's rows only, and its SFT term reads that teacher's outputs on the
  20,000 studies of `cxr_report`. The reasoning teacher is close to the student and needs no SFT.
- `grpo` trains with two sequences per GPU, since 2,048-token samples share the first GPU with the metric server.

`chord` and `two_stage` read the outputs of both teachers by default. Two recipes give them the report teacher's
outputs only, the same data AdaOPD reads:

```bash
medres train --task cxr_report_vqa --method adaopd --seed 42
medres train configs/recipes/adaopd/cxr_report_vqa_chord_report.yaml --seed 42
medres train configs/recipes/adaopd/cxr_report_vqa_two_stage_report.yaml --seed 42
medres eval run --task cxr_report_vqa --model runs/cxr_report_vqa_adaopd_s42/checkpoint-300 --green
```

## `modality_experts`: eight experts, one student

Med-R1 released eight experts, one Qwen2-VL-2B per imaging modality (CT, MRI, X-ray, fundus, dermoscopy,
microscopy, OCT, ultrasound), each trained with GRPO on the questions of its modality in OmniMedVQA. The task merges
them back into one Qwen2-VL-2B. Each prompt is scored by the expert of its modality, so eight teacher servers run
next to the student, four per GPU.

The protocol is Med-R1's rather than that of the other tasks: its prompt templates, images at 384 × 384, 500 steps
of 128 sequences and a response limit of 512 tokens.

| File in `data/modality_experts/` | Content |
|:--|:--|
| `train.jsonl` | the training questions, each tagged with the expert of its modality |
| `train_teacher.jsonl` | the experts' answers to them, one sample at temperature 1 |
| `train_teacher_far.jsonl` | the answers of the CT, MRI and ultrasound experts only |
| `test.jsonl` | the test questions, with the modality as `domain` |

Three experts are far from the base model and five are close (`medres probe gap`). The `overrides.adaopd` block
therefore measures the gap on CT, MRI and ultrasound and replays the outputs of these three. The other overrides
set stage 1 of `two_stage` to 200 steps, so that both stages end on a checkpoint, and add ms-swift's `format` reward
to `grpo`, as Med-R1 does.

```bash
medres train --task modality_experts --method adaopd --seed 42
medres eval run --task modality_experts --model runs/modality_experts_adaopd_s42/checkpoint-500
```

The accuracy is reported per modality and as micro and macro averages.

## Recipes

A recipe is a run that is more than a task and a method. The ones behind the benchmark are in
`configs/recipes/adaopd/`:

| Recipe | What it changes |
|:--|:--|
| `cxr_report_4b.yaml` | `cxr_report` with Qwen3-VL-4B as the student; names no method, add `--method` |
| `cxr_report_4b_adaopd.yaml` | AdaOPD for that student, with the threshold validation selected for it (τ = 0) |
| `cxr_report_vqa_chord_report.yaml` | CHORD on two teachers, replaying the report teacher only |
| `cxr_report_vqa_two_stage_report.yaml` | two-stage on two teachers, stage 1 on the report teacher only |
| `new_teacher.yaml` | `cxr_report_vqa` starting from a student already trained on `med_vqa`; add `--method` |

```bash
medres train configs/recipes/adaopd/cxr_report_4b.yaml --method gkd --seed 42
medres train configs/recipes/adaopd/new_teacher.yaml --method adaopd --seed 42
```

`new_teacher.yaml` starts from `runs/med_vqa_opd_s42/checkpoint-300`. For another seed, point `student.init` to the
student of that seed.

## Your own task

A task is one YAML file plus the data files it names; no code is needed as long as the rows follow the format of
[data.md](data.md#file-formats). [extending.md](extending.md#a-task) walks through an example.
