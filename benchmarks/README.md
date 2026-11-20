# Benchmark

Results of the methods in MedResidency on its tasks, with the commands that reproduce each row.

Within a table every run follows one protocol: the same student, data, number of optimizer steps, sequences per step,
learning rate, response limit and hardware. Only the method differs. Unless a table says otherwise, a number is the
mean ± standard deviation over three seeds (42, 2, 3) at the final checkpoint, with greedy decoding on the test set.

> **Where the numbers come from.** The tables were produced for the AdaOPD paper with the research code that
> MedResidency was refactored from. They have not been re-run with this package. What we did check: for 47 of the
> recorded runs we compared the training arguments MedResidency builds with the arguments that were recorded at the
> time. 40 are identical; 7 differ only in how checkpoints are saved (`save_steps`, `save_only_model`). We also
> rebuilt the task files with `medres data` and compared them with the files the recorded runs read
> ([data.md](../docs/data.md#check-against-the-benchmarks-files)). If you reproduce a row, please
> [send us the result](#contributing-results).

## Contents

- [Protocol](#protocol)
- [Metrics and test sets](#metrics-and-test-sets)
- [Report generation](#report-generation)
- [Two teachers](#two-teachers)
- [Reasoning VQA](#reasoning-vqa)
- [Modality experts](#modality-experts)
- [A larger student](#a-larger-student)
- [A new teacher for a trained student](#a-new-teacher-for-a-trained-student)
- [Ablations of AdaOPD](#ablations-of-adaopd)
- [Selecting τ](#selecting-τ)
- [Gap atlas](#gap-atlas)
- [Comparing two runs](#comparing-two-runs)
- [Contributing results](#contributing-results)

## Protocol

| | |
|:--|:--|
| Student | Qwen3-VL-2B-Instruct, full fine-tuning with the vision tower frozen, bf16, DeepSpeed ZeRO-2 |
| Teachers | 8B medical VLMs of the same model family, so teacher and student share a tokenizer. Each teacher runs in its own vLLM server and returns the log-probability of every sampled token (the top 20 for GKD) |
| Budget | 300 optimizer steps of 64 sequences, about one pass over 20,000 training prompts. Methods with an SFT term read 64 teacher outputs per step in addition |
| Learning rate | 5e-6 for reports and for two teachers, 1e-6 for reasoning VQA; constant, no warm-up |
| Response limit | 512 tokens for reports; 2,048 for reasoning VQA and for two teachers |
| Rollouts | sampled by the student at temperature 1, with a vLLM engine on the student's own GPUs |
| OPD loss | ms-swift's GRPO objective with the teacher term only: clipping ε = 0.2, no KL to a reference model |
| Hardware | four A40 (48 GB) per run, two for the student and two for the teacher servers; four H100 (80 GB) for the 4B student. Every evaluation runs on an A40 |
| Evaluation | final checkpoint, greedy decoding |

All of this is the default configuration: `configs/defaults.yaml` plus the task file. `medres train ... --print-config`
shows the resolved values of a run.

Training is noisy: the same recipe run twice differed by 0.048 in MIMIC-CXR 1/RadCliQ between two GPU types. Keep
the runs you compare on one GPU type, and do not read a single-seed difference of that size as a result.

## Metrics and test sets

Reports are scored on findings and impression together.

- **1/RadCliQ** is the headline metric: the reciprocal of the mean RadCliQ-v1 over the test studies. Higher is better.
- **RadGraph** (F1), **RaTE** (RaTEScore), **CheXbert-5** (micro-F1 over five conditions) and **GREEN** are shown on a
  0–100 scale.
- **Trunc.** is the share of MIMIC-CXR test reports that reach the generation limit of 1,024 tokens. A student that
  has collapsed into repetition shows up here.

VQA is scored by accuracy (%) over all questions. A response with no recognizable option counts as wrong.

| Test set (`--sets`) | Task | Size | Source |
|:--|:--|--:|:--|
| `mimic_cxr` | `cxr_report` | 1,624 studies | MIMIC-CXR test split of ReXrank |
| `iu_xray` | `cxr_report` | 590 studies | IU X-ray test split of ReXrank; no IU X-ray study is used in training |
| `pmc_vqa` | `med_vqa` | 1,000 questions | PMC-VQA v2 test |
| `medxpertqa_mm` | `med_vqa` | 1,581 questions | MedXpertQA MM, the single-image questions |
| `mmmu_health` | `med_vqa` | 136 questions | MMMU validation, health and medicine subjects (one image, at most five options) |
| `omnimedvqa` | `modality_experts` | 17,662 questions | Med-R1's test split of OmniMedVQA |

The task `cxr_report_vqa` is evaluated on the five sets of `cxr_report` and `med_vqa`.

In every table the best trained method of a column is in bold. The first rows (the untrained student, the teachers)
are references and are not ranked.

## Report generation

Task `cxr_report`: one teacher, UniRG-CXR, on 20,000 MIMIC-CXR training studies with one frontal image each. The
untrained student is far from this teacher (language gap 3.05 nats, see the [gap atlas](#gap-atlas)), and plain OPD
collapses into repetition on some seeds.

| Method | MIMIC-CXR 1/RadCliQ | MIMIC-CXR RadGraph | MIMIC-CXR RaTE | MIMIC-CXR CheXbert-5 | MIMIC-CXR GREEN | IU X-ray 1/RadCliQ | IU X-ray RadGraph | IU X-ray RaTE | Trunc. (%) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| General student (Qwen3-VL-2B) | 0.548 | 11.1 | 50.8 | 27.8 | 21.2 | 0.678 | 15.0 | 55.5 | 48.9 |
| Specialized teacher (UniRG-CXR-8B) | 1.278 | 29.7 | 63.9 | 65.6 | 40.4 | 3.228 | 37.6 | 70.9 | 0.0 |
| Label SFT | 0.818 ± 0.015 | 18.3 ± 0.4 | 56.4 ± 0.2 | 41.5 ± 3.5 | 29.3 ± 0.8 | 1.551 ± 0.234 | 24.1 ± 3.7 | 62.2 ± 3.7 | 4.1 ± 2.7 |
| Offline KD | 0.841 ± 0.019 | 20.0 ± 0.2 | 56.9 ± 0.4 | 43.4 ± 5.0 | 30.3 ± 0.1 | 1.441 ± 0.068 | 24.8 ± 0.4 | 66.0 ± 1.1 | 6.9 ± 2.7 |
| OPD | 0.726 ± 0.074 | 17.4 ± 0.9 | 56.5 ± 2.2 | 39.1 ± 10.3 | 24.8 ± 1.9 | 1.749 ± 0.094 | 26.9 ± 0.2 | 67.7 ± 0.8 | 22.4 ± 18.3 |
| GKD | 0.919 ± 0.013 | 21.6 ± 0.2 | 59.0 ± 0.3 | 42.7 ± 5.7 | 31.0 ± 0.4 | 1.892 ± 0.109 | **28.9** ± 0.2 | **68.1** ± 0.5 | **0.0** ± 0.0 |
| CHORD | 0.913 ± 0.015 | 21.5 ± 0.3 | 59.1 ± 0.6 | 51.5 ± 3.4 | 30.1 ± 1.4 | 1.702 ± 0.227 | 27.4 ± 1.2 | 67.1 ± 1.0 | 1.2 ± 0.3 |
| CHORD-φ | 0.742 ± 0.129 | 18.5 ± 1.4 | 57.6 ± 0.9 | 42.1 ± 6.6 | 26.2 ± 2.0 | 1.746 ± 0.157 | 26.9 ± 1.1 | 68.1 ± 1.3 | 27.5 ± 37.6 |
| Two-stage | 0.892 ± 0.034 | 21.4 ± 1.0 | 58.9 ± 1.2 | **52.1** ± 0.7 | 30.4 ± 3.1 | 1.563 ± 0.176 | 27.0 ± 0.9 | 67.0 ± 0.3 | 3.4 ± 5.5 |
| AdaOPD | **0.931** ± 0.005 | **22.0** ± 0.3 | **59.7** ± 0.6 | 50.3 ± 4.7 | **31.2** ± 1.0 | **1.950** ± 0.030 | 28.7 ± 0.0 | 68.0 ± 0.1 | 0.8 ± 0.9 |

```bash
# the two reference rows
medres eval run --task cxr_report --model models/Qwen3-VL-2B-Instruct --green
medres eval run --task cxr_report --model models/UniRG-CXR --green

# one method, three seeds
for seed in 42 2 3; do
    medres train --task cxr_report --method adaopd --seed $seed
    medres eval run --task cxr_report --model runs/cxr_report_adaopd_s$seed/checkpoint-300 --green
done
```

| Row | `--method` | |
|:--|:--|:--|
| Label SFT | `sft` | on the reference reports |
| Offline KD | `offline_kd` | SFT on the teacher's outputs |
| OPD | `opd` | |
| GKD | `gkd` | |
| CHORD | `chord` | cosine decay of μ from 1 to 0 over the 300 steps |
| CHORD-φ | `chord_phi` | |
| Two-stage | `two_stage` | 150 steps of offline KD, then 150 of OPD; the final checkpoint is `checkpoint-150` |
| AdaOPD | `adaopd` | τ = 1 |

## Two teachers

Task `cxr_report_vqa`: UniRG-CXR teaches 10,000 report prompts and MediX-R1-8B teaches 10,000 PMC-VQA questions, mixed
in one run. One student serves both tasks.

| Method | 1/RadCliQ | RaTE | GREEN | IU 1/RadCliQ | PMC-VQA | MedXpertQA | MMMU-H&M |
|:--|--:|--:|--:|--:|--:|--:|--:|
| General student (Qwen3-VL-2B) | 0.548 | 50.8 | 21.2 | 0.678 | 46.3 | 16.3 | 41.9 |
| Specialized teachers (UniRG-CXR-8B / MediX-R1-8B) | 1.278 | 63.9 | 40.4 | 3.228 | 53.0 | 30.2 | 68.4 |
| Offline KD | 0.788 ± 0.013 | 56.6 ± 0.3 | 28.7 ± 0.4 | 1.715 ± 0.166 | 46.0 ± 2.7 | 17.6 ± 0.9 | 43.6 ± 3.0 |
| OPD | 0.619 ± 0.143 | 53.9 ± 3.7 | 22.0 ± 5.9 | 1.478 ± 0.604 | 46.6 ± 2.7 | 17.1 ± 2.1 | 46.1 ± 1.1 |
| GKD | 0.847 ± 0.008 | 57.8 ± 0.6 | 28.4 ± 0.9 | 1.902 ± 0.082 | 47.9 ± 0.3 | 19.9 ± 0.4 | 46.4 ± 1.3 |
| CHORD (both) | 0.874 ± 0.027 | 58.9 ± 0.2 | 29.6 ± 0.8 | **1.933** ± 0.015 | 47.0 ± 1.6 | 19.8 ± 0.4 | 46.6 ± 2.8 |
| CHORD (report) | 0.892 ± 0.036 | 58.1 ± 1.1 | 28.9 ± 1.7 | 1.589 ± 0.243 | 49.1 ± 1.0 | 20.9 ± 0.6 | 47.5 ± 4.3 |
| Two-stage (both) | 0.888 ± 0.033 | 58.3 ± 1.1 | 29.7 ± 0.8 | 1.785 ± 0.205 | 48.7 ± 0.4 | 19.1 ± 0.5 | 46.8 ± 1.1 |
| Two-stage (report) | 0.888 ± 0.041 | 58.6 ± 0.5 | 29.4 ± 0.3 | 1.840 ± 0.069 | 48.1 ± 1.0 | **21.2** ± 1.0 | 47.3 ± 2.9 |
| AdaOPD | **0.920** ± 0.008 | **59.3** ± 0.6 | **30.7** ± 0.6 | 1.915 ± 0.018 | **49.5** ± 1.0 | 20.8 ± 0.7 | **49.5** ± 3.8 |

"Both" and "report" say which teacher outputs the SFT term reads: those of both teachers on the 20,000 mixed prompts,
or those of the report teacher only. AdaOPD measures the gap to each teacher and keeps the outputs of the teachers
whose gap is above τ. Here that is the report teacher alone, with its outputs on the 20,000 studies of the report
task, and the "report" rows give CHORD and the two-stage recipe the same data.

```bash
medres train --task cxr_report_vqa --method adaopd --seed 42
medres train --task cxr_report_vqa --method chord --seed 42                            # CHORD (both)
medres train configs/recipes/adaopd/cxr_report_vqa_chord_report.yaml --seed 42          # CHORD (report)
medres train --task cxr_report_vqa --method two_stage --seed 42                        # Two-stage (both)
medres train configs/recipes/adaopd/cxr_report_vqa_two_stage_report.yaml --seed 42      # Two-stage (report)

medres eval run --task cxr_report_vqa --model runs/cxr_report_vqa_adaopd_s42/checkpoint-300 --green
```

A run started from a recipe is named after the recipe file: `runs/cxr_report_vqa_chord_report_s42`.

## Reasoning VQA

Task `med_vqa`: MediX-R1-8B on 20,000 PMC-VQA training questions, with long step-by-step rationales. This is the
control setting. The student starts close to the teacher (gap 0.64 nats, below τ), so AdaOPD uses no SFT from the
first step and trains exactly as OPD. The two share one row.

| Method | PMC-VQA | MedXpertQA | MMMU-H&M |
|:--|--:|--:|--:|
| General student (Qwen3-VL-2B) | 46.3 | 16.3 | 41.9 |
| Specialized teacher (MediX-R1-8B) | 53.0 | 30.2 | 68.4 |
| Offline KD | 46.1 ± 0.3 | 17.3 ± 1.4 | 44.8 ± 2.0 |
| GKD | 46.7 ± 1.0 | 17.8 ± 0.7 | 42.6 ± 0.8 |
| CHORD | 47.3 ± 0.8 | **19.3** ± 0.6 | 46.3 ± 3.4 |
| Two-stage | 47.9 ± 0.7 | 19.0 ± 1.1 | **48.5** ± 0.8 |
| OPD = AdaOPD | **48.3** ± 1.0 | **19.3** ± 1.3 | 44.9 ± 3.0 |

```bash
medres train --task med_vqa --method opd --seed 42
medres eval run --task med_vqa --model runs/med_vqa_opd_s42/checkpoint-300
```

`--method adaopd` measures the gap, finds it below τ and keeps μ at 0.

## Modality experts

Task `modality_experts`: the eight Med-R1 experts (one Qwen2-VL-2B per imaging modality, trained with GRPO on
OmniMedVQA) are distilled back into the base Qwen2-VL-2B. Each prompt is scored by the expert of its modality. The
protocol differs from the other tasks: 500 steps of 128 sequences, learning rate 5e-6, response limit 512 tokens.

Three experts (CT, MRI, ultrasound) are far from the base model and five are close, so AdaOPD reads the outputs of the
three far experts. Their gap closes quickly: μ is below 0.1 by step 17–20 and 0 from step 19–24 on, depending on the
seed, and the rest of the run is OPD.

| Method | OmniMedVQA accuracy (%) |
|:--|--:|
| Offline KD | 94.28 |
| OPD | 95.48 (seeds: 94.07–97.12) |
| AdaOPD | 95.98 |

Means of three seeds, on the test questions with Med-R1's prompt. AdaOPD is within the range of OPD's seeds: imitating
the far experts for the first steps costs nothing here.

```bash
medres train --task modality_experts --method adaopd --seed 42
medres eval run --task modality_experts --model runs/modality_experts_adaopd_s42/checkpoint-500
```

In the recorded runs of this task the seed changed the sampling and initialization noise but not the order of the
data. MedResidency also seeds the data order, so seeds 2 and 3 will not see the prompts in the recorded order.

## A larger student

The report task with Qwen3-VL-4B-Instruct as the student, on four 80 GB GPUs. Everything else is unchanged. The 4B
student is further from the teacher than the 2B one (gap 3.66 nats), and validation selects τ = 0 for it
([Selecting τ](#selecting-τ)).

| Method | Seeds | MIMIC-CXR 1/RadCliQ | MIMIC-CXR RadGraph | MIMIC-CXR RaTE | MIMIC-CXR CheXbert-5 | MIMIC-CXR GREEN | IU X-ray 1/RadCliQ | IU X-ray RadGraph | IU X-ray RaTE | Trunc. (%) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| General student (Qwen3-VL-4B) | – | 0.688 | 12.2 | 52.2 | 35.3 | 23.4 | 0.816 | 12.6 | 54.4 | 0.7 |
| Specialized teacher (UniRG-CXR-8B) | – | 1.278 | 29.7 | 63.9 | 65.6 | 40.4 | 3.228 | 37.6 | 70.9 | 0.0 |
| Offline KD | 3 | 0.898 ± 0.010 | 21.7 ± 0.2 | 58.1 ± 0.2 | 46.2 ± 3.1 | 31.5 ± 0.3 | 1.442 ± 0.038 | 25.0 ± 0.5 | 65.9 ± 0.6 | 2.8 ± 1.0 |
| OPD | 3 | 0.525 ± 0.156 | 13.4 ± 4.7 | 50.7 ± 6.3 | 31.2 ± 7.9 | 18.2 ± 5.5 | 1.243 ± 0.715 | 26.4 ± 1.6 | 65.4 ± 3.4 | 66.9 ± 27.7 |
| GKD | 3 | 0.923 ± 0.022 | 22.0 ± 0.2 | 58.6 ± 1.0 | 43.9 ± 6.4 | 30.6 ± 0.3 | 1.785 ± 0.114 | 28.0 ± 0.3 | 67.1 ± 0.8 | 1.1 ± 1.8 |
| CHORD (fixed schedule) | 3 | 0.908 ± 0.037 | 22.0 ± 0.9 | **59.7** ± 0.7 | 52.9 ± 8.4 | **32.2** ± 0.2 | 1.666 ± 0.148 | 27.1 ± 1.4 | 67.6 ± 0.5 | 3.9 ± 3.6 |
| CHORD-φ (authors' best variant) | 3 | 0.746 ± 0.066 | 19.2 ± 1.2 | 56.7 ± 1.0 | 32.4 ± 1.5 | 25.0 ± 3.0 | **1.890** ± 0.061 | **28.4** ± 0.1 | **68.2** ± 0.1 | 21.2 ± 13.1 |
| Two-stage (offline KD → OPD) | 3 | 0.882 ± 0.008 | 21.1 ± 0.3 | 58.6 ± 0.5 | 44.0 ± 7.1 | 29.4 ± 0.9 | 1.499 ± 0.093 | 25.5 ± 0.5 | 66.5 ± 0.9 | 0.9 ± 1.0 |
| AdaOPD (τ = 0 selected on validation) | 3 | **0.934** ± 0.022 | **22.1** ± 0.8 | 59.3 ± 0.6 | **54.1** ± 0.8 | 30.9 ± 0.7 | 1.676 ± 0.165 | 27.0 ± 1.3 | 67.2 ± 0.5 | **0.5** ± 0.6 |

```bash
medres train configs/recipes/adaopd/cxr_report_4b.yaml --method gkd --seed 42      # any method
medres train configs/recipes/adaopd/cxr_report_4b_adaopd.yaml --seed 42            # AdaOPD with τ = 0
medres eval run --task cxr_report --model runs/cxr_report_4b_adaopd_s42/checkpoint-300 --green
```

## A new teacher for a trained student

The student has already been trained: it is the OPD student of the [reasoning VQA](#reasoning-vqa) task. The report
teacher is now added, and training continues with both teachers on the mixed prompts of `cxr_report_vqa`. The report
columns measure the new skill, the VQA columns the skill to keep.

| Method | Seeds | 1/RadCliQ | RaTE | GREEN | IU 1/RadCliQ | PMC-VQA | MedXpertQA | MMMU-H&M |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| VQA-trained student (Qwen3-VL-2B) | 3 | 0.570 ± 0.005 | 52.4 ± 0.2 | 21.9 ± 0.3 | 0.730 ± 0.003 | 48.3 ± 1.0 | 19.3 ± 1.3 | 44.9 ± 3.0 |
| Specialized teachers (UniRG-CXR-8B / MediX-R1-8B) | – | 1.278 | 63.9 | 40.4 | 3.228 | 53.0 | 30.2 | 68.4 |
| Offline KD | 3 | 0.784 ± 0.011 | 56.6 ± 0.1 | 27.8 ± 0.4 | 1.712 ± 0.268 | 45.9 ± 1.9 | 18.0 ± 0.3 | 42.4 ± 0.9 |
| OPD | 3 | 0.505 ± 0.008 | 57.4 ± 0.1 | 28.6 ± 2.8 | 0.595 ± 0.036 | 48.1 ± 2.3 | 17.2 ± 1.1 | 44.6 ± 2.4 |
| GKD | 3 | 0.855 ± 0.022 | 58.1 ± 0.6 | 29.3 ± 0.2 | **1.962** ± 0.022 | 47.2 ± 1.7 | 17.3 ± 1.4 | 44.4 ± 2.1 |
| CHORD (fixed schedule) | 3 | 0.901 ± 0.024 | 58.2 ± 0.3 | 29.6 ± 1.5 | 1.724 ± 0.137 | 48.9 ± 0.7 | **20.6** ± 0.6 | 43.9 ± 3.0 |
| Two-stage (offline KD → OPD) | 3 | 0.853 ± 0.036 | 58.0 ± 0.4 | 29.1 ± 1.4 | 1.935 ± 0.015 | 47.2 ± 0.7 | 18.7 ± 0.3 | 43.9 ± 1.2 |
| AdaOPD | 3 | **0.916** ± 0.017 | **59.4** ± 0.6 | **30.5** ± 0.9 | 1.939 ± 0.001 | **49.8** ± 0.8 | 19.9 ± 0.8 | **48.3** ± 2.4 |

On the trained student the report teacher's gap is 3.05 nats (seed 42), as large as on the untrained one, and the
VQA teacher's is 0.44, below τ. AdaOPD therefore reads the report teacher's outputs only. CHORD and the two-stage
recipe read the outputs of both teachers, which is the task's default.

```bash
medres train --task med_vqa --method opd --seed 42                                  # the trained student
medres train configs/recipes/adaopd/new_teacher.yaml --method adaopd --seed 42
medres eval run --task cxr_report_vqa --model runs/new_teacher_adaopd_s42/checkpoint-300 --green
```

Each seed starts from the student of the same seed:

```bash
medres train configs/recipes/adaopd/new_teacher.yaml --method adaopd --seed 2 \
    student.init=runs/med_vqa_opd_s2/checkpoint-300
```

## Ablations of AdaOPD

On the report task. Each row changes one component of AdaOPD and keeps the others.

| Variant | Loop | MIMIC-CXR 1/RadCliQ | Δ | IU X-ray 1/RadCliQ |
|:--|--:|--:|--:|--:|
| AdaOPD | 0 | 0.931 ± 0.005 | – | 1.950 ± 0.030 |
| *Schedule of μ* (AdaOPD: from the measured gap) |  |  |  |  |
| None, μ = 0 (OPD) | 12–24 | 0.726 ± 0.074 | -0.205 | 1.749 ± 0.094 |
| Constant, μ = 0.27 | 10–12 | 0.926 ± 0.007 | -0.005 | 1.937 ± 0.021 |
| Cosine over 300 steps (CHORD) | 0 | 0.913 ± 0.015 | -0.018 | 1.702 ± 0.227 |
| Cosine over 100 steps | 0 | 0.924 ± 0.010 | -0.007 | 1.893 ± 0.050 |
| Cosine over 50 steps | 0 | 0.860 ± 0.123 | -0.071 | 1.841 ± 0.153 |
| CHORD authors' schedule | 0 | 0.897 ± 0.099 | -0.034 | 1.710 ± 0.414 |
| *Signal behind μ* (AdaOPD: language gap on fresh tokens) |  |  |  |  |
| Language gap on all tokens | 0 | 0.931 ± 0.030 | +0.000 | 1.941 ± 0.012 |
| Reverse KL | 0 | 0.924 ± 0.020 | -0.007 | 1.895 ± 0.028 |
| Reverse KL, τ = 1.44 | 0 | 0.920 ± 0.048 | -0.011 | 1.927 ± 0.010 |
| *Smoothing and threshold* (AdaOPD: moving average, τ = 1) |  |  |  |  |
| No moving average | 0 | 0.940 ± 0.025 | +0.010 | 1.926 ± 0.014 |
| τ = 0.5 | 0 | 0.929 ± 0.015 | -0.002 | 1.936 ± 0.017 |
| τ = 1.5 | 0 | 0.914 ± 0.016 | -0.017 | 1.959 ± 0.048 |

- **Loop** is the longest run of consecutive training steps in which at least half of the rollouts reach the
  response limit. A range is over the seeds.
- **Δ** is the difference to AdaOPD in MIMIC-CXR 1/RadCliQ, as the mean of the per-seed differences.
- *Constant* uses AdaOPD's mean μ over training (0.27). *Cosine* decays μ from 1 to 0 over the given steps and trains
  as OPD afterwards. The *CHORD authors' schedule* decays from 0.9 to 0.05 over 200 steps and stays at 0.05.
- *Reverse KL* measures the gap against the teacher that sees the image, in place of the image-free teacher.

Every row is an override of one run. Give each variant its own `--name`, or it will collide with the default run:

```bash
# schedule of μ
medres train --task cxr_report --method chord method.peak=0.27 method.valley=0.27 --name cxr_report_constant_s42
medres train --task cxr_report --method chord method.decay_steps=100 --name cxr_report_cosine100_s42
medres train --task cxr_report --method chord method.decay_steps=50 --name cxr_report_cosine50_s42
medres train --task cxr_report --method chord method.peak=0.9 method.valley=0.05 method.decay_steps=200 \
    --name cxr_report_chord_authors_s42

# signal behind μ
medres train --task cxr_report --method adaopd method.fresh_tokens=false --name cxr_report_adaopd_alltokens_s42
medres train --task cxr_report --method adaopd method.signal=kl --name cxr_report_adaopd_kl_s42
medres train --task cxr_report --method adaopd method.signal=kl method.tau=1.44 \
    --name cxr_report_adaopd_kl_tau1.44_s42

# smoothing and threshold
medres train --task cxr_report --method adaopd method.ema=0 --name cxr_report_adaopd_noema_s42
medres train --task cxr_report --method adaopd method.tau=0.5 --name cxr_report_adaopd_tau0.5_s42
medres train --task cxr_report --method adaopd method.tau=1.5 --name cxr_report_adaopd_tau1.5_s42
```

Three common remedies for a collapsing OPD run, without an SFT term:

| Variant | μ | Seeds | Loop | 1/RadCliQ | Δ | IU 1/RadCliQ | Trunc. (%) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AdaOPD (τ = 1) | measured | 3 | 0 | 0.931 ± 0.005 | – | 1.950 ± 0.030 | 0.8 ± 0.9 |
| OPD | 0 | 3 | 12–24 | 0.726 ± 0.074 | -0.205 ± 0.070 | 1.749 ± 0.094 | 22.4 ± 18.3 |
| OPD + repetition penalty 1.2 | 0 | 1 | 171 | 0.316 | -0.611 | 0.331 | 100.0 |
| OPD, learning rate 1e-6 | 0 | 1 | 42 | 0.774 | -0.153 | 1.808 | 10.5 |
| OPD + KL to the initial student | 0 | 1 | 22 | 0.823 | -0.104 | 1.974 | 4.1 |

The rows with one seed are seed 42, and their Δ is against AdaOPD with the same seed.

```bash
medres train --task cxr_report --method opd method.repetition_penalty=1.2 --name cxr_report_opd_reppen_s42
medres train --task cxr_report --method opd train.lr=1e-6 --name cxr_report_opd_lr1e-6_s42
medres train --task cxr_report --method opd train.swift_args.beta=0.04 --name cxr_report_opd_kl_s42
```

The KL to the initial student has no key of its own: `beta` is the argument of ms-swift's trainer, passed through
`train.swift_args`.

## Selecting τ

AdaOPD has one hyper-parameter that matters, the gap threshold τ. Its default is 1 nat. The benchmark selects it on
validation data only, never on a test set:

- **Validation sets.** Reports: 500 studies of the official MIMIC-CXR validation split, scored by 1/RadCliQ. VQA: 1,000
  PMC-VQA training questions that are not among the 20,000 training prompts, on figures that none of those prompts
  uses (one question per figure).
- **Rule, fixed before the runs.** The value with the highest validation mean over the selection seeds wins, but
  τ = 1 is kept unless another value is better by more than 0.01 in 1/RadCliQ or one point of accuracy.

| Setting | Validation metric | Seeds | τ = -0.5 | τ = 0 | τ = 0.3 | τ = 0.5 | τ = 1 | τ = 1.5 | Selected |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Report generation, 2B student | 1/RadCliQ | 3 | – | – | – | 1.380 | **1.386** | 1.363 | 1 |
| Report generation, 4B student | 1/RadCliQ | 2 | 1.409 | **1.424** | – | 1.402 | 1.397 | – | 0 |
| Long-reasoning VQA | PMC-VQA acc. | 2 | – | – | 44.9 | – | **46.9** | – | 1 |

A dash marks a value outside the grid of that setting. The grid was run on the first two seeds (4B student, VQA) or
all three (2B student).

```bash
medres train --task cxr_report --method adaopd method.tau=0.5 --name cxr_report_adaopd_tau0.5_s42
medres eval run --task cxr_report --model runs/cxr_report_adaopd_tau0.5_s42/checkpoint-300 --split validation
```

Baselines run at their authors' settings. If you tune a baseline, tune it on the same validation set and say so.

## Gap atlas

The gap probe measures a teacher-student pair before any training. The untrained student writes responses to training
prompts, and both models score them:

- **ĝ**, the language gap: log p_student(y) − log p_teacher(y | no image), in nats per fresh token. It says how far
  the student's text is from text the teacher would write.
- **ē**, the visual evidence: log p_teacher(y | image) − log p_teacher(y | no image), averaged over the same tokens.

The table gives ĝ on responses sampled at temperature 1 and on greedy responses, and ē on the sampled ones. Bold marks
ĝ above τ = 1, where AdaOPD starts with SFT on teacher outputs. Teacher and student share a tokenizer in every pair.

| Teacher | Student | ĝ (sampled) | ĝ (greedy) | ē (sampled) |
|:--|:--|--:|--:|--:|
| *Chest X-ray reports (MIMIC-CXR training studies)* |  |  |  |  |
| UniRG-CXR-8B | Qwen3-VL-2B | **3.05** | 3.04 | -0.44 |
| UniRG-CXR-8B | Qwen3-VL-4B | **3.66** | 3.69 | -0.52 |
| QoQ-Med3-VL-8B | Qwen3-VL-2B | **1.07** | 1.03 | +0.18 |
| Lingshu-7B | Qwen2.5-VL-3B | 0.52 | 0.37 | -0.19 |
| NV-Reason-CXR-3B | Qwen2.5-VL-3B | **2.04** | 1.29 | -0.10 |
| Qwen2.5-VL-7B (general) | Qwen2.5-VL-3B | 0.47 | 0.41 | +0.07 |
| *Medical VQA* |  |  |  |  |
| MediX-R1-8B, PMC-VQA | Qwen3-VL-2B | 0.64 | 0.75 | +0.11 |
| QoQ-Med3-VL-8B, PMC-VQA | Qwen3-VL-2B | 0.80 | 0.76 | +0.12 |
| SonoReasoner-8B, ultrasound | Qwen3-VL-2B | 0.51 | 0.64 | +0.04 |
| *Med-R1 modality experts (OmniMedVQA)* |  |  |  |  |
| Med-R1 CT | Qwen2-VL-2B | **3.59** | 3.70 | -0.05 |
| Med-R1 MRI | Qwen2-VL-2B | **2.92** | 3.09 | -0.03 |
| Med-R1 ultrasound | Qwen2-VL-2B | **2.99** | 4.38 | -0.16 |
| Med-R1 X-ray | Qwen2-VL-2B | 0.22 | 0.32 | +0.13 |
| Med-R1 dermoscopy | Qwen2-VL-2B | 0.18 | 0.27 | +0.10 |
| Med-R1 fundus | Qwen2-VL-2B | 0.21 | 0.29 | +0.11 |
| Med-R1 microscopy | Qwen2-VL-2B | 0.17 | 0.32 | +0.08 |
| Med-R1 OCT | Qwen2-VL-2B | 0.15 | 0.22 | +0.08 |

Each pair was probed on 128 training prompts (32 per Med-R1 expert).

```bash
medres probe gap --data data/cxr_report/train.jsonl --student models/Qwen3-VL-2B-Instruct \
    --teacher models/UniRG-CXR --out probes/unirg_qwen3vl2b
medres probe gap --data data/med_vqa/train.jsonl --student models/Qwen3-VL-2B-Instruct \
    --teacher models/MediX-R1-8B --max-new-tokens 2048 --out probes/medix_qwen3vl2b
medres probe gap --data data/modality_experts/train.jsonl --tag CT --n 32 --student models/Qwen2-VL-2B-Instruct \
    --teacher models/Med-R1/VQA_CT --out probes/medr1_ct
```

The probe takes model directories, so a pair does not have to be registered in `configs/models.yaml`: the Qwen2.5-VL
rows were measured that way. SonoReasoner's ultrasound questions are the one prompt set of the table that
MedResidency does not build.

A large gap at the start does not by itself predict a collapse. Three modality experts start around 3 nats and OPD
trains fine, because there the gap closes early in training. With the report teacher it stays above τ, and OPD loops.
To watch the gap during an OPD run without changing the training, add `method.monitor_gap=true`.

## Comparing two runs

`medres eval run` writes one line per test item next to each summary. `medres stats compare` reads two such files and
gives the paired difference with a 95% bootstrap interval (1,000 resamples):

```bash
medres stats compare evals/cxr_report_adaopd_s42/checkpoint-300/mimic_cxr.scores.jsonl \
    evals/cxr_report_chord_s42/checkpoint-300/mimic_cxr.scores.jsonl \
    --metric inv_radcliq --cluster-data data/cxr_report/test_mimic.jsonl

medres stats compare evals/med_vqa_opd_s42/checkpoint-300/pmc_vqa.items.jsonl \
    evals/med_vqa_chord_s42/checkpoint-300/pmc_vqa.items.jsonl --metric accuracy
```

`--cluster-data` resamples patients instead of studies, because a patient can have several studies in the MIMIC-CXR
test set.

How the benchmark reads a comparison:

- Compare seed by seed. The interval covers the test set, not the randomness of training.
- A **win** or **loss** needs an interval that excludes zero.
- A **tie** is an interval that contains zero with a difference below 0.01 in 1/RadCliQ (0.5 points for the other
  metrics).
- Anything else is **undetermined**.
- "Better" in the tables means a higher mean over the seeds; significance is reported separately.

## Contributing results

New rows are welcome: a reproduction of a row above, a new method on an existing task, or a new task with its
baselines. Open a pull request that adds the row and gives

- the commands, and the commit of MedResidency they were run with;
- the GPUs, and the seeds (three, if you can afford them);
- the numbers from `evals/<run>/<checkpoint>/summary.json`.

Send aggregates only. Reports generated for MIMIC-CXR studies are credentialed data under the PhysioNet agreement and
must not be uploaded, and neither should per-study scores or logs that contain report text.

The method `grpo` (reinforcement learning with a task reward, `medres list rewards`) is implemented but has no row
yet. It is a good place to start.
