# Data

The repository contains no data. You download the sources yourself, and `medres data` turns them into the task
files that training and evaluation read:

```
raw/            the sources, as downloaded                      ($MEDRES_RAW)
data/<task>/    the task files: prompts, teacher outputs, test sets, resized images   ($MEDRES_DATA)
models/         students and teachers                           ($MEDRES_MODELS)
```

## Sources

| Source | Used by | Where it comes from | Access |
|:--|:--|:--|:--|
| MIMIC-CXR 2.0.0 (reports) | `cxr_report`, `cxr_report_vqa` | [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) | credentialed |
| MIMIC-CXR-JPG 2.0.0 (images, split and metadata tables) | `cxr_report`, `cxr_report_vqa` | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | credentialed |
| ReXrank test definitions | `cxr_report` | [rajpurkarlab/ReXrank](https://github.com/rajpurkarlab/ReXrank), `data/` | public download; the MIMIC-CXR file holds report text |
| IU X-ray images | `cxr_report` (test only) | hub dataset `dz-osamu/IU-Xray` | public |
| PMC-VQA (version 2) | `med_vqa`, `cxr_report_vqa` | hub dataset `RadGenome/PMC-VQA` | public |
| MedXpertQA | `med_vqa` (test only) | hub dataset `TsinghuaC3I/MedXpertQA` | public |
| MMMU, five health subjects | `med_vqa` (test only) | hub dataset `MMMU/MMMU` | public |
| OmniMedVQA, open-access part | `modality_experts` | hub dataset `foreverbeliever/OmniMedVQA` | public |
| Med-R1's splits of OmniMedVQA | `modality_experts` | [Yuxiang-Lai117/Med-R1](https://github.com/Yuxiang-Lai117/Med-R1), `Splits/modality` | public |

`scripts/download_models.sh <task>` fetches the public sources of a task together with its models. Every dataset and
every model has its own licence and terms of use. Read them where you download; MedResidency's Apache-2.0 licence
covers the code of this repository and nothing else.

## Credentialed data

MIMIC-CXR is released to credentialed PhysioNet users who have signed its data-use agreement. MedResidency does not
download it and cannot give you access. If you work with it, the agreement covers more than the original files:

- the task files built from it (`data/cxr_report/`, `data/cxr_report_vqa/`), which contain report text and images;
- `ReXRank_MIMICCXR_test.json`, which is public to download but contains MIMIC-CXR reports;
- what a teacher or a student writes about these images: teacher outputs, test-set generations, per-study scores;
- `<run>/completions.jsonl`, in which ms-swift logs the sampled rollouts next to their prompts and references
  (`train.log_completions=false` turns it off);
- anything you send elsewhere. The agreement does not allow sharing the data with third parties, and an online
  service such as a hosted LLM API is a third party.

Whether weights trained on the data may be released is also a question for the agreement, not for this framework.

The tools are written for this situation. The builders, the generator, the scorers and the gap probe print counts
and aggregate statistics, never text. `.gitignore` keeps `data/`, `raw/`, `runs/`, `evals/`, every `*.jsonl` and
every image out of version control, and `scripts/check_release.py` fails when such a file is tracked anyway.
Results enter the benchmark as corpus-level numbers ([benchmarks](../benchmarks/README.md#contributing-results)).

## Building the task files

All builders write to `$MEDRES_DATA/<task>` unless `--out` says otherwise, are deterministic, and print what they
wrote. `medres data <tool> --help` documents every option.

### `med_vqa`

```bash
bash scripts/download_models.sh med_vqa
medres data med-vqa --pmc-vqa raw/pmc_vqa --medxpertqa raw/medxpertqa --mmmu raw/mmmu
```

This writes `train.jsonl` (20,000 questions), `val.jsonl` (1,000), the three test files and `img/`, with images
resized so that the longest side is at most 896 pixels. `--n-train`, `--n-val` and `--n-test` change the sizes.
The validation questions are drawn from PMC-VQA's training file, on figures that no training prompt uses.

### `cxr_report`

```bash
bash scripts/download_models.sh cxr_report

medres data cxr-report --mimic-cxr /path/to/mimic-cxr/2.0.0 --mimic-cxr-jpg /path/to/mimic-cxr-jpg/2.0.0 \
    --rexrank raw/rexrank/ReXRank_MIMICCXR_test.json

IU_IMAGES=$(dirname "$(find raw/iu_xray/extracted -type d -name 'CXR1_*' | head -1)")
medres data cxr-iu --rexrank raw/rexrank/ReXRank_IUXray_test.json --images "$IU_IMAGES"
```

`medres data cxr-report` reads `mimic-cxr-reports.zip` from the first directory and `files/`,
`mimic-cxr-2.0.0-split.csv.gz` and `mimic-cxr-2.0.0-metadata.csv.gz` from the second. It only reads them; both
directories can be read-only. It writes

- `train.jsonl` and `train_labels.jsonl`: 20,000 studies of the official training split that have a findings and an
  impression section, one frontal image each (PA before AP);
- `val.jsonl`: 500 studies of the official validation split;
- `test_mimic.jsonl`: the test studies of ReXrank. No study of this file enters training or validation;
- `img/`: the images at 512 × 512.

The sections are extracted with the section parser of the MIMIC-CXR code repository, a copy of which is in
`src/medresidency/data/third_party/`.

`medres data cxr-iu` adds `test_iu.jsonl` and its images. It needs no credentialed data, so the IU X-ray test of a
model can be reproduced by anyone.

### `cxr_report_vqa`

The two-teacher task mixes the two tasks above. Build both first, including their teacher outputs
([below](#teacher-outputs)), then:

```bash
medres data mixed
```

By default this draws 10,000 prompts from `data/med_vqa` and 10,000 from `data/cxr_report`, shuffles them, and
writes `train.jsonl` and `train_teacher.jsonl` in the same order to `data/cxr_report_vqa/`. `--sources`, `--n` and
`--out` build other mixtures. Only prompts for which a teacher output exists are drawn. A mixture that contains
MIMIC-CXR rows is credentialed data.

### `modality_experts`

```bash
bash scripts/download_models.sh modality_experts
medres data modality-experts --splits raw/med_r1/Splits/modality --images raw/omnimedvqa/OmniMedVQA
```

`--images` is the directory that holds OmniMedVQA's `Images/`; the paths in Med-R1's split files are relative to
it. The builder writes `train.jsonl` (71,333 questions, each tagged with the expert of its modality), `test.jsonl`
(17,662 questions) and the images at 384 × 384.

## Teacher outputs

Offline KD and the SFT term of two-stage, CHORD, CHORD-φ and AdaOPD train on answers the teacher wrote to the
training prompts. They are made once per task, in two steps: the teacher generates, and
`medres data teacher-outputs` joins its responses with the prompts.

```bash
medres eval generate --task med_vqa --profile teacher_outputs --model models/MediX-R1-8B \
    --data data/med_vqa/train.jsonl --out gen/med_vqa.jsonl
medres data teacher-outputs --prompts data/med_vqa/train.jsonl --generations gen/med_vqa.jsonl \
    --out data/med_vqa/train_teacher.jsonl
```

`--profile teacher_outputs` takes the response limit and the decoding from the `generation.teacher_outputs` block of
the task file, so everyone's file is made the same way. Responses that hit the limit, or that fail the task's format
check, are dropped: `train_teacher.jsonl` can be shorter than `train.jsonl`, and the tool prints how many rows it
kept.

Generation runs on one GPU. For 20,000 prompts it is worth splitting: `--shard i/n` takes every n-th row, and
`scripts/slurm/teacher_outputs.sbatch` runs four shards as an array job.

```bash
sbatch -p <partition> scripts/slurm/teacher_outputs.sbatch --task cxr_report --model models/UniRG-CXR \
    --data data/cxr_report/train.jsonl
# when all shards are done:
medres data teacher-outputs --prompts data/cxr_report/train.jsonl --generations gen/shard*.jsonl \
    --out data/cxr_report/train_teacher.jsonl
```

With several teachers, each one answers the prompts that carry its tag (`--tag`), and all files are joined at once:

```bash
for tag in CT MRI X-Ray Fundus Dermo Micro OCT US; do
    medres eval generate --task modality_experts --profile teacher_outputs --tag $tag \
        --model models/Med-R1/VQA_$tag --data data/modality_experts/train.jsonl --out gen/$tag.jsonl
done
medres data teacher-outputs --prompts data/modality_experts/train.jsonl --generations gen/*.jsonl \
    --out data/modality_experts/train_teacher.jsonl
medres data teacher-outputs --prompts data/modality_experts/train.jsonl --generations gen/*.jsonl \
    --tags CT MRI US --out data/modality_experts/train_teacher_far.jsonl
```

The last command keeps the outputs of three teachers only. AdaOPD's SFT term reads the teachers that are far from
the student, and `medres probe gap` ([evaluation.md](evaluation.md#the-gap-probe)) says which ones those are.

The join refuses to write a file unless there is exactly one generation for every prompt, so a missing shard or a
missing teacher cannot go unnoticed.

## File formats

Every file is JSON lines: one object per line. Rows follow ms-swift's conversation format, with a few extra keys.

**Training prompts** (`train.jsonl`, the `data.prompts` of a task):

```json
{"messages": [{"role": "user", "content": "<image>Which organ is shown? A: Liver B: Kidney C: Spleen D: Heart ..."}],
 "images": ["/path/to/data/med_vqa/img/example.jpg"],
 "teacher_tag": "medix",
 "solution": "<answer>B</answer>"}
```

| Key | Meaning |
|:--|:--|
| `messages` | one user turn; `<image>` marks where the image goes |
| `images` | one image path |
| `teacher_tag` | the teacher that scores this row, a key of the task's `teachers` |
| `solution` | what a reward compares the response with (`grpo`); distillation methods ignore it |

**Supervised files** (`train_teacher.jsonl`, `train_labels.jsonl`; `data.teacher_outputs` and `data.labels`) hold
the same user turn followed by an assistant turn with the answer, and the `images`.

**Test and validation files** add what scoring needs to the prompt:

| Task type | Keys | Meaning |
|:--|:--|:--|
| VQA | `id`, `domain` | `<domain>/<number>`; accuracy is reported per domain |
| | `answer` | the correct option letter |
| | `parser` | how a response is read: `letter` (the option letter of the final answer) or `answer_tag` (the word inside `<answer></answer>`, compared exactly) |
| Report | `id` | the study |
| | `findings`, `impression`, `reference` | the reference sections, and both in the form `Findings: ... Impression: ...` |
| | `patient` | optional; groups studies for the patient-level bootstrap of `medres stats compare` |

**Generations** (`medres eval generate`) have one line per prompt, `{id, text, n_tokens, truncated}`, plus
`malformed` when a format check was requested. `id` is the row's own `id`, or its row number in files without one.

Two properties of these files are worth knowing:

- **Image paths are absolute.** The builders write the absolute path of each resized image. If you move `data/`,
  rebuild the files or rewrite the paths.
- **Files are written atomically.** A builder that is interrupted leaves the previous file in place, not half a
  file.

To use your own data, write files in these formats and name them in a task file
([extending.md](extending.md#a-task)). No builder is needed.

## References

- MIMIC-CXR: Johnson et al., "MIMIC-CXR, a de-identified publicly available database of chest radiographs with
  free-text reports", Scientific Data, 2019.
- IU X-ray: Demner-Fushman et al., "Preparing a collection of radiology examinations for distribution and
  retrieval", Journal of the American Medical Informatics Association, 2016.
- ReXrank: Zhang et al., "ReXrank: A public leaderboard for AI-powered radiology report generation",
  arXiv:2411.15122, 2024.
- PMC-VQA: Zhang et al., "PMC-VQA: Visual instruction tuning for medical visual question answering",
  arXiv:2305.10415, 2023.
- MedXpertQA: Zuo et al., "MedXpertQA: Benchmarking expert-level medical reasoning and understanding",
  arXiv:2501.18362, 2025.
- MMMU: Yue et al., "MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert
  AGI", CVPR 2024.
- OmniMedVQA: Hu et al., "OmniMedVQA: A new large-scale comprehensive evaluation benchmark for medical LVLM",
  CVPR 2024.
- Med-R1: Lai et al., "Med-R1: Reinforcement learning for generalizable medical reasoning in vision-language
  models", IEEE Transactions on Medical Imaging, 2026.
