# Extending

Most changes need no code. Before writing any, check whether one of the first two rows is enough:

| You want | Do this |
|:--|:--|
| another value of an existing key | `key=value` on the command line ([concepts.md](concepts.md#configuration)) |
| a combination you will run more than once | a recipe file |
| another model | [one line](#a-model) in `configs/models.yaml` |
| another dataset, teacher or budget | [a task file](#a-task) and data files |
| another loss or training schedule | [a method](#a-method): a YAML file and a builder function |
| another reward for `grpo` | [a reward class](#a-reward) |
| another test set for an existing task | [one line](#a-test-set) in the task file |

Whatever you add, `--dry-run` shows the commands it leads to before a GPU is involved.

## A model

Models are used through a name. The names are the keys of `configs/models.yaml`:

```yaml
my-teacher: {hub: my-org/My-CXR-8B, model_type: qwen3_vl}
```

| Key | Meaning |
|:--|:--|
| `hub` | the id on the Hugging Face hub; `scripts/download_models.sh model my-org/My-CXR-8B` fetches it |
| `path` | the directory of the weights, relative to `$MEDRES_MODELS` or absolute; default: the last part of `hub` |
| `model_type` | ms-swift's model type |
| `template` | ms-swift's chat template, when it is not the model type |

A recipe can define a model for itself in a `models:` block, which is handy while you try a model out. This one
replaces the teacher of the report task:

```yaml
# configs/recipes/mine/cxr_report_my_teacher.yaml
task: cxr_report
models:
  my-teacher: {hub: my-org/My-CXR-8B, model_type: qwen3_vl}
teachers: {unirg: my-teacher}
data: {teacher_outputs: cxr_report/train_my_teacher.jsonl}
```

Three things to keep in mind:

- **The tag stays, the model changes.** The rows of `train.jsonl` name their teacher by a tag (`unirg`), and
  `teachers` maps the tag to a model. Replacing the model under the same tag needs no new prompt file.
- **A new teacher needs its own teacher outputs** for the methods that read them
  ([data.md](data.md#teacher-outputs)), and it has its own gap to the student
  ([evaluation.md](evaluation.md#the-gap-probe)).
- **Teacher and student must share a tokenizer.** The teacher scores the student's tokens one by one, so in
  practice a teacher is a fine-tuned member of the student's model family.

Only the Qwen2-VL and Qwen3-VL families have been run. A student of another family at least needs its own
`train.banned_tokens`: the default list names Qwen's image placeholder tokens, which must never be sampled.

## A task

A task is a YAML file in `configs/tasks/` and the data files it names. As an example, take multiple-choice
questions on ultrasound images, with SonoReasoner-8B (already in the model zoo) as the teacher.

**1. Write the data files** in the formats of [data.md](data.md#file-formats). The helpers of
`medresidency.jsonl` build the rows:

```python
from medresidency import jsonl

rows = [
    jsonl.user_row("<image>" + question, image, teacher_tag="sono", solution=f"<answer>{letter}</answer>")
    for question, image, letter in questions
]
jsonl.write("data/us_vqa/train.jsonl", rows)
```

`teacher_tag` routes a row to its teacher. `solution` is only read by rewards, so a task that will never be trained
with `grpo` can leave it out. Test rows carry `id`, `domain`, `answer` and `parser` in addition.

**2. Write the task file.** It holds only what differs from `configs/defaults.yaml`:

```yaml
# configs/tasks/us_vqa.yaml
description: >-
  Ultrasound VQA. A 2B student learns from SonoReasoner-8B on multiple-choice questions about ultrasound images.
student: {model: qwen3-vl-2b}
teachers: {sono: sonoreasoner-8b}
data:
  prompts: us_vqa/train.jsonl
  teacher_outputs: us_vqa/train_teacher.jsonl
train:
  max_completion: 1024
eval:
  validation: us_vqa/val.jsonl
  test:
    us_vqa: us_vqa/test.jsonl
generation:
  eval: {max_tokens: 2048, max_model_len: 4096}
  teacher_outputs: {max_tokens: 1024, max_model_len: 4096}
overrides:
  grpo: {method: {reward: mcq_accuracy}}
```

| Block | Read by | Holds |
|:--|:--|:--|
| `description` | `medres list` | what the task is, in two or three sentences |
| `student`, `teachers`, `data`, `train`, `teacher_server`, `hardware` | `medres train` | the run |
| `eval` | `medres eval run` | `validation`: one file; `test`: name → file; `test_from`: test sets of other tasks |
| `generation` | `medres eval generate --task` | response limit and context length for evaluation (`eval`) and for writing teacher outputs (`teacher_outputs`) |
| `overrides.<method>` | `medres train` | what one method needs on this task, e.g. the reward of `grpo` |

**3. Check it.** The task now appears in `medres list tasks`, and every method can be built on it:

```bash
medres train --task us_vqa --method opd --dry-run
```

**4. Let the teacher answer the training prompts**, for `offline_kd` and the methods with an SFT term:

```bash
medres eval generate --task us_vqa --profile teacher_outputs --model models/SonoReasoner-8B \
    --data data/us_vqa/train.jsonl --out gen/us_vqa.jsonl
medres data teacher-outputs --prompts data/us_vqa/train.jsonl --generations gen/us_vqa.jsonl \
    --out data/us_vqa/train_teacher.jsonl
```

**5. Measure the gap** of the pair, which tells you what to expect from plain OPD:

```bash
medres probe gap --data data/us_vqa/train.jsonl --student models/Qwen3-VL-2B-Instruct \
    --teacher models/SonoReasoner-8B --max-new-tokens 1024 --out probes/sono_qwen3vl2b
```

After that the task trains and evaluates like the built-in ones:

```bash
medres train --task us_vqa --method adaopd --seed 42
medres eval run --task us_vqa --model runs/us_vqa_adaopd_s42/checkpoint-300
```

Choose the budget with the comparison in mind. Everything in the task file is shared by all methods, so a task
tuned until one method looks good is no longer a fair task. Fix the budget before comparing methods, and select
whatever you do tune on the validation file.

## A method

First make sure you need one. A variant that is a single ms-swift argument is an override
(`train.swift_args.beta=0.04`), and a fixed set of overrides is a recipe. A method is the right tool when the
variant has keys of its own, more than one stage, or code that runs inside the trainer.

A method has two parts. The YAML file declares its keys and their defaults; the builder turns a resolved
configuration into training stages. The example warms the student up on the reference answers and then switches to
OPD.

```yaml
# configs/methods/label_warmup.yaml
description: >-
  Label warm-up: SFT on the reference answers for `sft_steps` steps, then OPD for the rest.
method:
  name: label_warmup
  sft_steps: 100
```

```python
# src/medresidency/methods.py
@register("label_warmup")
def label_warmup(run: Run) -> list[Stage]:
    first, total = run.method["sft_steps"], run.train["steps"]
    if not 0 < first < total:
        raise ConfigError(f"method.sft_steps={first} must lie between 0 and train.steps={total}")
    stage1 = run.sft("stage1", "labels", first, f"{run.out}/stage1")
    stage1.args["save_steps"] = first
    stage2 = run.opd("stage2", total - first, run.out, model=f"{stage1.output_dir}/checkpoint-{first}")
    return [stage1, stage2]
```

That is the whole method: `medres list methods` shows it, `method.sft_steps=50` overrides its key, and a misspelt
key is an error because the keys of the YAML file are the only valid ones.

```bash
medres train --task cxr_report --method label_warmup --dry-run
```

What a builder works with:

| | |
|:--|:--|
| `run.method`, `run.train`, `run.cfg` | the method's keys, the `train` section, the whole configuration |
| `run.out`, `run.teachers` | the run directory; tag → model of every teacher |
| `run.data(key)` | the path of `data.<key>`; an error with a clear message if the task does not set it |
| `run.sft(name, dataset, steps, out)` | a supervised stage on `data.<dataset>`, on all GPUs |
| `run.opd(name, steps, out, model=None)` | an OPD stage with teacher servers; `model` is the checkpoint to start from |
| `run.replay(decay_steps, peak, valley)` | the arguments that add an SFT term on teacher outputs, with weight μ going from `peak` to `valley` |

A `Stage` is the ms-swift sub-command (`sft` or `rlhf`), its arguments as a dictionary, and what the launcher has
to start around it (`teachers`, `reward_server`). A builder changes a stage by editing `stage.args`; every key
becomes a `--key value` of the training command. `stage.env` holds environment variables for the trainer.

Code that has to run inside the trainer goes into a plugin, a Python file that ms-swift imports at start
(`stage.args["external_plugins"]`). `plugins/adaopd_plugin.py` is the example: it reads its settings from
`MEDRES_ADAOPD_*` variables, which the `adaopd` builder puts into `stage.env`.

Two conventions:

- **Check what the method needs in the builder** and raise `ConfigError`. `--dry-run` then fails in a second, not
  after the teacher servers have loaded.
- **Record the arguments.** `tests/golden_args.json` holds the ms-swift arguments of every task × method, so that a
  change of the protocol never goes unnoticed. After adding a method, run
  `MEDRES_UPDATE_GOLDEN=1 pytest tests/test_methods.py` and commit the new entries with it.

Add a section to [methods.md](methods.md) that says what the method optimizes; the tests check that every method is
documented.

## A reward

A reward of `grpo` is a class that maps sampled responses and the `solution` field of their rows to numbers. This
is the whole of `answer_exact` in `src/medresidency/plugins/reward_plugin.py`:

```python
class AnswerExact(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return [float(_answer(c) == _answer(s)) for c, s in zip(completions, solution)]


orms["answer_exact"] = AnswerExact
```

To add one:

1. write the class in `reward_plugin.py` and register it in `orms` under its name;
2. add the name to `REWARDS` in `src/medresidency/methods.py`, with `False` if the class computes the reward
   itself;
3. select it with `method.reward=<name>`, or in the `overrides.grpo` block of a task.

Read answers with the functions of `medresidency.answers`. The evaluation uses the same ones, so the reward and
the test accuracy agree on what a correct answer is.

A reward that needs a model on a GPU should not load it inside the trainer, where every training process would
hold a copy. `report_radcliq` shows the alternative: `True` in `REWARDS` makes the launcher start the report-metric
server (`src/medresidency/eval/reward_server.py`) next to the training and pass its address in
`MEDRES_REWARD_URL`; the reward class only sends the texts and receives the scores.

GRPO normalizes rewards within the samples of one prompt. A reward may therefore have any scale; what counts is how
it ranks the samples of a prompt.

## A test set

A test set is a file and one line under `eval.test` of its task:

```yaml
eval:
  test:
    pmc_vqa: med_vqa/test_pmc_vqa.jsonl
    my_set: med_vqa/test_my_set.jsonl
```

`medres eval run` decides how to score a file from its rows: rows with a `reference` are reports, all others are
questions with an `answer` ([data.md](data.md#file-formats)). A task can also borrow test sets, which are then
generated with the settings of the task they belong to:

```yaml
eval:
  test_from:
    med_vqa: [pmc_vqa, medxpertqa_mm]
```

When you add a test set to a built-in task, say so wherever you report numbers next to those of the
[benchmark](../benchmarks/README.md): its tables are defined on the built-in sets.

## Before you open a pull request

```bash
ruff format . && ruff check . && pytest
```

The tests run on a CPU in under a minute; none of them needs a model or data. [CONTRIBUTING.md](../CONTRIBUTING.md)
has the rest.
