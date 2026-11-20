"""Training methods.

A method turns a resolved configuration into one or two training stages: the ms-swift command, its arguments, and
what has to run next to it (teacher servers, a reward server). The launcher (launch.py) executes the stages.

All methods share one protocol, taken from the configuration: the same student, data, number of optimizer steps,
sequences per step, learning rate and response limit. They differ only in the loss.

To add a method, write `configs/methods/<name>.yaml` and register a builder here (docs/extending.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from medresidency import config
from medresidency.config import ConfigError

PLUGINS = Path(__file__).parent / "plugins"
TEACHER_SERVERS = "<teacher servers>"  # replaced by the launcher once the servers are up
REWARDS = {  # name (plugins/reward_plugin.py) -> needs the report-metric server
    "report_radcliq": True,
    "mixed_report_vqa": True,
    "mcq_accuracy": False,
    "answer_exact": False,
}
METHODS = {}


def register(name):
    def deco(fn):
        METHODS[name] = fn
        return fn

    return deco


@dataclass
class Stage:
    name: str
    command: str  # ms-swift sub-command: "sft" or "rlhf"
    args: dict  # ms-swift arguments
    n_gpus: int  # GPUs of the student
    teachers: bool = False  # needs the teacher servers
    teacher_logprobs: int = 1  # log-probabilities per position the teachers return
    reward_server: bool = False
    env: dict = field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return self.args["output_dir"]

    @property
    def steps(self) -> int:
        return self.args["max_steps"]


class Run:
    """A resolved configuration with the quantities every method derives from it."""

    def __init__(self, cfg: dict):
        self.cfg, self.train, self.method = cfg, cfg["train"], cfg["method"]
        self.seed = cfg["run"]["seed"]
        self.out = str(Path(cfg["run"]["output_root"]) / cfg["run"]["name"])
        self.student = config.model(cfg, cfg["student"]["model"])
        self.teachers = {tag: config.model(cfg, name) for tag, name in cfg["teachers"].items()}
        self.max_length = self.train["max_completion"] + self.train["prompt_budget"]

    def data(self, key: str) -> str:
        path = self.cfg["data"][key]
        if path is None:
            method, task = self.method["name"], self.cfg["task"]
            raise ConfigError(f"method {method} needs data.{key}, which task {task} does not set")
        return config.data_path(path)

    def n_gpus(self, with_teachers: bool) -> int:
        hw = self.cfg["hardware"]
        n = hw["gpus"] - hw["teacher_gpus"] if with_teachers else hw["gpus"]
        if n < 1:
            raise ConfigError(
                f"no GPU left for the student: hardware.gpus={hw['gpus']}, teacher_gpus={hw['teacher_gpus']}"
            )
        return n

    def base(self, dataset: str, steps: int, n_gpus: int, out: str, model: str | None = None) -> dict:
        """Arguments of every stage: full fine-tuning at a constant learning rate, `sequences_per_step` per update."""
        t = self.train
        acc, rest = divmod(t["sequences_per_step"], n_gpus * t["per_device_batch"])
        if rest or acc < 1:
            raise ConfigError(
                f"train.sequences_per_step={t['sequences_per_step']} is not a multiple of {n_gpus} student "
                f"GPUs x per_device_batch {t['per_device_batch']}"
            )
        return {
            "model": model or self.cfg["student"]["init"] or self.student["dir"],
            "model_type": self.student["model_type"],
            "template": self.student["template"],
            "dataset": self.data(dataset),
            "tuner_type": "full",
            "freeze_vit": t["freeze_vision"],
            "torch_dtype": "bfloat16",
            "attn_impl": "sdpa",
            "max_length": self.max_length,
            "per_device_train_batch_size": t["per_device_batch"],
            "gradient_accumulation_steps": acc,
            "learning_rate": t["lr"],
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0,
            "max_steps": steps,
            "save_steps": t["save_steps"],
            "save_only_model": False,  # saved with the optimizer state: an interrupted run can be resumed
            "deepspeed": "zero2",
            "seed": self.seed,
            "data_seed": self.seed,
            "dataloader_num_workers": 0,
            "dataset_num_proc": 4,
            "report_to": "tensorboard",
            "add_version": False,  # checkpoints in <run>/checkpoint-N, not in a time-stamped sub-directory
            "output_dir": out,
        }

    def rollout(self) -> dict:
        """The student samples its own responses with a vLLM engine on its training GPUs."""
        t = self.train
        return {
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": t["rollout_gpu_memory"],
            "vllm_max_model_len": self.max_length,
            "sleep_level": 1,
            "temperature": 1.0,
            "max_completion_length": t["max_completion"],
            "logging_steps": 1,
            "log_completions": t["log_completions"],
        }

    def sft(self, name: str, dataset: str, steps: int, out: str) -> Stage:
        n = self.n_gpus(with_teachers=False)
        return Stage(name, "sft", {**self.base(dataset, steps, n, out), "logging_steps": 5}, n)

    def opd(self, name: str, steps: int, out: str, model: str | None = None, **extra) -> Stage:
        """On-policy distillation: one sample per prompt, per-token advantage log p_teacher - log p_student."""
        if not self.teachers:
            raise ConfigError(f"method {self.method['name']} needs teachers, and task {self.cfg['task']} has none")
        n = self.n_gpus(with_teachers=True)
        args = {
            **self.base("prompts", steps, n, out, model),
            **self.rollout(),
            "rlhf_type": "grpo",  # ms-swift's GRPO trainer without a reward: only the teacher term is left
            "teacher_model_server": TEACHER_SERVERS,
            "teacher_tag_key": "teacher_tag",
            "teacher_kl_coef": 1.0,
            "beta": 0,
            "num_generations": 1,
            "loss_type": "grpo",
        }
        return Stage(name, "rlhf", args, n, teachers=True, **extra)

    def replay(self, decay_steps: int, peak: float, valley: float) -> dict:
        """Arguments of ms-swift's CHORD term: loss = (1 - mu) * OPD + mu * SFT on the teachers' outputs."""
        return {
            "chord_sft_dataset": self.data("teacher_outputs"),
            "chord_sft_per_device_train_batch_size": self.train["replay_per_device_batch"],
            "chord_mu_warmup_steps": 0,
            "chord_mu_decay_steps": decay_steps,
            "chord_mu_peak": peak,
            "chord_mu_valley": valley,
            # With the default (reentrant) checkpointing the second forward pass makes DeepSpeed ZeRO-2 fail with
            # "Gradient computed twice for this partition".
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }


def build(cfg: dict) -> list[Stage]:
    """The training stages of a run."""
    stages = METHODS[cfg["method"]["name"]](Run(cfg))
    for stage in stages:
        stage.args.update(cfg["train"]["swift_args"])  # the user's own ms-swift arguments win
        if stage.steps % stage.args["save_steps"]:
            raise ConfigError(
                f"stage {stage.name}: {stage.steps} steps is not a multiple of save_steps="
                f"{stage.args['save_steps']}, so its last checkpoint would not be the end of training"
            )
    return stages


@register("sft")
def sft(run: Run) -> list[Stage]:
    return [run.sft("train", "labels", run.train["steps"], run.out)]


@register("offline_kd")
def offline_kd(run: Run) -> list[Stage]:
    return [run.sft("train", "teacher_outputs", run.train["steps"], run.out)]


@register("opd")
def opd(run: Run) -> list[Stage]:
    stage = run.opd("train", run.train["steps"], run.out)
    if run.method["repetition_penalty"] is not None:
        stage.args["repetition_penalty"] = run.method["repetition_penalty"]
    if run.method["monitor_gap"]:
        stage.args["external_plugins"] = [str(PLUGINS / "adaopd_plugin.py")]
        stage.env["MEDRES_ADAOPD_MODE"] = "monitor"
    return [stage]


@register("gkd")
def gkd(run: Run) -> list[Stage]:
    m = run.method
    stage = run.opd("train", run.train["steps"], run.out, teacher_logprobs=m["topk"])
    for k in ("teacher_kl_coef", "num_generations", "loss_type"):
        del stage.args[k]
    stage.args.update({"rlhf_type": "gkd", "gkd_logits_topk": m["topk"], "lmbda": m["lmbda"], "beta": m["beta"]})
    if len(run.teachers) > 1:
        stage.args["remove_unused_columns"] = False  # keeps `teacher_tag` in the batch for routing
    return [stage]


@register("chord")
def chord(run: Run) -> list[Stage]:
    m = run.method
    stage = run.opd("train", run.train["steps"], run.out)
    stage.args.update(run.replay(m["decay_steps"] or run.train["steps"], m["peak"], m["valley"]))
    return [stage]


@register("chord_phi")
def chord_phi(run: Run) -> list[Stage]:
    m = run.method
    stage = run.opd("train", run.train["steps"], run.out)
    stage.args.update(run.replay(run.train["steps"], m["mu"], m["mu"]))
    stage.args["chord_enable_phi_function"] = True
    return [stage]


@register("two_stage")
def two_stage(run: Run) -> list[Stage]:
    m, total = run.method, run.train["steps"]
    first = m["sft_steps"]
    if not 0 < first < total:
        raise ConfigError(f"method.sft_steps={first} must lie between 0 and train.steps={total}")
    stages, ckpt = [], m["stage1_checkpoint"]
    if ckpt is None:
        stage1 = run.sft("stage1", "teacher_outputs", first, f"{run.out}/stage1")
        stage1.args["save_steps"] = first
        stages.append(stage1)
        ckpt = f"{stage1.output_dir}/checkpoint-{first}"
    return [*stages, run.opd("stage2", total - first, run.out, model=ckpt)]


@register("adaopd")
def adaopd(run: Run) -> list[Stage]:
    m = run.method
    if m["signal"] not in ("lang", "kl"):
        raise ConfigError(f"method.signal={m['signal']!r} must be lang or kl")
    unknown = sorted(set(m["measure_tags"]) - set(run.teachers))
    if unknown:
        raise ConfigError(f"method.measure_tags {unknown} are not teachers of this run ({', '.join(run.teachers)})")
    stage = run.opd("train", run.train["steps"], run.out)
    # the schedule arguments are those of CHORD; the plugin replaces the schedule by the measured mu
    stage.args.update(run.replay(run.train["steps"], 1.0, 0.0))
    stage.args["external_plugins"] = [str(PLUGINS / "adaopd_plugin.py")]
    stage.env.update(
        MEDRES_ADAOPD_MODE="control",
        MEDRES_ADAOPD_TAU=str(m["tau"]),
        MEDRES_ADAOPD_EMA=str(m["ema"]),
        MEDRES_ADAOPD_TAGS=",".join(m["measure_tags"]),
        MEDRES_ADAOPD_FRESH="1" if m["fresh_tokens"] else "0",
        MEDRES_ADAOPD_SIGNAL=m["signal"],
    )
    if m["gap0"] is not None:
        stage.env["MEDRES_ADAOPD_GAP0"] = str(m["gap0"])
    return [stage]


@register("grpo")
def grpo(run: Run) -> list[Stage]:
    m = run.method
    if m["reward"] not in REWARDS:
        raise ConfigError(f"method.reward={m['reward']!r}; available: {', '.join(REWARDS)}")
    n = run.n_gpus(with_teachers=False)
    args = {
        **run.base("prompts", run.train["steps"], n, run.out),
        **run.rollout(),
        "rlhf_type": "grpo",
        "external_plugins": [str(PLUGINS / "reward_plugin.py")],
        "reward_funcs": [m["reward"]],
        "beta": m["beta"],
        "num_generations": m["num_generations"],
    }
    return [Stage("train", "rlhf", args, n, reward_server=REWARDS[m["reward"]])]
