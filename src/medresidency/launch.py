"""Run a configuration: for every stage of its method, start what the stage needs and train with ms-swift.

One run uses `hardware.gpus` cards of one node (the first ones of CUDA_VISIBLE_DEVICES when it is set):

  - a stage without teachers trains on all of them;
  - a stage with teachers trains the student on the first `gpus - teacher_gpus` cards and serves the teachers on the
    last `teacher_gpus` cards: a single teacher takes all of them with tensor parallelism, several teachers are spread
    over them with one card each;
  - the report-metric server of a reward shares the first card with the student.

Everything the launcher starts is stopped when the stage ends, fails or is interrupted. What was run is recorded in
`<run>/medres/`: the resolved configuration, the commands, and one line per launch with the code and package versions.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from importlib import metadata
from pathlib import Path

import yaml

from medresidency import config, methods

SWIFT = [sys.executable, "-m", "swift.cli.main"]  # the `swift` command of this environment, whatever PATH says
ENV_DEFAULTS = {  # only set when the caller's environment does not
    "HF_HUB_OFFLINE": "1",  # models and data are local files
    "VLLM_NO_USAGE_STATS": "1",
    # FlashInfer's sampler compiles with nvcc at first use, and compute nodes rarely have it
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}
VERSIONS = ("medresidency", "ms-swift", "vllm", "torch", "transformers", "trl", "deepspeed")
READY = re.compile(r"Uvicorn running on http://127\.0\.0\.1:(\d+)")
PORT = "<port>"


class LaunchError(RuntimeError):
    pass


def to_argv(args: dict) -> list[str]:
    """Turn a dictionary of ms-swift arguments into command-line arguments."""
    out = []
    for k, v in args.items():
        if v is None:
            continue
        out.append(f"--{k}")
        if isinstance(v, bool):
            out.append("true" if v else "false")
        elif isinstance(v, (list, tuple)):
            out.extend(str(x) for x in v)
        elif isinstance(v, dict):
            out.append(json.dumps(v))
        else:
            out.append(str(v))
    return out


def gpu_ids(cfg: dict) -> list[str]:
    n = cfg["hardware"]["gpus"]
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    ids = visible or [str(i) for i in range(n)]
    if len(ids) < n:
        raise LaunchError(f"hardware.gpus={n}, but CUDA_VISIBLE_DEVICES lists {len(ids)} cards")
    return ids[:n]


def teacher_plan(cfg: dict, gpus: list[str]) -> list[dict]:
    """The model and the cards of every teacher tag."""
    cards = gpus[len(gpus) - cfg["hardware"]["teacher_gpus"] :]
    if not cards:
        raise LaunchError("hardware.teacher_gpus=0: the teachers of this method need at least one card")
    tags = list(cfg["teachers"])
    return [
        {
            "tag": tag,
            "model": config.model(cfg, cfg["teachers"][tag]),
            "gpus": cards if len(tags) == 1 else [cards[i * len(cards) // len(tags)]],
        }
        for i, tag in enumerate(tags)
    ]


def teacher_command(cfg: dict, stage: methods.Stage, teacher: dict, port) -> list[str]:
    ts, m = cfg["teacher_server"], teacher["model"]
    args = {
        "model": m["dir"],
        "model_type": m["model_type"],
        "template": m["template"],
        "infer_backend": "vllm",
        "host": "127.0.0.1",
        "port": port,
        "max_logprobs": stage.teacher_logprobs,
        "vllm_tensor_parallel_size": len(teacher["gpus"]),
        "vllm_gpu_memory_utilization": ts["gpu_memory"],
        "vllm_max_model_len": stage.args["max_length"],
        "vllm_enforce_eager": True,
        "vllm_max_num_seqs": ts["max_num_seqs"],
        # The log-probabilities of a scored response are a [tokens x vocabulary] float32 tensor outside vLLM's
        # memory budget: the cap on batched tokens is what keeps the card from running out of memory.
        "vllm_engine_kwargs": {"max_num_batched_tokens": ts["max_batched_tokens"]},
        "torch_dtype": "bfloat16",
    }
    return [*SWIFT, "deploy", *to_argv(args)]


def reward_command(port) -> list[str]:
    python = os.environ.get("MEDRES_METRICS_PYTHON")
    if not python:
        raise LaunchError(
            "this reward is scored by the report-metric server: set MEDRES_METRICS_PYTHON to the python "
            "of the metrics environment (scripts/install_metrics_env.sh)"
        )
    return [python, "-I", str(Path(__file__).parent / "eval" / "reward_server.py"), "--port", str(port)]


def shell(env: dict, cmd: list[str]) -> str:
    """A command as shell text, one option per line."""
    lines, cur = [], [f"{k}={shlex.quote(str(v))}" for k, v in env.items()]
    for tok in cmd:
        if tok.startswith("--") and cur:
            lines.append(" ".join(cur))
            cur = []
        cur.append(tok if tok == PORT else shlex.quote(tok))
    lines.append(" ".join(cur))
    return " \\\n    ".join(lines)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def last_checkpoint(output_dir) -> int:
    steps = [p.name.split("-", 1)[1] for p in Path(output_dir).glob("checkpoint-*")]
    return max((int(s) for s in steps if s.isdigit()), default=0)


def _get(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode()
    except OSError:
        return None


class Server:
    """A background process in its own process group, with its output in a log file."""

    def __init__(self, name: str, cmd: list[str], env: dict, log: Path):
        self.name, self.log = name, log
        if log.exists():  # keep the log of an earlier attempt
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(log.stat().st_mtime))
            log.rename(log.with_name(f"{log.stem}.{stamp}.log"))
        self.file = open(log, "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.file, stderr=subprocess.STDOUT, start_new_session=True)

    def check(self) -> None:
        if self.proc.poll() is not None:
            raise LaunchError(f"{self.name} exited with code {self.proc.returncode}; its log: {self.log}")

    def wait_ready(self, probe, timeout: float):
        """Poll `probe()` until it returns something; fail when the process exits or the time is over."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.check()
            result = probe()
            if result:
                return result
            time.sleep(5)
        raise LaunchError(f"{self.name} was not ready after {timeout:.0f} s; its log: {self.log}")

    def stop(self) -> None:
        # the whole group: `swift deploy` is a wrapper around the server process, and an orphaned server keeps its port
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(20)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()
        self.file.close()


def _teacher_url(server: Server, model_dir: str):
    """The address the server listens on, once it answers and serves the right model.

    The address is read from the log: ms-swift moves to the next port when the requested one is taken."""
    ports = READY.findall(server.log.read_text(errors="replace"))
    if not ports:
        return None
    url = f"http://127.0.0.1:{ports[-1]}"
    body = _get(f"{url}/v1/models")
    if body is None:
        return None
    served = [m.get("id") for m in json.loads(body).get("data", [])]
    if Path(model_dir).name not in served:
        raise LaunchError(f"{server.name}: {url} serves {served}, not {Path(model_dir).name}")
    return url


def environment(cfg: dict, cache: str) -> dict:
    """What the launcher adds to the caller's environment (variables the caller has set are kept)."""
    add = {
        **ENV_DEFAULTS,
        # vision placeholder tokens a student may not sample (scripts/install_train_env.sh patches ms-swift to read it)
        "MEDRES_BAD_WORDS": ",".join(cfg["train"]["banned_tokens"]),
        # ms-swift caches preprocessed datasets under one name: concurrent runs sharing the cache crash each other
        "MODELSCOPE_CACHE": f"{cache}/modelscope",
        "HF_DATASETS_CACHE": f"{cache}/hf_datasets",
    }
    return {k: v for k, v in add.items() if k not in os.environ}


def code_version() -> str | None:
    """The commit of this clone, or None when git is not installed or the package is not in a clone."""
    try:
        git = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
        )
    except OSError:  # compute nodes often have no git
        return None
    return git.stdout.strip() or None


def _record(stage: methods.Stage, gpus: list[str], resumed_from: int) -> dict:
    versions = {}
    for package in VERSIONS:
        with contextlib.suppress(metadata.PackageNotFoundError):
            versions[package] = metadata.version(package)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": stage.name,
        "resumed_from_step": resumed_from,
        "host": socket.gethostname(),
        "gpus": gpus,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "code": code_version(),
        "versions": versions,
    }


def run_stage(cfg: dict, stage: methods.Stage, gpus: list[str], env_add: dict, meta: Path | None, resumed_from=0):
    """Run one training stage; with `meta=None` only print its commands."""
    dry = meta is None
    args, env_stage = dict(stage.args), dict(stage.env)
    student = gpus[: stage.n_gpus]
    if resumed_from:
        args["resume_from_checkpoint"] = f"{stage.output_dir}/checkpoint-{resumed_from}"
        env_stage["MEDRES_RESUME_CHECKPOINT"] = args["resume_from_checkpoint"]
    base_env = {**os.environ, **env_add}
    timeout = cfg["teacher_server"]["startup_timeout"]
    text = [f"# stage {stage.name}: {stage.steps} steps, student on GPU {','.join(student)}"]
    servers: list[Server] = []

    def start(name, cmd, cards):
        env = {"CUDA_VISIBLE_DEVICES": ",".join(cards)}
        text.append(f"# {name}\n{shell(env, cmd)} &")
        if not dry:
            servers.append(Server(name, cmd, {**base_env, **env}, meta / f"{name.replace(' ', '_')}.log"))
            return servers[-1]

    try:
        if stage.teachers:
            routes = []
            # one at a time: two servers that load at once can take each other's port
            for t in teacher_plan(cfg, gpus):
                cmd = teacher_command(cfg, stage, t, PORT if dry else free_port())
                server = start(f"teacher {t['tag']}", cmd, t["gpus"])
                if dry:
                    url = f"http://127.0.0.1:{PORT}"
                else:
                    model_dir = t["model"]["dir"]
                    url = server.wait_ready(lambda s=server, d=model_dir: _teacher_url(s, d), timeout)
                routes.append({"url": url, "tags": [t["tag"]]})
            args["teacher_model_server"] = json.dumps(routes)
        if stage.reward_server:
            port = PORT if dry else free_port()
            server = start("reward server", reward_command(port), student[:1])
            env_stage["MEDRES_REWARD_URL"] = f"http://127.0.0.1:{port}"
            if not dry:
                server.wait_ready(lambda: _get(f"http://127.0.0.1:{port}/health"), timeout)
        for server in servers:  # a teacher may have died while a later one was loading
            server.check()

        env_stage["CUDA_VISIBLE_DEVICES"] = ",".join(student)
        # With NPROC_PER_NODE ms-swift starts one process per card with torch.distributed.run. GKD needs that on a
        # single card as well: the vLLM engine of its rollouts reads the rank variables of torchrun, and ms-swift sets
        # them itself only in its GRPO trainer.
        if stage.n_gpus > 1 or args.get("rlhf_type") == "gkd":
            env_stage.update(NPROC_PER_NODE=str(stage.n_gpus), MASTER_PORT=PORT if dry else str(free_port()))
        cmd = [*SWIFT, stage.command, *to_argv(args)]
        text.append(f"# trainer\n{shell(env_stage, cmd)}")
        if dry:
            print("\n".join(text) + "\n")
            return
        with open(meta / "commands.sh", "a") as f:
            f.write("\n".join(text) + "\n\n")
        with open(meta / "launches.jsonl", "a") as f:
            f.write(json.dumps(_record(stage, gpus, resumed_from)) + "\n")
        print(f"[medres] stage {stage.name}: training in {stage.output_dir} (commands: {meta}/commands.sh)", flush=True)
        trainer = subprocess.Popen(cmd, env={**base_env, **env_stage})
        try:
            while True:
                try:
                    code = trainer.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    for server in servers:
                        server.check()
        finally:
            if trainer.poll() is None:
                trainer.terminate()
                try:
                    trainer.wait(60)
                except subprocess.TimeoutExpired:
                    trainer.kill()
        if code:
            raise LaunchError(f"stage {stage.name}: ms-swift exited with code {code}")
    finally:
        for server in servers:
            server.stop()


def _exports(env: dict) -> str:
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())


def run(cfg: dict, dry_run: bool = False) -> None:
    stages = methods.build(cfg)
    gpus = gpu_ids(cfg)
    if dry_run:
        env_add = environment(cfg, "<per-run cache>")
        print(_exports(env_add))
        for stage in stages:
            run_stage(cfg, stage, gpus, env_add, None)
        return

    todo = []
    for stage in stages:
        done = last_checkpoint(stage.output_dir)
        if done and not cfg["run"]["resume"]:
            raise LaunchError(
                f"{stage.output_dir} already holds checkpoint-{done}: choose another run.name, or continue "
                "this run with run.resume=true"
            )
        if done < stage.steps:
            todo.append((stage, done))
    if not todo:
        print("[medres] nothing to do: every stage has its last checkpoint")
        return
    meta = Path(cfg["run"]["output_root"]) / cfg["run"]["name"] / "medres"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def terminate(signum, frame):  # SLURM cancels with SIGTERM: leave through the `finally` blocks
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    cache = tempfile.mkdtemp(prefix="medres_")
    try:
        env_add = environment(cfg, cache)
        with open(meta / "commands.sh", "a") as f:
            f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}\n{_exports(env_add)}\n")
        for stage, done in todo:
            run_stage(cfg, stage, gpus, env_add, meta, done)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
