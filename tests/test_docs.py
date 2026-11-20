"""The documentation is checked against the code: commands, options, names, files and links."""

import re
import shlex
import shutil
from pathlib import Path

import pytest

from medresidency import cli, config, methods

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "benchmarks" / "README.md", *sorted(DOCS.glob("*.md"))]

# what the SLURM templates pass their arguments to
SBATCH = {"train": "medres train", "eval": "medres eval run", "teacher_outputs": "medres eval generate"}


def code_blocks(page: Path, language: str) -> list[str]:
    return re.findall(rf"^```{language}\n(.*?)^```", page.read_text(), re.M | re.S)


def prose(page: Path) -> str:
    return re.sub(r"^```.*?^```", "", page.read_text(), flags=re.M | re.S)


def commands() -> list[tuple[str, str]]:
    """(page, line) for every line of the bash blocks, with continuation lines joined."""
    found = []
    for page in PAGES:
        for block in code_blocks(page, "bash"):
            for line in block.replace("\\\n", " ").splitlines():
                line = re.sub(r"sbatch .*scripts/slurm/(\w+)\.sbatch", lambda m: SBATCH[m[1]], line)
                found.append((str(page.relative_to(ROOT)), " ".join(line.split())))
    return found


def anchors(page: Path) -> set[str]:
    """The anchors GitHub gives the headings of a page."""
    titles = re.findall(r"^#+ (.+)$", prose(page), re.M)
    return {re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-") for title in titles}


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A working directory with the configs, to which the examples of docs/extending.md are added as it says."""
    shutil.copytree(config.config_dir(), tmp_path / "configs")
    for block in code_blocks(DOCS / "extending.md", "yaml"):
        first = block.splitlines()[0]
        if first.startswith("# configs/"):
            path = tmp_path / first[2:]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(block)
    monkeypatch.setattr(methods, "METHODS", dict(methods.METHODS))
    for block in code_blocks(DOCS / "extending.md", "python"):
        if block.startswith("# src/medresidency/methods.py"):
            exec(block, dict(vars(methods)))
    monkeypatch.setenv("MEDRES_CONFIGS", str(tmp_path / "configs"))
    monkeypatch.chdir(tmp_path)


def test_train_commands_resolve(checkout):
    lines = [(page, line) for page, line in commands() if line.startswith("medres train")]
    assert lines
    for page, line in lines:
        argv = shlex.split(line.replace("$seed", "42"), comments=True)
        try:
            cli.main([*argv[1:], "--dry-run"])
        except SystemExit as e:
            pytest.fail(f"{page}: `{line}`: {e}")


def test_examples_of_the_extending_guide(checkout, capsys):
    cli.main(["train", "configs/recipes/mine/cxr_report_my_teacher.yaml", "--method", "opd", "--dry-run"])
    assert "My-CXR-8B" in capsys.readouterr().out
    label_warmup = ["train", "--task", "cxr_report", "--method", "label_warmup", "--dry-run"]
    cli.main([*label_warmup, "method.sft_steps=50"])
    assert "checkpoint-50" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown configuration keys: method.sft_step"):
        cli.main([*label_warmup, "method.sft_step=50"])


def test_tool_options_exist():
    package = Path(cli.__file__).parent
    shown = set()
    for page, line in commands():
        for group, tool, rest in re.findall(r"medres (data|eval|probe|stats) (\S+)([^#|&;]*)", line):
            assert tool in cli.TOOLS[group], f"{page}: `{line}`"
            source = (package / (cli.TOOLS[group][tool][0].replace(".", "/") + ".py")).read_text()
            for option in re.findall(r" (--[\w-]+)", rest):
                assert f'"{option}"' in source, f"{page}: `medres {group} {tool}` has no option {option}"
            shown.add((group, tool))
    # every tool appears in a command of the documentation
    assert shown == {(group, tool) for group, tools in cli.TOOLS.items() for tool in tools}


def test_every_task_and_method_is_documented():
    for kind, option in (("tasks", "--task"), ("methods", "--method")):
        page = (DOCS / f"{kind}.md").read_text()
        for name in config.available(kind):
            assert re.search(rf"{option} {name}\b", page), f"docs/{kind}.md has no command with `{option} {name}`"


def test_named_files_exist(checkout):
    for page in PAGES:
        text = re.sub(r"https?://\S+", "", page.read_text())
        for name in re.findall(r"\b(?:configs|scripts|src|tests|docs|benchmarks)/[\w./-]*", text):
            name = name.rstrip(".")
            # in the repository, or one of the files the extending guide tells the reader to write
            assert (ROOT / name).exists() or Path(name).exists(), f"{page.relative_to(ROOT)}: {name}"


def test_links_resolve():
    for page in PAGES:
        for link in re.findall(r"\]\(([^)\s]+)\)", prose(page)):
            if link.startswith("http"):
                continue
            path, _, anchor = link.partition("#")
            target = page.parent / path if path else page
            assert target.exists(), f"{page.relative_to(ROOT)}: {link}"
            if anchor:
                assert anchor in anchors(target), f"{page.relative_to(ROOT)}: {link}"
