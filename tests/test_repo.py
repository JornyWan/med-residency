import importlib.util
import re
from pathlib import Path

import pytest

import medresidency

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def check_release():
    spec = importlib.util.spec_from_file_location("check_release", ROOT / "scripts" / "check_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_check_reports_what_must_not_be_published(check_release, tmp_path):
    home = "/".join(["", "home", "alice", "runs"])  # put together here, so that this file passes the check itself
    cases = [
        ("README.md", "nothing to report\n", None),
        ("data/train.txt", "a file in a data directory\n", "data/train.txt: in a directory of data or outputs"),
        ("notes/rows.jsonl", "{}\n", "notes/rows.jsonl: a .jsonl file"),
        ("notes/paths.md", f"first line\nsee {home}\n", "notes/paths.md:2: matches"),
        ("notes/host.md", "trained on node-17\n", "notes/host.md:1: matches 'node-"),
        ("notes/big.txt", "x" * (check_release.MAX_BYTES + 1), "notes/big.txt: larger than 1 MB"),
        ("notes/weights.dat", b"\xff\xfe\x00", "notes/weights.dat: not a text file"),
    ]
    for name, content, _ in cases:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())

    found = check_release.problems(tmp_path, [name for name, _, _ in cases], [r"node-\d+"])
    expected = [message for _, _, message in cases if message]
    assert len(found) == len(expected)
    assert all(line.startswith(message) for line, message in zip(found, expected))
    assert "alice" not in "\n".join(found)  # the matching text is never printed


def test_repository_passes_the_release_check(check_release):
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = check_release.repository_files(ROOT)
    assert check_release.problems(ROOT, files, check_release.blocklist(ROOT)) == []


def test_gitignore_covers_what_the_release_check_rejects(check_release):
    ignored = (ROOT / ".gitignore").read_text().split()
    assert all(f"/{name}/" in ignored for name in check_release.DIRS)
    assert all(f"*{suffix}" in ignored for suffix in check_release.SUFFIXES)


def test_one_version_everywhere():
    version = re.escape(medresidency.__version__)
    for name, line in (
        ("pyproject.toml", rf'^version = "{version}"$'),
        ("CITATION.cff", rf"^version: {version}$"),
        ("CHANGELOG.md", rf"^## {version}\b"),
    ):
        assert re.search(line, (ROOT / name).read_text(), re.M), name
