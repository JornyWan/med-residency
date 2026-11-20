import importlib
from pathlib import Path

import pytest

from medresidency import cli


def test_every_tool_exists():
    package = Path(cli.__file__).parent
    for tools in cli.TOOLS.values():
        for module, text in tools.values():
            assert (package / (module.replace(".", "/") + ".py")).is_file() and text
            if module not in cli.METRICS_ENV:  # those are run as files by the python of the metrics environment
                assert callable(importlib.import_module(f"medresidency.{module}").main)


def test_group_help(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["eval", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert all(name in out for name in cli.TOOLS["eval"])
    with pytest.raises(SystemExit) as e:
        cli.main(["data", "no-such-tool"])
    assert e.value.code == 2


def test_tool_help_needs_no_gpu_packages(capsys):
    for group, tools in cli.TOOLS.items():
        for name, (module, _) in tools.items():
            if module in cli.METRICS_ENV:
                continue
            with pytest.raises(SystemExit) as e:
                cli.main([group, name, "--help"])
            assert e.value.code == 0 and f"medres {group} {name}" in capsys.readouterr().out


def test_metrics_tools_need_their_environment():
    with pytest.raises(SystemExit, match="MEDRES_METRICS_PYTHON"):
        cli.main(["eval", "report-score", "--help"])
