import pytest


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """the tests read the repository's own configs, with default data and model locations"""
    for name in ("MEDRES_CONFIGS", "MEDRES_DATA", "MEDRES_MODELS", "MEDRES_METRICS_PYTHON", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
