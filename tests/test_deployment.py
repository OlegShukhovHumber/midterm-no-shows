"""Tests that the files the Space reads agree with each other."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import sklearn

ROOT = Path(__file__).parent.parent


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def readme_settings() -> dict[str, str]:
    """The key: value lines between the first two --- lines of the README."""
    block = read("README.md").split("---")[1]
    settings = {}
    for line in block.strip().splitlines():
        key, _, value = line.partition(":")
        settings[key.strip()] = value.strip().strip('"')
    return settings


def test_installed_sklearn_matches_model_info():
    info = json.loads(read("models/model_info.json"))
    assert sklearn.__version__ == info["sklearn_version"]


def test_installed_sklearn_matches_requirements_pin():
    pin = re.search(r"^scikit-learn==(\S+)$", read("requirements.txt"), re.M)
    assert pin is not None, "requirements.txt has no scikit-learn pin"
    assert sklearn.__version__ == pin.group(1)


def test_gradio_is_not_in_requirements():
    assert "gradio" not in read("requirements.txt").lower()


def test_readme_sdk_version_matches_installed_gradio():
    assert readme_settings()["sdk_version"] == version("gradio")


def test_app_file_named_in_readme_exists():
    assert (ROOT / readme_settings()["app_file"]).is_file()
