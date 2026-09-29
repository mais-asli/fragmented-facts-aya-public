from pathlib import Path
import pytest
from fragmented_facts.io import read_json
from fragmented_facts.verification import fixture_data, tiny_runner


@pytest.fixture(scope="session")
def facts():
    return fixture_data()


@pytest.fixture(scope="session")
def templates():
    return read_json(Path(__file__).resolve().parents[1] / "configs/templates.json")


@pytest.fixture(scope="session")
def runner(facts, templates):
    return tiny_runner(facts, templates)
