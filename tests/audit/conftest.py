"""The checkout fixture of the gate tests."""
from pathlib import Path

import pytest


from tests.audit.gate_fixture import Checkout


@pytest.fixture(params=["src/pkg", "src/other"])
def repo(tmp_path: Path, request) -> Checkout:
    checkout = tmp_path / "repository"
    checkout.mkdir()
    return Checkout(checkout, request.param)
