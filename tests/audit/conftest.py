"""The checkout fixture of the gate tests."""
from pathlib import Path

import pytest


from tests.audit.gate_fixture import Checkout


@pytest.fixture
def repo(tmp_path: Path) -> Checkout:
    """A checkout whose sources live under a root other than the conventional
    `src/pkg`, so a measure that assumes that root counts nothing and fails."""
    checkout = tmp_path / "repository"
    checkout.mkdir()
    return Checkout(checkout, "src/other")
