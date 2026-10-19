"""Tests that the example notebooks run as expected."""

from pathlib import Path

import pytest
from testbook import testbook

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("notebook_path", _EXAMPLES_DIR.glob("*.ipynb"))
def test_examples(notebook_path):
    with testbook(notebook_path, execute=False, timeout=600) as tb:
        tb.execute()
