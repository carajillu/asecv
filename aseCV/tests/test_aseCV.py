"""
Unit and regression test for the aseCV package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import aseCV


def test_aseCV_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "aseCV" in sys.modules
