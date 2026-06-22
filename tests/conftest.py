"""
Shared pytest fixtures and configuration for the AquaCrop-OSPy test suite.

Two things are centralised here so they don't have to be repeated in every
test module:

1. ``DEVELOPMENT`` is set before aquacrop is imported, which disables the
   ahead-of-time compilation path. conftest.py is imported by pytest before any
   test module, so setting it here covers the whole suite.
2. The weather data is loaded once per session (the climate files are large),
   and exposed as fixtures.

Numerical outputs are compared with a tolerance rather than exact equality.
Different operating systems, Python versions, and numpy/BLAS builds produce
sub-ULP differences in floating-point reductions, which is what caused the
original exact-equality tests to fail intermittently across the CI matrix.
``MODEL_TOL`` absorbs that drift while still catching real regressions; use it
as ``value == pytest.approx(expected, **MODEL_TOL)``.
"""
import os
import sys

# Must be set before aquacrop is imported anywhere.
os.environ.setdefault("DEVELOPMENT", "True")

# aquacrop's __init__ only binds AquaCropModel/Soil/Crop/etc. when "-m" is NOT
# in sys.argv. Running "pytest -m '<marker>'" puts "-m" in sys.argv and would
# leave aquacrop half-imported. Import it once here with a sanitized argv so the
# names are bound and cached before any test module needs them.
_orig_argv = sys.argv
try:
    sys.argv = [sys.argv[0]]
    import aquacrop  # noqa: F401
finally:
    sys.argv = _orig_argv

import pytest

from aquacrop import Soil
from aquacrop.utils import prepare_weather, get_filepath

# Relative tolerance of 0.1% with a small absolute floor for near-zero values.
MODEL_TOL = {"rel": 1e-3, "abs": 1e-3}


@pytest.fixture(scope="session")
def tunis_weather():
    """Tunis climate series (used by most Wheat scenarios)."""
    return prepare_weather(get_filepath("tunis_climate.txt"))


@pytest.fixture(scope="session")
def champion_weather():
    """Champion climate series (used by the Maize irrigation scenarios)."""
    return prepare_weather(get_filepath("champion_climate.txt"))


@pytest.fixture
def sandy_loam():
    """A fresh SandyLoam soil per test (the model can mutate soil state)."""
    return Soil(soil_type="SandyLoam")
