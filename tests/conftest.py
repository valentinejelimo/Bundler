"""Shared pytest configuration for the gdpval-qc test suite.

Adds the templates/harbor directory to sys.path so that the bundled
verifier_lib (a separate dependency world) can be imported by tests
without touching the project's packaging.
"""
import sys
from pathlib import Path

HARBOR = Path(__file__).resolve().parents[1] / "templates" / "harbor"
if str(HARBOR) not in sys.path:
    sys.path.insert(0, str(HARBOR))
