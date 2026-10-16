"""Fixtures wrapping the synthetic decompositions in :mod:`ocs.demo`.

The builders live in the package rather than here because
``scripts/make_demo_project.py`` needs them too -- it is how the editor gets
driven on a machine with no CUDA. See that module's docstring for why the
layout is shaped the way it is.
"""

from __future__ import annotations

import pytest

from ocs import demo
from ocs.demo import CANVAS, part, solid, two_blobs  # noqa: F401  (used by tests)
from ocs.psd_io import Decomposition


@pytest.fixture
def figure() -> Decomposition:
    """Clean figure, no junk layers."""
    return demo.figure()


@pytest.fixture
def figure_with_junk() -> Decomposition:
    """Same figure plus the dummy layers see-through leaves in unfiltered."""
    return demo.figure_with_junk()


@pytest.fixture
def blob_figure() -> Decomposition:
    """Worst case for requirement 2-2: one connected silhouette, one layer."""
    return demo.blob_figure()
