from __future__ import annotations

from pathlib import Path

import pytest

from fakes import write_video


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    return write_video(tmp_path / "source.mp4")
