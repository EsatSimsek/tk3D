from __future__ import annotations

import pytest

from src.video_io import ensure_output_tree, load_session


def test_session_rejects_duplicate_camera_ids(tmp_path) -> None:
    session_path = tmp_path / "session.yaml"
    session_path.write_text(
        """session_id: demo
cameras:
  - camera_id: cam
    video_path: one.mp4
  - camera_id: cam
    video_path: two.mp4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_session(session_path)


def test_output_tree_rejects_session_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError):
        ensure_output_tree(tmp_path, "../outside")
