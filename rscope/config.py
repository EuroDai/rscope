"""Rscope configuration."""

from pathlib import Path

BASE_PATH = Path("/tmp/rscope/active_run")
TEMP_PATH = Path("/tmp/rscope/temp")
META_PATH = BASE_PATH / "rscope_meta.pkl"
REMOTE_BASE_PATH = BASE_PATH
REMOTE_META_PATH = REMOTE_BASE_PATH / "rscope_meta.pkl"


def set_base_path(path: str | Path):
  """Update all paths derived from the local rollout cache directory."""
  global BASE_PATH, TEMP_PATH, META_PATH
  BASE_PATH = Path(path)
  TEMP_PATH = BASE_PATH / ".tmp"
  META_PATH = BASE_PATH / "rscope_meta.pkl"


def set_remote_base_path(path: str | Path):
  """Update all paths derived from the remote rollout directory."""
  global REMOTE_BASE_PATH, REMOTE_META_PATH
  REMOTE_BASE_PATH = Path(path)
  REMOTE_META_PATH = REMOTE_BASE_PATH / "rscope_meta.pkl"
