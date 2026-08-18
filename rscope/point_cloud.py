"""Render saved point-cloud observations in the MuJoCo user scene."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

_POINT_SIZE_M = 0.002
_POINT_RGBA = np.asarray((0.1, 0.85, 0.95, 0.9), dtype=np.float32)


def _palm_to_world(
    points: np.ndarray, position: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
  rotation = np.empty(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=np.float64))
  matrix = rotation.reshape(3, 3)
  return np.asarray(points) @ matrix.T + np.asarray(position)


def update(
    scene: mujoco.MjvScene,
    observations: Mapping[str, np.ndarray],
    frame_index: int,
    *,
    visible: bool,
) -> int:
  """Replace user geoms with the selected frame's valid point cloud."""
  scene.ngeom = 0
  if not visible or 'point_cloud_palm_m' not in observations:
    return 0
  points = observations['point_cloud_palm_m'][frame_index]
  mask = observations.get('point_mask')
  if mask is not None:
    points = points[np.asarray(mask[frame_index], dtype=bool)]
  if not len(points):
    return 0
  palm_position = observations['point_cloud_frame_position_w_m'][frame_index]
  palm_quaternion = observations[
      'point_cloud_frame_quaternion_wxyz'
  ][frame_index]
  points_w = _palm_to_world(points, palm_position, palm_quaternion)
  count = min(len(points_w), scene.maxgeom)
  size = np.full(3, _POINT_SIZE_M, dtype=np.float64)
  matrix = np.eye(3, dtype=np.float64).reshape(-1)
  for index, point in enumerate(points_w[:count]):
    mujoco.mjv_initGeom(
        scene.geoms[index],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        np.asarray(point, dtype=np.float64),
        matrix,
        _POINT_RGBA,
    )
  scene.ngeom = count
  return count
