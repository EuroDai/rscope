"""Render saved point-cloud observations in the MuJoCo user scene."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

_POINT_RADIUS_M = 0.002
_POINT_RGBA = (0.0, 0.8, 1.0, 0.9)
_TARGET_RADIUS_M = 0.03
_TARGET_RGBA = (1.0, 0.5, 0.0, 0.3)


def _palm_to_world(
    points: np.ndarray, position: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
  rotation = np.empty(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=np.float64))
  matrix = rotation.reshape(3, 3)
  return np.asarray(points) @ matrix.T + np.asarray(position)


def _style(
    visualization: Mapping[str, object] | None,
    name: str,
    default_radius: float,
    default_rgba: tuple[float, float, float, float],
) -> tuple[float, np.ndarray]:
  record = {} if visualization is None else visualization.get(name, {})
  if not isinstance(record, Mapping):
    raise ValueError(f'Visualization style {name!r} must be an object.')
  radius = float(record.get('radius_m', default_radius))
  rgba = np.asarray(record.get('rgba', default_rgba), dtype=np.float32)
  if not np.isfinite(radius) or radius <= 0.0:
    raise ValueError(f'Visualization radius for {name!r} must be positive.')
  if rgba.shape != (4,) or not np.isfinite(rgba).all():
    raise ValueError(f'Visualization RGBA for {name!r} must have 4 values.')
  return radius, rgba


def _add_sphere(
    scene: mujoco.MjvScene,
    index: int,
    position: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
  mujoco.mjv_initGeom(
      scene.geoms[index],
      mujoco.mjtGeom.mjGEOM_SPHERE,
      np.full(3, radius, dtype=np.float64),
      np.asarray(position, dtype=np.float64),
      np.eye(3, dtype=np.float64).reshape(-1),
      rgba,
  )


def update(
    scene: mujoco.MjvScene,
    observations: Mapping[str, np.ndarray],
    frame_index: int,
    *,
    visible: bool,
    visualization: Mapping[str, object] | None = None,
) -> int:
  """Replace user geoms with the selected frame's target and point cloud."""
  scene.ngeom = 0
  if not visible:
    return 0
  count = 0
  target = observations.get('target_position_w_m')
  if target is not None and scene.maxgeom > 0:
    target_position = np.asarray(target[frame_index])
    if target_position.shape != (3,) or not np.isfinite(target_position).all():
      raise ValueError('Replay target position must contain finite XYZ values.')
    target_radius, target_rgba = _style(
        visualization, 'target', _TARGET_RADIUS_M, _TARGET_RGBA
    )
    _add_sphere(
        scene, count, target_position, target_radius, target_rgba
    )
    count += 1

  if 'point_cloud_palm_m' not in observations:
    scene.ngeom = count
    return count
  points = observations['point_cloud_palm_m'][frame_index]
  mask = observations.get('point_mask')
  if mask is not None:
    points = points[np.asarray(mask[frame_index], dtype=bool)]
  if not len(points):
    scene.ngeom = count
    return count
  palm_position = observations['point_cloud_frame_position_w_m'][frame_index]
  palm_quaternion = observations[
      'point_cloud_frame_quaternion_wxyz'
  ][frame_index]
  points_w = _palm_to_world(points, palm_position, palm_quaternion)
  point_count = min(len(points_w), scene.maxgeom - count)
  point_radius, point_rgba = _style(
      visualization, 'point_cloud', _POINT_RADIUS_M, _POINT_RGBA
  )
  for offset, point in enumerate(points_w[:point_count]):
    _add_sphere(
        scene, count + offset, point, point_radius, point_rgba
    )
  scene.ngeom = count + point_count
  return scene.ngeom
