"""Loader for versioned, non-pickle MuJoCo replay bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

SCHEMA_NAME = 'mujoco-replay-bundle'
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding='utf-8'))
  if not isinstance(value, dict):
    raise TypeError(f'Replay manifest must be an object: {path}')
  return value


def _verified_path(root: Path, record: dict[str, Any]) -> Path:
  root = root.resolve()
  path = (root / record['path']).resolve()
  if root != path and root not in path.parents:
    raise ValueError(f'Replay path escapes its bundle root: {record["path"]}')
  if not path.is_file():
    raise FileNotFoundError(path)
  expected = record.get('sha256')
  if expected and _sha256(path) != expected:
    raise ValueError(f'Replay file SHA256 mismatch: {path}')
  return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    return {name: np.asarray(data[name]) for name in data.files}


def _pad(array: np.ndarray, length: int) -> np.ndarray:
  if len(array) == length:
    return array
  if not len(array):
    return np.zeros((length,) + array.shape[1:], dtype=array.dtype)
  padding = np.repeat(array[-1:], length - len(array), axis=0)
  return np.concatenate((array, padding), axis=0)


def _stack_policy_arrays(
    values: list[dict[str, np.ndarray]], name: str, length: int
) -> np.ndarray:
  present = [value[name] for value in values if name in value]
  if not present:
    return np.empty((length, len(values), 0), dtype=np.float32)
  shape = present[0].shape[1:]
  dtype = present[0].dtype
  columns = []
  for value in values:
    array = value.get(name)
    if array is None:
      array = np.zeros((0,) + shape, dtype=dtype)
    if array.shape[1:] != shape:
      raise ValueError(f'Inconsistent replay array shape for {name}.')
    columns.append(_pad(array, length))
  return np.stack(columns, axis=1)


def _metric_labels(
    name: str, width: int, metadata: dict[str, Any]
) -> list[str]:
  if name.startswith('fingertip_') and width == len(
      metadata.get('fingertip_names', ())
  ):
    return [f'{name}/{label}' for label in metadata['fingertip_names']]
  if name.startswith('robot_joint_') and width == len(
      metadata.get('robot_joint_names', ())
  ):
    return [f'{name}/{label}' for label in metadata['robot_joint_names']]
  return [f'{name}/{index}' for index in range(width)]


def _scalar_metrics(
    value: dict[str, np.ndarray], metadata: dict[str, Any]
) -> dict[str, np.ndarray]:
  result: dict[str, np.ndarray] = {}
  for name, array in value.items():
    if array.ndim == 1:
      result[name] = array
    elif array.ndim == 2:
      labels = _metric_labels(name, array.shape[1], metadata)
      result.update(
          {label: array[:, index] for index, label in enumerate(labels)}
      )
  return result


def _combine_metrics(
    values: list[dict[str, np.ndarray]],
    metadata: dict[str, Any],
    length: int,
) -> dict[str, np.ndarray]:
  scalar = [_scalar_metrics(value, metadata) for value in values]
  keys = sorted(set().union(*(value.keys() for value in scalar)))
  result = {}
  for key in keys:
    columns = []
    for value in scalar:
      array = value.get(key)
      if array is None:
        array = np.full(0, np.nan, dtype=np.float32)
      columns.append(_pad(array, length))
    result[key] = np.stack(columns, axis=1)
  return result


def _combine_observations(
    values: list[dict[str, np.ndarray]], length: int
) -> dict[str, np.ndarray]:
  keys = sorted(set().union(*(value.keys() for value in values)))
  return {
      key: _stack_policy_arrays(values, key, length)
      for key in keys
  }


def _validate_version(manifest: dict[str, Any]) -> None:
  if manifest.get('schema') != SCHEMA_NAME:
    raise ValueError(f"Unsupported replay schema: {manifest.get('schema')}")
  if manifest.get('schema_version') != SCHEMA_VERSION:
    raise ValueError(
        f"Unsupported replay schema version: {manifest.get('schema_version')}"
    )
  producer_version = str(manifest['model']['mujoco_version'])
  producer_minor = producer_version.split('.')[:2]
  viewer_minor = mujoco.__version__.split('.')[:2]
  if producer_minor != viewer_minor:
    raise ValueError(
        f'MuJoCo version mismatch: bundle {producer_version}, '
        f'viewer {mujoco.__version__}.'
    )


def load_bundle(base_path: str | Path) -> dict[str, Any]:
  """Load a replay bundle into existing rscope rollout-shaped arrays."""
  root = Path(base_path)
  manifest = _read_json(root / 'manifest.json')
  _validate_version(manifest)
  model_path = _verified_path(root, manifest['model'])
  loaded = []
  for example_record in manifest['examples']:
    example_manifest_path = root / example_record['path']
    example = _read_json(example_manifest_path)
    if (
        example.get('schema') != SCHEMA_NAME
        or example.get('schema_version') != SCHEMA_VERSION
    ):
      raise ValueError(f'Invalid example schema: {example_manifest_path}')
    example_root = example_manifest_path.parent
    model_fields = _load_npz(
        _verified_path(example_root, example['model_fields'])
    )
    policies = []
    for policy_name in ('student', 'teacher'):
      policy_record = example['policies'][policy_name]
      policy_root = example_root / policy_name
      states = _load_npz(_verified_path(policy_root, policy_record['state']))
      metrics = _load_npz(_verified_path(policy_root, policy_record['metrics']))
      observations = {}
      if 'observations' in policy_record:
        observations = _load_npz(
            _verified_path(policy_root, policy_record['observations'])
        )
      policies.append((states, metrics, observations, policy_record))
    lengths = np.asarray(
        [len(policy[0]['qpos']) for policy in policies], dtype=np.int64
    )
    max_length = int(lengths.max())
    states = [policy[0] for policy in policies]
    metric_values = [policy[1] for policy in policies]
    observation_values = [policy[2] for policy in policies]
    metadata = example['episode']
    qpos = _stack_policy_arrays(states, 'qpos', max_length)
    qvel = _stack_policy_arrays(states, 'qvel', max_length)
    if qpos.shape[-1] != int(manifest['model']['nq']):
      raise ValueError('Replay qpos width does not match its MuJoCo model.')
    if qvel.shape[-1] != int(manifest['model']['nv']):
      raise ValueError('Replay qvel width does not match its MuJoCo model.')
    mocap_pos = _stack_policy_arrays(states, 'mocap_pos', max_length)
    mocap_quat = _stack_policy_arrays(states, 'mocap_quat', max_length)
    time = _stack_policy_arrays(states, 'time_s', max_length)
    if time.ndim == 3 and time.shape[-1] == 1:
      time = time[..., 0]
    if time.ndim != 2:
      raise ValueError('Replay time must be scalar per frame and policy.')
    observations = _combine_observations(observation_values, max_length)
    for source_name, observation_name in (
        ('palm_position_w_m', 'point_cloud_frame_position_w_m'),
        ('palm_quaternion_wxyz', 'point_cloud_frame_quaternion_wxyz'),
        ('object_position_w_m', 'replay_focus_position_w_m'),
    ):
      if any(source_name in value for value in metric_values):
        observations[observation_name] = _stack_policy_arrays(
            metric_values, source_name, max_length
        )
    loaded.append(
        {
            'name': example['name'],
            'qpos': qpos,
            'qvel': qvel,
            'mocap_pos': mocap_pos,
            'mocap_quat': mocap_quat,
            'obs': observations,
            'reward': np.zeros((max_length, 2), dtype=np.float32),
            'time': time,
            'metrics': _combine_metrics(metric_values, metadata, max_length),
            'lengths': lengths,
            'env_labels': ('Student', 'Teacher'),
            'model_fields': model_fields,
            'metadata': {
                **metadata,
                'selection': example['selection'],
            },
        }
    )
  return {'manifest': manifest, 'model_path': model_path, 'rollouts': loaded}


def apply_model_fields(
    model: mujoco.MjModel, fields: dict[str, np.ndarray]
) -> None:
  """Apply one episode's generic host-model patch before playback."""
  for name, source in fields.items():
    if not hasattr(model, name):
      raise ValueError(f'MuJoCo model has no replay field {name!r}.')
    target = getattr(model, name)
    if source.size != target.size:
      raise ValueError(
          f'Replay model field {name!r} has shape {source.shape}, '
          f'expected {target.shape}.'
      )
    target[:] = source.reshape(target.shape)
  none = mujoco.mjtSameFrame.mjSAMEFRAME_NONE.value
  model.body_sameframe[:] = none
  model.geom_sameframe[:] = none
  model.site_sameframe[:] = none
  model.body_simple[:] = 0
