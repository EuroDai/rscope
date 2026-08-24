"""Rollout utilities."""

from bisect import bisect_right
from pathlib import Path
import pickle
from typing import Dict, Hashable, List, Mapping, NamedTuple, Union

from absl import logging
import numpy as np
from numpy.typing import NDArray

MAX_VIEWPORTS = 12


class Rollout(NamedTuple):
  qpos: NDArray
  qvel: NDArray
  mocap_pos: NDArray
  mocap_quat: NDArray
  obs: NDArray
  reward: NDArray
  time: NDArray
  metrics: NDArray


# Global rollout state.
rollouts: List[Rollout] = []
rollout_names: List[str] = []
rollout_lengths: List[NDArray] = []
rollout_env_labels: List[tuple[str, ...]] = []
rollout_model_fields: List[Dict] = []
rollout_metadata: List[Dict] = []
bundle_model_path: Path | None = None
bundle_manifest: Dict | None = None
bundle_object_groups: List[tuple[int, ...]] = []
bundle_replay_positions: Dict[int, tuple[int, int]] = {}
num_evals = 0
num_envs = 0
env_ctrl_dt = 0.0
change_rollout = False


def get_num_evals():
  return num_evals


def _update_metadata(rollout: Rollout):
  global num_evals, num_envs, env_ctrl_dt, change_rollout
  num_envs = rollout.qpos.shape[1]
  num_evals = len(rollouts)
  # TODO: this is wrong when it resets on timestep 1.
  env_ctrl_dt = rollout.time[1, 0] - rollout.time[0, 0]
  if len(rollouts) == 1:
    change_rollout = True


def append_unroll(fpath: Union[str, Path]):
  """Load an unroll file and insert it into the list in filename order."""
  logging.info(f'Loading unroll: {fpath}')
  fpath = Path(fpath)
  fname = fpath.name
  if fname in rollout_names:
    return

  with open(fpath, 'rb') as f:
    rollout = pickle.load(f)

  assert rollout.qpos.shape[:2] == rollout.qvel.shape[:2], (
      'qpos and qvel non-matching time or envs dimension:'
      f' {rollout.qpos.shape} vs {rollout.qvel.shape}'
  )

  insert_idx = bisect_right(rollout_names, fname)
  rollout_names.insert(insert_idx, fname)
  rollouts.insert(insert_idx, rollout)
  rollout_lengths.insert(
      insert_idx,
      np.full(rollout.qpos.shape[1], rollout.qpos.shape[0], dtype=int),
  )
  rollout_env_labels.insert(
      insert_idx,
      tuple(f'Env {index + 1}' for index in range(rollout.qpos.shape[1])),
  )
  rollout_model_fields.insert(insert_idx, {})
  rollout_metadata.insert(insert_idx, {})
  _update_metadata(rollout)


def load_all_local_unrolls(base_path: Union[str, Path]) -> List[str]:
  """Load all existing unroll files from base_path into rollouts."""
  global bundle_model_path, bundle_manifest
  base = Path(base_path)
  manifest_path = base / 'manifest.json'
  if manifest_path.is_file():
    from rscope.replay_bundle import load_bundle

    bundle = load_bundle(base)
    bundle_model_path = bundle['model_path']
    bundle_manifest = bundle['manifest']
    for item in bundle['rollouts']:
      value = Rollout(
          qpos=item['qpos'],
          qvel=item['qvel'],
          mocap_pos=item['mocap_pos'],
          mocap_quat=item['mocap_quat'],
          obs=item['obs'],
          reward=item['reward'],
          time=item['time'],
          metrics=item['metrics'],
      )
      rollouts.append(value)
      rollout_names.append(item['name'])
      rollout_lengths.append(item['lengths'])
      rollout_env_labels.append(item['env_labels'])
      rollout_model_fields.append(item['model_fields'])
      rollout_metadata.append(item['metadata'])
      _update_metadata(value)
    _rebuild_bundle_navigation()
    return rollout_names.copy()
  unroll_files = sorted(
      f.name for f in base.iterdir() if f.name.endswith('.mj_unroll')
  )
  for fname in unroll_files:
    append_unroll(base / fname)
  return unroll_files


def get_replay_length(eval_index: int, env_index: int) -> int:
  """Return the unpadded frame count for one rollout environment."""
  return int(rollout_lengths[eval_index][env_index])


def get_env_label(eval_index: int, env_index: int) -> str:
  return rollout_env_labels[eval_index][env_index]


def get_num_envs_for_eval(eval_index: int) -> int:
  """Return the number of policies/environments available for one replay."""
  return len(rollout_env_labels[eval_index])


def get_rollout_metadata(eval_index: int) -> Dict:
  return rollout_metadata[eval_index]


def object_name_from_metadata(metadata: Mapping, fallback: str) -> str:
  """Return the readable replay object name for viewer overlays."""
  selection = metadata.get('selection')
  if isinstance(selection, Mapping):
    object_name = selection.get('object_name')
    if isinstance(object_name, str) and object_name.strip():
      return object_name
  variant_name = metadata.get('object_variant_name')
  if isinstance(variant_name, str) and variant_name.strip():
    return variant_name
  return fallback


def _bundle_object_key(metadata: Mapping, fallback: str) -> Hashable:
  """Return a stable object-variant key, including for older bundles."""
  selection = metadata.get('selection')
  if isinstance(selection, Mapping):
    variant_index = selection.get('dataset_variant_index')
    if variant_index is not None:
      return ('dataset_variant_index', str(variant_index))
    variant_name = selection.get('object_variant_name')
    if variant_name:
      return ('object_variant_name', str(variant_name))
    object_name = selection.get('object_name')
    if object_name:
      return ('object_name_scale', str(object_name), selection.get('scale'))
  variant_name = metadata.get('object_variant_name')
  if variant_name:
    return ('object_variant_name', str(variant_name))
  if '/sample-' in fallback:
    return ('replay_prefix', fallback.rsplit('/sample-', 1)[0])
  return ('replay', fallback)


def _bundle_sample_index(metadata: Mapping, fallback: int) -> int:
  selection = metadata.get('selection')
  if isinstance(selection, Mapping):
    value = selection.get('sample_index')
    if isinstance(value, int) and not isinstance(value, bool):
      return value
  return fallback


def _rebuild_bundle_navigation() -> None:
  """Group flattened bundle examples by object while preserving order."""
  global bundle_object_groups, bundle_replay_positions
  grouped: Dict[Hashable, List[int]] = {}
  for eval_index, (name, metadata) in enumerate(
      zip(rollout_names, rollout_metadata, strict=True)
  ):
    key = _bundle_object_key(metadata, name)
    grouped.setdefault(key, []).append(eval_index)
  bundle_object_groups = [
      tuple(
          sorted(
              indices,
              key=lambda index: _bundle_sample_index(
                  rollout_metadata[index], index
              ),
          )
      )
      for indices in grouped.values()
  ]
  bundle_replay_positions = {
      eval_index: (object_index, sample_index)
      for object_index, indices in enumerate(bundle_object_groups)
      for sample_index, eval_index in enumerate(indices)
  }


def has_bundle_navigation() -> bool:
  """Return whether portable bundle object/sample navigation is active."""
  return bundle_manifest is not None and bool(bundle_object_groups)


def navigate_bundle_object(eval_index: int, delta: int) -> int:
  """Move to another object and select its first saved episode."""
  object_index, _ = bundle_replay_positions[eval_index]
  target = (object_index + delta) % len(bundle_object_groups)
  return bundle_object_groups[target][0]


def navigate_bundle_sample(eval_index: int, delta: int) -> int:
  """Move between saved episodes belonging to the current object."""
  object_index, sample_index = bundle_replay_positions[eval_index]
  samples = bundle_object_groups[object_index]
  return samples[(sample_index + delta) % len(samples)]


def get_bundle_position(eval_index: int) -> tuple[int, int, int, int]:
  """Return one-based object/sample positions and their totals."""
  object_index, sample_index = bundle_replay_positions[eval_index]
  return (
      object_index + 1,
      len(bundle_object_groups),
      sample_index + 1,
      len(bundle_object_groups[object_index]),
  )


def apply_current_model_fields(model, eval_index: int) -> None:
  fields = rollout_model_fields[eval_index]
  if fields:
    from rscope.replay_bundle import apply_model_fields

    apply_model_fields(model, fields)


def dict_obs_pixels_env_select(obs: Dict, i_env: int) -> Dict:
  """
  Select the first MAX_VIEWPORTS keys from the observation dictionary
  that start with 'pixels/' (excluding ones with 'latent') and extract column i_env.
  """
  obs_pixels = {}
  num_shown = 0
  for key in obs.keys():
    if num_shown >= MAX_VIEWPORTS:
      break
    if key.startswith('pixels/') and 'latent' not in key:
      obs_pixels[key] = obs[key][:, i_env]
      num_shown += 1
  return obs_pixels


def dict_obs_env_select(obs: Dict, i_env: int) -> Dict:
  """Select one environment while retaining all observation modalities."""
  return {key: value[:, i_env] for key, value in obs.items()}


def dict_obs_t_select(obs: Dict, t: int) -> Dict:
  """
  Select the first MAX_VIEWPORTS keys from the observation dictionary
  that start with 'pixels/' (excluding ones with 'latent') and extract index t.
  """
  obs_t = {}
  num_shown = 0
  for key in obs.keys():
    if num_shown >= MAX_VIEWPORTS:
      break
    if key.startswith('pixels/') and 'latent' not in key:
      obs_t[key] = obs[key][t]
      num_shown += 1
  return obs_t


def metrics_env_select(metrics: Dict, i_env: int) -> Dict:
  """Select column i_env from each metric."""
  return {key: metrics[key][:, i_env] for key in metrics.keys()}


def get_num_metric_pages(metrics: Dict, page_size: int = MAX_VIEWPORTS) -> int:
  """Return the number of metric pages needed for display."""
  if not metrics:
    return 0
  return (len(metrics) + page_size - 1) // page_size


def metrics_page_select(
    metrics: Dict, page: int, page_size: int = MAX_VIEWPORTS
) -> Dict:
  """Select one page of metrics while preserving key order."""
  num_pages = get_num_metric_pages(metrics, page_size)
  if not num_pages:
    return {}
  page = page % num_pages
  start = page * page_size
  end = start + page_size
  keys = list(metrics.keys())[start:end]
  return {key: metrics[key] for key in keys}
