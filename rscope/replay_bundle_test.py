"""Tests for portable replay bundle loading."""

import hashlib
import json
from pathlib import Path
import tempfile

from absl.testing import absltest
import mujoco
import numpy as np

from rscope import replay_bundle
from rscope import point_cloud


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(path: Path, **values) -> dict:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path, **values)
  return {'path': path.name, 'sha256': _sha256(path)}


def _write_json(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value), encoding='utf-8')


def _create_bundle(root: Path) -> Path:
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body><joint/><geom size='0.1'/>"
      "</body></worldbody></mujoco>"
  )
  model_path = root / 'models' / 'template.mjb'
  model_path.parent.mkdir(parents=True)
  mujoco.mj_saveModel(model, str(model_path), None)
  example_root = root / 'examples' / 'both-success'
  model_fields = _write_npz(
      example_root / 'model-fields.npz',
      geom_rgba=np.full_like(model.geom_rgba, 0.5),
  )
  policies = {}
  for policy_name, length in (('student', 3), ('teacher', 2)):
    policy_root = example_root / policy_name
    states = _write_npz(
        policy_root / 'states.npz',
        qpos=np.zeros((length, model.nq)),
        qvel=np.zeros((length, model.nv)),
        time_s=np.arange(length, dtype=np.float64) * 0.05,
    )
    metrics = _write_npz(
        policy_root / 'metrics.npz',
        object_target_distance_m=np.linspace(0.1, 0.02, length),
        palm_position_w_m=np.zeros((length, 3)),
        palm_quaternion_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (length, 1)),
    )
    record = {'state': states, 'metrics': metrics}
    if policy_name == 'student':
      record['observations'] = _write_npz(
          policy_root / 'observations.npz',
          point_cloud_palm_m=np.ones((length, 4, 3)),
          point_mask=np.ones((length, 4), dtype=bool),
      )
    policies[policy_name] = record
  example = {
      'schema': replay_bundle.SCHEMA_NAME,
      'schema_version': replay_bundle.SCHEMA_VERSION,
      'name': 'both-success',
      'selection': {'category': 'both_success'},
      'episode': {
          'batch_seed': 123,
          'env_index': 1,
          'fingertip_names': ['thumb', 'index'],
          'robot_joint_names': [],
      },
      'model_fields': model_fields,
      'policies': policies,
  }
  _write_json(example_root / 'manifest.json', example)
  root_manifest = {
      'schema': replay_bundle.SCHEMA_NAME,
      'schema_version': replay_bundle.SCHEMA_VERSION,
      'model': {
          'path': 'models/template.mjb',
          'sha256': _sha256(model_path),
          'mujoco_version': mujoco.__version__,
          'nq': model.nq,
          'nv': model.nv,
      },
      'examples': [
          {
              'name': 'both-success',
              'path': 'examples/both-success/manifest.json',
          }
      ],
  }
  _write_json(root / 'manifest.json', root_manifest)
  return model_path


class ReplayBundleTest(absltest.TestCase):

  def test_loads_paired_replay_and_preserves_lengths(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _create_bundle(root)
      result = replay_bundle.load_bundle(root)
      loaded = result['rollouts'][0]
      self.assertEqual(loaded['qpos'].shape[1], 2)
      np.testing.assert_array_equal(loaded['lengths'], (3, 2))
      self.assertEqual(loaded['env_labels'], ('Student', 'Teacher'))
      self.assertTrue(loaded['obs']['point_mask'][:, 0].all())
      self.assertFalse(loaded['obs']['point_mask'][:, 1].any())
      self.assertIn('object_target_distance_m', loaded['metrics'])

  def test_applies_model_patch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model_path = _create_bundle(root)
      result = replay_bundle.load_bundle(root)
      model = mujoco.MjModel.from_binary_path(str(model_path))
      replay_bundle.apply_model_fields(
          model, result['rollouts'][0]['model_fields']
      )
      np.testing.assert_allclose(model.geom_rgba, 0.5)

  def test_rejects_changed_file(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model_path = _create_bundle(root)
      model_path.write_bytes(model_path.read_bytes() + b'changed')
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        replay_bundle.load_bundle(root)

  def test_palm_point_transform_uses_wxyz_quaternion(self):
    points = np.asarray([[1.0, 0.0, 0.0]])
    transformed = point_cloud._palm_to_world(
        points,
        np.asarray([0.0, 2.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(transformed, [[1.0, 2.0, 0.0]])

  def test_point_cloud_populates_user_scene(self):
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    scene = mujoco.MjvScene(model, maxgeom=8)
    count = point_cloud.update(
        scene,
        {
            'point_cloud_palm_m': np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
            ),
            'point_mask': np.asarray([[True, False]]),
            'point_cloud_frame_position_w_m': np.asarray([[0.0, 2.0, 0.0]]),
            'point_cloud_frame_quaternion_wxyz': np.asarray(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
        },
        0,
        visible=True,
    )
    self.assertEqual(count, 1)
    self.assertEqual(scene.ngeom, 1)
    np.testing.assert_allclose(scene.geoms[0].pos, (0.0, 2.0, 0.0))


if __name__ == '__main__':
  absltest.main()
