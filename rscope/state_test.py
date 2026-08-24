"""Tests for viewer navigation state."""

from absl.testing import absltest
import glfw

from rscope import rollout
from rscope.state import ViewerState


class ViewerStateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rollout.num_evals = 4
    rollout.num_envs = 2
    rollout.rollout_env_labels = [
        ('Student', 'Teacher'),
        ('Student', 'Teacher'),
        ('Student', 'Teacher'),
        ('Student', 'Teacher'),
    ]
    rollout.bundle_manifest = {}
    rollout.bundle_object_groups = [(0, 2), (1, 3)]
    rollout.bundle_replay_positions = {
        0: (0, 0),
        2: (0, 1),
        1: (1, 0),
        3: (1, 1),
    }

  def tearDown(self):
    rollout.bundle_manifest = None
    rollout.bundle_object_groups = []
    rollout.bundle_replay_positions = {}
    super().tearDown()

  def test_bundle_arrows_navigate_objects_and_episodes(self):
    state = ViewerState()

    state.key_callback(glfw.KEY_RIGHT)
    self.assertEqual(state.cur_eval, 2)
    state.key_callback(glfw.KEY_UP)
    self.assertEqual(state.cur_eval, 1)
    state.key_callback(glfw.KEY_LEFT)
    self.assertEqual(state.cur_eval, 3)

  def test_bundle_p_switches_policy(self):
    state = ViewerState()

    state.key_callback(glfw.KEY_P)
    self.assertEqual(state.cur_env, 1)
    state.key_callback(glfw.KEY_P)
    self.assertEqual(state.cur_env, 0)

  def test_legacy_arrows_keep_native_navigation(self):
    rollout.bundle_manifest = None
    state = ViewerState()

    state.key_callback(glfw.KEY_RIGHT)
    self.assertEqual(state.cur_env, 1)
    state.key_callback(glfw.KEY_UP)
    self.assertEqual(state.cur_eval, 1)


if __name__ == '__main__':
  absltest.main()
