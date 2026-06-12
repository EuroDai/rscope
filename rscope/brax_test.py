from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

import rscope.brax as brax


class BraxRolloutSaverTest(absltest.TestCase):

  @mock.patch("rscope.brax.rscope_utils.dump_eval")
  @mock.patch("rscope.brax.rscope_utils.rscope_init")
  def test_dump_rollout_reuses_cached_jit(self, _, mock_dump_eval):
    compiled_rollout = mock.Mock(
        side_effect=[
            (mock.sentinel.trace1, mock.sentinel.obs1, mock.sentinel.rew1, mock.sentinel.done1),
            (mock.sentinel.trace2, mock.sentinel.obs2, mock.sentinel.rew2, mock.sentinel.done2),
        ]
    )

    with mock.patch("rscope.brax.jax.jit", return_value=compiled_rollout) as mock_jit:
      saver = brax.BraxRolloutSaver(
          trace_env=SimpleNamespace(xml_path="scene.xml", model_assets={}),
          ppo_params=SimpleNamespace(num_envs=4, episode_length=8, action_repeat=1),
          vision=False,
          rscope_envs=2,
          determistic=True,
          callback_fn=mock.Mock(),
      )
      saver.set_make_policy(mock.sentinel.make_policy)

      saver.dump_rollout(mock.sentinel.params1)
      saver.dump_rollout(mock.sentinel.params2)

    mock_jit.assert_called_once()
    self.assertEqual(compiled_rollout.call_count, 2)
    saver.callback_fn.assert_has_calls(
        [
            mock.call(
                mock.sentinel.trace1,
                mock.sentinel.obs1,
                mock.sentinel.rew1,
                mock.sentinel.done1,
            ),
            mock.call(
                mock.sentinel.trace2,
                mock.sentinel.obs2,
                mock.sentinel.rew2,
                mock.sentinel.done2,
            ),
        ]
    )
    mock_dump_eval.assert_has_calls(
        [
            mock.call(mock.sentinel.trace1, mock.sentinel.obs1, mock.sentinel.rew1),
            mock.call(mock.sentinel.trace2, mock.sentinel.obs2, mock.sentinel.rew2),
        ]
    )

  @mock.patch("rscope.brax.rscope_utils.rscope_init")
  def test_dump_rollout_requires_make_policy(self, _):
    with mock.patch("rscope.brax.jax.jit", return_value=mock.Mock()):
      saver = brax.BraxRolloutSaver(
          trace_env=SimpleNamespace(xml_path="scene.xml", model_assets={}),
          ppo_params=SimpleNamespace(num_envs=4, episode_length=8, action_repeat=1),
          vision=False,
          rscope_envs=2,
          determistic=True,
      )

    with self.assertRaisesRegex(ValueError, "set_make_policy"):
      saver.dump_rollout(mock.sentinel.params)


if __name__ == "__main__":
  absltest.main()
