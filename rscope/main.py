"""Rscope main script."""

from collections.abc import Mapping
from pathlib import Path
from queue import Queue
import shutil
import threading
import time

from absl import app
from absl import flags
from absl import logging
import mujoco
import mujoco.viewer as mujoco_viewer
from watchdog.observers import Observer

import rscope.config as config
import rscope.event_handler as event_handler
import rscope.image_processing as image_processing
import rscope.model_loader as model_loader
import rscope.point_cloud as point_cloud
import rscope.rollout as rollout
from rscope.ssh_utils import prime_ssh_password_cache
from rscope.ssh_utils import SSHFileTransfer
from rscope.ssh_utils import SSHFileWatcher
from rscope.state import ViewerState
import rscope.viewer_utils as vu


def main(ssh_enabled=False, polling_interval=10, path=None, remote_path=None):

  if path is not None:
    config.set_base_path(path)
  if ssh_enabled:
    config.set_remote_base_path(
        remote_path if remote_path is not None else config.BASE_PATH
    )

  # if BASE_PATH does not exist, make it
  if not config.BASE_PATH.exists():
    config.BASE_PATH.mkdir(parents=True, exist_ok=True)
  bundle_mode = (config.BASE_PATH / "manifest.json").is_file()
  if not bundle_mode and not config.TEMP_PATH.exists():
    config.TEMP_PATH.mkdir(parents=True, exist_ok=True)

  # Create an instance of ViewerState to encapsulate state.
  viewer_state = ViewerState()

  if ssh_enabled:
    logging.info(
        "SSH file watching enabled with polling interval:"
        f" {polling_interval}s; remote path: {config.REMOTE_BASE_PATH};"
        f" local cache: {config.BASE_PATH}"
    )
    prime_ssh_password_cache()

    # Delete all existing files in the base path to prevent duplication.
    for file in config.BASE_PATH.glob("*"):
      try:
        if file.is_dir():
          shutil.rmtree(file)
        else:
          file.unlink()
      except Exception as e:
        logging.error(f"Error deleting {file}: {e}")

    file_queue = Queue()
    known_files = set()
    stop_event = threading.Event()

    watcher_thread = SSHFileWatcher(
        file_queue, known_files, stop_event, polling_interval=polling_interval
    )
    transfer_thread = SSHFileTransfer(file_queue, stop_event, viewer_state)

  # Setup file system observer.
  observer = None
  if not bundle_mode:
    event_handler_instance = event_handler.MjUnrollHandler()
    observer = Observer()
    observer.schedule(
        event_handler_instance, str(config.BASE_PATH), recursive=False
    )
    try:
      observer.start()
    except Exception as e:
      logging.error(f"Error starting observer: {e}")

  if ssh_enabled:
    watcher_thread.start()
    transfer_thread.start()
  else:
    rollout.load_all_local_unrolls(config.BASE_PATH)

  # Wait for new rollouts to trickle in.
  print("Waiting for rollouts...")
  while not rollout.rollouts:
    time.sleep(3)
  print(
      f"Found {len(rollout.rollouts)} rollouts in {config.BASE_PATH}, "
      "starting viewer..."
  )

  # Initialize figures using metrics keys from the first rollout.
  metrics_keys = vu.get_ordered_metric_keys(
      list(rollout.rollouts[0].metrics.keys())
  )
  vu.reset_figures(metrics_keys)

  # Determine the initial replay length.
  replay_len = rollout.get_replay_length(0, 0)

  # Load the Mujoco model and data.
  mj_model, mj_data, meta = model_loader.load_model_and_data(ssh_enabled)

  loop_cnt = 0

  with mujoco_viewer.launch_passive(
      mj_model,
      mj_data,
      show_left_ui=False,
      show_right_ui=False,
      key_callback=viewer_state.key_callback,
  ) as viewer:

    while viewer.is_running():
      step_start = time.time()

      # Trajectory selection: if a new rollout is requested.
      if viewer_state.change_rollout:
        viewer_state.change_rollout = False
        full_rollout = rollout.rollouts[viewer_state.cur_eval]
        metrics_keys = vu.get_ordered_metric_keys(
            list(full_rollout.metrics.keys())
        )
        vu.reset_figures(metrics_keys)
        if isinstance(full_rollout.obs, dict):
          obs = rollout.dict_obs_env_select(
              full_rollout.obs, viewer_state.cur_env
          )
        else:
          obs = full_rollout.obs[:, viewer_state.cur_env]
        cur_rollout = full_rollout._replace(
            qpos=full_rollout.qpos[:, viewer_state.cur_env],
            qvel=full_rollout.qvel[:, viewer_state.cur_env],
            mocap_pos=full_rollout.mocap_pos[:, viewer_state.cur_env],
            mocap_quat=full_rollout.mocap_quat[:, viewer_state.cur_env],
            obs=obs,
            reward=full_rollout.reward[:, viewer_state.cur_env],
            time=full_rollout.time[:, viewer_state.cur_env],
            metrics=rollout.metrics_env_select(
                full_rollout.metrics, viewer_state.cur_env
            ),
        )
        replay_index = 0
        replay_len = rollout.get_replay_length(
            viewer_state.cur_eval, viewer_state.cur_env
        )
        with viewer.lock():
          rollout.apply_current_model_fields(mj_model, viewer_state.cur_eval)
          if (
              isinstance(cur_rollout.obs, Mapping)
              and "replay_focus_position_w_m" in cur_rollout.obs
          ):
            viewer.cam.lookat[:] = cur_rollout.obs["replay_focus_position_w_m"][
                0
            ]
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0

      # Check if the transfer status message has expired.
      if (
          ssh_enabled
          and viewer_state.transfer_status
          and viewer_state.transfer_until
      ):
        current_time = time.time()
        if current_time > viewer_state.transfer_until:
          viewer_state.transfer_status = None
          viewer_state.transfer_until = None

      metrics_keys = vu.get_ordered_metric_keys(
          list(cur_rollout.metrics.keys())
      )
      ordered_metrics = {key: cur_rollout.metrics[key] for key in metrics_keys}
      num_metric_pages = rollout.get_num_metric_pages(
          ordered_metrics, vu.MAX_VIEWPORTS
      )
      if num_metric_pages:
        viewer_state.cur_metric_page %= num_metric_pages
        metric_status = (
            f"{viewer_state.cur_metric_page + 1}/{num_metric_pages}"
            if viewer_state.show_metrics
            else "off"
        )
      else:
        viewer_state.cur_metric_page = 0
        metric_status = "0/0" if viewer_state.show_metrics else "off"

      # Overlay text.
      replay_metadata = rollout.get_rollout_metadata(viewer_state.cur_eval)
      object_name = rollout.object_name_from_metadata(
          replay_metadata,
          rollout.rollout_names[viewer_state.cur_eval],
      )
      policy_label = rollout.get_env_label(
          viewer_state.cur_eval, viewer_state.cur_env
      )
      selection = replay_metadata.get("selection", {})
      if rollout.has_bundle_navigation():
        object_index, object_count, sample_index, sample_count = (
            rollout.get_bundle_position(viewer_state.cur_eval)
        )
        scale = selection.get("scale")
        scale_text = f" @{float(scale):.2f}" if scale is not None else ""
        text_1 = "Object\nEpisode\nPolicy\nStep\nMetrics\nStatus\nSpeed"
        text_2 = (
            f"{object_index}/{object_count} {object_name}{scale_text}\n"
            f"{sample_index}/{sample_count}\n"
            f"{policy_label}\n"
            f"{replay_index + 1}/{replay_len}\n"
            f"{metric_status}\n"
        )
      else:
        text_1 = "Object\nReplay\nPolicy\nStep\nMetrics\nStatus\nSpeed"
        text_2 = (
            f"{object_name}\n"
            f"{rollout.rollout_names[viewer_state.cur_eval]} "
            f"({viewer_state.cur_eval+1}/{len(rollout.rollouts)})\n"
            f"{policy_label}\n"
            f"{replay_index}\n"
            f"{metric_status}\n"
        )
      text_2 += "Pause" if viewer_state.pause else "Play"
      text_2 += f"\n{viewer_state.playback_speed * 100:.1f}%"
      overlays = [(
          mujoco.mjtFontScale.mjFONTSCALE_150,
          mujoco.mjtGridPos.mjGRID_TOPLEFT,
          text_1,
          text_2,
      )]
      if replay_metadata:
        group = selection.get("group", selection.get("category", "unknown"))
        variant_index = selection.get("dataset_variant_index")
        variant = (
            f"#{variant_index}" if variant_index is not None else "unknown"
        )
        source_rate = selection.get("source_success_rate")
        source_rate_text = (
            f"{float(source_rate) * 100:.1f}%"
            if source_rate is not None
            else "unknown"
        )
        outcomes = replay_metadata.get("policy_outcomes", {}).get(
            policy_label.lower(), {}
        )
        outcome_text = ""
        if outcomes:
          outcome_text = "  ".join(
              f"{label}:{'Y' if outcomes[key] else 'N'}"
              if key in outcomes
              else f"{label}:?"
              for label, key in (
                  ("G", "grasp_success"),
                  ("H", "task_hold_success"),
                  ("F", "final_position_success"),
              )
            )
        info_1 = "Group\nVariant\nSource SR\nSeed / Env\nOutcome"
        info_2 = (
            f"{group}\n"
            f"{variant}\n"
            f"{source_rate_text}\n"
            f"{replay_metadata.get('batch_seed', '?')} / "
            f"{replay_metadata.get('env_index', '?')}\n"
            f"{outcome_text or 'unknown'}"
        )
        overlays.append((
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            info_1,
            info_2,
        ))

      if viewer_state.show_help:
        menu_text_1, menu_text_2 = vu.get_menu_text(
            bundle_mode=rollout.has_bundle_navigation()
        )
        overlays.append((
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            menu_text_1,
            menu_text_2,
        ))

      if ssh_enabled and viewer_state.transfer_status:
        overlays.append((
            mujoco.mjtFontScale.mjFONTSCALE_250,
            mujoco.mjtGridPos.mjGRID_BOTTOM,
            viewer_state.transfer_status,
            "",
        ))

      loop_cnt += 1
      if loop_cnt % 2 == 0:
        # These viewer APIs lock internally in MuJoCo 3.11.
        viewer.set_texts(overlays)

        if viewer_state.show_metrics:
          if not viewer_state.pause:
            for key, metrics in ordered_metrics.items():
              vu.add_data_to_fig(key, metrics[replay_index])
          cur_metrics = rollout.metrics_page_select(
              ordered_metrics,
              viewer_state.cur_metric_page,
              vu.MAX_VIEWPORTS,
          )
          viewports = vu.get_viewports(len(cur_metrics), viewer.viewport)
          viewport_figures = [
              (viewport, vu.figures[key])
              for key, viewport in zip(cur_metrics.keys(), viewports)
          ]
          if viewport_figures:
            viewer.set_figures(viewport_figures)
          else:
            viewer.clear_figures()
        else:
          viewer.clear_figures()

      # Render pixel observations if available.
      if isinstance(cur_rollout.obs, Mapping):
        if any(key.startswith("pixels/") for key in cur_rollout.obs.keys()):
          if viewer_state.show_pixel_obs:
            cur_obs = rollout.dict_obs_t_select(cur_rollout.obs, replay_index)
            viewports = vu.get_viewports(len(cur_obs), viewer.viewport)
            processed_obs = {
                key: image_processing.process_img(
                    cur_obs[key], viewport.height, viewport.width
                )
                for key, viewport in zip(cur_obs.keys(), viewports)
            }
            viewer.set_images(
                list(zip(viewports, list(processed_obs.values())))
            )
          else:
            viewer.clear_images()

      # Advance simulation: update the state.
      def advance_rollout(mj_model, mj_data, idx):
        mj_data.qpos, mj_data.qvel = (
            cur_rollout.qpos[idx],
            cur_rollout.qvel[idx],
        )
        if cur_rollout.mocap_pos.size:
          mj_data.mocap_pos, mj_data.mocap_quat = (
              cur_rollout.mocap_pos[idx],
              cur_rollout.mocap_quat[idx],
          )
        mj_data.time = cur_rollout.time[idx]
        mujoco.mj_forward(mj_model, mj_data)

      with viewer.lock():
        advance_rollout(mj_model, mj_data, replay_index)
        if isinstance(cur_rollout.obs, Mapping):
          point_cloud.update(
              viewer.user_scn,
              cur_rollout.obs,
              replay_index,
              visible=viewer_state.show_pixel_obs,
              visualization=(rollout.bundle_manifest or {}).get(
                  'visualization'
              ),
          )
      if not viewer_state.pause:
        replay_index = (replay_index + 1) % replay_len
      viewer.sync()

      time_until_next_step = float(
          rollout.env_ctrl_dt
      ) / viewer_state.playback_speed - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)

  # Modify cleanup section
  if ssh_enabled:
    stop_event.set()
    watcher_thread.join()
    transfer_thread.join()

  if observer is not None:
    observer.stop()
    observer.join()
