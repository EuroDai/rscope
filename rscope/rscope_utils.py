"""Brax training rscope utils."""

import datetime
import os
import pathlib
from pathlib import PosixPath
import pickle
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np

from rscope import config
from rscope import rollout


def clear_dir(path: pathlib.Path):
  for child in path.iterdir():
    if child.is_dir():
      shutil.rmtree(child)
    else:
      child.unlink()


def _xml_has_external_file_refs(xml_bytes: bytes) -> bool:
  try:
    root = ET.fromstring(xml_bytes)
  except ET.ParseError:
    return True
  return any("file" in elem.attrib for elem in root.iter())


def rscope_init(
    xml_path: Union[PosixPath, str],
    model_assets: Optional[Dict[str, Any]] = None,
):
  # clear the active run directory.
  if os.path.exists(config.BASE_PATH):
    clear_dir(config.BASE_PATH)
  else:
    os.makedirs(config.BASE_PATH)

  # Save the xml into assets for remote rscope usage. Some generated MJCFs are
  # already self-contained; keeping every environment asset can introduce many
  # duplicate basenames such as YCB texture_map.png files, which MuJoCo rejects
  # when loading from an in-memory assets dict.
  xml_bytes = pathlib.Path(xml_path).read_bytes()
  xml_asset_name = pathlib.Path(xml_path).name
  if model_assets is None or not _xml_has_external_file_refs(xml_bytes):
    model_assets = {}
  else:
    model_assets = dict(model_assets)
  model_assets[xml_asset_name] = xml_bytes

  if not isinstance(xml_path, str):
    xml_path = xml_path.as_posix()

  rscope_meta = {"xml_path": xml_path, "model_assets": model_assets}
  # Make the base path and temp path if they don't exist.
  if not os.path.exists(config.BASE_PATH):
    os.makedirs(config.BASE_PATH)
  if not os.path.exists(config.TEMP_PATH):
    os.makedirs(config.TEMP_PATH)

  with open(os.path.join(config.BASE_PATH, "rscope_meta.pkl"), "wb") as f:
    pickle.dump(rscope_meta, f)


def dump_eval(trace: dict, obs: Union[jp.ndarray, dict], rew: jp.ndarray):
  # write to <datetime>.mj_unroll.
  now = datetime.datetime.now()
  now_str = now.strftime("%Y_%m_%d-%H_%M_%S")
  # ensure it's numpy.
  trace, obs, rew = jax.tree.map(lambda x: np.array(x), (trace, obs, rew))

  # save as dict rather than brax Transition.
  eval_rollout = rollout.Rollout(
      qpos=trace["qpos"],
      qvel=trace["qvel"],
      mocap_pos=trace["mocap_pos"],
      mocap_quat=trace["mocap_quat"],
      obs=obs,
      reward=rew,
      time=trace["time"],
      metrics=trace["metrics"],
  )

  # 2 stages to ensure atomicity.
  temp_path = os.path.join(config.TEMP_PATH, f"partial_transition.tmp")
  final_path = os.path.join(config.BASE_PATH, f"{now_str}.mj_unroll")
  with open(temp_path, "wb") as f:
    pickle.dump(eval_rollout, f)
  os.rename(temp_path, final_path)
