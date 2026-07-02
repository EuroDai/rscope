"""Model loader."""

import pathlib
import pickle
import time

import mujoco
import paramiko

import rscope.config as config
from rscope.ssh_utils import ssh_connect


def _load_mj_model_from_meta(meta):
  xml_asset_name = pathlib.Path(meta["xml_path"]).name
  stub_file = "<mujoco><include file='{}'/></mujoco>".format(xml_asset_name)
  assets = meta["model_assets"]
  try:
    return mujoco.MjModel.from_xml_string(stub_file, assets=assets)
  except ValueError as exc:
    if "Repeated file name in assets dict" not in str(exc):
      raise
    if xml_asset_name not in assets:
      raise
    # Older rscope dumps may include large asset dictionaries with repeated
    # basenames. Self-contained generated XMLs do not need those extra assets.
    return mujoco.MjModel.from_xml_string(
        stub_file,
        assets={xml_asset_name: assets[xml_asset_name]},
    )


def load_model_and_data(ssh_enabled=False):
  """Load meta information and create the Mujoco model and data."""
  # Create the active run directory if it doesn't exist
  config.BASE_PATH.mkdir(parents=True, exist_ok=True)
  if ssh_enabled:
    while True:
      try:
        # Connect to remote SSH
        ssh = paramiko.SSHClient()
        ssh_connect(ssh)
        sftp = ssh.open_sftp()

        try:
          # Copy meta file from remote to local
          remote_meta = str(config.REMOTE_META_PATH)
          local_meta = str(config.META_PATH)
          sftp.get(remote_meta, local_meta)
          break
        except FileNotFoundError:
          print(f"Meta file not found at {remote_meta}, waiting...")
          time.sleep(4)
        finally:
          sftp.close()
          ssh.close()
      except Exception as e:
        print(f"SSH error: {e}, retrying...")
        time.sleep(4)

  # Load meta file and create model
  with open(config.META_PATH, "rb") as f:
    meta = pickle.load(f)
  mj_model = _load_mj_model_from_meta(meta)
  mj_data = mujoco.MjData(mj_model)
  return mj_model, mj_data, meta
