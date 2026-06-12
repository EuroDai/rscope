import os
from pathlib import Path
import pickle
import paramiko
from queue import Queue
import shutil
import tempfile
import threading
import time
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver

import rscope.__main__  # noqa: F401
import rscope.config as config
import rscope.model_loader as model_loader
import rscope.ssh_utils as ssh_utils
from rscope.ssh_utils import SSHFileTransfer
from rscope.ssh_utils import SSHFileWatcher


class MockSFTP:

  def __init__(self, remote_dir):
    self.remote_dir = remote_dir
    self.listdir_calls = []
    self.get_calls = []

  def listdir(self, path):
    self.listdir_calls.append(path)
    return os.listdir(self.remote_dir)

  def get(self, remote_path, local_path):
    self.get_calls.append((remote_path, local_path))
    # Extract just the filename from the remote path
    filename = os.path.basename(remote_path)
    src_path = os.path.join(self.remote_dir, filename)
    shutil.copy(src_path, local_path)

  def close(self):
    pass


class MockSSHClient:

  def __init__(self, remote_dir):
    self.remote_dir = remote_dir
    self.sftp = MockSFTP(self.remote_dir)

  def set_missing_host_key_policy(self, policy):
    pass

  def connect(self, hostname, port=None, username=None, **kwargs):
    pass

  def open_sftp(self):
    return self.sftp

  def close(self):
    pass


class SSHUtilsTest(absltest.TestCase):

  def setUp(self):
    # Create temporary directories for test
    self.local_temp_dir = tempfile.mkdtemp()
    self.remote_temp_dir = tempfile.mkdtemp()
    self.ssh_config_dir = tempfile.mkdtemp()
    self.ssh_config_path = Path(self.ssh_config_dir) / "config"
    ssh_utils._PASSWORD_CACHE.clear()
    if not flags.FLAGS.is_parsed():
      flags.FLAGS(["ssh_utils_test"])

    # Store original config values
    self.orig_base_path = config.BASE_PATH
    self.orig_temp_path = config.TEMP_PATH
    self.orig_meta_path = config.META_PATH
    self.orig_remote_base_path = config.REMOTE_BASE_PATH
    self.orig_remote_meta_path = config.REMOTE_META_PATH

    # Set config to use temporary directories
    config.BASE_PATH = Path(self.local_temp_dir)
    config.TEMP_PATH = Path(self.local_temp_dir)
    config.META_PATH = config.BASE_PATH / "rscope_meta.pkl"
    config.set_remote_base_path("/remote/rollouts")

    self.ssh_config_path_patcher = mock.patch(
        "rscope.ssh_utils._get_ssh_config_path", return_value=self.ssh_config_path
    )
    self.ssh_config_path_patcher.start()

    # Set up the SSH client mock
    self.mock_ssh_instance = MockSSHClient(self.remote_temp_dir)
    self.ssh_client_patcher = mock.patch(
        "paramiko.SSHClient", return_value=self.mock_ssh_instance
    )
    self.mock_ssh_client = self.ssh_client_patcher.start()

  def tearDown(self):
    # Stop the patchers
    self.ssh_client_patcher.stop()
    self.ssh_config_path_patcher.stop()

    # Restore original config values
    config.BASE_PATH = self.orig_base_path
    config.TEMP_PATH = self.orig_temp_path
    config.META_PATH = self.orig_meta_path
    config.REMOTE_BASE_PATH = self.orig_remote_base_path
    config.REMOTE_META_PATH = self.orig_remote_meta_path

    # Clean up temp directories
    shutil.rmtree(self.local_temp_dir, ignore_errors=True)
    shutil.rmtree(self.remote_temp_dir, ignore_errors=True)
    shutil.rmtree(self.ssh_config_dir, ignore_errors=True)

  def _write_ssh_config(self, text: str):
    self.ssh_config_path.write_text(text, encoding="utf-8")

  @flagsaver.flagsaver(ssh_to="alice@myalias:2222", ssh_key="~/.ssh/test_key")
  def test_ssh_connect_uses_explicit_key_when_provided(self):
    self._write_ssh_config("Host myalias\n  Hostname real.example.com\n")
    ssh = mock.Mock()

    ssh_utils.ssh_connect(ssh)

    ssh.set_missing_host_key_policy.assert_called_once()
    ssh.connect.assert_called_once_with(
        "real.example.com",
        port=2222,
        username="alice",
        timeout=10,
        key_filename=os.path.expanduser("~/.ssh/test_key"),
    )

  @flagsaver.flagsaver(ssh_to="myalias", ssh_key=None)
  def test_ssh_connect_uses_ssh_config_identity_when_available(self):
    with mock.patch.dict(os.environ, {"USERPROFILE": "/tmp/windows-home"}, clear=False):
      self._write_ssh_config(
          "\n".join([
              "Host myalias",
              "  Hostname real.example.com",
              "  User configuser",
              "  Port 2200",
              "  IdentityFile ~/.ssh/id_config",
              "  IdentityFile %USERPROFILE%/.ssh/id_windows",
              "",
          ])
      )
      ssh = mock.Mock()

      ssh_utils.ssh_connect(ssh)

    ssh.connect.assert_called_once_with(
        "real.example.com",
        port=2200,
        username="configuser",
        timeout=10,
        key_filename=[
            os.path.expanduser("~/.ssh/id_config"),
            "/tmp/windows-home/.ssh/id_windows",
        ],
    )

  @flagsaver.flagsaver(ssh_to="alice@myalias:2022", ssh_key=None)
  def test_cli_username_and_port_override_ssh_config(self):
    self._write_ssh_config(
        "\n".join([
            "Host myalias",
            "  Hostname real.example.com",
            "  User configuser",
            "  Port 2200",
            "  IdentityFile ~/.ssh/id_config",
            "",
        ])
    )
    ssh = mock.Mock()

    ssh_utils.ssh_connect(ssh)

    ssh.connect.assert_called_once_with(
        "real.example.com",
        port=2022,
        username="alice",
        timeout=10,
        key_filename=[os.path.expanduser("~/.ssh/id_config")],
    )

  @flagsaver.flagsaver(ssh_to="127.0.0.1:8080", ssh_key=None)
  @mock.patch("rscope.ssh_utils.getpass.getpass", return_value="secret")
  @mock.patch("rscope.ssh_utils.getpass.getuser", return_value="localuser")
  def test_ssh_connect_falls_back_to_password_and_caches_it(
      self, mock_getuser, mock_getpass
  ):
    ssh = mock.Mock()
    ssh.connect.side_effect = [paramiko.AuthenticationException("bad key"), None]

    ssh_utils.ssh_connect(ssh)

    self.assertEqual(ssh.connect.call_count, 2)
    self.assertEqual(
        ssh.connect.call_args_list[0],
        mock.call(
            "127.0.0.1",
            port=8080,
            username="localuser",
            timeout=10,
        ),
    )
    self.assertEqual(
        ssh.connect.call_args_list[1],
        mock.call(
            "127.0.0.1",
            port=8080,
            username="localuser",
            timeout=10,
            password="secret",
            look_for_keys=False,
            allow_agent=False,
        ),
    )
    mock_getuser.assert_called_once()
    mock_getpass.assert_called_once_with(
        prompt="SSH password for localuser@127.0.0.1: "
    )

    second_ssh = mock.Mock()
    ssh_utils.ssh_connect(second_ssh)
    second_ssh.connect.assert_called_once_with(
        "127.0.0.1",
        port=8080,
        username="localuser",
        timeout=10,
        password="secret",
        look_for_keys=False,
        allow_agent=False,
    )
    self.assertEqual(mock_getpass.call_count, 1)

  @flagsaver.flagsaver(ssh_to="127.0.0.1:8080", ssh_key=None)
  @mock.patch("rscope.ssh_utils.getpass.getpass")
  @mock.patch("rscope.ssh_utils.getpass.getuser")
  def test_prime_ssh_password_cache_no_longer_prompts(
      self, mock_getuser, mock_getpass
  ):
    ssh_utils.prime_ssh_password_cache()

    mock_getuser.assert_not_called()
    mock_getpass.assert_not_called()

  @mock.patch("rscope.ssh_utils.ssh_connect")
  def test_remote_base_path_is_used_for_ssh_file_operations(self, _):
    self.mock_ssh_instance.sftp.listdir_calls.clear()
    self.mock_ssh_instance.sftp.get_calls.clear()

    filename = "remote_only.mj_unroll"
    with open(os.path.join(self.remote_temp_dir, filename), "w") as f:
      f.write("remote content")

    file_queue = Queue()
    known_files = set()
    stop_event = threading.Event()

    watcher_thread = SSHFileWatcher(
        file_queue, known_files, stop_event, polling_interval=0.1
    )
    transfer_thread = SSHFileTransfer(file_queue, stop_event)

    watcher_thread.start()
    transfer_thread.start()

    time.sleep(0.5)

    stop_event.set()
    watcher_thread.join(timeout=1)
    transfer_thread.join(timeout=1)

    self.assertIn(str(config.REMOTE_BASE_PATH), self.mock_ssh_instance.sftp.listdir_calls)
    self.assertIn(
        (
            str(config.REMOTE_BASE_PATH / filename),
            str(config.TEMP_PATH / f".tmp_{filename}"),
        ),
        self.mock_ssh_instance.sftp.get_calls,
    )
    self.assertTrue((config.BASE_PATH / filename).exists())

  @mock.patch("rscope.ssh_utils.ssh_connect")
  def test_transfer_recreates_missing_temp_directory(self, _):
    filename = "needs_temp_dir.mj_unroll"
    with open(os.path.join(self.remote_temp_dir, filename), "w") as f:
      f.write("temp dir recovered")

    config.TEMP_PATH = config.BASE_PATH / ".tmp"

    file_queue = Queue()
    file_queue.put(filename)
    stop_event = threading.Event()

    transfer_thread = SSHFileTransfer(file_queue, stop_event)
    shutil.rmtree(config.TEMP_PATH)

    transfer_thread.start()
    time.sleep(0.3)
    stop_event.set()
    transfer_thread.join(timeout=1)

    local_path = os.path.join(self.local_temp_dir, filename)
    self.assertTrue(
        os.path.exists(local_path), f"File {filename} was not transferred"
    )
    with open(local_path, "r") as f:
      content = f.read()
    self.assertEqual(content, "temp dir recovered")

  @mock.patch("rscope.model_loader.mujoco.MjData", return_value=mock.sentinel.mj_data)
  @mock.patch(
      "rscope.model_loader.mujoco.MjModel.from_xml_string",
      return_value=mock.sentinel.mj_model,
  )
  @mock.patch("rscope.model_loader.ssh_connect")
  def test_model_loader_uses_remote_meta_path(self, _, __, ___):
    self.mock_ssh_instance.sftp.get_calls.clear()

    with open(os.path.join(self.remote_temp_dir, "rscope_meta.pkl"), "wb") as f:
      pickle.dump(
          {
              "xml_path": "/remote/rollouts/scene.xml",
              "model_assets": {"scene.xml": "<mujoco/>"},
          },
          f,
      )

    mj_model, mj_data, meta = model_loader.load_model_and_data(ssh_enabled=True)

    self.assertEqual(mj_model, mock.sentinel.mj_model)
    self.assertEqual(mj_data, mock.sentinel.mj_data)
    self.assertEqual(meta["xml_path"], "/remote/rollouts/scene.xml")
    self.assertIn(
        (str(config.REMOTE_META_PATH), str(config.META_PATH)),
        self.mock_ssh_instance.sftp.get_calls,
    )

  def test_set_remote_base_path_preserves_posix_separators(self):
    config.set_remote_base_path("/home/handcraft/learning/rscope/")

    self.assertEqual(
        str(config.REMOTE_BASE_PATH), "/home/handcraft/learning/rscope"
    )
    self.assertEqual(
        str(config.REMOTE_META_PATH),
        "/home/handcraft/learning/rscope/rscope_meta.pkl",
    )

  def test_existing_files(self):
    """Test that existing files on remote are discovered and transferred."""
    with mock.patch("rscope.ssh_utils.ssh_connect"):
      # Create test files in remote directory
      test_files = []
      for i in range(3):  # Reduced from 5 to 3 files
        filename = f"test_{i}.mj_unroll"
        file_path = os.path.join(self.remote_temp_dir, filename)
        with open(file_path, "w") as f:
          f.write(f"Content for {filename}")
        test_files.append(filename)

      # Set up threads
      file_queue = Queue()
      known_files = set()
      stop_event = threading.Event()

      # Use shorter polling interval for testing
      watcher_thread = SSHFileWatcher(
          file_queue, known_files, stop_event, polling_interval=0.1
      )
      transfer_thread = SSHFileTransfer(file_queue, stop_event)

      # Start the threads
      watcher_thread.start()
      transfer_thread.start()

      # Give some time for the watcher to find files and transfer to process them
      time.sleep(0.5)  # Reduced from 3s to 0.5s

      # Stop the threads
      stop_event.set()
      watcher_thread.join(timeout=1)  # Reduced timeout from 5s to 1s
      transfer_thread.join(timeout=1)

      # Verify all files were discovered and transferred
      for filename in test_files:
        local_path = os.path.join(self.local_temp_dir, filename)
        self.assertTrue(
            os.path.exists(local_path), f"File {filename} was not transferred"
        )

        # Verify content matches
        with open(local_path, "r") as f:
          content = f.read()
        self.assertEqual(content, f"Content for {filename}")

      # Verify known_files set contains all test files
      self.assertEqual(known_files, set(test_files))

  def test_watcher_retries_when_remote_directory_is_missing(self):
    with mock.patch("rscope.ssh_utils.ssh_connect"):
      filename = "recovered.mj_unroll"
      with open(os.path.join(self.remote_temp_dir, filename), "w") as f:
        f.write("recovered content")

      listdir_calls = {"count": 0}

      def listdir_side_effect(path):
        listdir_calls["count"] += 1
        if listdir_calls["count"] == 1:
          raise FileNotFoundError()
        return [filename]

      self.mock_ssh_instance.sftp.listdir = mock.Mock(side_effect=listdir_side_effect)

      file_queue = Queue()
      known_files = set()
      stop_event = threading.Event()

      watcher_thread = SSHFileWatcher(
          file_queue, known_files, stop_event, polling_interval=0.1
      )
      transfer_thread = SSHFileTransfer(file_queue, stop_event)

      watcher_thread.start()
      transfer_thread.start()

      time.sleep(0.5)

      stop_event.set()
      watcher_thread.join(timeout=1)
      transfer_thread.join(timeout=1)

      local_path = os.path.join(self.local_temp_dir, filename)
      self.assertTrue(
          os.path.exists(local_path), f"File {filename} was not transferred"
      )
      with open(local_path, "r") as f:
        content = f.read()
      self.assertEqual(content, "recovered content")
      self.assertGreaterEqual(listdir_calls["count"], 2)

  def test_trickling_files(self):
    """Test that files added over time are discovered and transferred."""
    with mock.patch("rscope.ssh_utils.ssh_connect"):
      file_queue = Queue()
      known_files = set()
      stop_event = threading.Event()

      # Use shorter polling interval for testing
      watcher_thread = SSHFileWatcher(
          file_queue, known_files, stop_event, polling_interval=0.1
      )
      transfer_thread = SSHFileTransfer(file_queue, stop_event)

      # Start the threads
      watcher_thread.start()
      transfer_thread.start()

      # Create 3 files quickly
      test_files = []
      for i in range(3):  # Reduced from 5 to 3 files
        filename = f"trickle_{i}.mj_unroll"
        file_path = os.path.join(self.remote_temp_dir, filename)
        with open(file_path, "w") as f:
          f.write(f"Trickle content for {filename}")
        test_files.append(filename)

        # Wait a shorter time before adding the next file
        time.sleep(0.1)  # Reduced from 1s to 0.1s

      # Give some additional time for the last file to be processed
      time.sleep(0.5)  # Reduced from 3s to 0.5s

      # Stop the threads
      stop_event.set()
      watcher_thread.join(timeout=1)  # Reduced timeout from 5s to 1s
      transfer_thread.join(timeout=1)

      # Verify all files were discovered and transferred
      for filename in test_files:
        local_path = os.path.join(self.local_temp_dir, filename)
        self.assertTrue(
            os.path.exists(local_path), f"File {filename} was not transferred"
        )

        # Verify content matches
        with open(local_path, "r") as f:
          content = f.read()
        self.assertEqual(content, f"Trickle content for {filename}")

      # Verify known_files set contains all test files
      for filename in test_files:
        self.assertIn(filename, known_files)


if __name__ == "__main__":
  absltest.main()
