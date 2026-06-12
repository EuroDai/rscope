#!/usr/bin/env python3

"""Entry point for the rscope package when executed with python -m rscope."""

from absl import app
from absl import flags
from absl import logging

from rscope.main import main

# Configure absl flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'ssh_to', None, 'SSH connection string in the format [username@]host[:port]'
)
flags.DEFINE_string(
    'ssh_key',
    None,
    'Optional SSH private key override; otherwise use ~/.ssh/config or default SSH keys and prompt for a password if needed',
)
flags.DEFINE_integer(
    'polling_interval', 10, 'Interval in seconds for SSH file polling'
)
flags.DEFINE_string(
    'path', None, 'Path to local rollout directory or SSH cache directory.'
)
flags.DEFINE_string(
    'remote_path', None, 'Path to remote rollout directory when using SSH.'
)


def _main(argv):
  ssh_enabled = FLAGS.ssh_to is not None
  main(
      ssh_enabled=ssh_enabled,
      polling_interval=FLAGS.polling_interval,
      path=FLAGS.path,
      remote_path=FLAGS.remote_path,
  )


if __name__ == '__main__':
  logging.set_verbosity(
      logging.WARNING
  )  # Set to INFO to debug SSH and file watcher, WARNING to mute.
  app.run(_main)
