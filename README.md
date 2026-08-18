[![PyPI](https://img.shields.io/pypi/v/rscope.svg)](https://pypi.org/project/rscope/) [![Python Versions](https://img.shields.io/pypi/pyversions/rscope.svg)](https://pypi.org/project/rscope/) [![License: MIT](https://img.shields.io/pypi/l/rscope.svg)](https://opensource.org/licenses/MIT)

## Welcome to rscope!

A light-weight package to collect and interactively visualize trajectories while training Mujoco Playground environments. Rscope can visualize both local and remote (potentially headless) training runs.

![rscope_header](https://github.com/user-attachments/assets/225d0290-501d-4a2e-ace9-f2122786ffb6)

### Installation
> [!IMPORTANT]
> - Requires Python 3.10 or later.

```bash
pip install rscope
```

For local development from a clone:

```bash
pip install -e .
```

---

### Usage
> [!IMPORTANT]
> Mac users must run `mjpython` instead of `python`, e.g. `mjpython -m rscope`

#### Local training runs
To visualize rollouts from the default active-run directory:

```bash
python -m rscope
```

By default, rscope reads from:

- `/tmp/rscope/active_run`

This directory is expected to contain:

- `rscope_meta.pkl`
- one or more `*.mj_unroll` rollout files

#### Visualize old rollouts from a specific directory
You can now point rscope at any rollout directory instead of only using `/tmp/rscope/active_run`:

```bash
python -m rscope --path /path/to/rollout_dir
```

This is useful if you have copied rollouts into an experiment directory such as:

```text
logs/<experiment>/checkpoints/rscope/
```

#### Visualize a portable replay bundle

Rscope also accepts a versioned `mujoco-replay-bundle` directory. These bundles
store a shared compiled MuJoCo model, per-episode model-field patches, and
non-pickle NumPy trajectories:

```bash
python -m rscope --path /path/to/replay-bundle
```

For paired policy exports, use left/right to switch policies and up/down to
switch replay examples. `SHIFT+M` displays scalar trajectory metrics and
`SHIFT+O` overlays saved point-cloud observations in the MuJoCo scene.

Replay bundles are state playback artifacts. Rscope does not load checkpoints,
run policies, recompute task success, or integrate the recorded physics again.
The bundle producer remains responsible for those semantics.

#### Remote training runs
Below, **update `user@remote_host`**, for example `alice@168.42.4.8`.

If you want key-based SSH, first set up a password-free key-based SSH connection with the remote device:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/rsync_key -N ""
ssh-copy-id -i ~/.ssh/rsync_key.pub user@remote_host
```

If this worked, you should be able to ssh in without using a password:

```bash
ssh -i ~/.ssh/rsync_key user@remote_host
echo hello
exit
```

To visualize rollouts stored on a remote server via SSH with a private key:

```bash
python -m rscope --ssh_to user@remote_host[:port] --ssh_key ~/.ssh/rsync_key --polling_interval 5
```

If the remote rollout directory is not the default `/tmp/rscope/active_run`, pass it separately:

```bash
python -m rscope \
  --ssh_to user@remote_host[:port] \
  --ssh_key ~/.ssh/rsync_key \
  --remote_path /path/to/remote/rollouts \
  --polling_interval 5
```

If you also want to change the local SSH cache directory, use `--path` for the local cache and `--remote_path` for the remote rollout directory:

```bash
python -m rscope \
  --ssh_to user@remote_host[:port] \
  --path /tmp/my-local-rscope-cache \
  --remote_path /path/to/remote/rollouts \
  --polling_interval 5
```

If you do not pass `--ssh_key`, `rscope` prompts once in the terminal for the SSH password and reuses it for subsequent connections:

```bash
python -m rscope --ssh_to [user@]remote_host[:port] --polling_interval 5
```

> `port` defaults to `22`.

---

### Where rollout files are stored
When used with the default local setup, rscope writes rollout files to:

```text
/tmp/rscope/
├── active_run/
│   ├── rscope_meta.pkl
│   ├── 2026_05_16-12_26_53.mj_unroll
│   ├── 2026_05_16-12_31_10.mj_unroll
│   └── ...
└── temp/
    └── partial_transition.tmp
```

Notes:

- `active_run/` holds the current set of rollout files that the viewer reads.
- `temp/partial_transition.tmp` is an intermediate file used for atomic writes.
- Each `*.mj_unroll` file contains one saved rollout across all selected `rscope_envs`; it is **not** one file per environment.
- `rscope_init()` clears the active run directory at the start of a new run.

---

### Rollout ordering
Rollouts are ordered chronologically by filename timestamp.

- Existing local rollouts are loaded in sorted filename order.
- Newly discovered rollouts are inserted into the in-memory list in sorted filename order.
- Remote SSH polling also sorts newly discovered rollout filenames before loading them.

Filename format:

```text
YYYY_MM_DD-HH_MM_SS.mj_unroll
``` 

This means the viewer shows rollouts from oldest to newest.

---

### Features

1. Most features from [MuJoCo viewer](https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate)
2. Browse through trajectories. Use left/right arrow keys to switch through parallel environments and up/down for newer/older trajectories.
3. Live plotting. Use `SHIFT+M` to plot trajectory rewards and the contents of `state.metrics`. When there are more than 12 metrics, use `[` and `]` to page through them.
4. Pixel observations. Use `SHIFT+O` to overlay pixel observations if available. To use this feature, the observation must be a `dict` and the pixel keys must be prefixed with `pixels/`.

---

### Sharp bits

Some background on how rscope works: between policy updates, `rscope` unrolls multiple trajectories in parallel and then visualizes them on CPU. While this is simpler to implement and less expensive than tracing *training* runs like in IsaacLab, this and other implementation details lead to some unexpected gotchas:

- Typically, stochastic policies are used for evaluating training progress while deterministic ones are deployed. While you can use rscope on stochastic policies to get a feel for the agent's training exploration, we recommend [deterministic evals](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py#L232).
- Renders incorrectly for domain-randomized training because the loaded assets are from the nominal model definition.
- Plots metric keys 12 at a time and does not automatically filter shaping rewards.
- Visualizes only a limited number of pixel observations.
- Cannot capture curriculum progression during training, as curriculums depend on `state.info`, which is reset at the start of an evaluator run.
- Currently supports only PPO-based training.

---

### Contribution Guidelines

Please run the following before making a PR:

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```
