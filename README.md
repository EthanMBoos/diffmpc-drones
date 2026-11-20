# diffmpc-drones

diffmpc-drones tunes a drone's model predictive controller by training it.
It flies a quadrotor toward a hover, scores the flight with a reward, and
adjusts the controller's cost weights to raise that reward.

The solver is Toyota Research Institute's
[DiffMPC](https://github.com/ToyotaResearchInstitute/diffmpc)
([paper](https://arxiv.org/abs/2510.06179)). It solves the controller's
problem and returns the gradient of the solution with respect to the
controller's parameters, which is what makes the training possible.

So far this repo is DiffMPC's quadrotor example and nothing more. The script
follows `notebooks/RL_quadrotor.ipynb` step for step, with the same settings and
the same random start. Thrust limits, wind, and
randomization are the work ahead; see [docs/TODO.md](docs/TODO.md).

## Try it

With [uv](https://docs.astral.sh/uv/) installed:

```sh
git clone https://github.com/EthanMBoos/diffmpc-drones.git
cd diffmpc-drones
git submodule update --init
uv sync
uv run diffmpc-drones    # 20 training updates, about 3.5 minutes on a CPU
```

The reward goes from -241.91 at the first update to -130.90 at the last, and it
does not rise at every update. This was run on an Apple M-series CPU with JAX
0.12.1 and has not been run on a GPU.

The script differs from the notebook in one line. DiffMPC keeps a warm start for
its backward solve in a field of the starting guess, and JAX 0.12.1 crashes when
that field is the same for every drone in a batch. The script adds zero times
the drone's state to the field, which makes it one per drone without changing
its value. On JAX 0.7.2 the script prints the same rewards with and without that
line; there the reward goes from -242.00 to -131.86, so the small difference
above comes from the JAX version.

Do not pass `--recursive` to the submodule command. DiffMPC has a submodule of
its own that this repo does not use.

## Development

```sh
uv run ruff check . && uv run ruff format .
uv run pytest -m "not slow"   # skips the test that compiles a training update
uv run pytest
```

## DiffMPC's license

DiffMPC is licensed
[CC BY-NC-ND 4.0](https://github.com/ToyotaResearchInstitute/diffmpc/blob/main/LICENSE.txt).
This repo uses it unchanged, as a submodule, with its own code written around
it. Do not edit the files under `third_party/diffmpc` or copy them into `src/`,
because the license does not allow publishing modified versions. A change to the
solver itself would be written from the paper, as new code.
