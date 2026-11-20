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
uv run diffmpc-drones    # crashes at the first training update
```

**Training currently crashes.** JAX is now 0.12.1, and the first update fails
with:

```
ValueError: under vmap, the custom_vjp function solve has a primal output that
is shared across the batch (unbatched) with a nonzero right cotangent, and its
bwd rule produced a batched left cotangent for an unbatched primal input.
```

The script is unchanged from the previous commit, where JAX was pinned to 0.7.2
and training ran for 20 updates, with the reward going from -242.00 to -131.86.
The notebook, run unchanged, crashes the same way on JAX 0.12.1. The solve
before training still runs.

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
