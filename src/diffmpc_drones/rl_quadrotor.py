"""DiffMPC's quadrotor example, as a script.

This follows `notebooks/RL_quadrotor.ipynb` in DiffMPC step for step, with the
same settings and the same random start. It solves the hover problem once with
the starting cost weights, then trains the weights by gradient ascent on the
reward of a closed-loop flight. The notebook's plots are left out.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from diffmpc.dynamics.integrators import DiscretizationScheme, predict_next_state
from diffmpc.dynamics.quadrotor_dynamics import QuadrotorDynamics
from diffmpc.problems.optimal_control_problem import OptimalControlProblem
from diffmpc.solvers.sqp import SQPSolver
from diffmpc.utils.load_params import load_problem_params, load_solver_params

jax.config.update("jax_enable_x64", True)

# The notebook's settings.
HORIZON = 20
SIM_STEPS = 10
UPDATES = 20
RATE = 1e-4
BATCH = 1
SEED = 0

WEIGHT_KEYS = (
    "weights_penalization_reference_state_trajectory",
    "weights_penalization_control_squared",
    "weights_penalization_final_state",
)


def start_states(batch: int = BATCH, seed: int = SEED) -> jnp.ndarray:
    """The notebook's random start states, drawn in the same order from NumPy."""
    np.random.seed(seed)
    states = np.zeros((batch, 13))
    states[:, :3] = -4.0 + 8.0 * np.random.rand(batch, 3)  # position, m
    states[:, 3:6] = -0.1 + 0.2 * np.random.rand(batch, 3)  # velocity, m/s
    # One tilted attitude, shared by the whole batch.
    delta = 0.5 * np.random.randn(4)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0]) + delta / np.linalg.norm(delta)
    states[:, 6:10] = quaternion / np.linalg.norm(quaternion)
    return jnp.array(states)


def build(initial_state):
    """The notebook's problem, solver and trainable cost weights."""
    params = load_problem_params("quadrotor.yaml")
    steps = HORIZON - 1  # DiffMPC counts a horizon of N as N + 1 states
    params["horizon"] = steps
    params["mass"] = 0.1 * jnp.ones(1)
    params["inertia"] = jnp.array([0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01, 0.1])
    params["initial_state"] = initial_state
    for key in ("reference_state_trajectory", "reference_control_trajectory"):
        params[key] = jnp.repeat(params[key][:1], steps + 1, axis=0)

    solver_params = load_solver_params("sqp.yaml")
    solver_params["num_scp_iteration_max"] = 50
    solver_params["pcg"]["tol_epsilon"] = 1.0e-12
    solver_params["linesearch"] = True

    dynamics = QuadrotorDynamics()
    solver = SQPSolver(
        program=OptimalControlProblem(dynamics=dynamics, params=params), params=solver_params
    )
    weights = {key: params[key] for key in WEIGHT_KEYS}
    return dynamics, solver, params, weights


def reward(state, control, params):
    """Negative squared distance from the goal, plus a cost on control effort."""
    goal = params["reference_state_trajectory"][0]
    trim = params["reference_control_trajectory"][0]
    return -(jnp.sum((state[:3] - goal[:3]) ** 2) + 0.1 * jnp.sum((control - trim) ** 2))


def rollout(dynamics, solver, params, state, weights):
    """Fly SIM_STEPS control steps in closed loop and return the summed reward."""

    def first_guess(state):
        guess = solver.initial_guess({**params, "initial_state": state})
        quaternions = guess.states[:, 6:10]
        quaternions = quaternions / jnp.linalg.norm(quaternions, axis=1, keepdims=True)
        controls = jnp.repeat(jnp.array([[1.0, 0.0, 0.0, 0.0]]), HORIZON, axis=0)
        return guess._replace(states=guess.states.at[:, 6:10].set(quaternions), controls=controls)

    def step(carry, _):
        state, solution, total = carry
        step_params = {**params, "initial_state": state}
        solution = solver.solve(solution, step_params, weights)
        control = solution.controls[0]
        state = predict_next_state(
            dynamics,
            step_params["discretization_resolution"],
            DiscretizationScheme.RUNGEKUTTA4,
            step_params,
            state,
            control,
        )
        return (state, solution, total + reward(state, control, params)), None

    carry = (state, first_guess(state), 0.0)
    (_, _, total), _ = jax.lax.scan(step, carry, None, length=SIM_STEPS)
    return total


def make_update(dynamics, solver, params, starts):
    """One gradient ascent step on the cost weights, as the notebook takes it."""

    def batch_reward(weights):
        totals = jax.vmap(lambda s: rollout(dynamics, solver, params, s, weights))(starts)
        return jnp.sum(totals)

    @jax.jit
    def update(weights):
        value, grad = jax.value_and_grad(batch_reward)(weights)
        # Cost weights are kept positive.
        weights = {
            k: jnp.maximum(1e-5, weights[k] + RATE * jnp.nan_to_num(grad[k])) for k in weights
        }
        return weights, value

    return update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--updates", type=int, default=UPDATES, help="gradient updates")
    args = parser.parse_args()

    starts = start_states()
    dynamics, solver, params, weights = build(starts[0])
    print(f"JAX {jax.__version__} on {jax.devices()[0]}")

    baseline = solver.solve(solver.initial_guess(params), params, weights)
    print(
        f"solve with the starting weights: status {int(baseline.status)}, "
        f"{int(baseline.num_iter)} iterations"
    )

    update = make_update(dynamics, solver, params, starts)
    start = time.monotonic()
    for i in range(args.updates):
        weights, value = update(weights)
        print(
            f"update {i + 1}/{args.updates}: reward {float(value):.4f}, "
            f"{time.monotonic() - start:.0f} s"
        )


if __name__ == "__main__":
    main()
