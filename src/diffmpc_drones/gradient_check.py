"""Check DiffMPC's gradient on one solve of the quadrotor example.

DiffMPC's backward pass ignores the curvature of the drone model. This script
measures what that costs on one settled solve. It computes the gradient of the
first control step's reward with respect to the cost weights four ways: from
DiffMPC, from a dense solve of the same equations with the same simplification,
from a dense solve that keeps the model's curvature, and by finite differences.
The first two should agree, and so should the last two. The finite differences
are noisy for the weights that start at 0.0001.

    uv run python -m diffmpc_drones.gradient_check
"""

import jax
import jax.numpy as jnp
import numpy as np
from diffmpc.dynamics.integrators import DiscretizationScheme, predict_next_state
from jax.flatten_util import ravel_pytree

from diffmpc_drones import rl_quadrotor as rq

# Entries of the flattened weights to compare: 0-3 are control weights, 4-16
# final state weights and 17-29 running state weights.
ENTRIES = [0, 3, 4, 7, 10, 14, 17, 20, 23, 27]
STEPS = rq.HORIZON - 1
NX, NU = 13, 4


def main() -> None:
    start = rq.start_states()[0]
    dynamics, solver, params, weights = rq.build(start, rq.SETTLING_SOLVER)
    program = solver.program
    theta, unravel = ravel_pytree(weights)

    def first_step_reward(control):
        state = predict_next_state(
            dynamics,
            params["discretization_resolution"],
            DiscretizationScheme.RUNGEKUTTA4,
            params,
            start,
            control,
        )
        return rq.reward(state, control, params)

    def solve(theta):
        guess = solver.initial_guess(params)
        quaternions = guess.states[:, 6:10]
        quaternions = quaternions / jnp.linalg.norm(quaternions, axis=1, keepdims=True)
        guess = guess._replace(
            states=guess.states.at[:, 6:10].set(quaternions),
            controls=jnp.repeat(jnp.array([[1.0, 0.0, 0.0, 0.0]]), rq.HORIZON, axis=0),
        )
        solution = solver.solve(guess, params, unravel(theta))
        return first_step_reward(solution.controls[0]), solution

    value = jax.jit(lambda t: solve(t)[0])
    solution = jax.jit(lambda t: solve(t)[1])(theta)
    from_diffmpc = jax.jit(jax.grad(lambda t: solve(t)[0]))(theta)
    print(
        f"solve: status {int(solution.status)}, {int(solution.num_iter)} iterations, "
        f"last change in the plan {float(solution.convergence_error):.1e}"
    )

    # The plan as one vector z, and the problem's cost and constraints of z.
    def unpack(z):
        states = z[: rq.HORIZON * NX].reshape(rq.HORIZON, NX)
        controls = z[rq.HORIZON * NX :].reshape(STEPS, NU)
        return states, jnp.concatenate([controls, jnp.zeros((1, NU))])

    def cost(z, theta):
        return program.cost(*unpack(z), {**params, **unravel(theta)})

    def constraints(z, theta):
        return jnp.ravel(program.equality_constraints(*unpack(z), {**params, **unravel(theta)}))

    z = jnp.concatenate([solution.states.ravel(), solution.controls[:STEPS].ravel()])
    jacobian = jax.jacfwd(constraints)(z, theta)
    # Multipliers from the first KKT condition, by least squares.
    multipliers = -jnp.linalg.solve(jacobian @ jacobian.T, jacobian @ jax.grad(cost)(z, theta))

    def kkt_residual(theta):
        stationarity = jax.grad(cost)(z, theta) + jax.jacfwd(constraints)(z, theta).T @ multipliers
        return jnp.concatenate([stationarity, constraints(z, theta)])

    residual_dtheta = jax.jacfwd(kkt_residual)(theta)
    reward_dz = jax.grad(lambda z: first_step_reward(unpack(z)[1][0]))(z)
    rhs = jnp.concatenate([reward_dz, jnp.zeros(multipliers.size)])

    def implicit_gradient(hessian):
        zeros = jnp.zeros((multipliers.size, multipliers.size))
        kkt = jnp.block([[hessian, jacobian.T], [jacobian, zeros]])
        return -residual_dtheta.T @ jnp.linalg.solve(kkt, rhs)

    cost_curvature = jax.hessian(cost)(z, theta)
    model_curvature = jax.hessian(lambda z: multipliers @ constraints(z, theta))(z)

    def finite_difference(i):
        step = 1e-2 * float(theta[i])
        return (float(value(theta.at[i].add(step))) - float(value(theta.at[i].add(-step)))) / (
            2 * step
        )

    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print("DiffMPC                    ", np.array(from_diffmpc)[ENTRIES])
    print("dense, curvature ignored   ", np.array(implicit_gradient(cost_curvature))[ENTRIES])
    print(
        "dense, curvature kept      ",
        np.array(implicit_gradient(cost_curvature + model_curvature))[ENTRIES],
    )
    print("finite differences         ", np.array([finite_difference(i) for i in ENTRIES]))


if __name__ == "__main__":
    main()
