import jax.numpy as jnp
import pytest

from diffmpc_drones import rl_quadrotor as rq


def test_start_state_is_the_notebooks():
    state = rq.start_states()[0]
    assert state.shape == (13,)
    # Seed 0 of NumPy's legacy generator, in the notebook's order of draws.
    assert jnp.allclose(state[:3], jnp.array([0.39050803, 1.72151493, 0.82210701]))
    assert jnp.isclose(jnp.linalg.norm(state[6:10]), 1.0)


def test_rollout_reward_is_finite():
    starts = rq.start_states()
    dynamics, solver, params, weights = rq.build(starts[0])
    total = rq.rollout(dynamics, solver, params, starts[0], weights)
    assert jnp.isfinite(total) and total < 0


@pytest.mark.slow
@pytest.mark.xfail(
    raises=ValueError, strict=True, reason="JAX 0.12.1 refuses to batch the gradient"
)
def test_one_update_changes_the_weights():
    starts = rq.start_states()
    dynamics, solver, params, weights = rq.build(starts[0])
    new, value = rq.make_update(dynamics, solver, params, starts)(dict(weights))
    assert jnp.isfinite(value)
    assert any(not jnp.allclose(new[k], weights[k]) for k in weights)
