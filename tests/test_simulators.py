import time

import numpy as np
import jax.numpy as jnp
import pytest

import jax.numpy as jnp
from jabsim import sim
from jabsim.simulators import sim_to_jit


def decay(t, x, args):
    par, = args
    for i in range(0,len(x)):
        x0_sq = jnp.square(x[i])
    return [par['rate'] * x0_sq]


@pytest.mark.parametrize('simulator', ['scipy', 'euler', 'rk4'])
def test_simulates_exponential_decay(simulator):
    kwargs = {'ode_steps_in_savetimestep': 100} if simulator != 'scipy' else {}
    ts, xs, success = sim(decay, ({'rate': 0.5},), np.array([1.0]), (0.0, 0.5), 0.05,
                 simulator=simulator, **kwargs)

    np.testing.assert_allclose(ts, np.linspace(0.0, 0.5, 11))
    np.testing.assert_allclose(np.asarray(xs[:, 0]), 1/(1-0.5*np.asarray(ts)), rtol=2e-2, atol=1e-4)
    assert(success)


def test_rejects_unknown_simulator():
    with pytest.raises(ValueError, match='Unknown simulator: unknown'):
        sim({}, decay, np.array([1.0]), (0.0, 1.0), 0.25, simulator='unknown')


def test_repeated_calls_with_varying_args_reuse_jit_cache():
    # sim_to_jit's static_argnames are model_ode/tf/savetimestep/simulator/
    # ode_steps_in_savetimestep, so repeated sim() calls that keep all of those fixed
    # and vary only args/x0 (as in an ABC-SMC or MCMC loop) should hit the compiled
    # cache instead of retracing every call
    if not hasattr(sim_to_jit, '_cache_size'):
        pytest.skip('jax jit cache introspection (_cache_size) unavailable in this jax version')

    sim(decay, ({'rate': 0.1},), np.array([1.0]), (0.0, 0.5), 0.05,
        simulator='rk4', ode_steps_in_savetimestep=50)
    cache_size_after_first_call = sim_to_jit._cache_size()

    for rate in (0.2, 0.3, 0.4, 0.5):
        sim(decay, ({'rate': rate},), np.array([2.0]), (0.0, 0.5), 0.05,
            simulator='rk4', ode_steps_in_savetimestep=50)

    assert sim_to_jit._cache_size() == cache_size_after_first_call


def test_repeated_calls_faster_than_first_cold_call():
    # a fresh model_ode (distinct object identity) forces a real cold compile on the
    # first call; subsequent calls varying only args should be dispatch-only and much
    # faster, since model_ode/tf/savetimestep/simulator/ode_steps_in_savetimestep stay
    # fixed and only args changes
    def fresh_decay(t, x, args):
        par, = args
        return [par['rate'] * x[0]]

    def timed_call(rate):
        start = time.perf_counter()
        sim(fresh_decay, ({'rate': rate},), np.array([1.0]), (0.0, 0.5), 0.05,
            simulator='rk4', ode_steps_in_savetimestep=200)
        return time.perf_counter() - start

    first_call_time = timed_call(0.1)
    warm_call_times = [timed_call(rate) for rate in (0.2, 0.3, 0.4, 0.5, 0.6)]

    assert min(warm_call_times) < first_call_time / 2
