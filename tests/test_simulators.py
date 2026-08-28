import numpy as np
import jax.numpy as jnp
import pytest

import jax.numpy as jnp
from jabsim import sim


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
