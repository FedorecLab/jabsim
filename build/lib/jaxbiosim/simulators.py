# JAXBIOSIM/SIMULATORS.PY - functions for ODE simulation

# PACKAGE IMPORTS
# import and configure jax
from jax import config
config.update('jax_enable_x64', True)
import jax
# import everything else
import numpy as np
import scipy.integrate
import jax.numpy as jnp


# CUSTOM EULER SOLVER --------------------------------------------------------------------------------------------------

# Euler solver using jax.lax.scan
def scanning_euler(model_ode,
                   sim_state,  # simulator state
                   ode_step,  # simulation time step
                   ode_steps_in_savetimestep,
                   args):
    """
        Use Euler integration to get the simulator state to be recorded at the next saving point - to be used with
        jax.lax.scan.

        The ODE must have the signature ``model_ode(t, x, args)``. As in the original simulator, ``args`` is a tuple
        whose first item is the model parameter object.
    """

    # EULER STEP FUNCTION (with non-negativity constraints)
    def euler_step(step_cntr, t_x):
        return {
            # entries updated over the course of the Euler step
            't': t_x['t'] + ode_step,
            'x': jnp.maximum(t_x['x'] + ode_step * model_ode(t_x['t'], t_x['x'], args), 0),
        }

    # GET NEXT TIME POINT AND STATE (by Euler integration)
    next_sim_state = jax.lax.fori_loop(0, ode_steps_in_savetimestep, euler_step, sim_state)

    # RETURN UPDATED SIMULATOR STATE - AND CURRENT STATE VECTOR
    return next_sim_state, next_sim_state['x']


# Euler solver using jax.lax.fori_loop
def loopy_euler(model_ode,
                save_cntr,
                sim_state_record,
                ode_step,
                ode_steps_in_savetimestep,
                args):
    """Integrate one saving interval with Euler integration."""

    # EULER STEP FUNCTION (with non-negativity constraints)
    def euler_step(step_cntr, t_x):
        return {
            't': t_x['t'] + ode_step,
            'x': jnp.maximum(t_x['x'] + ode_step * model_ode(t_x['t'], t_x['x'], args), 0),
        }

    # GET THE PRESENT TIME POINT AND STATE
    last_t_x = {'t': sim_state_record['ts'][save_cntr - 1], 'x': sim_state_record['xs'][save_cntr - 1, :]}
    this_t_x = jax.lax.fori_loop(0, ode_steps_in_savetimestep, euler_step, last_t_x)

    # RETURN UPDATED SIMULATOR STATE
    return {'ts': sim_state_record['ts'], 'xs': sim_state_record['xs'].at[save_cntr, :].set(this_t_x['x'])}


# Fourth-order Runge-Kutta solver using jax.lax.fori_loop
def loopy_rk4(model_ode,
              save_cntr,
              sim_state_record,
              ode_step,
              ode_steps_in_savetimestep,
              args):
    """Integrate one saving interval with fourth-order Runge-Kutta integration."""

    # FOURTH-ORDER RUNGE-KUTTA STEP FUNCTION (with non-negativity constraints)
    def rk4_step(step_cntr, t_x):
        k1 = model_ode(t_x['t'], t_x['x'], args)
        k2 = model_ode(t_x['t'] + ode_step / 2, jnp.maximum(t_x['x'] + ode_step * k1 / 2, 0), args)
        k3 = model_ode(t_x['t'] + ode_step / 2, jnp.maximum(t_x['x'] + ode_step * k2 / 2, 0), args)
        k4 = model_ode(t_x['t'] + ode_step, jnp.maximum(t_x['x'] + ode_step * k3, 0), args)
        return {
            't': t_x['t'] + ode_step,
            'x': jnp.maximum(t_x['x'] + ode_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0),
        }

    # GET THE PRESENT TIME POINT AND STATE
    last_t_x = {'t': sim_state_record['ts'][save_cntr - 1], 'x': sim_state_record['xs'][save_cntr - 1, :]}
    this_t_x = jax.lax.fori_loop(0, ode_steps_in_savetimestep, rk4_step, last_t_x)

    # RETURN UPDATED SIMULATOR STATE
    return {'ts': sim_state_record['ts'], 'xs': sim_state_record['xs'].at[save_cntr, :].set(this_t_x['x'])}


# DIFFRAX UTILS --------------------------------------------------------------------------------------------------------
def stop_when_neg_x_cond(t, x, args, **kwargs):
    """Return True if any state variable is negative."""
    return jnp.any(x < 0)


def _save_times(tf, savetimestep, array_module):
    """Make saving times, including each endpoint exactly once."""
    if len(tf) != 2 or tf[1] <= tf[0]:
        raise ValueError('tf must contain two increasing time points')
    if savetimestep <= 0:
        raise ValueError('savetimestep must be positive')

    ts = array_module.arange(tf[0], tf[1], savetimestep)
    return array_module.concatenate((ts, array_module.asarray([tf[1]])))


# SIMULATOR FUNCTION ---------------------------------------------------------------------------------------------------
def sim(par, model_ode, x0, tf, savetimestep, simulator='scipy', **kwargs):
    """
        Simulate an ODE model.

        Args:
            par: model parameters
            model_ode: ODE function handle with signature model_ode(t, x, args)
            x0: initial condition
            tf: two-item simulation time span
            savetimestep: save the simulation every savetimestep time units
            simulator: one of 'scipy', 'diffrax', 'scan_euler', 'loopy_euler', or 'loopy_rk4'
            **kwargs: additional arguments specific to the simulator

        Returns:
            (ts, xs): saving times and system state at each saving time
    """

    if simulator == 'diffrax':
        try:
            import diffrax
        except ImportError as exc:
            raise ImportError("The 'diffrax' simulator requires the optional 'diffrax' dependency") from exc

        # define the ODE term and saving points
        term = diffrax.ODETerm(model_ode)
        ts = _save_times(tf, savetimestep, jnp).astype(jnp.float64)

        # define diffrax parameters if not specified
        solver = kwargs.get('solver', diffrax.Dopri5())
        tols = kwargs.get('tols', {'rtol': 1e-6, 'atol': 1e-9})
        stepsize_controller = diffrax.PIDController(rtol=tols['rtol'], atol=tols['atol'])
        dt0 = kwargs.get('dt0', 0.1)
        stop_event = diffrax.Event(stop_when_neg_x_cond) if kwargs.get('stop_when_neg', True) else None

        # solve the ODE
        sol = diffrax.diffeqsolve(term, solver,
                                  args=(par,),
                                  t0=tf[0], t1=tf[1], dt0=dt0, y0=jnp.asarray(x0),
                                  saveat=diffrax.SaveAt(ts=ts), max_steps=None,
                                  stepsize_controller=stepsize_controller, event=stop_event)
        return sol.ts, sol.ys

    elif simulator in ('scan_euler', 'loopy_euler', 'loopy_rk4'):
        ode_steps_in_savetimestep = int(kwargs.get('ode_steps_in_savetimestep', 1000))
        if ode_steps_in_savetimestep < 1:
            raise ValueError('ode_steps_in_savetimestep must be at least 1')

        args = (par,)
        ts = _save_times(tf, savetimestep, jnp).astype(jnp.float64)
        x0 = jnp.asarray(x0)

        # Fixed-step methods require equal saving intervals.
        if not np.allclose(np.diff(np.asarray(ts)), savetimestep):
            raise ValueError('fixed-step simulators require tf[1] - tf[0] to be divisible by savetimestep')

        ode_step = savetimestep / ode_steps_in_savetimestep

        if simulator == 'scan_euler':
            scan_step = lambda sim_state, t: scanning_euler(model_ode, sim_state, ode_step,
                                                             ode_steps_in_savetimestep, args)
            sim_state0 = {'t': jnp.asarray(tf[0]), 'x': x0}
            sim_state_final, saved_xs = jax.lax.scan(scan_step, sim_state0, ts[1:])
            xs = jnp.concatenate((x0[jnp.newaxis, :], saved_xs), axis=0)
        else:
            loop_function = loopy_euler if simulator == 'loopy_euler' else loopy_rk4
            loop_step = lambda save_cntr, record: loop_function(model_ode, save_cntr, record, ode_step,
                                                                 ode_steps_in_savetimestep, args)
            sim_state_record = {'ts': ts, 'xs': jnp.tile(x0, (ts.shape[0], 1))}
            sim_state_final = jax.lax.fori_loop(1, ts.shape[0], loop_step, sim_state_record)
            xs = sim_state_final['xs']

        return ts, xs

    elif simulator == 'scipy':
        args = (par,)
        term = lambda t, x: np.asarray(model_ode(t, x, args), dtype=float)
        ts = _save_times(tf, savetimestep, np).astype(float)

        solver = kwargs.get('solver', 'LSODA')
        tols = kwargs.get('tols', {'rtol': 1e-6, 'atol': 1e-9})
        dt0 = kwargs.get('dt0', None)

        result = scipy.integrate.solve_ivp(term,
                                           t_span=(tf[0], tf[1]), y0=np.asarray(x0, dtype=float), t_eval=ts,
                                           method=solver, rtol=tols['rtol'], atol=tols['atol'], first_step=dt0)
        if not result.success:
            raise RuntimeError('ODE simulation failed: ' + result.message)
        return result.t, result.y.T

    raise ValueError("simulator must be 'scipy', 'diffrax', 'scan_euler', 'loopy_euler', or 'loopy_rk4'")

