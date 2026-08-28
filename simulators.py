# JABSIM/SIMULATORS.PY - functions for ODE simulation

# PACKAGE IMPORTS
# import configured jax
import jax
# import everything else
import numpy as np
import scipy.integrate
import jax.numpy as jnp


# SOLVERS --------------------------------------------------------------------------------------------------------------

# Euler solver using jax.lax.fori_loop
def loopy_euler(model_ode,
                save_cntr,
                sim_state_record,
                ode_step,
                ode_steps_in_savetimestep,
                args):
    """
        Use Euler integration to get the simulator state to be recorded at the current saving point into
        sim_state_record - to be used with jax.lax.fori_loop

        Args:
            model_ode: ODE function handle
            save_cntr: current saving point
            sim_state_record: dictionary containing a record of simulation time points and state vector values
            ode_step: simulation time step duration
            ode_steps_in_savetimestep: number of ODE integration steps between each recording
            args: additional arguments to pass to the ODE function

        Returns:
            new_sim_state_record: record of simulation time points and state vector values - now with the right x at xs[save_cntr,:]
    """

    # EULER STEP FUNCTION (with non-negativity constraints)
    def euler_step(step_cntr, t_x):
        return {
            # entries updated over the course of the Euler step
            't': t_x['t'] + ode_step,
            'x': jnp.maximum(t_x['x'] + ode_step * model_ode(t_x['t'], t_x['x'], args),0),
        }
    # GET THE PRESENT TIME POINT AND STATE
    last_t_x = {'t': sim_state_record['ts'][save_cntr-1], 'x': sim_state_record['xs'][save_cntr-1, :]}
    this_t_x = jax.lax.fori_loop(0, ode_steps_in_savetimestep, euler_step, last_t_x)

    # RETURN UPDATED SIMULATOR STATE
    new_sim_state_record = {'ts': sim_state_record['ts'], 'xs': sim_state_record['xs'].at[save_cntr, :].set(this_t_x['x'])}
    return new_sim_state_record

# Fourth-order Runge-Kutta solver using jax.lax.fori_loop
def loopy_rk4(model_ode,
              save_cntr,
              sim_state_record,
              ode_step,
              ode_steps_in_savetimestep,
              args):
    """
        Use fourth-order Runge-Kutta integration to get the simulator state to be recorded at the current saving point
        into sim_state_record - to be used with jax.lax.fori_loop

        Args:
            model_ode: ODE function handle
            save_cntr: current saving point
            sim_state_record: dictionary containing a record of simulation time points and state vector values
            ode_step: simulation time step duration
            ode_steps_in_savetimestep: number of ODE integration steps between each recording
            args: additional arguments to pass to the ODE function

        Returns:
            new_sim_state_record: record of simulation time points and state vector values - now with the right x at xs[save_cntr,:]
    """

    # FOURTH-ORDER RUNGE-KUTTA STEP FUNCTION (with non-negativity constraints)
    def rk4_step(step_cntr, t_x):
        k1 = model_ode(t_x['t'], t_x['x'], args)
        k2 = model_ode(t_x['t'] + ode_step / 2, jnp.maximum(t_x['x'] + ode_step * k1 / 2, 0), args)
        k3 = model_ode(t_x['t'] + ode_step / 2, jnp.maximum(t_x['x'] + ode_step * k2 / 2, 0), args)
        k4 = model_ode(t_x['t'] + ode_step, jnp.maximum(t_x['x'] + ode_step * k3, 0), args)
        return {
            # entries updated over the course of the Runge-Kutta step
            't': t_x['t'] + ode_step,
            'x': jnp.maximum(t_x['x'] + ode_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0),
        }

    # GET THE PRESENT TIME POINT AND STATE
    last_t_x = {'t': sim_state_record['ts'][save_cntr-1], 'x': sim_state_record['xs'][save_cntr-1, :]}
    this_t_x = jax.lax.fori_loop(0, ode_steps_in_savetimestep, rk4_step, last_t_x)

    # RETURN UPDATED SIMULATOR STATE
    new_sim_state_record = {'ts': sim_state_record['ts'], 'xs': sim_state_record['xs'].at[save_cntr, :].set(this_t_x['x'])}
    return new_sim_state_record


# SIMULATOR FUNCTION ---------------------------------------------------------------------------------------------------
# simulator function
def sim(model_ode, args, x0, tf, savetimestep, simulator='rk4', return_numpy=True, **kwargs):
    """
        Simulate an ODE model.

        Args:
            model_ode: ODE function handle
            args: extra arguments for the ODE function
            x0: initial condition
            tf: time span of the simulation
            savetimestep: saving the siulation every savetimestep hours
            simulator: simulation method
            return_numpy: True if returning numpy arrays, False is jax.numpy arrays
            **kwargs: additional arguments to pass to the simulation method - specific to the simulator

        Returns:
            xs: system state at each time point specfied
    """

    if (simulator == 'euler'):
        # define custom Euler parameters if not specified
        ## number of ODE integration steps between each record point
        ode_steps_in_savetimestep = kwargs.get('ode_steps_in_savetimestep', 1e4)

        # define the time points at which we save the solution
        ts = jnp.concatenate((jnp.arange(tf[0], tf[1], savetimestep), jnp.array([tf[1]])), dtype=jnp.float64)

        # calculate the time step for the Euler integration
        ode_step = savetimestep / ode_steps_in_savetimestep

        # make the model ode function return a jnp.array
        model_ode_jnp = lambda t, x, args: jnp.array(model_ode(t, x, args))

        # make the retrieval of next x a lambda-function for jax.lax.scanning
        loop_step = lambda save_cntr, sim_state_record: loopy_euler(model_ode_jnp,
                                                                    save_cntr,
                                                                    sim_state_record,  # simulator state
                                                                    ode_step,  # simulation time step
                                                                    int(ode_steps_in_savetimestep), # number of ODE integration steps between each recording
                                                                    args)

        # initalise the simulator state: (t, x) - x initialised with initial conditions
        sim_state_record = {'ts': ts, 'xs': jnp.tile(jnp.array(x0), (ts.shape[0], 1))}

        # simulate
        sim_state_rec_final = jax.lax.fori_loop(1, ts.shape[0], loop_step, sim_state_record)
        xs = sim_state_rec_final['xs']

        # check for simulation success (i.e. no nans or infs in x)
        success = not bool(jnp.any(jnp.isnan(xs)) or jnp.any(jnp.isinf(xs)))

        # return numpy or jax.numpy arrays
        if(return_numpy):
            return np.array(ts), np.array(xs), success
        else:
            return ts, xs, success

    elif (simulator == 'rk4'):
        # define custom fourth-order Runge-Kutta parameters if not specified
        ## number of ODE integration steps between each record point
        ode_steps_in_savetimestep = kwargs.get('ode_steps_in_savetimestep', 1e4)

        # define the time points at which we save the solution
        ts = jnp.concatenate((jnp.arange(tf[0], tf[1], savetimestep), jnp.array([tf[1]])), dtype=jnp.float64)

        # calculate the time step for the fourth-order Runge-Kutta integration
        ode_step = savetimestep / ode_steps_in_savetimestep

        # make the model ode function return a jnp.array
        model_ode_jnp = lambda t, x, args: jnp.array(model_ode(t, x, args))

        # make the retrieval of next x a lambda-function for jax.lax.fori_loop
        loop_step = lambda save_cntr, sim_state_record: loopy_rk4(model_ode_jnp,
                                                                  save_cntr,
                                                                  sim_state_record,  # simulator state
                                                                  ode_step,  # simulation time step
                                                                  int(ode_steps_in_savetimestep), # number of ODE integration steps between each recording
                                                                  args)

        # initalise the simulator state: (t, x) - x initialised with initial conditions
        sim_state_record = {'ts': ts, 'xs': jnp.tile(jnp.array(x0), (ts.shape[0], 1))}

        # simulate
        sim_state_rec_final = jax.lax.fori_loop(1, ts.shape[0], loop_step, sim_state_record)
        xs = sim_state_rec_final['xs']

        # check for simulation success (i.e. no nans or infs in x)
        success = not bool(jnp.any(jnp.isnan(xs)) or jnp.any(jnp.isinf(xs)))

        # return numpy or jax.numpy arrays
        if (return_numpy):
            return np.array(ts), np.array(xs), success
        else:
            return ts, xs, success

    elif (simulator == 'scipy'):
        # define ODE integration term
        term = lambda t, x: model_ode(t, x, args)

        # define the time points at which we save the solution
        ts = np.concatenate((np.arange(tf[0], tf[1], savetimestep), np.array([tf[1]])))

        # define diffrax parameters if not specified
        ## ODE solver
        solver = kwargs.get('solver', 'LSODA')
        ## ODE integration tolerances to specify the step size controller
        tols = kwargs.get('tols', {'rtol': 1e-6, 'atol': 1e-9})
        ## initial time step
        dt0 = kwargs.get('dt0', 0.1)

        # solve the ODE
        result = scipy.integrate.solve_ivp(term,
                                           t_span=(tf[0], tf[-1]),
                                           y0=x0,
                                           t_eval=ts,
                                           method=solver,
                                           rtol=tols['rtol'], atol=tols['atol'],
                                           first_step=dt0)
        xs = (result.y).T

        # return numpy or jax.numpy arrays
        if (return_numpy):
            return ts, xs, result.success
        else:
            return jnp.array(ts), jnp.array(xs), result.success

    else:
        raise ValueError("Unknown simulator: {}".format(simulator))
