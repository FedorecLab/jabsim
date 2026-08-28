# jabsim

A jax-based package for simulating ODE models of biological systems,
where **all variables are non-negative**. This enforcement of non-negativity, which neither `scipy.solve_ivp` nor `diffrax`
solvers can give you, is why you may need jabsim.
This is achieved by clamping the state variables to zero whenever they are negative 

jabsim is powered by the [jax](https://github.com/jax-ml/jax) package for high-performance computing and parallelisation.
This means that jabsim simulations can be jit-compiled and parallelised on a GPU or TPU using `jax.vmap` or shard mapping.

## How to use jabsim

1. Make an ODE function which calculates the derivative $\frac{dx}{dt}$ from the arguments `t, x, par` in this order.
   - What do the arguments stand for?     
     - `t` is the time at this point in the simulation
     - `x` is the state vector at this time point
     - `args` is a tuple of extra arguments passed on to the ODE function.
   - **IMPORTANT**: if you don't know jax, there are a few differences to keep in mind:
     - If you normally use `numpy` functions in your ODE, import `jax.numpy` as `jnp` and use it instead of `np`. jax with `jax.numpy` will get automatically installed as a dependency of jabsim when you install it.
     - Be careful using loops and if-statements. If you're an amateur programmer, just avoid doing all that. Otherwise, have a look at the [JAX documentation](https://docs.jax.dev/en/latest/index.html) .
2. Import `jabsim` and call `jabsim.sim` to simulate. The arguments are as follows:
   - `par`: a list, array or dict of model parameters as in your ODE function
   - `model_ode`: the ODE function you created
   - `x0`: the initial state vector as a 1D `np.array` or `jnp.array`
   - `tf`: tuple, array or list. The ODE will be simulated 
   - `savetimestep`: interval between the time points at which the trajectory is saved
   - `simulator`: string specifyingthe simulation method to use
        - `"euler"`: Euler simulator.
        - `"rk4"`: Runge-Kutta 4th order simulator. Slower per ODE integration step but more accurate, hence allowing larger steps for the same accuracy.
   - `ode_steps_in_savetimestep`: number of ODE integration steps within one timestep
        - e.g. if `savetimestep=0.5` hours and `ode_step_in_savetimesteps=100`, there will be 100 integration steps per 0.5 hour, so the ODE integration step size will be 0.5/100=0.05 hours.
        - high `ode_steps_in_savetimestep` number increases accuracy but increases runtimes
   - `return_numpy`: if `True` (by default, it is) the output will be in the `np.array` format, otherwise it will be `jnp.array`.
3. Running `jabsim.sim()` will return the arrays `ts` and `xs` as `np.array` or `jnp.array`, as well as a boolean value `success`.
   - `ts`: array of timepoints between `tf[0]` and `tf[1]` with `savetimestep` hours, seconds or whatever units you are using between each two consecutive point
   - `xs`: system trajectory saved as an array at the time points in `ts` - axis 0 for time, axis 1 for entries in the state vector (i.e. `xs.shape[0]=len(ts)`).
   - `success`: boolean value; `True` if no entry in `xs` is `nan` or `inf`, `False` otherwise.

## Notes
- In practice, 500 steps per hour (e.g. `savetimestep=0.5, ode_steps_in_savetimestep=250` or `savetimestep=1.0, ode_steps_in_savetimestep=500`) works well for the RK4 solver. For the Euler solver, 1e4 steps per hours is reasonably good.
- For benchmarking, you can also set `simulator="scipy"` to simulate your ODE with `scipy.solve_ivp` (but without any of the delicious jax features of the solvers above). In that case, don't use the arguments `ode_steps_in_savetimestep` and `savetimestep`. Instead, you can *optionally* specify:
  - `solver`: string describing any solver which may be used with `scipy.solve_ivp`. By default, we have `solver="LSODA"`.
  - `tols`: dictionary of relative and absolute tolerances for the scipy solver. By default, `tols={'rtol': 1e-6, 'atol': 1e-9}`.
  - `dt0`: starting integration step size. By default, `dt0=0.1`.
- If you want to make use of jax parallelisation, make sure to set `return_numpy=False` so that the solver would operate with `jnp.array` objects only.

## Example

Let us integrate a simple one-dimensional ODE $\frac{dx}{dt} = a x^2$. For the initial condition $x_0=1$ and $a=0.4$, 
this has the analytical solution $x = \frac{1}{1-0.4t}$. This means we can verify that for  `savetimestep=0.5`, 
`jabsim.sim()` produces `ts=np.array([0, 0.5, 1.0])` and `xs=np.array([1.0, 1.25, 1.66666667])`. 
All entries in `xs` are finite, hence `success=True`.

```python
# import jabsim
import jabsim

# import jax.numpy for numpy operations
import jax.numpy as jnp

# our model ODE function returning a list of one element
def model_ode(t, x, args):
    # unpack args - get the dictiory of parameters
    par, = args
    
    # use jnp to square x 
    # (here you could just as well use x[0]**2, we just want to make a point)
    x_squared = jnp.square(x[0])
    
    # return dx/dt as a list - with one entry for a one-dimensional ODE
    return [par['a'] * x_squared]

# our dictionary of paramneters
par = {'a': 0.4}

ts, xs, success = jabsim.sim(
    model_ode=model_ode,
    args=(par,),
    x0=jnp.array([1.0]),
    tf=(0.0, 1.0),
    savetimestep=0.5,
    simulator='rk4',
    ode_steps_in_savetimestep=10,
)

# print the timne
print(ts)
print(xs)
print(success)
```

# Citation
If you find this package useful in your work, please cite the paper below:
code for its Showcase 2 served as jabsim's direct ideological precursor.
```bibtex
@article{Gallup2024,
	author = {Gallup, Olivia and Sechkar, Kirill and Towers, Sebastian and Steel, Harrison},
	title = {Computational Synthetic Biology Enabled through JAX: A Showcase},
	journal = {ACS Synth. Biol.},
	volume = {13},
	number = {9},
	pages = {3046},
	year = {2024},
	doi = {10.1021/acssynbio.4c00307}
}
```

The original JAX package should be cited as:
```bibtex
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Yash Katariya and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```