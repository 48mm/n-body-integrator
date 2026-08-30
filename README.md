# N-Body Solar System Integrator

Numerical integration of the gravitational N-body problem, comparing the
**forward Euler**, **semi-implicit Euler-Cromer**, and **fourth-order
Runge-Kutta (RK4)** methods. This code produces the figures and tables in the
accompanying report *"Comparative Analysis of Numerical Integration Methods for
Solar System N-Body Simulations."* Schemes validated against NASA's JPL Ephemerides
alongside Lyapunov exponent calculation to analyse how the systems evolve.

**NOTE**: Report is available on request.

## Quick start

```bash
pip install -r requirements.txt
python main.py twobody          # two-body validation (no network needed)
python main.py all --no-show    # run every study, save CSV + PNG to results/
python test_nbody.py            # run the test suite
```

Every study is launched from `main.py`:

```
python main.py <study> [--source jpl|demo|auto] [--outdir DIR] [--dt SECONDS] [--no-show]

studies:
  twobody       Two-body circular orbit vs Kepler's law
  conservation  Energy & momentum over 30 days
  longterm      Energy drift over 100 years
  validation    Position/velocity vs ephemerides, 30 days
  chaos         Lyapunov / sensitivity analysis
  benchmark     Cost per step for each method
  all           Run all of the above
```

Each run writes a CSV of the numbers and a PNG of the figure into `--outdir`
(default `results/`). Use `--no-show` for batch runs.

## Initial conditions (`--source`)

Studies that need solar-system state vectors get them from one of two sources:

* `jpl`  - NASA [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/), downloaded
  over HTTP and **cached** under `cache/` so each (body, date) is fetched only
  once. This is the source used for the published results.
* `demo` - a network-free approximation: circular orbits at the planets' mean
  radii. Not physically precise, but a genuine conservative N-body system, so
  it is useful for running offline (and it backs the tests).
* `auto` - try JPL by default, fall back to `demo` if the network is down.

## Files

| File              | Purpose                                                              |
|-------------------|----------------------------------------------------------------------|
| `nbody.py`        | Core engine: `Body`, `NBodySystem`, the three integrators, conserved quantities, and the shared `simulate()` loop. Vectorised with NumPy. |
| `ephemerides.py`  | Solar-system initial conditions (JPL Horizons with disk cache + offline demo). |
| `analysis.py`     | The five studies plus the timing benchmark. Each returns data, writes a CSV and draws its figure. |
| `main.py`         | (Now) sole housing that maps each study to a function. (No more duplicated loops!)                                     |
| `test_nbody.py`   | Quantitative tests (conservation laws, Kepler period, RK4 convergence, bug regression). |

## Adding a new body

Add one line to the `BODIES` table in `ephemerides.py`:

```python
BODIES = {
    ...
    "pluto": ("999", 1.303e22, 5.906e12),   # Horizons id, mass [kg], mean radius [m]
}
```

Then include its name in any study, e.g. via the `names` argument, and every
part of the pipeline (fetching, caching, simulating, plotting) works for it
with no further changes:

```python
from nbody import NBodySystem, Body
system = NBodySystem([Body("Sun", 1.989e30), Body("Earth", 5.97e24, [1.496e11, 0, 0], [0, 29780, 0])])
traj = system.simulate("rk4", dt=3600, n_steps=8760)
```

## Notes on the methods

* **Euler** - first order; not symplectic so energy drifts and orbits decay.
* **Euler-Cromer** - first order but symplectic, thus energy error stays bounded,
  and it conserves linear and angular momentum much more precisely.
* **RK4** - fourth order, thus far more accurate short-term. ~4.5x the cost per step,
  but not symplectic causing slow long-term energy drift.

The acceleration is evaluated as a single vectorised pairwise sum shared by all
three integrators, so all methods and all system sizes use the same routine.

## TO DO:
- [x] Fix `getEphemerides` to avoid re-fetching same state vectors as live GET leads to throttling.
- [x] Fix acceleration accumulation bug with `Particle.updateGravitationalAcceleration` that overwrites `self.acceleration` after each call instead of adding it, leading to equally non-physical energy curves for Euler and Euler-Cromer.
- [x] Massively rework file structure so that each analysis can be in one main housing instead of separate files, hopefully addressing duplicate loop problem.
- [x] Continue reworking structure so that conservation tests consolidated to one file.
- [x] Implement vectorised array operations to handle force maths instead of slow `list` of `Particle` objects.
- [x] Implement caching and offline demo system in case JPL network down. 
- [x] Implement actual `argparse` CLI instead of just changing code between runs.
- [x] Rework `BODIES` structure so that all details per body are stored in one place instead of spread out, so that system can be edited easily.
- [x] Clean out leftover test + dependencies from earlier assignments / drafts.
- [ ] Rework report figures and captions.
- [ ] Update report conclusions based on updated energy conservation simulation.
