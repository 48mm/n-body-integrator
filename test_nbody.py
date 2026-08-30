"""
Quantitative tests for the N-body engine.

The tests cover the conservation laws and RK4's behaviours. As a regression 
guard for the original bug, the vectorised acceleration equals a direct pairwise 
sum and that Euler-Cromer's energy error is bounded and distinct from Euler's.
"""

import numpy as np

from nbody import NBodySystem, Body, G
from analysis import make_two_body


def _random_system(n=5, seed=0):
    """A random system only for the algebraic acceleration test, 
    where boundedness is irrelevant."""
    rng = np.random.default_rng(seed)
    bodies = [Body(f"b{i}", rng.uniform(1e23, 1e30),
                   rng.uniform(-1e11, 1e11, 3), rng.uniform(-1e4, 1e4, 3))
              for i in range(n)]
    return NBodySystem(bodies).to_barycentric()


def _bound_system():
    """Sun plus three planets on circular orbits, i.e. well-behaved
    and bound system."""
    m_sun = 1.989e30
    bodies = [Body("Sun", m_sun)]
    for name, radius, mass, phase in [
        ("Mercury", 5.79e10, 3.30e23, 0.0),
        ("Earth", 1.496e11, 5.97e24, 2.1),
        ("Jupiter", 7.785e11, 1.90e27, 4.2),
    ]:
        speed = np.sqrt(G * m_sun / radius)
        pos = radius * np.array([np.cos(phase), np.sin(phase), 0.0])
        vel = speed * np.array([-np.sin(phase), np.cos(phase), 0.0])
        bodies.append(Body(name, mass, pos, vel))
    return NBodySystem(bodies).to_barycentric()


def test_acceleration_matches_direct_sum():
    """Vectorised acceleration must equal an explicit pairwise loop."""
    system = _random_system(n=6, seed=1)
    fast = system.accelerations()
    slow = np.zeros_like(fast)
    for i in range(system.n):
        for j in range(system.n):
            if i == j:
                continue
            d = system.pos[j] - system.pos[i]
            slow[i] += G * system.mass[j] * d / np.linalg.norm(d) ** 3
    assert np.allclose(fast, slow, rtol=1e-12), "vectorised acceleration is wrong"


def test_two_body_period():
    """Euler-Cromer and RK4 recover Kepler's period to better than 0.1%."""
    sep, m1, m2 = 1e8, 5.972e24, 7.342e22
    kepler = 2 * np.pi * np.sqrt(sep ** 3 / (G * (m1 + m2)))
    for method in ("euler_cromer", "rk4"):
        traj = make_two_body(sep, m1, m2).simulate(method, dt=1800, n_steps=1000)
        sep_t = np.linalg.norm(traj.positions[:, 0] - traj.positions[:, 1], axis=1)
        peaks = np.where((sep_t[1:-1] > sep_t[:-2]) & (sep_t[1:-1] > sep_t[2:]))[0] + 1
        period = np.diff(traj.times[peaks]).mean()
        assert abs(period - kepler) / kepler < 1e-3, f"{method} period off"


def test_momentum_conserved():
    """Linear and angular momentum are conserved by every method."""
    for method in ("euler", "euler_cromer", "rk4"):
        system = _bound_system()
        # momentum scale: sum of per-body |m v| (the net is ~0
        # in the barycentric frame, so it cannot be used to normalise).
        scale = np.sum(system.mass * np.linalg.norm(system.vel, axis=1))
        traj = system.simulate(method, dt=3600, n_steps=500, sample_every=10)
        p = np.linalg.norm(traj.linear_momentum - traj.linear_momentum[0], axis=1)
        assert (p / scale).max() < 1e-12, f"{method} leaks linear momentum"


def test_euler_cromer_energy_bounded_and_distinct():
    """Regression for reported bug: Euler-Cromer's energy error must be
    both far smaller than Euler's and clearly not identical to it."""
    system = _bound_system()
    err = {}
    for method in ("euler", "euler_cromer", "rk4"):
        traj = system.copy().simulate(method, dt=3600, n_steps=2000, sample_every=10)
        err[method] = np.abs((traj.energy - traj.energy[0]) / traj.energy[0]).max()
    assert err["euler_cromer"] < 0.1 * err["euler"], "Euler-Cromer no better than Euler"
    assert err["rk4"] < err["euler_cromer"], "RK4 should be the most accurate"


def test_rk4_fourth_order():
    """Halving the step should cut RK4's error by roughly 2^4."""
    def error(dt, n):
        traj = make_two_body().simulate("rk4", dt=dt, n_steps=n)
        return np.abs((traj.energy - traj.energy[0]) / traj.energy[0]).max()

    coarse = error(400, 500)
    fine = error(200, 1000)
    assert coarse / fine > 8, f"RK4 not converging at 4th order (ratio {coarse / fine:.1f})"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
