"""
Core N-body engine.

A gravitational N-body system is stored as three NumPy arrays (masses,
positions, velocities) and advanced in time with one of three explicit
integrators: forward Euler, semi-implicit Euler-Cromer, and fourth-order
Runge-Kutta (RK4).  All the pairwise force work is vectorised, so the same
code evolves as many bodies as you want. Adding a body is just adding an 
entry to the list handed to ``NBodySystem``.

The physics follows Newtonian gravity for point masses:

    a_i = sum_{j != i} G m_j (r_j - r_i) / |r_j - r_i|^3

and the conserved quantities (total energy, linear and angular momentum) are
provided so the integrators can be validated against conservation laws.
"""

from dataclasses import dataclass, field
import numpy as np

# Newtonian constant of gravitation (CODATA 2018), SI units m^3 kg^-1 s^-2.
G = 6.67430e-11

# The integrators understood by ``NBodySystem.step`` / ``simulate``.
METHODS = ("euler", "euler_cromer", "rk4")


@dataclass
class Body:
    """A single point mass.

    Positions are in metres and velocities in metres per second, expressed in
    whatever inertial frame the caller uses (the solar-system helpers use the
    barycentric frame).
    """

    name: str
    mass: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        self.mass = float(self.mass)


@dataclass
class Trajectory:
    """Sampled output of a simulation run.

    Every array is indexed by sample number.  ``positions`` and ``velocities``
    have shape ``(n_samples, n_bodies, 3)``; the conserved quantities are
    stored as time series so accuracy can be plotted directly.
    """

    names: list
    times: np.ndarray                 # (n_samples,)              seconds
    positions: np.ndarray             # (n_samples, n_bodies, 3)  metres
    velocities: np.ndarray            # (n_samples, n_bodies, 3)  m/s
    energy: np.ndarray                # (n_samples,)              joules
    linear_momentum: np.ndarray       # (n_samples, 3)            kg m/s
    angular_momentum: np.ndarray      # (n_samples, 3)            kg m^2/s
    method: str
    dt: float

    def index(self, name):
        """Return the body index for ``name`` (case-insensitive)."""
        lowered = [n.lower() for n in self.names]
        return lowered.index(name.lower())


class NBodySystem:
    """A set of gravitationally interacting point masses.

    Parameters
    ----------
    bodies : sequence of Body
        The bodies making up the system.
    gravitational_constant : float, optional
        Override for ``G`` (mainly useful for testing in scaled units).
    """

    def __init__(self, bodies, gravitational_constant=G):
        self.names = [b.name for b in bodies]
        self.mass = np.array([b.mass for b in bodies], dtype=float)
        self.pos = np.array([b.position for b in bodies], dtype=float)
        self.vel = np.array([b.velocity for b in bodies], dtype=float)
        self.G = float(gravitational_constant)
        self.time = 0.0

    # ---- construction helpers ------------------------------------------

    @property
    def n(self):
        """Number of bodies."""
        return len(self.mass)

    def copy(self):
        """Return an independent copy of the system (state included)."""
        clone = NBodySystem.__new__(NBodySystem)
        clone.names = list(self.names)
        clone.mass = self.mass.copy()
        clone.pos = self.pos.copy()
        clone.vel = self.vel.copy()
        clone.G = self.G
        clone.time = self.time
        return clone

    def to_barycentric(self):
        """Shift into the centre-of-mass frame in place and return self.

        Removes any net drift of the whole system so that total linear
        momentum is zero.
        """
        total = self.mass.sum()
        self.pos -= (self.mass[:, None] * self.pos).sum(0) / total
        self.vel -= (self.mass[:, None] * self.vel).sum(0) / total
        return self

    # ---- forces --------------------------------------------------------

    def accelerations(self, pos=None):
        """Gravitational acceleration on every body.

        Vectorised evaluation of ``a_i = sum_j G m_j (r_j - r_i)/r_ij^3``.
        The self-interaction term is removed by sending the diagonal
        separation to infinity, so its contribution is exactly zero.

        Parameters
        ----------
        pos : ndarray, optional
            Positions to evaluate at, shape ``(n, 3)``.  Defaults to the
            current positions; RK4 passes trial positions here.
        """
        if pos is None:
            pos = self.pos
        # r_ij[i, j] = r_j - r_i  ->  vector pointing from body i to body j.
        r_ij = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]      # (n, n, 3)
        dist2 = np.einsum("ijk,ijk->ij", r_ij, r_ij)             # (n, n)
        np.fill_diagonal(dist2, np.inf)                          # drop self term
        inv_dist3 = dist2 ** -1.5                                # (n, n)
        weight = self.G * inv_dist3 * self.mass[np.newaxis, :]   # (n, n)
        return np.einsum("ij,ijk->ik", weight, r_ij)            # (n, 3)

    # ---- integrators ---------------------------------------------------

    def _step_euler(self, dt):
        """Forward Euler: update position with the *old* velocity."""
        acc = self.accelerations()
        self.pos = self.pos + self.vel * dt
        self.vel = self.vel + acc * dt

    def _step_euler_cromer(self, dt):
        """Semi-implicit Euler-Cromer: update velocity first, then position
        with the *new* velocity.  This reordering is symplectic, giving
        bounded energy error instead of secular drift."""
        acc = self.accelerations()
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

    def _step_rk4(self, dt):
        """Classic fourth-order Runge-Kutta on the state (position, velocity).

        Four acceleration evaluations per step give O(dt^4) global error at
        roughly four times the cost of a single-evaluation method.
        """
        p0, v0 = self.pos, self.vel

        k1_p, k1_v = v0, self.accelerations(p0)
        k2_p, k2_v = v0 + 0.5 * dt * k1_v, self.accelerations(p0 + 0.5 * dt * k1_p)
        k3_p, k3_v = v0 + 0.5 * dt * k2_v, self.accelerations(p0 + 0.5 * dt * k2_p)
        k4_p, k4_v = v0 + dt * k3_v, self.accelerations(p0 + dt * k3_p)

        self.pos = p0 + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        self.vel = v0 + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    def step(self, method, dt):
        """Advance the system by one time step using ``method``."""
        stepper = {
            "euler": self._step_euler,
            "euler_cromer": self._step_euler_cromer,
            "rk4": self._step_rk4,
        }.get(method)
        if stepper is None:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
        stepper(dt)
        self.time += dt

    # ---- conserved quantities ------------------------------------------

    def kinetic_energy(self):
        return 0.5 * np.sum(self.mass * np.sum(self.vel ** 2, axis=1))

    def potential_energy(self):
        r_ij = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", r_ij, r_ij))
        iu = np.triu_indices(self.n, k=1)          # unordered pairs i < j
        mass_pairs = self.mass[iu[0]] * self.mass[iu[1]]
        return -self.G * np.sum(mass_pairs / dist[iu])

    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy()

    def linear_momentum(self):
        return (self.mass[:, None] * self.vel).sum(0)

    def angular_momentum(self):
        return np.cross(self.pos, self.mass[:, None] * self.vel).sum(0)

    def center_of_mass(self):
        return (self.mass[:, None] * self.pos).sum(0) / self.mass.sum()

    # ---- driver --------------------------------------------------------

    def simulate(self, method, dt, n_steps, sample_every=1):
        """Evolve the system and return a :class:`Trajectory`.

        The system is advanced for ``n_steps`` steps of size ``dt``; the state
        and conserved quantities are recorded at step 0 and then every
        ``sample_every`` steps. 

        Parameters
        ----------
        method : {'euler', 'euler_cromer', 'rk4'}
        dt : float
            Time step in seconds.
        n_steps : int
            Number of integration steps.
        sample_every : int
            Record one sample per this many steps.
        """
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")

        times, positions, velocities = [], [], []
        energy, lin_mom, ang_mom = [], [], []

        def record():
            times.append(self.time)
            positions.append(self.pos.copy())
            velocities.append(self.vel.copy())
            energy.append(self.total_energy())
            lin_mom.append(self.linear_momentum())
            ang_mom.append(self.angular_momentum())

        record()
        for i in range(n_steps):
            self.step(method, dt)
            if (i + 1) % sample_every == 0:
                record()

        return Trajectory(
            names=list(self.names),
            times=np.array(times),
            positions=np.array(positions),
            velocities=np.array(velocities),
            energy=np.array(energy),
            linear_momentum=np.array(lin_mom),
            angular_momentum=np.array(ang_mom),
            method=method,
            dt=dt,
        )

    def __repr__(self):
        return f"NBodySystem({self.n} bodies: {', '.join(self.names)})"
