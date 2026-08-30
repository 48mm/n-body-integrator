"""
Solar-system initial conditions.

Two sources of state vectors (position + velocity) are supported:

* ``"jpl"``  - NASA JPL Horizons, queried over HTTP and cached on disk so a
  given (body, date) is only downloaded once.
* ``"demo"`` - Network-free approximation built from circular orbits at the
  planets' mean orbital radii.  Useful for running rough estimates for the 
  conservation / long-term / chaos analyses offline.

``load_system("auto", ...)`` tries JPL first and silently falls back to the
demo system if the network is unavailable.
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import requests

from nbody import Body, NBodySystem, G

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# name -> (Horizons id, mass [kg], mean orbital radius [m] for the demo system).
# The Sun's radius is 0 (it sits at the centre); masses are JPL fact-sheet values.
BODIES = {
    "sun":     ("10",  1.98892e30, 0.0),
    "mercury": ("199", 3.3011e23,  5.791e10),
    "venus":   ("299", 4.8675e24,  1.0821e11),
    "earth":   ("399", 5.97237e24, 1.4960e11),
    "mars":    ("499", 6.4171e23,  2.2794e11),
    "jupiter": ("599", 1.8982e27,  7.7857e11),
    "saturn":  ("699", 5.6834e26,  1.4335e12),
    "uranus":  ("799", 8.6810e25,  2.8725e12),
    "neptune": ("899", 1.02413e26, 4.4951e12),
}

DEFAULT_BODIES = list(BODIES)  # Sun + eight planets, in orbital order.


# ---------------------------------------------------------------------------
# JPL Horizons
# ---------------------------------------------------------------------------

def _cache_path(name, date, center):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{name}_{date}_{center}".replace("@", "at").replace("'", "")
    return os.path.join(CACHE_DIR, tag + ".json")


def fetch_state(name, date, center="500@0"):
    """Return ``(position, velocity)`` in SI units for ``name`` on ``date``.

    Positions are metres and velocities metres/second in the requested frame
    (default ``500@0`` = solar-system barycentre).  Results are cached under
    ``cache/`` so repeated calls do not hit the network again.
    """
    name = name.lower()
    if name not in BODIES:
        raise ValueError(f"Unknown body {name!r}; known: {list(BODIES)}")

    path = _cache_path(name, date, center)
    if os.path.exists(path):
        with open(path) as fh:
            cached = json.load(fh)
        return np.array(cached["position"]), np.array(cached["velocity"])

    stop = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "format": "text", "COMMAND": f"'{BODIES[name][0]}'", "EPHEM_TYPE": "VECTORS",
        "CENTER": f"'{center}'", "START_TIME": f"'{date}'", "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": "'1d'", "VEC_TABLE": "2", "CSV_FORMAT": "YES",
    }
    response = requests.get(HORIZONS_URL, params=params, timeout=60)
    response.raise_for_status()
    position, velocity = _parse_vectors(response.text)

    with open(path, "w") as fh:
        json.dump({"position": position.tolist(), "velocity": velocity.tolist()}, fh)
    return position, velocity


def _parse_vectors(text):
    """Pull the first state vector out of a Horizons CSV response (km, km/s)."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if "$$SOE" in ln) + 1
        end = next(i for i, ln in enumerate(lines) if "$$EOE" in ln)
    except StopIteration:
        raise RuntimeError("Could not parse Horizons response (no $$SOE/$$EOE).")

    for line in lines[start:end]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            values = [float(p) for p in parts[2:8]]
            position = np.array(values[:3]) * 1e3   # km  -> m
            velocity = np.array(values[3:]) * 1e3   # km/s -> m/s
            return position, velocity
    raise RuntimeError("Horizons response contained no usable data rows.")


# ---------------------------------------------------------------------------
# Offline demo system
# ---------------------------------------------------------------------------

def demo_system(names):
    """Build a network-free system from circular orbits at mean radii.

    Each planet is placed on a circular orbit of radius ``BODIES[name][2]`` at
    a fixed spread of phase angles and given the corresponding circular speed
    about the Sun.  The result is shifted to the barycentric frame.  It is an
    approximation (no eccentricity or inclination) but it conserves energy and
    momentum, so it exercises every analysis correctly.
    """
    bodies = []
    for k, name in enumerate(names):
        _, mass, radius = BODIES[name.lower()]
        if radius == 0.0:
            bodies.append(Body(name.capitalize(), mass))
            continue
        theta = 0.7 * k  # deterministic, keeps the planets from lining up
        speed = np.sqrt(G * BODIES["sun"][1] / radius)
        position = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
        velocity = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
        bodies.append(Body(name.capitalize(), mass, position, velocity))
    return NBodySystem(bodies).to_barycentric()


# ---------------------------------------------------------------------------
# N-Body system loader
# ---------------------------s------------------------------------------------

def load_system(source, names=None, date="2024-01-01"):
    """Return an :class:`NBodySystem` for the requested bodies.

    Parameters
    ----------
    source : {'jpl', 'demo', 'auto'}
        Where to get the initial conditions.  ``'auto'`` uses JPL if reachable
        and falls back to the demo system otherwise.
    names : list of str, optional
        Body names; defaults to the Sun and eight planets.
    date : str
        Snapshot date ``YYYY-MM-DD``.
    """
    names = names or DEFAULT_BODIES

    if source == "demo":
        return demo_system(names)

    if source in ("jpl", "auto"):
        try:
            bodies = []
            for name in names:
                pos, vel = fetch_state(name, date)
                bodies.append(Body(name.capitalize(), BODIES[name.lower()][1], pos, vel))
            return NBodySystem(bodies).to_barycentric()
        except (requests.RequestException, RuntimeError) as err:
            if source == "jpl":
                raise
            print(f"  [ephemerides] JPL unavailable ({err}); using offline demo system.")
            return demo_system(names)

    raise ValueError(f"Unknown source {source!r}; choose 'jpl', 'demo' or 'auto'.")
