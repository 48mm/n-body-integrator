"""
Analyses for report figures.

Every function here shares the same simulation core (``NBodySystem.simulate``)
and the same conserved-quantity code, which is what keeps the whole project
small: there is one integration loop, not one per experiment.  Each function
returns a plain results dict, writes a CSV of the numbers into ``outdir`` and,
unless ``show`` is disabled, draws the corresponding figure.
"""

import csv
import os
from datetime import datetime, timedelta
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from nbody import NBodySystem, Body, G
import ephemerides

SECONDS_PER_DAY = 86400
START_DATE = "2024-01-01"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path}")


def _finish(fig, path, show):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _steps(days, dt):
    return int(round(days * SECONDS_PER_DAY / dt))


def _energy_error_percent(traj):
    return np.abs((traj.energy - traj.energy[0]) / traj.energy[0]) * 100


def lyapunov_fit(times, divergence, initial_perturbation):
    """Fit ``ln(d/d0) = lambda * t`` over the central half of the run.

    Returns ``(lambda [1/s], lyapunov_time [days], r_squared)``.  The ends are
    trimmed to avoid the initial transient and any late-time saturation.
    """
    lo, hi = len(times) // 4, 3 * len(times) // 4
    t = times[lo:hi]
    d = divergence[lo:hi]
    good = d > 0
    t, d = t[good], d[good]
    if len(t) < 2:
        return 0.0, np.inf, 0.0

    log_ratio = np.log(d / initial_perturbation)
    slope, intercept = np.polyfit(t, log_ratio, 1)
    fitted = slope * t + intercept
    ss_res = np.sum((log_ratio - fitted) ** 2)
    ss_tot = np.sum((log_ratio - log_ratio.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lyap_time_days = (1.0 / slope) / SECONDS_PER_DAY if slope > 0 else np.inf
    return slope, lyap_time_days, r2


# ---------------------------------------------------------------------------
# Figure 1 / Table 1 - two-body circular orbit
# ---------------------------------------------------------------------------

def make_two_body(separation=1e8, mass1=5.972e24, mass2=7.342e22):
    """Two bodies on a circular orbit about their common centre of mass."""
    speed = np.sqrt(G * (mass1 + mass2) / separation)
    r1, r2 = mass2 / (mass1 + mass2) * separation, mass1 / (mass1 + mass2) * separation
    v1, v2 = mass2 / (mass1 + mass2) * speed, mass1 / (mass1 + mass2) * speed
    return NBodySystem([
        Body("Primary", mass1, [-r1, 0, 0], [0, v1, 0]),
        Body("Secondary", mass2, [r2, 0, 0], [0, -v2, 0]),
    ])


def _detect_period(times, separation):
    """Orbital period from peak spacing and from the dominant FFT frequency."""
    peaks, _ = find_peaks(separation, height=separation.mean())
    peak_period = np.diff(times[peaks]).mean() if len(peaks) > 1 else np.nan

    dt = np.diff(times).mean()
    # zero-pad so the frequency grid is fine enough to resolve the period from
    # the few oscillation cycles available.
    n_fft = 16 * len(times)
    spectrum = np.abs(np.fft.rfft(separation - separation.mean(), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    dominant = np.argmax(spectrum[1:]) + 1        # skip the DC bin
    fft_period = 1.0 / freqs[dominant] if freqs[dominant] > 0 else np.nan
    return peak_period, fft_period


def two_body(outdir, dt=1800, hours=500, methods=("euler", "euler_cromer", "rk4"), show=True):
    separation, mass1, mass2 = 1e8, 5.972e24, 7.342e22
    kepler = 2 * np.pi * np.sqrt(separation ** 3 / (G * (mass1 + mass2)))
    n_steps = int(round(hours * 3600 / dt))
    print(f"Two-body validation: Kepler period = {kepler / 3600:.3f} h")

    rows, trajs = [], {}
    for method in methods:
        traj = make_two_body(separation, mass1, mass2).simulate(method, dt, n_steps)
        trajs[method] = traj
        sep = np.linalg.norm(traj.positions[:, 0] - traj.positions[:, 1], axis=1)
        peak_p, fft_p = _detect_period(traj.times, sep)
        rows.append([method, peak_p / 3600, fft_p / 3600,
                     abs(peak_p - kepler) / kepler * 100, abs(fft_p - kepler) / kepler * 100])
        print(f"  {method:13s} peak {peak_p / 3600:7.3f} h  fft {fft_p / 3600:7.3f} h  "
              f"(rel err {abs(peak_p - kepler) / kepler * 100:.3f}%)")

    _save_csv(os.path.join(outdir, "two_body_periods.csv"),
              ["method", "period_peak_h", "period_fft_h", "err_peak_%", "err_fft_%"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for method in methods:
        traj = trajs[method]
        sep = np.linalg.norm(traj.positions[:, 0] - traj.positions[:, 1], axis=1)
        ax1.plot(traj.times / 3600, sep / 1e9, label=method.upper(), alpha=0.8)
    ax1.set(xlabel="Time (h)", ylabel="Separation (10^9 m)", title="Center-to-center separation")
    ax1.legend()

    x = np.arange(len(methods))
    ax2.bar(x - 0.2, [r[1] for r in rows], 0.4, label="Peak")
    ax2.bar(x + 0.2, [r[2] for r in rows], 0.4, label="FFT")
    ax2.axhline(kepler / 3600, color="r", ls="--", label="Kepler")
    ax2.set(xticks=x, ylabel="Period (h)", title="Detected orbital period")
    ax2.set_xticklabels([m.upper() for m in methods])
    ax2.legend()
    _finish(fig, os.path.join(outdir, "fig1_two_body.png"), show)
    return {"kepler_h": kepler / 3600, "rows": rows}


# ---------------------------------------------------------------------------
# Figure 2 - energy & momentum conservation (30 days)
# ---------------------------------------------------------------------------

def conservation(outdir, source="auto", dt=3600, days=30,
                 methods=("euler_cromer", "rk4"), sample_every=10, show=True):
    system = ephemerides.load_system(source, ephemerides.DEFAULT_BODIES, START_DATE)
    n_steps = _steps(days, dt)
    print(f"Conservation: {system.n} bodies, {days} days, dt={dt}s")

    data, header, columns = {}, ["time_days"], None
    for method in methods:
        traj = system.copy().simulate(method, dt, n_steps, sample_every)
        e_err = _energy_error_percent(traj)
        p_err = np.linalg.norm(traj.linear_momentum - traj.linear_momentum[0], axis=1)
        l_err = np.linalg.norm(traj.angular_momentum - traj.angular_momentum[0], axis=1)
        data[method] = (traj.times / SECONDS_PER_DAY, e_err, p_err, l_err)
        header += [f"{method}_energy_%", f"{method}_dP", f"{method}_dL"]
        print(f"  {method:13s} max |dE| {e_err.max():.3e}%   final {e_err[-1]:.3e}%")
        if columns is None:
            columns = [data[method][0]]
        columns += [e_err, p_err, l_err]

    _save_csv(os.path.join(outdir, "conservation.csv"), header, np.column_stack(columns))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["|Energy error| (%)", "Linear momentum error (kg m/s)", "Angular momentum error (kg m^2/s)"]
    for ax, col, label in zip(axes, (1, 2, 3), labels):
        for method in methods:
            t = data[method][0]
            ax.semilogy(t, data[method][col], label=method.upper(), lw=2, alpha=0.8)
        ax.set(xlabel="Time (days)", ylabel=label)
        ax.legend()
    fig.suptitle(f"Conservation over {days} days (dt = {dt/3600:.1f} h)", fontweight="bold")
    _finish(fig, os.path.join(outdir, "fig2_conservation.png"), show)
    return data


# ---------------------------------------------------------------------------
# Figure 3 - long-term energy drift (100 years)
# ---------------------------------------------------------------------------

def long_term(outdir, source="auto", dt=3600, years=100,
              methods=("euler_cromer", "rk4"), sample_every=240, show=True):
    names = ["sun", "earth", "jupiter"]
    system = ephemerides.load_system(source, names, START_DATE)
    n_steps = _steps(years * 365, dt)
    print(f"Long-term drift: Sun-Earth-Jupiter, {years} years, {n_steps} steps")

    series, header, columns = {}, ["time_days"], None
    for method in methods:
        traj = system.copy().simulate(method, dt, n_steps, sample_every)
        drift = (traj.energy - traj.energy[0]) / abs(traj.energy[0]) * 100  # signed
        series[method] = (traj.times / SECONDS_PER_DAY, drift)
        header.append(f"{method}_energy_drift_%")
        print(f"  {method:13s} final drift {drift[-1]:.3e}%   max |drift| {np.abs(drift).max():.3e}%")
        if columns is None:
            columns = [series[method][0]]
        columns.append(drift)

    _save_csv(os.path.join(outdir, "long_term_energy.csv"), header, np.column_stack(columns))

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for method in methods:
        t, drift = series[method]
        axes[0].plot(t, drift, label=method.upper(), lw=1.2)
        axes[1].semilogy(t, np.abs(drift), label=method.upper(), lw=1.2)
        axes[2].plot(t[:-1], np.diff(drift) / np.diff(t), lw=1.1)
    axes[0].set(ylabel="Energy drift (%)", title="Energy drift (linear)")
    axes[0].axhline(0, color="k", ls="--", lw=0.7); axes[0].legend()
    axes[1].set(ylabel="|drift| (%)", title="Energy drift (log)")
    axes[2].set(xlabel="Time (days)", ylabel="d/dt drift (%/day)", title="Drift rate")
    axes[2].axhline(0, color="k", ls="--", lw=0.7)
    fig.suptitle(f"Energy drift over {years} years", fontweight="bold")
    _finish(fig, os.path.join(outdir, "fig3_long_term.png"), show)
    return series


# ---------------------------------------------------------------------------
# Figure 4 - validation against ephemerides (30 days)
# ---------------------------------------------------------------------------

def _end_date(start, days):
    return (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def _truth_states(source, planets, start, end, initial):
    """Return {planet: (position, velocity)} to validate against.

    Uses JPL Horizons at the end date when available; falls back to a
    fine-step RK4 reference (dt = 450 s) when running offline, clearly labelled.
    """
    if source != "demo":
        try:
            return {p: ephemerides.fetch_state(p, end) for p in planets}, "JPL Horizons"
        except Exception as err:                                  # network fallback
            print(f"  [validation] JPL unavailable ({err}); using RK4 reference as truth.")
    ref = initial.copy().simulate("rk4", 450, _steps((datetime.strptime(end, '%Y-%m-%d')
                                                      - datetime.strptime(start, '%Y-%m-%d')).days, 450))
    return ({p: (ref.positions[-1, ref.index(p)], ref.velocities[-1, ref.index(p)]) for p in planets},
            "RK4 dt=450 s reference (offline surrogate)")


def validation(outdir, source="auto", dt=3600, days=30,
               methods=("euler_cromer", "rk4"), show=True):
    initial = ephemerides.load_system(source, ephemerides.DEFAULT_BODIES, START_DATE)
    planets = [n for n in initial.names if n.lower() != "sun"]
    end = _end_date(START_DATE, days)
    truth, truth_label = _truth_states(source, [p.lower() for p in planets], START_DATE, end, initial)
    print(f"Validation: {days} days, truth = {truth_label}")

    n_steps = _steps(days, dt)
    rows = []
    errors = {m: {"pos": [], "vel": []} for m in methods}
    for method in methods:
        traj = initial.copy().simulate(method, dt, n_steps)
        for planet in planets:
            idx = traj.index(planet)
            true_pos, true_vel = truth[planet.lower()]
            pos_err = np.linalg.norm(traj.positions[-1, idx] - true_pos)
            vel_err = np.linalg.norm(traj.velocities[-1, idx] - true_vel)
            pos_pct = pos_err / np.linalg.norm(true_pos) * 100
            vel_pct = vel_err / np.linalg.norm(true_vel) * 100
            errors[method]["pos"].append(pos_err)
            errors[method]["vel"].append(vel_err)
            rows.append([method, planet, pos_err, pos_pct, vel_err, vel_pct])
        print(f"  {method:13s} position error range: "
              f"{min(r[2] for r in rows if r[0]==method):.2e} - "
              f"{max(r[2] for r in rows if r[0]==method):.2e} m")

    _save_csv(os.path.join(outdir, "validation.csv"),
              ["method", "body", "pos_err_m", "pos_err_%", "vel_err_ms", "vel_err_%"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(planets))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        off = (i - (len(methods) - 1) / 2) * width
        ax1.bar(x + off, np.array(errors[method]["pos"]) / 1e9, width, label=method.upper())
        ax2.bar(x + off, np.array(errors[method]["vel"]) / 1e3, width, label=method.upper())
    for ax, ylabel in ((ax1, "Position error (10^9 m)"), (ax2, "Velocity error (km/s)")):
        ax.set(yscale="log", ylabel=ylabel, xticks=x)
        ax.set_xticklabels([p.capitalize() for p in planets], rotation=45)
        ax.legend()
    fig.suptitle(f"Error vs {truth_label} after {days} days", fontweight="bold")
    _finish(fig, os.path.join(outdir, "fig4_validation.png"), show)
    return rows


# ---------------------------------------------------------------------------
# Table 2 - Lyapunov analysis (1 year)
# ---------------------------------------------------------------------------

def chaos(outdir, source="auto", dt=3600, days=365,
          perturbations=(1.0, 1e3, 1e6, 1e9), perturb_body="earth",
          methods=("euler_cromer", "rk4"), sample_every=24, show=True):
    names = ["sun", "mercury", "venus", "earth", "mars", "jupiter"]
    base = ephemerides.load_system(source, names, START_DATE)
    idx = base.names.index(perturb_body.capitalize())
    n_steps = _steps(days, dt)
    print(f"Chaos: perturbing {perturb_body} over {days} days")

    rows = []
    for method in methods:
        reference = base.copy().simulate(method, dt, n_steps, sample_every)
        ref_track = reference.positions[:, idx, :]
        for delta in perturbations:
            perturbed = base.copy()
            perturbed.pos[idx, 0] += delta
            run = perturbed.simulate(method, dt, n_steps, sample_every)
            divergence = np.linalg.norm(run.positions[:, idx, :] - ref_track, axis=1)
            lam, lyap_days, r2 = lyapunov_fit(run.times, divergence, delta)
            rows.append([method, delta, divergence[-1] / 1e3, lam, lyap_days, r2])
            print(f"  {method:13s} d0={delta:8.0e} m  final div {divergence[-1]/1e3:12.3f} km  "
                  f"lambda {lam:.3e}/s  T_L {lyap_days:.3e} d  R2 {r2:.3f}")

    _save_csv(os.path.join(outdir, "chaos_lyapunov.csv"),
              ["method", "perturbation_m", "final_divergence_km",
               "lyapunov_s^-1", "lyapunov_time_days", "r_squared"], rows)
    return rows


# ---------------------------------------------------------------------------
# Computational cost per step
# ---------------------------------------------------------------------------

def benchmark(outdir, source="auto", dt=3600, n_steps=2000,
              methods=("euler", "euler_cromer", "rk4"), show=False):
    system = ephemerides.load_system(source, ephemerides.DEFAULT_BODIES, START_DATE)
    print(f"Benchmark: {system.n} bodies, {n_steps} steps")

    timings = {}
    for method in methods:
        clone = system.copy()
        start = perf_counter()
        for _ in range(n_steps):
            clone.step(method, dt)
        timings[method] = (perf_counter() - start) / n_steps
    ref = timings.get("euler_cromer", min(timings.values()))

    rows = [[m, timings[m] * 1e6, timings[m] / ref] for m in methods]
    for m, us, ratio in rows:
        print(f"  {m:13s} {us:8.2f} us/step   {ratio:.2f}x")
    _save_csv(os.path.join(outdir, "benchmark_timing.csv"),
              ["method", "microseconds_per_step", "ratio_vs_euler_cromer"], rows)
    return rows
