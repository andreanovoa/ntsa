"""Lyapunov exponent tools for continuous-time dynamical-system models.

Benettin QR spectrum (joint RK4 state+tangent integration), leading-exponent
estimation from perturbation growth, analytic/finite-difference Jacobians, and
a data-only Rosenstein estimator for measured series (no equations needed).
Reference: Kantz & Schreiber, "Nonlinear Time Series Analysis", Ch. 11.
"""

import warnings

import numpy as np

from ntsa.tools.maps import stationary_start
from ntsa.tools.runners import respawn, run_long

# ---------------------------------------------------------------------------
# RHS and Jacobians
# ---------------------------------------------------------------------------

def get_rhs(model):
    """Return f(u) -> du/dt for a continuous-time model.

    Parameters read once from ``{**model.alpha0, **model.governing_eqns_params}``.
    Raises AttributeError for discrete-map models (no ``time_derivative``).
    """
    p = {**model.alpha0, **model.governing_eqns_params}
    try:
        td = model.time_derivative
    except AttributeError as exc:
        raise AttributeError(f"{type(model).__name__} is a discrete-map model (no time_derivative); "
                             "Lyapunov tools require a continuous RHS.") from exc

    def f(u):
        # time_derivative may return a tuple zero-padded for augmented rows — keep first Nphi.
        return np.asarray(td(0.0, u, **p), dtype=float)[:len(u)]

    return f


def _jac_lorenz63(model):
    """Analytic Jacobian factory for Lorenz63 (params frozen at call-creation)."""
    s, r, b = model.sigma, model.rho, model.beta

    def jac(u):
        return np.array([[-s, s, 0.0],
                         [r - u[2], -1.0, -u[0]],
                         [u[1], u[0], -b]])

    return jac


def _jac_lorenz96(model):
    """Analytic Jacobian factory for Lorenz96: dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F."""
    N = model.Nphi
    assert N >= 4, "Lorenz96 Jacobian requires Nx >= 4"
    i = np.arange(N)
    ip1, im1, im2 = (i + 1) % N, (i - 1) % N, (i - 2) % N

    def jac(u):
        J = np.zeros((N, N))
        J[i, i] = -1.0
        J[i, ip1] = u[im1]
        J[i, im1] = u[ip1] - u[im2]
        J[i, im2] = -u[im1]
        return J

    return jac


_ANALYTIC_JACOBIANS = {'Lorenz63': _jac_lorenz63, 'Lorenz96': _jac_lorenz96}


def fd_jacobian(f, u, eps=1e-6):
    """Central-difference Jacobian of f at u, per-column step h = eps*max(1, |u_j|)."""
    u = np.asarray(u, dtype=float)
    n = len(u)
    J = np.empty((n, n))
    for j in range(n):
        h = eps * max(1.0, abs(u[j]))
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        J[:, j] = (f(up) - f(um)) / (2.0 * h)
    return J


def get_jacobian(model, f=None):
    """Return jac(u) -> (N, N): analytic for Lorenz63/Lorenz96, else finite differences."""
    builder = _ANALYTIC_JACOBIANS.get(type(model).__name__)
    if builder is not None:
        return builder(model)
    f = f or get_rhs(model)
    return lambda u: fd_jacobian(f, u)


# ---------------------------------------------------------------------------
# Benettin QR spectrum
# ---------------------------------------------------------------------------

def _rk4_tangent_step(f, jac, u, U, dt):
    """One RK4 step of state u and tangent matrix U, sharing the RK stages."""
    k1 = f(u)
    K1 = jac(u) @ U
    u2 = u + 0.5 * dt * k1
    k2 = f(u2)
    K2 = jac(u2) @ (U + 0.5 * dt * K1)
    u3 = u + 0.5 * dt * k2
    k3 = f(u3)
    K3 = jac(u3) @ (U + 0.5 * dt * K2)
    u4 = u + dt * k3
    k4 = f(u4)
    K4 = jac(u4) @ (U + dt * K3)
    return (u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4),
            U + dt / 6.0 * (K1 + 2 * K2 + 2 * K3 + K4))


def _settled_state(model, t_ref):
    """State on the attractor: extend the discarded transient until a probe run shows no drift.

    Near-marginally-stable orbits (e.g. just past a Hopf point) outlive any fixed
    transient multiple; a still-decaying/growing start biases finite-time Lyapunov
    estimates upward on limit cycles. Doubles the transient (up to 4x) while
    ``stationary_start`` still detects amplitude drift in a 20*t_ref probe.
    """
    tt = max(model.t_transient, 10 * t_ref)
    psi = None
    for _ in range(4):
        _, y, psi = run_long(respawn(model), t_run=20 * t_ref, t_transient=tt,
                             trim_transient=False)
        if stationary_start(y[:, 0]) == 0:
            break
        tt *= 2
    return psi[-1]


def lyapunov_spectrum(model, n_exp=None, dt=None, t_transient=None, t_run=None,
                      t_max=None, atol=2e-3, rtol=0.05, N_gs=4, u0=None, seed=0,
                      verbose=True, full_output=False):
    """Lyapunov spectrum via Gram-Schmidt reorthonormalisation (Benettin et al.).

    Integrates state + tangent space with a fixed-step joint RK4, so `dt`
    defaults to min(model.dt, 0.01) to keep the tangent propagation accurate.
    Times default to multiples of t_lyap (t_CR fallback): warmup 10x, run 200x.

    `t_run` is a floor, not the horizon: integration continues (up to `t_max`,
    default 20*t_run) until every running exponent satisfies the halving test
    |lam_j(T) - lam_j(T/2)| < max(atol, rtol*|lam_j(T)|) at two consecutive
    checks (one pass can be a phase coincidence on a periodic orbit). This
    catches both the O(1/T) drift of neutral exponents on limit cycles — atol
    is effectively how close to zero they must land — and the slow statistical
    convergence of chaotic ones. Warns if `t_max` is hit unconverged.

    Returns
    -------
    np.ndarray, shape (n_exp,), sorted descending. With ``full_output=True``,
    returns ``(exponents, info)`` where info has T, converged, t_hist and
    lam_hist (running estimates, one row per QR step, unsorted).
    """
    Nphi = model.Nphi
    n_exp = Nphi if n_exp is None else min(n_exp, Nphi)
    t_ref = getattr(model, 't_lyap', model.t_CR)
    t_transient = 10 * t_ref if t_transient is None else t_transient
    t_run = 200 * t_ref if t_run is None else t_run
    t_max = 20 * t_run if t_max is None else t_max
    dt = min(model.dt, 0.01) if dt is None else dt

    f = get_rhs(model)
    jac = get_jacobian(model, f=f)
    rng = np.random.default_rng(seed)

    if u0 is None:
        u0 = _settled_state(model, t_ref)
    u = np.asarray(u0, dtype=float).copy()
    U = np.linalg.qr(rng.standard_normal((Nphi, n_exp)))[0]

    for _ in range(int(t_transient / dt)):
        u, U = _rk4_tangent_step(f, jac, u, U, dt)
    if not np.isfinite(u).all():
        raise RuntimeError("State diverged during transient phase.")
    U, _ = np.linalg.qr(U)

    n_floor = int(t_run / dt)
    n_max = max(int(t_max / dt), n_floor)
    log_R_sum = np.zeros(n_exp)
    log_R_hist = []  # cumulative sum after each QR step, for the halving test
    check_every = max(1, n_floor // (10 * N_gs))  # QR steps between convergence checks
    report_every = max(500, n_floor // 10)
    converged = False
    n_ok = 0
    n = 0
    while n < n_max:
        u, U = _rk4_tangent_step(f, jac, u, U, dt)
        n += 1
        if n % N_gs:
            continue
        if not np.isfinite(u).all():
            raise RuntimeError(f"State diverged at step {n} of accumulation phase.")
        U, R = np.linalg.qr(U)
        log_R_sum += np.log(np.abs(np.diag(R)))
        log_R_hist.append(log_R_sum.copy())
        k = len(log_R_hist)
        if verbose and n % report_every == 0:
            print(f"  GS {n}/{n_max}  lam1={(log_R_sum / (k * N_gs * dt)).max():.4f}")
        if n >= n_floor and k % check_every == 0 and k >= 2:
            lam = log_R_sum / (k * N_gs * dt)
            k2 = k // 2
            lam_half = log_R_hist[k2 - 1] / (k2 * N_gs * dt)
            n_ok = n_ok + 1 if np.all(np.abs(lam - lam_half)
                                      < np.maximum(atol, rtol * np.abs(lam))) else 0
            if n_ok >= 2:
                converged = True
                break

    T = len(log_R_hist) * N_gs * dt
    exponents = np.sort(log_R_hist[-1] / T)[::-1]
    if not converged:
        warnings.warn(f"lyapunov_spectrum: not converged at T={T:g} "
                      f"(|lam(T)-lam(T/2)| still above tol) — raise t_max.", stacklevel=2)
    if verbose:
        print(f"Lyapunov spectrum (T={T:g}, {'converged' if converged else 'NOT converged'}): "
              f"{np.array2string(exponents, precision=4)}")
    if full_output:
        k_arr = np.arange(1, len(log_R_hist) + 1)
        info = dict(T=T, converged=converged, t_hist=k_arr * N_gs * dt,
                    lam_hist=np.array(log_R_hist) / (k_arr[:, None] * N_gs * dt))
        return exponents, info
    return exponents


# ---------------------------------------------------------------------------
# Leading exponent from perturbation growth
# ---------------------------------------------------------------------------

def fit_log_growth(t, log_seps):
    """Fit the exponential-growth window of mean log-separation curves.

    Parameters
    ----------
    t : (Nt,) times; log_seps : (Nt, n_pert) per-member log separations.

    Returns
    -------
    (lam1, lam1_std, i1, i2, r2); lam1 is nan when r2 < 0.5 or window too short.
    """
    mean = log_seps.mean(axis=1)
    n = len(mean)
    i1 = int(np.argmin(mean))
    # genuine exponential growth starts from the initial eps at t ~ 0; on a stable orbit
    # the separation *oscillates* and argmin lands in a late dip whose 20-sample rebound
    # would otherwise fit as a huge spurious slope (seen on VdP: lam1 ~ 1800, r2 > 0.5)
    if i1 > n // 4:
        return np.nan, np.nan, i1, min(i1 + 20, n - 1), 0.0
    sat = float(mean[int(0.8 * n):].mean())
    threshold = mean[i1] + 0.70 * (sat - mean[i1])  # sign-safe version of the 0.70*sat_level heuristic
    above = np.where(mean[i1 + 1:] >= threshold)[0]
    i2 = i1 + 1 + int(above[0]) if len(above) else n - 1
    # floor at 20 samples only: a %-of-record floor drags the window into the
    # saturated plateau for fast systems whose growth phase is short (e.g. Rijke)
    i2 = max(i2, i1 + 20)
    i2 = min(i2, n - 1)
    # demand a real exponential-growth range: chaos grows eps -> attractor size
    # (~ln(1e6) = 14 nats); a stable orbit only shows a ~1-nat noise-floor bounce,
    # which the short window would otherwise fit as a spurious steep slope
    if i2 <= i1 + 5 or (sat - mean[i1]) < 2.0:
        return np.nan, np.nan, i1, i2, 0.0

    x, y = t[i1:i2], mean[i1:i2]
    coeffs = np.polyfit(x, y, 1)
    yhat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-14)
    lam1 = float(coeffs[0])
    slopes = [np.polyfit(x, log_seps[i1:i2, j], 1)[0] for j in range(log_seps.shape[1])]
    lam1_std = float(np.std(slopes))
    if r2 < 0.5:
        lam1 = np.nan
    return lam1, lam1_std, i1, i2, r2


def _integrate_from(model, psi_init, Nt):
    """Integrate a respawned copy of `model` from psi_init for Nt steps, no transient.

    Returns (t (Nt,), psi (Nt, Nphi)) for member 0. Used instead of
    run_long(..., t_transient=0) so a zero-transient run never hits the
    reset-history path.
    """
    mj = respawn(model, psi0=psi_init)
    psi, t = mj.time_integrate(Nt=Nt)
    mj.close()
    return t, psi[:, :mj.Nphi, 0]


def leading_lyapunov(model, n_pert=8, eps=1e-6, t_run=None, seed=0):
    """Leading Lyapunov exponent from the growth of small random perturbations.

    Integrates one reference and `n_pert` perturbed trajectories SERIALLY
    (m=1 each) and fits the linear window of the mean log separation.

    Returns
    -------
    (lam1, lam1_std, res) with res holding t, log_sep, mean_log_sep, i1, i2,
    r2, lam1, lam1_std, sat, diam. lam1 is nan when the fit is rejected —
    R^2 < 0.5, no real growth range, or saturation far below the attractor
    diameter (non-normal transient amplification, not chaos).
    """
    t_ref = getattr(model, 't_lyap', model.t_CR)
    t_run = 20 * t_ref if t_run is None else t_run
    rng = np.random.default_rng(seed)

    u0 = _settled_state(model, t_ref)  # drift-checked warmup: a still-settling start reads as growth
    scale = np.abs(u0).max()

    Nt = int(round(t_run / model.dt))
    t, psi_ref = _integrate_from(model, u0, Nt)
    seps = np.empty((len(t), n_pert))
    for j in range(n_pert):
        uj = u0 + eps * scale * rng.standard_normal(len(u0))
        _, psi_j = _integrate_from(model, uj, Nt)
        seps[:, j] = np.linalg.norm(psi_j - psi_ref, axis=1)

    log_sep = np.log(np.maximum(seps, 1e-300))
    lam1, lam1_std, i1, i2, r2 = fit_log_growth(t, log_sep)
    # non-normal transient amplification on a stable orbit grows eps by a large gain
    # and then PLATEAUS far below the attractor size, whereas genuine chaos
    # decorrelates to O(attractor diameter). Reject fits whose tail is both low
    # (< 5% of the diameter) and flat (still-growing separation just means the
    # horizon ended mid-growth — e.g. L96 at 20*t_lyap — and must not be rejected).
    # Thermoacoustic limit cycles otherwise read lam1 ~ 200 from the transient alone.
    diam = float(np.linalg.norm(np.ptp(psi_ref, axis=0)))
    mean_ls = log_sep.mean(axis=1)
    i_tail = int(0.8 * len(t))
    sat = float(np.exp(np.mean(mean_ls[i_tail:])))
    tail_slope = float(np.polyfit(t[i_tail:], mean_ls[i_tail:], 1)[0])
    if np.isfinite(lam1) and sat < 0.05 * diam and tail_slope < 0.1 * lam1:
        lam1 = np.nan
    res = dict(t=t, log_sep=log_sep, mean_log_sep=log_sep.mean(axis=1),
               i1=i1, i2=i2, r2=r2, lam1=lam1, lam1_std=lam1_std, sat=sat, diam=diam)
    return lam1, lam1_std, res


def rosenstein_lyapunov(x, dt, zeta=None, dim=None, theiler=None, k_max=None, n_ref=1000):
    """Data-only leading Lyapunov exponent from a scalar series (Rosenstein et al. 1993).

    Delay-embeds `x`, pairs each reference point with its nearest neighbour at
    least a Theiler window away in time, tracks the pairwise divergence, and fits
    the linear window of the mean log divergence with `fit_log_growth` — so the
    same rejection guards apply as in `leading_lyapunov` (R^2 >= 0.5 and a real
    exponential-growth range): periodic or noise-floor-bound data returns nan
    rather than a spurious slope.

    Parameters
    ----------
    x : np.ndarray
        Scalar series, shape (Nt,).
    dt : float
        Sampling time.
    zeta, dim : int, optional
        Embedding delay/dimension; default `optimal_lag` / `false_nearest_neighbours`.
    theiler : int, optional
        Temporal exclusion window (samples) for the neighbour search; default the
        mean inter-maximum spacing (the signal's mean period), floored at `zeta`.
    k_max : int, optional
        Divergence-tracking horizon (samples); default 10 Theiler windows,
        capped so every pair can be tracked to the end of the record.
    n_ref : int
        Maximum number of reference points, strided over the record.

    Returns
    -------
    (lam1, lam1_std, res) mirroring `leading_lyapunov`; `res` plugs into
    `characterize.plot_lyapunov_fit` and adds zeta, dim, theiler, n_pairs.
    `lam1_std` is the slope spread across 8 blocks of pairs (pseudo-members),
    not the raw pair-to-pair scatter. Raises ValueError when the record is too
    short to form a divergence horizon or enough neighbour pairs.
    """
    from scipy.spatial import cKDTree

    from ntsa.tools.embedding import delay_embed, false_nearest_neighbours, optimal_lag
    from ntsa.tools.maps import first_return_map

    x = np.asarray(x, dtype=float).ravel()
    if zeta is None:
        zeta = optimal_lag(x)
    if dim is None:
        dim, _ = false_nearest_neighbours(x, zeta)
    Y = delay_embed(x, dim, zeta)
    N = len(Y)
    if theiler is None:
        _, _, pk = first_return_map(x)
        theiler = int(np.mean(np.diff(pk))) if pk.size > 2 else zeta * dim
    w = max(int(theiler), zeta, 1)
    if k_max is None:
        k_max = 10 * w
    k_max = int(min(k_max, N - w - 2))
    if k_max < 20:
        raise ValueError(f'series too short: divergence horizon k_max={k_max} (N={N}, theiler={w})')

    # nearest neighbour of each reference point, excluding |i-j| <= w (Theiler
    # window) and start points whose divergence track would run off the record
    valid = N - k_max
    ref = np.arange(0, valid, max(1, valid // n_ref))[:n_ref]
    tree = cKDTree(Y[:valid])
    k_query = int(min(valid, 2 * w + 2))
    _, nbrs = tree.query(Y[ref], k=k_query)
    pairs_i, pairs_j = [], []
    for i, row in zip(ref, np.atleast_2d(nbrs)):
        ok = row[np.abs(row - i) > w]
        if ok.size:
            pairs_i.append(int(i))
            pairs_j.append(int(ok[0]))
    if len(pairs_i) < 10:
        raise ValueError(f'only {len(pairs_i)} temporally separated neighbour pairs (theiler={w})')

    ks = np.arange(k_max + 1)
    d = np.linalg.norm(Y[np.asarray(pairs_i)[:, None] + ks] - Y[np.asarray(pairs_j)[:, None] + ks], axis=-1)
    logd = np.log(np.maximum(d, 1e-300)).T  # (k_max+1, n_pairs)

    # 8 blocks of pairs as pseudo-members: fit_log_growth's member spread then
    # measures block-to-block slope variability, not raw pair-to-pair scatter
    blocks = np.array_split(np.arange(logd.shape[1]), min(8, logd.shape[1]))
    log_sep = np.column_stack([logd[:, b].mean(axis=1) for b in blocks])
    t = ks * dt
    lam1, lam1_std, i1, i2, r2 = fit_log_growth(t, log_sep)
    res = dict(t=t, log_sep=log_sep, mean_log_sep=log_sep.mean(axis=1),
               i1=i1, i2=i2, r2=r2, lam1=lam1, lam1_std=lam1_std,
               zeta=zeta, dim=dim, theiler=w, n_pairs=logd.shape[1])
    return lam1, lam1_std, res


def kaplan_yorke(exponents, tol=2e-3):
    """Kaplan-Yorke dimension from a sorted-descending Lyapunov spectrum.

    D_KY = j + sum_{i<=j} lam_i / |lam_{j+1}| with j the largest index keeping the
    cumulative sum >= -tol (tol absorbs the finite-time jitter of neutral
    exponents, so a limit cycle reads ~1 instead of 0). Returns 0.0 for an
    all-contracting spectrum (fixed point) and Nphi if the sum never turns
    negative within the resolved exponents.
    """
    lam = np.sort(np.asarray(exponents, dtype=float))[::-1]
    csum = np.cumsum(lam)
    j = np.flatnonzero(csum >= -tol)
    if j.size == 0:
        return 0.0
    j = int(j[-1])
    if j + 1 >= len(lam):
        return float(len(lam))
    return float(j + 1 + csum[j] / abs(lam[j + 1]))


def covariant_lyapunov_vectors(*args, **kwargs):
    """Covariant Lyapunov vectors — not implemented."""
    raise NotImplementedError("CLVs: Ginelli et al. 2007 — store Q,R in forward Benettin pass, "
                              "backward-iterate upper-triangular coeffs. Not implemented yet.")


if __name__ == '__main__':
    from dynamodels.physical import Lorenz63

    model = Lorenz63(dt=0.01)
    exponents = lyapunov_spectrum(model)
    reference = np.array([0.906, 0.0, -14.57])
    print(f"Benettin spectrum : {np.array2string(exponents, precision=4)}")
    print(f"Reference values  : {reference}")
    print(f"Difference        : {np.array2string(exponents - reference, precision=4)}")

    lam1, lam1_std, _ = leading_lyapunov(model)
    print(f"Perturbation-growth lam1 = {lam1:.4f} +/- {lam1_std:.4f}")
