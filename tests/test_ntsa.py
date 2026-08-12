"""Integration tests for the `ntsa` package (tools, lyapunov, characterize).

Plain assert-based tests; collected by pytest with the rest of the suite, or run
standalone with ``python tests/test_ntsa.py``.
"""

import sys
import time
import traceback

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402

from ntsa import lyapunov as lyap  # noqa: E402
from ntsa import tools as ntsa_tools  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy caches for expensive runs (shared across tests)
# ---------------------------------------------------------------------------

_cache = {}


def _l63_model():
    if 'l63' not in _cache:
        from dynamodels.physical import Lorenz63
        _cache['l63'] = Lorenz63(dt=0.01)
    return _cache['l63']


def _l63_traj():
    """60*t_lyap trajectory of Lorenz63 (t, y, psi, dt)."""
    if 'l63_traj' not in _cache:
        model = _l63_model()
        t, y, psi = ntsa_tools.run_long(ntsa_tools.respawn(model), t_run=60 * model.t_lyap)
        assert np.all(np.diff(t) > 0), 'run_long time stamps not strictly increasing'
        _cache['l63_traj'] = (t, y, psi, float(t[1] - t[0]))
    return _cache['l63_traj']


def _l63_leading():
    """(lam1, lam1_std, res) from leading_lyapunov on Lorenz63."""
    if 'l63_leading' not in _cache:
        model = _l63_model()
        _cache['l63_leading'] = lyap.leading_lyapunov(model, n_pert=5, t_run=20 * model.t_lyap)
    return _cache['l63_leading']


def _sine():
    """Noiseless sine, period T=100 samples, 30 periods."""
    if 'sine' not in _cache:
        T = 100
        n = np.arange(30 * T)
        _cache['sine'] = np.sin(2 * np.pi * n / T)
    return _cache['sine']


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ami_fnn_sine():
    x = _sine()
    lag = ntsa_tools.optimal_lag(x, max_lag=60)
    assert 17 <= lag <= 33, f'optimal_lag = {lag}, expected ~T/4 = 25 in [17, 33]'
    d, fractions = ntsa_tools.false_nearest_neighbours(x, lag)
    assert d == 2, f'FNN dimension = {d} (fractions {fractions}), expected 2 for a sine'


def test_delay_embed_recurrence():
    x = _sine()
    Y = ntsa_tools.delay_embed(x, dim=3, lag=25)
    assert Y.shape == (len(x) - 2 * 25, 3), f'delay_embed shape {Y.shape}'
    assert np.allclose(Y[0], [x[0], x[25], x[50]])
    R = ntsa_tools.recurrence_matrix(Y[-400:])
    assert R.dtype == bool
    assert np.array_equal(R, R.T), 'recurrence matrix not symmetric'
    assert R.diagonal().all(), 'recurrence matrix diagonal not all True'
    assert 0.0 < R.mean() < 1.0, f'recurrence density {R.mean()}'
    R10 = ntsa_tools.recurrence_matrix(Y[-400:], rr=0.10)
    assert abs(R10.mean() - 0.10) < 0.05, f'rr-target recurrence density {R10.mean()}'


def test_stationary_start():
    t = np.arange(0, 40, 0.01)
    ramp = np.sin(2 * np.pi * t) * np.minimum(t / 15, 1.0)  # Hopf-like amplitude growth to t=15
    i0 = ntsa_tools.stationary_start(ramp)
    assert 8 <= t[i0] <= 15.5, f'ramp trimmed at t={t[i0]}, expected near 15'
    assert ntsa_tools.stationary_start(np.sin(2 * np.pi * t)) == 0, 'flat sine trimmed'
    rng = np.random.default_rng(0)
    modulated = np.sin(2 * np.pi * t) * (1 + 0.5 * np.sin(0.22 * np.pi * t)) \
        + 0.2 * rng.standard_normal(len(t))
    assert ntsa_tools.stationary_start(modulated) == 0, 'stationary QP/noisy envelope trimmed'


def test_l63_spectrum():
    model = _l63_model()
    exps, info = lyap.lyapunov_spectrum(model, n_exp=3, verbose=False, full_output=True)
    assert exps.shape == (3,)
    assert info['converged'], f'spectrum not converged at T={info["T"]}'
    assert abs(exps[0] - 0.906) < 0.1, f'lambda1 = {exps[0]}'
    assert abs(exps[1]) < 2e-3, f'lambda2 = {exps[1]}, neutral exponent not converged to 0'
    assert abs(exps.sum() + 13.667) < 0.5, f'sum(lambda) = {exps.sum()}, expected -sigma-1-beta'
    # FD Jacobian path vs the analytic one at 5 random states
    f = lyap.get_rhs(model)
    jac = lyap.get_jacobian(model)
    rng = np.random.default_rng(0)
    for _ in range(5):
        u = rng.uniform(-20, 20, size=model.Nphi)
        J_an, J_fd = jac(u), lyap.fd_jacobian(f, u)
        rel = np.abs(J_fd - J_an).max() / max(np.abs(J_an).max(), 1e-12)
        assert rel < 1e-5, f'FD vs analytic Jacobian rel err {rel:.2e}'


def test_l63_leading():
    lam1, lam1_std, res = _l63_leading()
    assert np.isfinite(lam1), f'leading_lyapunov returned lam1 = {lam1} (r2 = {res["r2"]})'
    assert 0.6 < lam1 < 1.2, f'lam1 = {lam1}, expected in (0.6, 1.2)'
    assert res['log_sep'].shape == (len(res['t']), 5)
    assert res['i1'] < res['i2']


def test_l96_leading_guard():
    # regression: the non-normal saturation guard must NOT reject genuine chaos whose
    # default horizon ends mid-growth (tail low but still rising)
    from dynamodels.physical import Lorenz96
    lam1, _, res = lyap.leading_lyapunov(Lorenz96(Nx=10, F=8., dt=0.01), n_pert=3)
    assert np.isfinite(lam1), f'L96 F=8 lam1 rejected (sat={res["sat"]:.3g}, diam={res["diam"]:.3g})'
    assert 0.5 < lam1 < 2.5, f'L96 F=8 lam1 = {lam1}'


def test_vdp_leading_not_chaotic():
    # regression: a stable limit cycle must NOT yield a finite positive lam1 from the
    # noise-floor bounce of decayed perturbations (fit_log_growth dynamic-range guard)
    from dynamodels.physical import VdP
    lam1, _, _ = lyap.leading_lyapunov(VdP(), n_pert=3, t_run=0.3, seed=0)
    assert not (np.isfinite(lam1) and lam1 > 1.0), f'VdP lam1 = {lam1}, spurious chaos'


def test_l63_fnn():
    model = _l63_model()
    t, y, psi, dt = _l63_traj()
    x = y[:, 0]
    zeta = ntsa_tools.optimal_lag(x, max_lag=max(2, int(model.t_CR / model.dt)))
    assert 0.05 < zeta * dt < 0.5, f'zeta*dt = {zeta * dt} t.u., expected in (0.05, 0.5)'
    d, fractions = ntsa_tools.false_nearest_neighbours(x, zeta)
    assert d in (3, 4), f'FNN dimension = {d} (fractions {fractions}), expected 3 or 4'


def test_classify():
    # sine -> period-1 limit cycle
    label, ev = ntsa_tools.classify_regime(_sine(), dt=1.0)
    assert label.startswith('limit_cycle'), f'sine classified as {label}'
    assert ev['n_clusters'] == 1, f'sine n_clusters = {ev["n_clusters"]}'

    # Lorenz63 (rho=28) with its measured lam1 -> chaotic
    t, y, psi, dt = _l63_traj()
    lam1, lam1_std, _ = _l63_leading()
    label, ev = ntsa_tools.classify_regime(y[:, 0], dt, lam1=lam1, lam1_std=lam1_std,
                                           t_total=t[-1] - t[0])
    assert label == 'chaotic', f'L63 classified as {label} (evidence {ev})'

    # L63 without a Lyapunov estimate: the continuum of maxima must NOT collapse
    # to a small-period limit cycle (count_peak_clusters sentinel)
    label, ev = ntsa_tools.classify_regime(y[:, 0], dt)
    assert not label.startswith('limit_cycle_period'), \
        f'L63 without lam1 classified as {label} (evidence {ev})'

    # two incommensurate tones -> quasiperiodic (default cluster_tol)
    tt = np.arange(0, 500, 0.05)
    x = np.sin(tt) + np.sin(np.sqrt(2) * tt)
    label, ev = ntsa_tools.classify_regime(x, dt=0.05)
    assert label == 'quasiperiodic', f'two-tone classified as {label} (evidence {ev})'

    # same two tones with a large mean: DC must not swamp the PSD peak prominence
    label, ev = ntsa_tools.classify_regime(x + 50.0, dt=0.05)
    assert label == 'quasiperiodic', f'offset two-tone classified as {label} (evidence {ev})'

    # constant + tiny noise -> fixed point
    rng = np.random.default_rng(0)
    x = 1.0 + 1e-9 * rng.standard_normal(1000)
    label, _ = ntsa_tools.classify_regime(x, dt=0.01)
    assert label == 'fixed_point', f'constant classified as {label}'

    # a clean period-2 signal must stay a limit cycle even against a large, noisy lam1
    # (non-normal transient amplification inflates perturbation fits on stable orbits;
    # tight maxima clusters outrank the Lyapunov step)
    tt = np.arange(0, 300, 0.05)
    x2 = np.sin(tt) + 0.6 * np.sin(0.5 * tt)
    label, ev = ntsa_tools.classify_regime(x2, dt=0.05, lam1=150.0, lam1_std=20.0, t_total=300.0)
    assert label == 'limit_cycle_period_2', f'period-2 + bogus lam1 classified as {label} ({ev})'

    # spectrum refinements: two neutral exponents -> quasiperiodic even when the PSD
    # ratio spuriously matches a dense rational (as at L96 F=4.4: 0.552 ~ 5/9);
    # all-negative spectrum -> fixed point even when solver noise wiggles the tail
    tq = np.arange(0, 2000, 0.05)
    xq = np.sin(0.5 * tq) + 0.5 * np.sin(0.2793 * tq)  # ratio 0.5586: irrational, ~5/9 within 0.005
    label, ev = ntsa_tools.classify_regime(xq, dt=0.05)
    assert label == 'frequency_locked', f'near-rational two-tone without spectrum: {label} ({ev})'
    label, ev = ntsa_tools.classify_regime(xq, dt=0.05,
                                           exponents=np.array([1e-4, -1e-4, -0.5]))
    assert label == 'quasiperiodic', f'two-neutral spectrum classified as {label} ({ev})'
    assert ev['n_neutral'] == 2, ev['n_neutral']
    xfp = 4.9 + 1e-2 * np.sin(tq) * rng.standard_normal(len(tq))
    label, _ = ntsa_tools.classify_regime(xfp, dt=0.05,
                                          exponents=np.array([-0.6, -0.6, -12.5]))
    assert label == 'fixed_point', f'all-negative spectrum classified as {label}'


def test_respawn_observation():
    from dynamodels.physical import Lorenz63, Lorenz96
    m = Lorenz63(observe_dims=[0])
    r = ntsa_tools.respawn(m)
    assert r.Nq == 1 and list(r.observe_dims) == [0], \
        f'respawn dropped observe_dims: Nq={r.Nq}, observe_dims={r.observe_dims}'
    m96 = Lorenz96(Nx=10, observed_idx=[3])
    r96 = ntsa_tools.respawn(m96)
    assert list(r96.observed_idx) == [3], f'respawn dropped observed_idx: {r96.observed_idx}'
    for mm in (m, r, m96, r96):
        mm.close()


def test_bifurcation_smoke():
    model = _l63_model()
    values, peaks = ntsa_tools.bifurcation_sweep(model, 'rho', [20., 28., 35.],
                                                 t_transient=5., t_sample=10.)
    assert values.shape == (3,)
    assert set(peaks) == {'max'}
    assert len(peaks['max']) == model.Nq
    assert all(len(per_obs) == 3 for per_obs in peaks['max'])
    i28 = int(np.argmin(np.abs(values - 28.)))
    assert peaks['max'][0][i28].size > 0, 'no maxima collected at rho=28'



def test_bifurcation_ensemble():
    # the default sweep is ONE ensemble forecast: member k at values[k]
    model = _l63_model()
    vals = [20., 28., 35.]
    values, peaks = ntsa_tools.bifurcation_sweep(model, 'rho', vals,
                                                 t_transient=5., t_sample=10.)
    assert all(len(per_obs) == 3 for per_obs in peaks['max'])
    i28 = int(np.argmin(np.abs(values - 28.)))
    assert peaks['max'][0][i28].size > 0, 'no maxima collected at rho=28'
    # rho=20 is a fixed point, rho=28 chaotic: their extrema must differ
    assert peaks['max'][0][0].size != peaks['max'][0][i28].size or not np.allclose(
        peaks['max'][0][0].mean(), peaks['max'][0][i28].mean()), \
        'members look identical — per-member parameters were not applied'


def test_correlation_dimension_poincare():
    t = np.arange(0, 400, 0.05)
    D2c, _ = ntsa_tools.correlation_dimension(np.column_stack([np.cos(t), np.sin(t)]))
    assert 0.8 < D2c < 1.2, f'circle D2 = {D2c}'
    a = 2 + np.cos(np.sqrt(2) * t)
    torus = np.column_stack([a * np.cos(t), a * np.sin(t), np.sin(np.sqrt(2) * t)])
    D2t, _ = ntsa_tools.correlation_dimension(torus)
    assert 1.6 < D2t < 2.4, f'2-torus D2 = {D2t}'
    # plane-crossing section: two-tone -> a loop (D2 ~ 1); pure sine -> a tight cluster
    x = np.sin(t) + 0.5 * np.sin(np.sqrt(2) * t)
    P = ntsa_tools.poincare_section(x, ntsa_tools.optimal_lag(x))
    assert len(P) > 20, f'only {len(P)} crossings'
    D2p, _ = ntsa_tools.correlation_dimension(P)
    assert 0.6 < D2p < 1.4, f'two-tone section D2 = {D2p}'
    xs = np.sin(t)
    Ps = ntsa_tools.poincare_section(xs, ntsa_tools.optimal_lag(xs))
    assert np.ptp(Ps, axis=0).max() < 0.05 * np.ptp(xs), 'sine section not a tight cluster'


def test_kaplan_yorke():
    assert abs(lyap.kaplan_yorke([0.906, 0.0, -14.57]) - 2.062) < 0.01
    assert lyap.kaplan_yorke([-0.5, -1.0]) == 0.0, 'all-negative spectrum must give 0'
    d = lyap.kaplan_yorke([0.0005, -0.0014, -0.0347])  # 2-torus (two neutrals) reads ~2
    assert 1.8 < d < 2.2, f'2-torus D_KY = {d}'


def test_mds_smoke():
    t, y, psi, dt = _l63_traj()
    gamma, idx = ntsa_tools.classical_mds(psi)
    n = min(psi.shape[0], 2000)
    assert gamma.shape == (n, 3), f'gamma shape {gamma.shape}'
    assert len(idx) == n
    v = gamma.var(axis=0)
    assert v[0] >= v[1] >= v[2], f'MDS coordinate variances not descending: {v}'


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [test_ami_fnn_sine, test_delay_embed_recurrence, test_stationary_start,
         test_l63_spectrum,
         test_l63_leading, test_l96_leading_guard, test_vdp_leading_not_chaotic,
         test_l63_fnn, test_classify,
         test_respawn_observation,
         test_correlation_dimension_poincare, test_kaplan_yorke,
         test_bifurcation_smoke, test_bifurcation_ensemble, test_mds_smoke]

if __name__ == '__main__':
    n_fail = 0
    t_all = time.time()
    for test in TESTS:
        t0 = time.time()
        try:
            test()
            print(f'PASS  {test.__name__}  ({time.time() - t0:.1f} s)')
        except Exception:
            n_fail += 1
            print(f'FAIL  {test.__name__}  ({time.time() - t0:.1f} s)')
            traceback.print_exc()
    total = time.time() - t_all
    print(f'\n{len(TESTS) - n_fail}/{len(TESTS)} tests passed in {total:.1f} s')
    sys.exit(1 if n_fail else 0)


def test_respawn_preserves_structure_and_dtype():
    # complex spectral states must survive respawn (no float cast), and the
    # structural fixed_params (KS: Nx, nu, L) must be carried over
    from dynamodels.physical import KS

    m = KS(Nx=64, dt=0.1, nu=0.08)
    m2 = ntsa_tools.respawn(m)
    assert m2.Nx == m.Nx and m2.dt == m.dt
    assert np.iscomplexobj(m2.psi0)
    assert np.isclose(m2.L, m.L)
    pa, _ = m.time_step(Nt=20)
    pb, _ = m2.time_step(Nt=20)
    assert np.allclose(pa, pb)
