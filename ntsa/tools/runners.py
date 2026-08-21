"""Model runners: fresh instances and long integrations past the transient."""

import numpy as np

from ntsa.tools.maps import stationary_start


def respawn(model, psi0=None, dt=None, **overrides):
    """Fresh instance of ``type(model)`` with `alpha0` and fixed params carried over.

    Parameters in `overrides` win over the carried-over values. `psi0` defaults
    to ``model.psi0[:, 0]``; `dt` defaults to ``model.dt``.
    """
    if psi0 is None:
        psi0 = model.psi0[:, 0]
    params = {**model.alpha0,
              **{key: getattr(model, key) for key in model.fixed_params}}
    # observation config is neither an alpha nor a fixed param — carry it over too
    for key in ('observe_dims', 'observed_idx'):
        if hasattr(model, key):
            params[key] = list(getattr(model, key))
    params.update(overrides)
    psi0 = np.asarray(psi0)
    if psi0.dtype == object:
        psi0 = psi0.astype(float)
    return type(model)(psi0=psi0, dt=dt or model.dt, **params)


def run_long(model, t_run, t_transient=None, trim_transient=True):
    """Integrate `model` past the transient, then record `t_run` time units.

    Mutates `model` (history is reset to t=0 after the transient) and closes it.
    With ``trim_transient`` (default), residual transient drift that outlives
    ``t_transient`` is detected on the first observable (``stationary_start``)
    and the leading drifting samples are dropped, with a printed note — raise
    ``t_transient`` to keep the full horizon.

    Returns
    -------
    t : ndarray (Nt,)
    y : ndarray (Nt, Nq)
        Observable history of ensemble member 0.
    psi : ndarray (Nt, Nphi)
        State history of ensemble member 0.
    """
    if t_transient is None:
        t_transient = model.t_transient
    Nt_tr = int(round(t_transient / model.dt))
    if Nt_tr > 0:
        psi, _ = model.time_integrate(Nt=Nt_tr)
        model.update_history(psi[-1:], t=np.array([0.]), reset=True)
    psi, tt = model.time_integrate(int(round(t_run / model.dt)))
    model.update_history(psi, tt)  # keep the integrator's stamps (dt..Nt*dt after the reset to t=0)
    model.close()
    t, y, psi = model.hist_t, model.get_observable_hist()[:, :, 0], model.hist[:, :model.Nphi, 0]
    if trim_transient:
        i0 = stationary_start(y[:, 0])
        if i0:
            print(f'[run_long] trimmed {t[i0] - t[0]:.3g} t.u. of residual transient drift '
                  f'({100 * i0 / len(t):.0f}% of the record) — raise t_transient to keep the full horizon')
            t, y, psi = t[i0:], y[i0:], psi[i0:]
    return t, y, psi
