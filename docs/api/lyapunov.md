# `ntsa.lyapunov`

## At a glance

| Function | One-liner |
| --- | --- |
| `get_rhs(model)` | RHS callable `f(u)` from `model.time_derivative` (continuous models only). |
| `fd_jacobian(f, u, eps=1e-6)` | Finite-difference Jacobian. |
| `get_jacobian(model, f=None)` | Analytic Jacobian for Lorenz63/96, finite-difference otherwise. |
| `lyapunov_spectrum(model, ...)` | Full spectrum via tangent-space QR (Benettin / Gram-Schmidt), sorted descending; runs until the running exponents converge (halving test), `full_output=True` returns the convergence history. |
| `leading_lyapunov(model, n_pert=8, ...)` | $\lambda_1 \pm$ std from perturbation-growth fit; works for **any** Model. |
| `fit_log_growth(t, log_seps)` | Linear-region fit of log separation; `nan` if $R^2 < 0.5$. |
| `kaplan_yorke(exponents)` | Kaplan–Yorke dimension from a spectrum (shown as a box on the delay-portrait panel). |
| `covariant_lyapunov_vectors(...)` | **TODO stub** (Ginelli et al. 2007) — raises `NotImplementedError`. |


## Full reference

::: ntsa.lyapunov
