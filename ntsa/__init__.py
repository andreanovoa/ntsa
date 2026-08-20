"""ntsa — nonlinear time-series analysis for dynamical-system models.

- `ntsa.tools` — delay embeddings, return/Poincaré maps, recurrence, MDS,
  regime classification, bifurcation sweeps.
- `ntsa.lyapunov` — Benettin QR spectrum, leading exponent, Kaplan-Yorke, and
  the data-only Rosenstein estimator.
- `ntsa.characterize` — per-case diagnostic figure rows and a demo driver
  (``python -m ntsa.characterize``).
- `ntsa.data` — `DataSeries`, the equation-free pipeline for raw measured series
  (no model needed; Lyapunov/bifurcation methods are bypassed).

Works with any model implementing the `dynamodels.Model` interface (see the
"Model protocol" section of the README) — `dynamodels` is the reference
implementation used by the demos and tests, but any duck-type is accepted.

Reference: Kantz & Schreiber, *Nonlinear Time Series Analysis* (2nd ed., CUP 2004).
"""

__version__ = "0.2.0"
