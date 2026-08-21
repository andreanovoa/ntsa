"""ntsa — nonlinear time-series analysis for dynamical-system models.

- `ntsa.tools` — the primitives, one module per job: delay embeddings,
  return/Poincaré maps, recurrence, MDS, signal statistics, and Lyapunov
  exponents (Benettin QR spectrum, leading exponent, data-only Rosenstein).
- `ntsa.classification` — `classify_regime`, the regime decision tree.
- `ntsa.bifurcation` — bifurcation sweeps and diagrams, plus a demo driver
  (``python -m ntsa.bifurcation``).
- `ntsa.characterize` — per-case diagnostic figure rows and a demo driver
  (``python -m ntsa.characterize``).
- `ntsa.data` — `DataSeries`, the equation-free pipeline for raw measured series
  (no model needed; Lyapunov/bifurcation methods are bypassed).

Works with any model implementing the `dynamodels.Model` interface (see the
"Model protocol" section of the README) — `dynamodels` is the reference
implementation used by the demos and tests, but any duck-type is accepted.

Reference: Kantz & Schreiber, *Nonlinear Time Series Analysis* (2nd ed., CUP 2004).
"""

__version__ = "0.3.0"
