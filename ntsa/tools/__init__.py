"""Nonlinear time-series analysis tools for dynamical-system models.

One module per job, everything re-exported here (``from ntsa.tools import ...``):

- `runners` — `respawn`, `run_long`
- `embedding` — delay embeddings and lag/dimension selection
- `maps` — return maps, Poincare sections, peak-based transient/period detection
- `geometry` — recurrence matrices, classical MDS, correlation dimension
- `statistics` — PSD, autocorrelation, summary statistics
- `lyapunov` — Benettin QR spectrum, leading exponent, data-only Rosenstein

The higher-level analyses built on these live one level up: `ntsa.classification`
(regime decision tree) and `ntsa.bifurcation` (sweeps and diagrams).

Reference: Kantz & Schreiber, *Nonlinear Time Series Analysis* (2nd ed., CUP 2004).
"""

from ntsa.tools import lyapunov
from ntsa.tools.embedding import average_mutual_information, delay_embed, false_nearest_neighbours, optimal_lag
from ntsa.tools.geometry import classical_mds, correlation_dimension, recurrence_matrix
from ntsa.tools.lyapunov import kaplan_yorke, leading_lyapunov, lyapunov_spectrum, rosenstein_lyapunov
from ntsa.tools.maps import count_peak_clusters, first_return_map, poincare_section, stationary_start
from ntsa.tools.runners import respawn, run_long
from ntsa.tools.statistics import autocorrelation, fun_PSD, signal_stats

__all__ = [
    'autocorrelation', 'average_mutual_information', 'classical_mds', 'correlation_dimension',
    'count_peak_clusters', 'delay_embed', 'false_nearest_neighbours', 'first_return_map',
    'fun_PSD', 'kaplan_yorke', 'leading_lyapunov', 'lyapunov', 'lyapunov_spectrum',
    'optimal_lag', 'poincare_section', 'recurrence_matrix', 'respawn', 'rosenstein_lyapunov',
    'run_long', 'signal_stats', 'stationary_start',
]

_MOVED = {'classify_regime': 'ntsa.classification',
          'EXTREMA_COLORS': 'ntsa.bifurcation',
          'bifurcation_sweep': 'ntsa.bifurcation',
          'plot_bifurcation': 'ntsa.bifurcation'}


def __getattr__(name):
    # back-compat aliases, resolved lazily to avoid a circular import — the
    # canonical homes are ntsa.classification / ntsa.bifurcation
    if name in _MOVED:
        import importlib
        return getattr(importlib.import_module(_MOVED[name]), name)
    raise AttributeError(f"module 'ntsa.tools' has no attribute {name!r}")
