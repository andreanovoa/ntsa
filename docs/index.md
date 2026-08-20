# ntsa — nonlinear time-series analysis

Characterizes the dynamical regime of a model — or of a measured time series
alone (`ntsa.data.DataSeries`, no equations needed) — from a single long trajectory:
delay embedding (optimal lag + false nearest neighbours), Lyapunov exponents, regime
classification (fixed point / limit cycle period-k / frequency-locked / quasiperiodic / chaotic),
and a per-case diagnostic figure — one row of 8 panels
[time series (red, with zoom inset) | PSD (purple, semilogy) | 3-D delay portrait (green,
with $D_\mathrm{KY}$ box) | first-return map of local maxima (blue) | plane-crossing Poincaré section
(orange) | recurrence plot (black/white) | 3-D MDS | Lyapunov spectrum] — plus Lyapunov-fit,
Lyapunov-spectrum and classical-MDS pages, all in one multi-page PDF.

Reference: Kantz & Schreiber, *Nonlinear Time Series Analysis* (2004).

![8-panel characterization rows for the demo cases](assets/ntsa_defaults.png)

*Demo output (`python -m ntsa.characterize`): Lorenz63 chaotic and period-1, Van der Pol, Lorenz96.*

The same pipeline runs **from data alone** — a measured scalar record plus,
optionally, state snapshots for the MDS panel, no model equations: embedding,
$D_2$, a Rosenstein leading-exponent estimate, and the regime label
([`ntsa.data`](api/data.md)).

![8-panel characterization of Lorenz63 from measurements only](assets/ntsa_data_l63.png)

*Snapshot-only route (`ntsa.data.DataSeries`): the Lorenz63 case re-characterized
without the model — $\lambda_1 = 0.97 \pm 0.06$ (true 0.906), classified chaotic.*

## Ecosystem

- [`dynamodels`](https://github.com/andreanovoa/dynamodels) — the reference Model
  implementation (physical models used by the demos and tests).
- [romda](https://github.com/andreanovoa/real-time-bias-aware-DA) — real-time
  bias-aware data assimilation built on both packages.

## Citation

If you use this repository, please cite the software archive:

```bibtex
@software{novoa_ntsa,
  author = {Nóvoa},
  title = {ntsa: nonlinear time-series analysis for dynamical-system models},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.21840914},
  url = {https://doi.org/10.5281/zenodo.21840914},
}
```

The routines in this package were developed from the codes published as
supplementary material of Nóvoa & Magri (2022):

```bibtex
@article{novoa2022jfm,
  title = {Real-time thermoacoustic data assimilation},
  journal = {Journal of Fluid Mechanics},
  volume = {948},
  pages = {A35},
  year = {2022},
  doi = {10.1017/jfm.2022.653},
  url = {https://doi.org/10.1017/jfm.2022.653},
  eprint = {2106.06409},
  archivePrefix = {arXiv},
  author = {Nóvoa and Magri},
}
```

## References

- Kantz & Schreiber (2004). *Nonlinear Time Series Analysis*, 2nd ed., Cambridge Univ. Press.
- Kennel, Brown & Abarbanel (1992). Determining embedding dimension for phase-space
  reconstruction using a geometrical construction. *Phys. Rev. A* 45, 3403.
- Benettin, Galgani, Giorgilli & Strelcyn (1980). Lyapunov characteristic exponents for smooth
  dynamical systems and for Hamiltonian systems. *Meccanica* 15, 9–30.
- Rosenstein, Collins & De Luca (1993). A practical method for calculating largest Lyapunov
  exponents from small data sets. *Physica D* 65, 117–134.
- Ginelli, Poggi, Turchi, Chaté, Livi & Politi (2007). Characterizing dynamics with covariant
  Lyapunov vectors. *Phys. Rev. Lett.* 99, 130601.
