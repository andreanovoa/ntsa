# ntsa — nonlinear time-series analysis for dynamical-system models

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21843575-blue.svg)](https://doi.org/10.5281/zenodo.21843575)
[![PyPI](https://img.shields.io/pypi/v/ntsa)](https://pypi.org/project/ntsa/)
[![docs](https://github.com/andreanovoa/ntsa/actions/workflows/docs.yml/badge.svg)](https://andreanovoa.github.io/ntsa/)

Characterizes the dynamical regime of a model from a single long trajectory:
- Lyapunov exponents
- Delay embedding (optimal lag + false nearest neighbours)
- Regime classification (fixed point / limit cycle period-k / frequency-locked / quasiperiodic / chaotic)
- Yields a per-case diagnostic figure — one row of 8 panels. Example:

**Figure 1: Dynamical systems diagnostics** Time series | PSD | 3-D delay portrait | first-return map of local maxima | 
plane-crossing Poincaré section | recurrence plot | 3-D MDS | Lyapunov spectrum
<img width="2877" height="1659" alt="image" src="https://github.com/user-attachments/assets/d366a1f1-6980-4bee-ae1c-a00c514c33b4" />



**Documentation: [andreanovoa.github.io/ntsa](https://andreanovoa.github.io/ntsa/)**
| Tutorial: [`tutorial_ntsa.ipynb`](tutorial_ntsa.ipynb)


**Key Reference**: 
Kantz & Schreiber, *Nonlinear Time Series Analysis* (2004).

## Install

```bash
pip install ntsa
```

## Quickstart

```python
from dynamodels.physical import Lorenz63
from ntsa import characterize as chz

chz.characterize([Lorenz63()], pdf_name='figs/l63.pdf')
```

```bash
python -m ntsa.characterize          # 4-case demo -> figs/ntsa_defaults.pdf (+ .png)
```


Works with any model implementing the
[model protocol](https://andreanovoa.github.io/ntsa/protocol/) —
[`dynamodels`](https://github.com/andreanovoa/dynamodels) is the reference
implementation. Part of the same ecosystem as
[romda](https://github.com/andreanovoa/real-time-bias-aware-DA) (real-time
bias-aware data assimilation).

## Citation

If you use this repository, please cite the software archive:

```bibtex
@software{novoa_ntsa,
  author = {Nóvoa},
  title = {ntsa: nonlinear time-series analysis for dynamical-system models},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.21843575},
  url = {https://doi.org/10.5281/zenodo.21843575},
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

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
ruff check ntsa/ tests/
```


Releases: bump `version` in `pyproject.toml`, then `git tag vX.Y.Z && git push --tags`
(publishes to PyPI); docs deploy to GitHub Pages on every push to `main`.

## License

MIT
