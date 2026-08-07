# Getting started

## Install

```bash
pip install ntsa            # once released; until then (dynamodels is not on PyPI yet either):
pip install "dynamodels @ git+https://github.com/andreanovoa/dynamodels" \
            "ntsa @ git+https://github.com/andreanovoa/ntsa"
```

## Quickstart

```python
from dynamodels.physical import Lorenz63
from ntsa import characterize as chz

chz.characterize([Lorenz63()], pdf_name='figs/l63.pdf')
```

```bash
python -m ntsa.characterize                     # 4-case demo -> figs/ntsa_defaults.pdf (+ .png)
python -m ntsa.characterize --model lorenz63 --param rho --values 20 28 100 350
python -m ntsa.tools --model lorenz63 --param rho --nP 40   # bifurcation sweep demo
```

Figures are written relative to the working directory (`figs/`), so `cd` to where
you want them first.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
ruff check ntsa/ tests/
```

## Releasing (maintainer note)

Releases publish to PyPI via GitHub Actions trusted publishing on version tags:
configure a trusted publisher for `andreanovoa/ntsa` (workflow `release.yml`,
environment `pypi`) at pypi.org, then `git tag v0.1.0 && git push --tags`.
