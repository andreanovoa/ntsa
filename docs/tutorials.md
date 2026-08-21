# Tutorials

Executed notebooks, rendered to HTML (outputs and figures included) on every docs
build; the GitHub column opens the `.ipynb` itself.

| Notebook | What it covers | |
| --- | --- | --- |
| [NTSA with `ntsa` + `dynamodels`](tutorials/tutorial_ntsa.html) | Long clean trajectories, signal statistics, Takens delay embedding, first-return map and recurrence plot, Lyapunov exponents (leading, Benettin QR spectrum, Kaplan–Yorke and torus dimensions), classical MDS portraits, regime classification, the one-call `characterize()`, and a Lorenz-96 bifurcation sweep. | [GitHub](https://github.com/andreanovoa/ntsa/blob/main/tutorial_ntsa.ipynb) |
| [Data-driven NTSA of an annular combustor](tutorials/tutorial_annular_data.html) | The equation-free pipeline (`ntsa.data.DataSeries`) on measured acoustic pressure from a turbulent annular combustor ([Zenodo 15609832](https://zenodo.org/records/15609832)): loading and conditioning the records, embedding diagnostics, Rosenstein $\lambda_1$, regime classification across four equivalence ratios, a band-passed vs raw comparison (overlaid PSDs, attractors, recurrence plots — and why the two branches classify differently), and an experimental bifurcation diagram. | [GitHub](https://github.com/andreanovoa/ntsa/blob/main/tutorial_annular_data.ipynb) |

To regenerate the HTML locally: `python -m nbconvert --to html tutorial_*.ipynb --output-dir docs/tutorials`.
