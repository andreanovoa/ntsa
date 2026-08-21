# Tutorials

Executed notebooks, rendered to HTML (outputs and figures included) on every docs
build; the GitHub column opens the `.ipynb` itself.

| Notebook | What it covers | |
| --- | --- | --- |
| [NTSA with `ntsa` + `dynamodels`](tutorials/tutorial_ntsa.html) | Long clean trajectories, signal statistics, Takens delay embedding, first-return map and recurrence plot, Lyapunov exponents (leading, Benettin QR spectrum, Kaplan–Yorke and torus dimensions), classical MDS portraits, regime classification, the one-call `characterize()`, and a Lorenz-96 bifurcation sweep. | [GitHub](https://github.com/andreanovoa/ntsa/blob/main/tutorial_ntsa.ipynb) |

To regenerate the HTML locally: `python -m nbconvert --to html tutorial_ntsa.ipynb --output-dir docs/tutorials`.
