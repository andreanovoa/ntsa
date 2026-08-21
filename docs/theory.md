# Theory

The equations behind each method, with the notation used throughout the package:
a scalar observable $x(t)$ sampled at $\Delta t$ (so $x_i = x(t_i)$, $N$ samples),
a model state $\psi \in \mathbb{R}^{N_\phi}$, and delay $\zeta$ expressed in
samples.

## At a glance

| Section | What it covers |
| --- | --- |
| [Phase-space reconstruction](theory/reconstruction.md) | Delay embedding (Takens) and its two free parameters — AMI lag, FNN dimension — plus classical MDS from the full state trajectory. |
| [Signal and attractor diagnostics](theory/diagnostics.md) | PSD peaks, first-return map and peak clusters, recurrence plot, Poincaré section, and the correlation dimension $D_2$. |
| [Lyapunov exponents](theory/lyapunov.md) | Benettin QR spectrum, perturbation-growth leading exponent, the data-only Rosenstein estimator, and the Kaplan–Yorke dimension. |
| [Regime classification](theory/classification.md) | The decision tree, the Lorenz-96 F-route bifurcation diagram, measured $\lambda_1$ tables, defaults, and notes. |

The Lorenz96 Ruelle–Takens–Newhouse route ties the four together in the
[worked example](torus.md).

## References

- Takens (1981). Detecting strange attractors in turbulence. *Lecture Notes in Math.* 898.
- Fraser & Swinney (1986). Independent coordinates for strange attractors from mutual information. *Phys. Rev. A* 33, 1134.
- Kennel, Brown & Abarbanel (1992). Determining embedding dimension for phase-space reconstruction. *Phys. Rev. A* 45, 3403.
- Eckmann, Kamphorst & Ruelle (1987). Recurrence plots of dynamical systems. *Europhys. Lett.* 4, 973.
- Grassberger & Procaccia (1983). Characterization of strange attractors. *Phys. Rev. Lett.* 50, 346.
- Benettin, Galgani, Giorgilli & Strelcyn (1980). Lyapunov characteristic exponents for smooth dynamical systems. *Meccanica* 15, 9.
- Rosenstein, Collins & De Luca (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D* 65, 117.
- Kaplan & Yorke (1979). Chaotic behavior of multidimensional difference equations. *Lecture Notes in Math.* 730.
- Kantz & Schreiber (2004). *Nonlinear Time Series Analysis*, 2nd ed., Cambridge University Press.
