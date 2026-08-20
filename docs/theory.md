# Theory

The equations behind each method, with the notation used throughout the package:
a scalar observable $x(t)$ sampled at $\Delta t$ (so $x_i = x(t_i)$, $N$ samples),
a model state $\psi \in \mathbb{R}^{N_\phi}$, and delay $\zeta$ expressed in
samples. The methods are grouped by what they do: reconstructing the phase space
from a record, diagnosing the signal and the reconstructed attractor, computing
Lyapunov exponents, and classifying the regime. References are collected at the
bottom.

## Phase-space reconstruction

How coordinates for the attractor are obtained — from a scalar record (delay
embedding, with its two free parameters) or from the full state trajectory (MDS).

### Delay embedding (Takens)

The attractor is reconstructed from the scalar record by the delay map

$$
\mathbf{Y}_i = \left( x_i,\ x_{i+\zeta},\ \dots,\ x_{i+(d-1)\zeta} \right)
\in \mathbb{R}^d ,
$$

which is an embedding (diffeomorphic to the true attractor) for
$d > 2\,d_A$ with $d_A$ the attractor's box-counting dimension [Takens 1981].
`delay_embed(x, dim, lag)` builds $\mathbf{Y}$; the two free parameters are
chosen as follows.

### Optimal lag: average mutual information

The AMI between the signal and its $\zeta$-shifted copy,

$$
I(\zeta) = \sum_{k,l} p_{kl}(\zeta)\,
\ln \frac{p_{kl}(\zeta)}{p_k\, p_l},
$$

is estimated on a histogram ($p_k$: marginal bin probabilities, $p_{kl}$:
joint probabilities of $x_i$ falling in bin $k$ and $x_{i+\zeta}$ in bin $l$).
`optimal_lag` returns the **first local minimum** of $I(\zeta)$ — the classic
compromise between independence (large $I$ drop) and staying on the same
attractor fold [Fraser & Swinney 1986].

### Embedding dimension: false nearest neighbours

For each point $\mathbf{Y}_i$ in dimension $d$ with nearest neighbour
$\mathbf{Y}_{n(i)}$ at distance $R_d(i)$, the neighbour is declared **false** if
unfolding to $d+1$ stretches it too much [Kennel et al. 1992]:

$$
\frac{\left| x_{i+d\zeta} - x_{n(i)+d\zeta} \right|}{R_d(i)} > R_\mathrm{tol}
\quad \text{or} \quad
\frac{R_{d+1}(i)}{\sigma_x} > A_\mathrm{tol},
$$

with $R_\mathrm{tol} = 10$, $A_\mathrm{tol} = 2$ and $\sigma_x$ the signal's
standard deviation. `false_nearest_neighbours` returns the smallest $d$ whose
false-neighbour fraction drops below 1%.

### Classical multidimensional scaling

From the pairwise distance matrix $D_{ij}$ of the (subsampled) state
trajectory, the Gram matrix is recovered by double centring,

$$
B = -\tfrac{1}{2}\, H\, D^{(2)}\, H, \qquad
H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^{\!\top},
$$

where $D^{(2)}$ holds squared distances. The embedding coordinates are
$\gamma_j = \sqrt{\mu_j}\, v_j$ from the top eigenpairs $(\mu_j, v_j)$ of $B$
— a distance-preserving 3-D view of the attractor that requires no choice of
observable.

## Signal and attractor diagnostics

Qualitative fingerprints of the regime (PSD peaks, return map, recurrence,
Poincaré section) and the quantitative attractor dimension $D_2$.

### Power spectral density

`fun_PSD` uses the one-sided FFT amplitude spectrum,

$$
\mathrm{PSD}(f_k) = \frac{2}{N} \left| \sum_{j=0}^{N-1} x_j\,
e^{-2\pi i jk/N} \right| , \qquad f_k = \frac{k}{N\,\Delta t},\quad
k < N/2 .
$$

The classifier extracts the dominant peaks $f_1, f_2, \dots$ from it
(`find_peaks` with prominence thresholds) and tests their ratios for
rationality.

### First-return map and peak clusters

Successive local maxima of the signal form the return map
$x_{\max}(i+1)$ vs $x_{\max}(i)$. A period-$k$ orbit produces $k$ tight
clusters; `count_peak_clusters` counts maxima levels that differ by more than
`tol` $\times$ the signal range. This check outranks a positive
perturbation-growth exponent in the classifier: $k$ tight maxima levels over a
long record are incompatible with chaos (see the
[decision tree](classification.md)).

### Recurrence plot

$$
R_{ij} = \Theta\!\left( \varepsilon - \left\| \mathbf{Y}_i - \mathbf{Y}_j
\right\| \right),
$$

with $\Theta$ the Heaviside step and $\varepsilon$ set to 10% of the maximum
pairwise distance (retargeted to a fixed 10% recurrence rate when the density
degenerates). Periodic dynamics give unbroken diagonals; quasiperiodic
dynamics give diagonals of varying spacing; chaos gives short, broken
diagonals [Eckmann et al. 1987].

### Poincaré section

`poincare_section` intersects the 3-D delay embedding with the plane
$x(t + 2\zeta) = \text{level}$ (median by default), keeping upward crossings.
Between bracketing samples $i$ and $i+1$ the crossing is linearly
interpolated,

$$
\mathbf{P} = \mathbf{Y}_i + w\,(\mathbf{Y}_{i+1} - \mathbf{Y}_i),
\qquad w = \frac{s_i}{s_i - s_{i+1}},
$$

where $s = Y_3 - \text{level}$ is the signed distance to the plane. A
limit cycle of period $k$ appears as $k$ points, a 2-torus as a closed loop,
chaos as a fractal scatter.

### Correlation dimension (Grassberger–Procaccia)

The correlation sum over pairs of points on the attractor (or section),

$$
C(r) = \frac{2}{n(n-1)} \sum_{i<j} \Theta\!\left( r -
\left\| \mathbf{Y}_i - \mathbf{Y}_j \right\| \right)
\;\propto\; r^{D_2} \quad (r \to 0),
$$

gives $D_2$ as the slope of $\ln C$ vs $\ln r$ fitted between the 2nd and
50th distance percentiles [Grassberger & Procaccia 1983].
For a $k$-torus $D_2 \approx k$ on the attractor and $\approx k-1$ on its
Poincaré section — one of the three torus-dimension witnesses used in the
[worked example](torus.md).

## Lyapunov exponents

Three routes, by what is available: the full spectrum needs the model's
right-hand side (and ideally its Jacobian); the perturbation-growth estimate
only needs a model that can be re-integrated; the Rosenstein estimate needs
nothing but the measured series. The Kaplan–Yorke dimension follows from the
spectrum.

### Full spectrum: Benettin QR

The full spectrum comes from evolving the state jointly with a tangent basis
$\Psi \in \mathbb{R}^{N_\phi \times n_\mathrm{exp}}$ under the variational
equation

$$
\dot{\psi} = f(\psi), \qquad
\dot{\Psi} = J(\psi)\,\Psi, \qquad
J = \frac{\partial f}{\partial \psi},
$$

(RK4, analytic $J$ for Lorenz63/96, finite differences otherwise), with QR
re-orthonormalization $\Psi = QR$ every $N_\mathrm{gs}$ steps. The exponents
are the time-averaged log growth of the diagonal of $R$ [Benettin et al. 1980]:

$$
\lambda_j = \lim_{T \to \infty} \frac{1}{T} \sum_{m} \ln R_{jj}^{(m)} .
$$

**Convergence control.** The neutral (along-flow) exponent decays to zero only
as $\mathcal{O}(1/T)$, so the horizon is extended (up to `t_max`) until the
halving test

$$
\left| \lambda_j(T) - \lambda_j(T/2) \right| <
\max\!\left( \text{atol},\ \text{rtol} \cdot |\lambda_j| \right)
$$

passes at two consecutive checks (one can be a phase coincidence).

### Leading exponent from perturbation growth

Without any Jacobian (so it also works for discrete maps), `leading_lyapunov`
evolves an ensemble of perturbed copies $\psi + \delta_0 \mathbf{e}$ and fits
the linear region of the mean log separation,

$$
\lambda_1 \approx \frac{d}{dt} \left\langle \ln \left\|
\psi_\mathrm{pert}(t) - \psi_\mathrm{ref}(t) \right\| \right\rangle ,
$$

between the transient and the saturation plateau (automatic window; fit
rejected if $R^2 < 0.5$).

**Saturation guard.** On a stable orbit, non-normal transient amplification
can produce a large positive slope even though there is no chaos
(thermoacoustic systems read $\lambda_1 \sim 200$ this way). The fit is
rejected when the separation saturates far below the attractor diameter while
the tail has stopped growing:

$$
\text{sat} < 0.05 \cdot \mathrm{diam}
\quad \text{and} \quad
\text{tail slope} < 0.1\,\lambda_1 .
$$

### Data-only: Rosenstein estimator

When only a measured series is available, `rosenstein_lyapunov` applies the same
idea to the delay embedding [Rosenstein et al. 1993]: each reference point
$\mathbf{y}_i$ is paired with its nearest neighbour $\mathbf{y}_j$ at least one
Theiler window $w$ away in time ($|i - j| > w$, with $w$ the mean inter-maximum
spacing, so pairs sample different orbits rather than adjacent samples of the
same one), and the mean log divergence

$$
S(k) = \left\langle \ln \left\| \mathbf{y}_{i+k} - \mathbf{y}_{j+k} \right\|
\right\rangle_{(i,j)}
$$

is fitted over its linear window with the same machinery — and the same $R^2$
and growth-range guards — as above, so periodic or noise-bound data returns nan
rather than a spurious slope. Pairs are averaged in 8 blocks whose slope spread
gives the quoted uncertainty. This is what `DataSeries.analyze` runs by default,
letting measured data reach the `chaotic` label with no model at all:

![Rosenstein log-divergence fit on Lorenz63 measurements](assets/ntsa_data_l63_rosenstein.png)

*Lorenz63 from measurements only: $\lambda_1 = 0.97 \pm 0.06$ against the true 0.906.*

### Kaplan–Yorke dimension

With the exponents sorted $\lambda_1 \geq \lambda_2 \geq \dots$ and $j$ the
largest index with $\sum_{i=1}^{j} \lambda_i \geq 0$,

$$
D_\mathrm{KY} = j + \frac{\sum_{i=1}^{j} \lambda_i}{\left| \lambda_{j+1}
\right|} ,
$$

conjectured equal to the information dimension [Kaplan & Yorke 1979]. Shown as
a text box on the delay-portrait panel.

## Regime classification

A $k$-torus carries exactly $k$ neutral exponents ($|\lambda| <$ `neutral_tol`)
and no positive one; chaos replaces one neutral direction with a positive
exponent. Together with $D_2$ on the attractor ($\approx k$) and on the
section ($\approx k-1$) this fixes the torus dimension — the full decision
order (fixed point $\to$ period-$k$ clusters $\to$ chaos $\to$
quasiperiodic/locked) is on the [classification page](classification.md), and
the Lorenz96 Ruelle–Takens–Newhouse route is the [worked example](torus.md).

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
