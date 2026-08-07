# Worked example: torus dimensions along a Ruelle–Takens–Newhouse route

The torus dimension T^k is read off three independent witnesses: the number of
**neutral Lyapunov exponents** (k zeros: limit cycle 1, 2-torus 2, 3-torus 3 — chaos has
a positive one instead), the **correlation dimension** D2 of the attractor (≈ k, fractal
for chaos) and of the plane-crossing **Poincaré section** (≈ k−1: dots → loop → band).
Measured on Lorenz-96 (Nx=10), the route is textbook RTN, with locking windows
interleaving near the breakdown and no stable 3-torus (as RTN predicts — T³ is
generically unstable and the attractor turns strange directly):

| F | λ signature (n₀ = neutrals) | D2 state / section | regime |
| --- | --- | --- | --- |
| 2–3.95 | one zero | 1.05 / — | limit cycle (T¹) |
| 4.0–4.06 | one zero | — | locked windows (periodic on the torus) |
| 4.2–4.4 | **two zeros** | 1.8 / 0.9 | quasiperiodic (T²) |
| 4.45 | λ1=+0.016, n₀=2 | 2.0 / 1.0 | first chaos, interleaved with… |
| 4.5–4.55 | two zeros | 2.1–2.2 / 1.0 | …re-locked/QP windows (Arnold tongues) |
| 4.6 | λ1=+0.039 (converged) | 2.15 / 1.02 | chaotic **wrinkled torus** — section still loop-like |
| 5 | λ1=+0.07 (plateau, λ1·T grows) | 2.9 / 1.1 | chaos on a thickened torus remnant |
| 8 | 3 positive exponents | 4.8 / 1.8 | developed chaos (D_KY ≈ 6.5) |

This is why F=4.6 and F=5 "look QP": the strange attractor inherits the torus geometry
(D2 barely above 2, section barely above a loop) while the dynamics on it are already
exponentially divergent — the spectrum, not the geometry, makes the call.
