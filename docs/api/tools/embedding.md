# `ntsa.tools.embedding`

## At a glance

| Function | One-liner |
| --- | --- |
| `delay_embed(x, dim, lag)` | Takens delay-embedding matrix `(N-(dim-1)*lag, dim)`. |
| `average_mutual_information(x, max_lag, n_bins=64)` | AMI for lags `1..max_lag`. |
| `optimal_lag(x, max_lag=None, n_bins=64)` | Delay $\zeta$ = first local minimum of the AMI (samples). |
| `false_nearest_neighbours(x, lag, ...)` | Kennel FNN embedding dimension `d` and per-dimension fractions. |
| `cao_embedding_dimension(x, lag, d_max=10, theiler=0, tol=0.05, ...)` | Cao (1997) minimum embedding dimension `d` from the saturation of $E_1$, plus $E_1$ and $E_2$ ($E_2 \approx 1$ for every $d$ flags stochastic data). |
| `takens_bound(D=None, x=None, ...)` | Upper bound $\lfloor 2D+1 \rfloor$ on the minimum embedding dimension (Takens; Sauer, Yorke & Casdagli), with $D$ given or estimated as $D_2$ from the series. |

## Full reference

::: ntsa.tools.embedding
