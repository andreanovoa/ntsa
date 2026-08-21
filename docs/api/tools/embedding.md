# `ntsa.tools.embedding`

## At a glance

| Function | One-liner |
| --- | --- |
| `delay_embed(x, dim, lag)` | Takens delay-embedding matrix `(N-(dim-1)*lag, dim)`. |
| `average_mutual_information(x, max_lag, n_bins=64)` | AMI for lags `1..max_lag`. |
| `optimal_lag(x, max_lag=None, n_bins=64)` | Delay $\zeta$ = first local minimum of the AMI (samples). |
| `false_nearest_neighbours(x, lag, ...)` | Kennel FNN embedding dimension `d` and per-dimension fractions. |

## Full reference

::: ntsa.tools.embedding
