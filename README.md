# Polymarket FAA Outage Attribution — Forecast Entry

Forecasts the Yes price of every market in Polymarket's FAA outage
attribution family, minute by minute, 4 hours ahead — five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per market per minute.

## Entry point

`main.py` (Python 3.13, standard library only, no network, deterministic).
Run locally on the kit's practice cases:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory and writes one `<id>.out`
per case (`{"forecasts": [{"id", "q": [[...5 quantiles...] x 240]}]}`).

## Model

Persistence median + adaptive, heavy-tailed spread:

```
q_tau(h) = clamp(x + z_tau * sigma * h**0.4, 0, 1)
```

- **Median = last price.** A prediction-market price is (close to) a
  martingale, so the last price is the best point forecast. Every drift
  variant tried in backtesting scored worse than zero drift.
- **sigma per market**, from the median absolute deviation of its 1-minute
  price moves over the last 6 hours (robust to long flat stretches),
  floored at 0.0012 and capped at 0.02. If the 6-hour window holds fewer
  than 20 nonzero moves, it extends back until enough real moves are seen —
  a quiet market that jumped 0.01 three times should not be forecast as
  frozen.
- **Outer quantiles (5% / 95%) widened 3x** vs the normal z: these prices
  move in steps and jump, so the true conditional distribution has far
  heavier tails than a Gaussian random walk. Inner quantiles keep normal
  width.
- **Spread grows like h**0.4**, not sqrt(h): realised 4-hour moves grow
  more slowly than a random walk because prices sit still for long
  stretches and partially mean-revert inside the day.

## Backtest (practice cases, NVIDIA CEO event)

Forecast each practice case p00–p04 and score against the next case's
realised prices over the 240-minute horizon (mean pinball loss):

| transition | this entry | house baseline |
|---|---|---|
| p00 → p01 | 0.5289 | 0.5361 |
| p01 → p02 | 0.3313 | 0.3804 |
| p02 → p03 | 0.3717 | 0.4084 |
| p03 → p04 | 0.3008 | 0.3686 |
| p04 → p05 | 0.2432 | 0.2988 |
| **mean** | **0.3552** | **0.3985** |

~11% better than the baseline on the same five transitions. (The official
practice baseline is 0.363589 on all six cases; p05's own horizon has no
locally available answer, so it is not in this table.) The practice is a
different event from the final, so this is a sanity check on the method,
not a promise about the final score.

## Verification

- All six practice outputs: valid shape, every market exactly once,
  240 rows x 5 non-decreasing quantiles in [0, 1], each file < 1 MiB.
- Two runs byte-identical (deterministic, no randomness).
- Full run of all six practice cases: ~0.33 s (limit is 8 s per case).
- Edge cases verified: a market with only 3 prices, a market with a
  single price, and a 61-minute flat market all produce valid outputs
  (sigma falls back to the floor).

## License

MIT — see LICENSE.
