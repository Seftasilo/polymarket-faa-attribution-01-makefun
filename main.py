"""Forecast Polymarket FAA outage attribution Yes prices, 4 hours ahead.

Model: persistence median + adaptive, heavy-tailed spread.

    q_tau(h) = clamp(x + z_tau * sigma * h**0.4, 0, 1)

- x is the market's last price. A prediction-market price is (close to) a
  martingale, so the last price is the best point forecast: the median
  forecast is x, with no drift.
- sigma is per market, from the median absolute deviation of its 1-minute
  price moves over the last 6 hours (360 moves), floored at 0.0012 and
  capped at 0.02. If that window holds fewer than 20 nonzero moves, the
  window is extended backwards until 20 nonzero moves are seen, so a
  thin market that jumped a few times is not forecast as frozen. If there
  is no nonzero move at all in the whole history, sigma is the floor.
- The outer quantiles (5% / 95%) use a z widened 3x: these prices move in
  steps and jump, so the true conditional distribution has far heavier
  tails than a Gaussian random walk. The inner quantiles keep the normal z.
- The spread grows like h**0.4, not sqrt(h): realised 4-hour moves grow
  more slowly than a random walk because prices sit still for long
  stretches and partially mean-revert inside the day.

Reads every *.in in MAKEFUN_INPUT_DIR (default /input) and writes one
<id>.out per case to MAKEFUN_OUTPUT_DIR (default /output). Standard
library only, deterministic, no network.
"""

import json
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Standard normal quantiles at 0.05, 0.25, 0.5, 0.75, 0.95.
# The outer two are widened 3x for the heavy tails of step-and-jump moves.
Z = (-1.644854 * 3.0, -0.674490, 0.0, 0.674490, 1.644854 * 3.0)

HORIZON_DEFAULT = 240
WINDOW_MINUTES = 360          # 6 hours of 1-minute moves
MIN_NONZERO_MOVES = 20        # extend the window back until this many
SIGMA_FLOOR = 0.0012
SIGMA_CAP = 0.02
MAD_TO_SIGMA = 1.4826         # consistency factor for a normal distribution


def market_sigma(prices):
    """Robust per-minute volatility: the MAD of the 1-minute moves.

    The window is the last 6 hours (360 moves). If it holds fewer than 20
    nonzero moves, it is extended backwards until 20 nonzero moves are
    in hand (or the history runs out), so a thin market that jumped a few
    times is not forecast as frozen. The MAD is the classic median
    absolute deviation from the window's median move, scaled by the
    normal consistency factor 1.4826.
    """
    n = len(prices)
    if n < 2:
        return SIGMA_FLOOR

    start = max(0, n - 1 - WINDOW_MINUTES)
    nonzero = 0
    for i in range(n - 1, start, -1):
        if prices[i] != prices[i - 1]:
            nonzero += 1
    while nonzero < MIN_NONZERO_MOVES and start > 0:
        start -= 1
        if prices[start + 1] != prices[start]:
            nonzero += 1

    moves = [prices[i] - prices[i - 1] for i in range(start + 1, n)]
    med = statistics.median(moves)
    sigma = MAD_TO_SIGMA * statistics.median([abs(d - med) for d in moves])
    if sigma < SIGMA_FLOOR:
        return SIGMA_FLOOR
    if sigma > SIGMA_CAP:
        return SIGMA_CAP
    return sigma


def forecast_market(prices, horizon, h_factor):
    """240 (or horizon) rows of 5 quantiles for one market."""
    # A market with no prices at all cannot happen per the rules (its grid
    # starts at its first point); fall back to 0.5 rather than crashing.
    x = prices[-1] if prices else 0.5
    sigma = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * h_factor[h - 1]
        row = []
        prev = 0.0
        for z in Z:
            q = x + z * spread
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            q = round(q, 6)
            # Rounding can only move a value by 1e-6; keep rows
            # non-decreasing exactly as the format requires.
            if q < prev:
                q = prev
            row.append(q)
            prev = q
        rows.append(row)
    return rows


def forecast_case(case):
    horizon = case.get("horizon_minutes", HORIZON_DEFAULT)
    h_factor = [h ** 0.4 for h in range(1, horizon + 1)]
    forecasts = []
    for market in case["markets"]:
        forecasts.append({
            "id": market["id"],
            "q": forecast_market(market["prices"], horizon, h_factor),
        })
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(
        name for name in os.listdir(INPUT_DIR) if name.endswith(".in")
    )
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
