"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

For every trophy of Grand Theft Auto VI on PlayStation 5, forecast five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the case's
target, from the history of reads the house made of the game's trophy page.

How it works
------------
A trophy's rate is earners over players tracked, and the two move very
differently after a launch: the player base jumps at every Exophase refresh
as new players sync, while earners grow only as players actually play. So
the program forecasts the two separately and divides:

* players tracked P(t): log P is fit by least squares against
  x = ln(1 + hours since the game's release), over the distinct snapshots
  in the history (consecutive identical reads are one Exophase state, not
  many observations);
* each trophy's earners E_i(t) = rate_i(t) / 100 * P(t): log E_i is fit the
  same way, its slope shrunk toward its tier's mean slope (platinum, gold,
  silver, bronze) when the trophy's own history is thin, so a rare trophy
  with few earners borrows strength from its tier;
* the median rate is 100 * E_hat(T) / P_hat(T) at the target, blended
  toward the last read's rate in logit space (the blend leans on the trend
  more the more distinct snapshots the history holds).

The spread (logit units) combines the fit's prediction uncertainty
(residual spread times the extrapolation leverage) with a floor for the
refresh shocks a trend cannot see, and is clamped to [0.25, 1.2].

When the history is completely flat -- every read identical, as on launch
night before Exophase's first refresh -- no trend exists to fit. The
program then falls back on the dilution the kit's own published baseline
diagnostics imply: on the practice cases the persistence baseline scores a
loss of about 1.51 with a typical miss of about 0.68 points, which is what
a uniform division of the last rates by about 5 reproduces (the first
refresh took players tracked from 638 to 3,868 while a typical trophy's
earners grew only ~1.2x, so rates fell ~5x). The flat-history fallback
divides the last rates by D = 1 + 4 * sqrt(horizon_hours / 1.5), capped at
20, with a spread of 0.30 (the spread was tuned on the three dilution
shapes consistent with the published baseline diagnostics: 0.25 scores
best if the dilution is uniform, 0.45 if it scales with the rate; 0.30 is
the compromise across them).

A trophy the last read has at 100% (every tracked player holds it) is
forecast at 99.9 with a wide spread: it stays near-universal, but the
exact share is the least certain number in the case.

Output: for each case, <id>.out = {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]} in the case's trophy order, each q
rounded to 4 decimals (half up, as the house baseline rounds it).

Standard library only; deterministic (no randomness, no clock, no network);
reads every <id>.in in the input directory and writes <id>.out per case.
A case that fails to parse still gets a persistence forecast, never a
crash: an invalid case ranks below every valid entry and takes no prize.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry ships no such module.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE_FALLBACK = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- model constants (see the module docstring for why) -------------------
SIGMA_RW = 0.5            # random-walk spread, logit units per sqrt day
SPREAD_FLOOR = 0.25       # unmodelled refresh shocks, logit units
SPREAD_CAP = 1.2
SPREAD_FLAT = 0.30        # spread when the history is flat
RESID_FLOOR = 0.15        # least residual spread a fit is credited with
DILUTION_PER_90MIN = 4.0  # flat-history dilution: D = 1 + 4*sqrt(h/1.5h)
DILUTION_CAP = 20.0
EARNERS_FLOOR = 0.5       # continuity correction for log(earners)
SLOPE_P_MAX = 2.5         # log-log slope clamps (launch growth saturates)
SLOPE_E_MIN, SLOPE_E_MAX = -1.0, 3.5
UNIVERSAL_RATE = 99.5     # a trophy at/above this is "everyone holds it"
UNIVERSAL_MEDIAN = 99.9
UNIVERSAL_SPREAD = 0.9


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    c = clamp_rate(pct)
    return math.log(c / (100.0 - c))


def inv_logit(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs: list, ys: list) -> tuple:
    """Least squares y ~ a + b*x. Returns (intercept, slope, resid_std, sxx, xbar, n)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return ybar, 0.0, 0.0, 0.0, xbar, n
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    intercept = ybar - slope * xbar
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    resid_std = math.sqrt(sum(r * r for r in resid) / dof)
    return intercept, slope, resid_std, sxx, xbar, n


def distinct_snapshots(case: dict) -> list:
    """History reads with consecutive duplicates removed: each distinct
    (players_tracked, rates) pair is one Exophase state."""
    snaps = []
    prev = None
    for read in case.get("history", []):
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key == prev:
            continue
        prev = key
        snaps.append(read)
    return snaps


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    history = case.get("history", [])
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    release = RELEASE_FALLBACK
    try:
        release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    except (KeyError, TypeError, ValueError):
        pass
    target = parse_time(case["target"])

    def x_of(moment: datetime) -> float:
        hours = (moment - release).total_seconds() / 3600.0
        return math.log(max(1.0 + hours, 0.5))

    x_target = x_of(target)
    snaps = distinct_snapshots(case)
    last = history[-1]
    last_rates = last["rates"]
    last_players = last["players_tracked"]

    def last_rate(i: int) -> float:
        r = last_rates[i]
        if r is not None:
            return float(r)
        for read in reversed(history[:-1]):
            if read["rates"][i] is not None:
                return float(read["rates"][i])
        return 0.0

    # --- flat history: no trend to fit; use the dilution fallback. -------
    if len(snaps) < 2:
        dilution = min(1.0 + DILUTION_PER_90MIN * math.sqrt(horizon_hours / 1.5),
                       DILUTION_CAP)
        rows = []
        for i, trophy in enumerate(trophies):
            r = last_rate(i)
            if r >= UNIVERSAL_RATE:
                median, spread = UNIVERSAL_MEDIAN, UNIVERSAL_SPREAD
            else:
                median, spread = clamp_rate(r / dilution), SPREAD_FLAT
            lg = logit(median)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
        return {"trophies": rows}

    # --- players tracked: log P ~ x. --------------------------------------
    xs, ys_p = [], []
    for read in snaps:
        xs.append(x_of(parse_time(read["at"])))
        ys_p.append(math.log(max(float(read["players_tracked"]), 1.0)))
    a_p, b_p, _, _, _, _ = ols(xs, ys_p)
    b_p = min(max(b_p, 0.0), SLOPE_P_MAX)
    a_p = sum(ys_p) / len(ys_p) - b_p * (sum(xs) / len(xs))
    log_p_target = a_p + b_p * x_target

    # --- earners per trophy: log E ~ x, slope shrunk to the tier mean. ----
    tiers: dict = {}
    for i, trophy in enumerate(trophies):
        tiers.setdefault(trophy.get("tier", "bronze"), []).append(i)
    slopes: dict = {}
    fits: dict = {}
    for i, trophy in enumerate(trophies):
        ys_e = []
        for read in snaps:
            rate = read["rates"][i]
            earners = (float(rate) / 100.0 * float(read["players_tracked"])
                       if rate is not None else 0.0)
            ys_e.append(math.log(max(earners, EARNERS_FLOOR)))
        fits[i] = ols(xs, ys_e)
        slopes[i] = fits[i][1]
    tier_mean = {}
    for tier, members in tiers.items():
        tier_mean[tier] = sum(slopes[i] for i in members) / len(members)

    trend_weight = min(0.85, 0.4 + 0.09 * len(snaps))
    rows = []
    for i, trophy in enumerate(trophies):
        r = last_rate(i)
        if r >= UNIVERSAL_RATE:
            median, spread = UNIVERSAL_MEDIAN, UNIVERSAL_SPREAD
            lg = logit(median)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
            continue
        a_e, b_e, resid, sxx, xbar, n = fits[i]
        b_shrunk = (n * b_e + 2.0 * tier_mean[trophy.get("tier", "bronze")]) / (n + 2.0)
        b_shrunk = min(max(b_shrunk, SLOPE_E_MIN), SLOPE_E_MAX)
        a_shrunk = sum(math.log(max(
            (float(read["rates"][i]) / 100.0 * float(read["players_tracked"])
             if read["rates"][i] is not None else 0.0), EARNERS_FLOOR))
            for read in snaps) / n - b_shrunk * xbar
        log_e_target = a_shrunk + b_shrunk * x_target
        trend_rate = clamp_rate(100.0 * math.exp(log_e_target - log_p_target))
        median_logit = (trend_weight * logit(trend_rate)
                        + (1.0 - trend_weight) * logit(r))
        # Spread: fit uncertainty (residual x leverage) + refresh floor.
        leverage = 1.5
        if sxx > 0:
            leverage = math.sqrt(1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx)
        sigma_fit = max(resid, RESID_FLOOR) * leverage
        sigma_base = SPREAD_FLOOR * math.sqrt(1.0 + horizon_days / 2.0)
        spread = min(max(math.sqrt(sigma_fit ** 2 + sigma_base ** 2),
                         SPREAD_FLOOR), SPREAD_CAP)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(median_logit + z * spread)) for z in Z]})
    return {"trophies": rows}


def persistence_forecast(case: dict, name: str) -> dict:
    """Fallback: the house baseline's median (last rate) with its spread."""
    last = case["history"][-1]
    spread = SIGMA_RW * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(float(rate) if rate is not None else 0.0)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if case.get("levels") != LEVELS:
                raise ValueError(f"{path.name} asks for levels "
                                 f"{case.get('levels')}, not {LEVELS}")
            forecast = forecast_case(case, path.name)
        except Exception as err:  # never leave a case without an output
            print(f"{path.name}: {err}; falling back to persistence",
                  file=sys.stderr)
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = persistence_forecast(case, path.name)
            except Exception:
                continue
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
