# 🏆 World Cup 2026 Predictor

A Monte Carlo simulation engine that predicts the FIFA World Cup 2026 outcome using Elo ratings and Poisson regression.

## How It Works

1. **Elo Ratings** — Computes strength ratings for every international team from 49,000+ historical matches. Factors in tournament importance, recency, home advantage, and goal margin.

2. **Poisson Model** — A trained `PoissonRegressor` (scikit-learn) converts Elo gaps into expected goals per team. Uses 8 years of match data with 5 features: attack Elo, defense Elo, Elo difference, home advantage, and tournament weight.

3. **Tournament Simulation** — Runs 10,000 full tournament simulations covering all 48 teams across 12 groups, third-place qualification, and the complete knockout bracket (R32 → R16 → QF → SF → Final). Knockout ties go to extra time, then Elo-weighted penalty shootouts.

4. **Head-to-Head Predictor** — Interactive CLI to compare any two teams with win/draw/loss probabilities, expected goals, and most likely scoreline. If the most likely result is a draw, shows knockout progression with ET & penalties.

## Quick Start

```bash
pip install pandas numpy scipy scikit-learn tqdm tabulate requests
python oracle.py
```

Match data is downloaded automatically on first run from [martj42/international_results](https://github.com/martj42/international_results) and cached locally.

## Sample Output

```
  Rank  Team                      Elo  Champion%    Final%    Semi%
------  ----------------------  -----  -----------  --------  -------
     1  France                   2222  16.4%        26.8%     38.3%
     2  Spain                    2221  15.6%        24.4%     37.2%
     3  Argentina                2209  15.2%        23.8%     37.1%
     4  England                  2111  6.0%         11.8%     20.6%
     5  Brazil                   2085  5.0%         10.0%     19.9%
     ...
```

## Commands

| Command | Description |
|---------|-------------|
| `Team A vs Team B` | Head-to-head prediction |
| `titles` | Reprint the full 48-team results table |
| `teams` | List all valid team names |
| `quit` | Exit |

## Project Structure

| File | Purpose |
|------|---------|
| `worldcup2026.py` | Teams, groups, bracket structure, data download |
| `elo.py` | Elo ratings and Poisson goal model |
| `simulation.py` | Match prediction and tournament simulation |
| `oracle.py` | CLI entry point and head-to-head predictor |

## Methodology

- **Elo K-factor**: `30 × tournament_weight × recency_weight × goal_margin_multiplier`
- **Tournament weights**: World Cup 2.0, Continental cups 1.5, Qualifiers 1.25, Friendlies 0.5
- **Recency weights**: Last 2 years 1.4×, last 5 years 1.1×, older 0.8×
- **Goal margin**: `min(1 + goal_diff/3, 2.0)` — big wins count more
- **Home advantage**: +100 Elo for non-neutral matches
- **Penalties**: Elo-weighted probability (divisor 600) instead of 50/50
- **Simulations**: 10,000 full tournaments with `np.random.seed(42)` for reproducibility

