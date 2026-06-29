# World Cup 2026 Predictor

Predicts the FIFA World Cup 2026 using Elo ratings, Poisson regression, and Monte Carlo simulation.

## Features

- **Elo ratings** from 49,000+ historical international matches (1872–present)
- **Goal margin weighting** — a 4-0 win moves Elo more than 1-0
- **Recency bias** — recent matches (last 2 years) count 1.4× more than older ones
- **Tournament weighting** — World Cup matches carry 2× weight, friendlies 0.5×
- **ML-trained Poisson model** — scikit-learn `PoissonRegressor` with 5 features, trained on 8 years of data
- **Actual R32 knockout bracket** — uses the real draw, not simulated groups, so path difficulty affects title odds
- **WC 2026 group stage results** already baked into Elo ratings from the live dataset
- **10,000 Monte Carlo simulations** of the full knockout bracket (R32 → Final)
- **Elo-weighted penalty shootouts** — stronger teams get a slight edge, not a coin flip
- **Extra time simulation** — drawn matches go to ET at 25% intensity before penalties
- **Head-to-head predictor** — win/draw/loss %, expected goals, most likely scoreline, ET result for draws
- **Fuzzy team name matching** — misspell a name and it suggests the closest match
- **Flag emoji** for all 48 teams
- **Live data updates** — `--update` flag re-downloads the latest match results

## Run

```
pip install -r requirements.txt
python oracle.py
```

Refresh data after new matches are played:

```
python oracle.py --update
```

## Commands

```
> Brazil vs France      # head-to-head prediction
> titles                # reprint title odds table
> teams                 # list all 48 teams
> quit                  # exit
```

## Project Structure

| File | Role |
|------|------|
| `worldcup2026.py` | Teams, groups, R32 bracket, data download |
| `elo.py` | Elo computation + Poisson model training |
| `simulation.py` | Match simulation + knockout tournament Monte Carlo |
| `oracle.py` | CLI entry point + head-to-head predictor |
