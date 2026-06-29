# World Cup 2026 Predictor

A Python CLI that predicts the 2026 FIFA World Cup using machine-learned Elo ratings, Poisson regression, and Monte Carlo simulation.

It replays ~49,000 historical international matches to build Elo strength ratings, trains a Poisson goal model on 8 years of data, then simulates the knockout bracket 10,000 times to estimate every team's chances of winning it all. After the simulation, an interactive prompt lets you pit any two teams against each other for a head-to-head prediction.

## Run

```
pip install -r requirements.txt
python oracle.py
```

To refresh match data (picks up newly played results):

```
python oracle.py --update
```

On first run it downloads the historical results dataset and caches it locally as `results.csv`. Later runs use the cached file and load instantly. The dataset is live — it already includes played 2026 World Cup matches, which feed straight into the Elo ratings. Fixtures not yet played carry no score and are ignored.

## Using the prompt

```
> Brazil vs France      # head-to-head match prediction
> titles                # reprint the title-odds table
> teams                 # list all 48 qualified teams
> quit                  # exit
```

When the most likely scoreline is a draw, the predictor also shows the most likely result after extra time and each team's knockout advancement probability.

## How it works

| File | Role |
|------|------|
| `worldcup2026.py` | The 48 qualified teams, group assignments, actual R32 bracket, dataset name mapping, and data download. |
| `elo.py` | Downloads results, replays them chronologically, computes Elo ratings, and trains the Poisson goal model. |
| `simulation.py` | Poisson expected-goals predictions, vectorized Monte Carlo match simulation, and full knockout bracket simulation. |
| `oracle.py` | The CLI — wires everything together, renders tables, and runs the interactive head-to-head prompt. |

## The model

**Elo ratings.** Every historical match nudges each team's rating toward its result. Updates are weighted by tournament importance (World Cup > continental cup > qualifier > friendly), recency (recent matches count up to 1.4× more), goal margin (a 3-0 win moves the needle more than 1-0), and home advantage (+100 Elo for non-neutral venues). Teams start at 1500.

**Expected goals.** A scikit-learn `PoissonRegressor` trained on 8 years of match data with 5 features: attack Elo, defense Elo, Elo difference, home advantage, and tournament weight. The model learns how Elo gaps translate into goals scored — unlike a hand-tuned formula, the coefficients come from the data via maximum likelihood.

**Match simulation.** Each team's goals are drawn from a Poisson distribution parameterized by the model's predicted lambda. For knockout matches: if drawn after 90 minutes, extra time is simulated at 25% intensity (lambda × 0.25). Still drawn after ET → Elo-weighted penalty shootout (stronger team gets a slight edge, not a coin flip).

**Tournament simulation.** The actual R32 bracket from the 2026 World Cup is used as the starting point. Knockout matches are simulated from R32 through the Final, repeated 10,000 times with a fixed seed for reproducibility. Each run tracks how far every team progresses, producing title odds, final odds, and semifinal odds.

## Teams

The 48 teams and groups follow the official FIFA draw (13 December 2025, Zurich).

| | | | |
|---|---|---|---|
| **A** Mexico · South Africa · Korea Republic · Czechia | **B** Canada · Qatar · Switzerland · Bosnia & Herzegovina | **C** Brazil · Morocco · Haiti · Scotland | **D** USA · Paraguay · Australia · Türkiye |
| **E** Germany · Curaçao · Côte d'Ivoire · Ecuador | **F** Netherlands · Japan · Tunisia · Sweden | **G** Belgium · Egypt · IR Iran · New Zealand | **H** Spain · Cabo Verde · Saudi Arabia · Uruguay |
| **I** France · Iraq · Senegal · Norway | **J** Argentina · Algeria · Austria · Jordan | **K** Congo DR · Portugal · Uzbekistan · Colombia | **L** England · Croatia · Ghana · Panama |
