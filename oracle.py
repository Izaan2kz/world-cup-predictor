"""CLI entry point for World Cup 2026 predictions."""

import sys
import io
import re
import difflib

sys.dont_write_bytecode = True

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

import numpy as np
from scipy.stats import poisson
from tabulate import tabulate

from worldcup2026 import ALL_TEAMS, FLAG_EMOJI, download_data
from elo import compute_elo_and_model
from simulation import predict_match, simulate_match, run_full_simulation, R32_TEAMS

# Download data (--update forces re-download)
force_update = "--update" in sys.argv
download_data(force=force_update)

# Build ratings and model
elo, team_elo, model, lambda_cache = compute_elo_and_model()

# Run tournament simulation
table_data = run_full_simulation(elo, team_elo, lambda_cache)

print("\n" + tabulate(table_data, headers=["Rank", "Team", "Elo", "Elo#", "Champion%", "Final%", "Semi%"],
                       tablefmt="simple"))


def print_h2h(ta, tb):
    la, lb, pw_a, pd, pw_b, top5 = predict_match(model, elo, ta, tb, neutral=True)
    sw_a, sd, sw_b, *_ = simulate_match(model, elo, ta, tb, neutral=True, n=10000, allow_draw=True)

    fa = FLAG_EMOJI.get(ta, "")
    fb = FLAG_EMOJI.get(tb, "")
    ea = round(elo.get(ta, 1500))
    eb = round(elo.get(tb, 1500))

    best_score = f"{top5[0][0]}-{top5[0][1]}"

    print(f"\n{fa} {ta}  (Elo {ea})    vs    {fb} {tb}  (Elo {eb})")
    print("-" * 56)
    print(f"{ta} win : {sw_a:5.1f}%")
    print(f"Draw      : {sd:5.1f}%")
    print(f"{tb} win: {sw_b:5.1f}%")
    print(f"Expected goals : {ta} {la:.2f} — {lb:.2f} {tb}")
    print(f"Most likely score : {best_score}  ({ta} — {tb})")

    if top5[0][0] == top5[0][1]:
        draw_goals = top5[0][0]
        et_la, et_lb = la * 0.25, lb * 0.25
        et_pa = np.array([poisson.pmf(i, et_la) for i in range(5)])
        et_pb = np.array([poisson.pmf(j, et_lb) for j in range(5)])
        et_matrix = np.outer(et_pa, et_pb)
        np.fill_diagonal(et_matrix, 0)
        et_flat = et_matrix.flatten()
        et_best = np.argmax(et_flat)
        et_ga, et_gb = et_best // 5, et_best % 5
        total_a = int(draw_goals + et_ga)
        total_b = int(draw_goals + et_gb)
        print(f"Most likely after ET : {total_a}-{total_b}  ({ta} — {tb})")

        ko_a, _, ko_b, _, _, _, _ = simulate_match(model, elo, ta, tb, neutral=True, n=10000, allow_draw=False)
        print(f"\nIf knockout → ET & penalties:")
        print(f"  {ta} advances : {ko_a:5.1f}%")
        print(f"  {tb} advances : {ko_b:5.1f}%")


print("\nHead-to-head predictor. Enter two teams, e.g. Brazil vs France")
print("Commands: titles | teams | quit")

while True:
    try:
        inp = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not inp:
        continue
    low = inp.lower()
    if low == "quit":
        break
    if low == "titles":
        print("\n" + tabulate(table_data, headers=["Rank", "Team", "Elo", "Elo#", "Champion%", "Final%", "Semi%"],
                               tablefmt="simple"))
        continue
    if low == "teams":
        for t in sorted(ALL_TEAMS):
            print(f"  {t}")
        continue

    parts = inp.split(" vs " if " vs " in inp else " VS ")
    if len(parts) != 2:
        parts = inp.split(" Vs ")
    if len(parts) != 2:
        parts = re.split(r'\s+vs\s+', inp, flags=re.IGNORECASE)
    if len(parts) != 2:
        print("Format: Team A vs Team B")
        continue

    ta, tb = parts[0].strip(), parts[1].strip()
    valid = True
    for name in [ta, tb]:
        if name not in ALL_TEAMS:
            matches = difflib.get_close_matches(name, ALL_TEAMS, n=3, cutoff=0.4)
            if matches:
                print(f"'{name}' not found. Did you mean: {', '.join(matches)}?")
            else:
                print(f"'{name}' not found. Type 'teams' to see valid names.")
            valid = False
    if valid:
        print_h2h(ta, tb)
