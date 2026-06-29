"""Match prediction and tournament simulation."""

import numpy as np
from scipy.stats import poisson
from tqdm import trange

from worldcup2026 import ALL_TEAMS, GROUPS, R32_SLOTS, R16_PAIRS, STAGES


def predict_match(model, elo, team_a, team_b, neutral=True):
    from elo import get_lambdas
    la, lb = get_lambdas(model, elo, team_a, team_b, neutral)
    pa = np.array([poisson.pmf(i, la) for i in range(10)])
    pb = np.array([poisson.pmf(j, lb) for j in range(10)])
    matrix = np.outer(pa, pb)

    p_win_a = np.sum(np.tril(matrix, -1))
    p_draw = np.sum(np.diag(matrix))
    p_win_b = np.sum(np.triu(matrix, 1))

    flat = matrix.flatten()
    top_idx = np.argsort(flat)[::-1][:5]
    top5 = [(idx // 10, idx % 10, flat[idx]) for idx in top_idx]
    return la, lb, p_win_a, p_draw, p_win_b, top5


def simulate_match(model, elo, team_a, team_b, neutral=True, n=10000, allow_draw=True):
    from elo import get_lambdas
    la, lb = get_lambdas(model, elo, team_a, team_b, neutral)
    ga = np.random.poisson(la, n)
    gb = np.random.poisson(lb, n)

    went_to_et = np.zeros(n, dtype=bool)
    went_to_pen = np.zeros(n, dtype=bool)

    if not allow_draw:
        drawn = ga == gb
        if np.any(drawn):
            went_to_et[drawn] = True
            nd = np.sum(drawn)
            et_a = np.random.poisson(la * 0.25, nd)
            et_b = np.random.poisson(lb * 0.25, nd)
            ga[drawn] += et_a
            gb[drawn] += et_b
            still_drawn = drawn.copy()
            still_drawn[drawn] = ga[drawn] == gb[drawn]
            if np.any(still_drawn):
                went_to_pen[still_drawn] = True
                ns = np.sum(still_drawn)
                ea = elo.get(team_a, 1500)
                eb = elo.get(team_b, 1500)
                pen_prob_a = 1 / (1 + 10 ** ((eb - ea) / 600))
                pen = (np.random.random(ns) < pen_prob_a).astype(int)
                ga[still_drawn] += pen
                gb[still_drawn] += 1 - pen

    wa = np.mean(ga > gb) * 100
    d = np.mean(ga == gb) * 100
    wb = np.mean(ga < gb) * 100
    p_et = np.mean(went_to_et) * 100
    p_pen = np.mean(went_to_pen) * 100
    pen_a = np.mean((went_to_pen) & (ga > gb)) * 100
    pen_b = np.mean((went_to_pen) & (gb > ga)) * 100
    return wa, d, wb, p_et, p_pen, pen_a, pen_b


def _rank_teams(team_stats):
    return sorted(team_stats, key=lambda x: (x[1], x[2], x[3]), reverse=True)


def _sim_group(group_teams, lambda_cache):
    stats = {t: [0, 0, 0] for t in group_teams}
    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            ta, tb = group_teams[i], group_teams[j]
            la, lb = lambda_cache[(ta, tb)]
            ga = np.random.poisson(la)
            gb = np.random.poisson(lb)
            stats[ta][1] += ga - gb
            stats[ta][2] += ga
            stats[tb][1] += gb - ga
            stats[tb][2] += gb
            if ga > gb:
                stats[ta][0] += 3
            elif ga < gb:
                stats[tb][0] += 3
            else:
                stats[ta][0] += 1
                stats[tb][0] += 1
    return _rank_teams([(t, s[0], s[1], s[2]) for t, s in stats.items()])


def _sim_knockout(ta, tb, elo, lambda_cache):
    la, lb = lambda_cache[(ta, tb)]
    ga = np.random.poisson(la)
    gb = np.random.poisson(lb)
    if ga != gb:
        return ta if ga > gb else tb
    ga2 = np.random.poisson(la * 0.25)
    gb2 = np.random.poisson(lb * 0.25)
    ga += ga2
    gb += gb2
    if ga != gb:
        return ta if ga > gb else tb
    ea = elo.get(ta, 1500)
    eb = elo.get(tb, 1500)
    pen_prob_a = 1 / (1 + 10 ** ((eb - ea) / 600))
    return ta if np.random.random() < pen_prob_a else tb


def _run_tournament(elo, lambda_cache):
    results = {t: "group_exit" for t in ALL_TEAMS}
    group_results = {}
    for gletter, gteams in GROUPS.items():
        group_results[gletter] = _sim_group(gteams, lambda_cache)

    thirds = []
    for gletter in sorted(GROUPS.keys()):
        t, pts, gd, gf = group_results[gletter][2]
        thirds.append((t, pts, gd, gf, gletter))

    thirds_ranked = sorted(thirds, key=lambda x: (x[1], x[2], x[3]), reverse=True)
    qualified_thirds = thirds_ranked[:8]

    for gletter in GROUPS:
        for i in range(2):
            results[group_results[gletter][i][0]] = "r32"
    for x in qualified_thirds:
        results[x[0]] = "r32"

    def get_team(code):
        if code.startswith("1"):
            return group_results[code[1]][0][0]
        elif code.startswith("2"):
            return group_results[code[1]][1][0]
        return None

    available_thirds = list(qualified_thirds)
    third_assignments = {}
    third_slots_order = [74, 77, 79, 80, 81, 82, 85, 87]
    slot_pools = {s[0]: s[3] for s in R32_SLOTS if s[3] is not None}

    def assign_thirds(slots, available, assignments):
        if not slots:
            return True
        match_num = slots[0]
        pool = slot_pools[match_num]
        for i, (t, pts, gd, gf, gletter) in enumerate(available):
            if gletter in pool:
                assignments[match_num] = t
                remaining = available[:i] + available[i + 1:]
                if assign_thirds(slots[1:], remaining, assignments):
                    return True
                del assignments[match_num]
        return False

    assign_thirds(third_slots_order, available_thirds, third_assignments)

    r32_winners = {}
    for match_num, code_a, code_b, pool in R32_SLOTS:
        ta = get_team(code_a) if code_a else third_assignments[match_num]
        tb = get_team(code_b) if code_b else third_assignments[match_num]
        r32_winners[match_num] = _sim_knockout(ta, tb, elo, lambda_cache)

    for t in r32_winners.values():
        if results[t] == "r32":
            results[t] = "r16"

    r16_winners = {}
    for m1, m2 in R16_PAIRS:
        r16_winners[(m1, m2)] = _sim_knockout(r32_winners[m1], r32_winners[m2], elo, lambda_cache)

    for t in r16_winners.values():
        if results[t] == "r16":
            results[t] = "quarters"

    r16_list = list(r16_winners.values())
    qf_winners, qf_losers = [], []
    for i in range(0, 8, 2):
        winner = _sim_knockout(r16_list[i], r16_list[i + 1], elo, lambda_cache)
        loser = r16_list[i + 1] if winner == r16_list[i] else r16_list[i]
        qf_winners.append(winner)
        qf_losers.append(loser)

    for t in qf_winners:
        if results[t] == "quarters":
            results[t] = "semis"

    sf_winners, sf_losers = [], []
    for i in range(0, 4, 2):
        winner = _sim_knockout(qf_winners[i], qf_winners[i + 1], elo, lambda_cache)
        loser = qf_winners[i + 1] if winner == qf_winners[i] else qf_winners[i]
        sf_winners.append(winner)
        sf_losers.append(loser)

    for t in sf_winners:
        if results[t] == "semis":
            results[t] = "final"

    _sim_knockout(sf_losers[0], sf_losers[1], elo, lambda_cache)

    champion = _sim_knockout(sf_winners[0], sf_winners[1], elo, lambda_cache)
    results[champion] = "champion"
    return results


def run_full_simulation(elo, team_elo, lambda_cache, n_sims=10000):
    print(f"\nRunning {n_sims:,} tournament simulations...")
    np.random.seed(42)
    counts = {t: {s: 0 for s in STAGES} for t in ALL_TEAMS}

    for _ in trange(n_sims):
        res = _run_tournament(elo, lambda_cache)
        for t, stage in res.items():
            counts[t][stage] += 1

    rows = []
    for t in ALL_TEAMS:
        c = counts[t]
        champ = c["champion"] / (n_sims / 100)
        final = (c["champion"] + c["final"]) / (n_sims / 100)
        semi = (c["champion"] + c["final"] + c["semis"]) / (n_sims / 100)
        rows.append((t, round(team_elo[t]), champ, final, semi))

    rows.sort(key=lambda x: x[2], reverse=True)
    table_data = []
    for i, (t, e, ch, fi, se) in enumerate(rows, 1):
        table_data.append([i, t, e, f"{ch:.1f}%", f"{fi:.1f}%", f"{se:.1f}%"])

    return table_data
