import numpy as np
import pandas as pd

CATEGORICAL = ["team1", "team2", "toss_winner", "toss_decision", "venue", "city"]
NUMERIC = [
    "season", "team1_win_rate", "team2_win_rate",
    "team1_recent_form", "team2_recent_form", "h2h_team1_rate",
    "team1_venue_win_rate", "team2_venue_win_rate",
    "team1_toss_win_rate", "team2_toss_win_rate",
    "team1_toss_bat_win_rate", "team2_toss_bat_win_rate",
]
FEATURES = CATEGORICAL + NUMERIC

def safe_rate(wins, games):
    return wins / games if games else 0.5

def build_sequential_features(df):
    df = df.sort_values(["date", "season"]).reset_index(drop=True).copy()

    games, wins, recent = {}, {}, {}
    venue_games, venue_wins = {}, {}
    toss_games, toss_wins = {}, {}
    toss_bat_games, toss_bat_wins = {}, {}
    h2h = {}
    rows = []

    for _, r in df.iterrows():
        a, b, venue = r.team1, r.team2, r.venue
        pair_key = frozenset((a, b))
        pair = h2h.get(pair_key, {})

        a_games, b_games = games.get(a, 0), games.get(b, 0)
        a_vg = venue_games.get((a, venue), 0)
        b_vg = venue_games.get((b, venue), 0)
        a_tg, b_tg = toss_games.get(a, 0), toss_games.get(b, 0)
        a_bg, b_bg = toss_bat_games.get(a, 0), toss_bat_games.get(b, 0)

        rows.append({
            "team1": a, "team2": b,
            "toss_winner": r.toss_winner,
            "toss_decision": r.toss_decision,
            "venue": venue, "city": r.city,
            "season": int(r.season),
            "team1_win_rate": safe_rate(wins.get(a, 0), a_games),
            "team2_win_rate": safe_rate(wins.get(b, 0), b_games),
            "team1_recent_form": float(np.mean(recent.get(a, [0.5])[-5:])),
            "team2_recent_form": float(np.mean(recent.get(b, [0.5])[-5:])),
            "h2h_team1_rate": safe_rate(pair.get(a, 0), pair.get("total", 0)),
            "team1_venue_win_rate": safe_rate(
                venue_wins.get((a, venue), 0), a_vg
            ),
            "team2_venue_win_rate": safe_rate(
                venue_wins.get((b, venue), 0), b_vg
            ),
            "team1_toss_win_rate": safe_rate(toss_wins.get(a, 0), a_tg),
            "team2_toss_win_rate": safe_rate(toss_wins.get(b, 0), b_tg),
            "team1_toss_bat_win_rate": safe_rate(toss_bat_wins.get(a, 0), a_bg),
            "team2_toss_bat_win_rate": safe_rate(toss_bat_wins.get(b, 0), b_bg),
            "target_team1_win": int(r.winner == a),
            "winner": r.winner,
            "date": r.date,
        })

        winner = r.winner
        for team in (a, b):
            games[team] = games.get(team, 0) + 1
            wins[team] = wins.get(team, 0) + int(winner == team)
            recent.setdefault(team, []).append(float(winner == team))
            key = (team, venue)
            venue_games[key] = venue_games.get(key, 0) + 1
            venue_wins[key] = venue_wins.get(key, 0) + int(winner == team)

        tw = r.toss_winner
        if tw in (a, b):
            toss_games[tw] = toss_games.get(tw, 0) + 1
            toss_wins[tw] = toss_wins.get(tw, 0) + 1
            if r.toss_decision == "bat":
                toss_bat_games[tw] = toss_bat_games.get(tw, 0) + 1
                toss_bat_wins[tw] = toss_bat_wins.get(tw, 0) + int(winner == tw)

        h2h.setdefault(pair_key, {"total": 0})
        h2h[pair_key][winner] = h2h[pair_key].get(winner, 0) + 1
        h2h[pair_key]["total"] += 1

    return pd.DataFrame(rows)
