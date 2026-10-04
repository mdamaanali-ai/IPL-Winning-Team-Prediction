import pandas as pd

ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}

REQUIRED = ["season", "date", "team1", "team2", "toss_winner",
            "toss_decision", "winner"]

def normalize_team(value):
    if pd.isna(value):
        return value
    return ALIASES.get(str(value).strip(), str(value).strip())

def clean_matches(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_")
                  for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    if "venue" not in df.columns:
        df["venue"] = "Unknown Venue"
    if "city" not in df.columns:
        df["city"] = "Unknown City"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    for c in ["team1", "team2", "toss_winner", "winner", "venue", "city"]:
        df[c] = df[c].map(normalize_team)

    df["toss_decision"] = df["toss_decision"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["season", "date", "team1", "team2", "winner"])
    df = df[df["winner"].isin(df["team1"]) | df["winner"].isin(df["team2"])]

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    else:
        df = df.drop_duplicates(subset=["date", "team1", "team2", "winner"])

    return df.sort_values(["date", "season"]).reset_index(drop=True)
