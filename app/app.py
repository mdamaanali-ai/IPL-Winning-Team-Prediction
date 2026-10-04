import json, sys
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import MODEL_FILE, MATCHES_FILE, METRICS_FILE, CONFUSION_FILE, FEATURE_IMPORTANCE_FILE
from src.data_utils import clean_matches
from src.features import FEATURES, build_sequential_features

st.set_page_config(page_title="IPL Winning Team Predictor", page_icon="🏏", layout="wide")

st.title("🏏 IPL Winning Team Predictor")
st.caption("Pre-match ML • Historical form • Head-to-head • Toss • Venue")

if not MODEL_FILE.exists() or not MATCHES_FILE.exists():
    st.error("Model/data missing. Run `python scripts/download_data.py` and then `python scripts/train.py`.")
    st.stop()

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    return clean_matches(pd.read_csv(MATCHES_FILE))

bundle = load_bundle()
matches = load_data()
feat = build_sequential_features(matches)

teams = bundle["teams"]
venues = bundle["venues"]
cities = bundle["cities"]

with st.sidebar:
    st.header("🎯 Match Setup")
    team1 = st.selectbox("Team 1", teams)
    team2 = st.selectbox("Team 2", [x for x in teams if x != team1])
    toss_winner = st.selectbox("Toss winner", [team1, team2])
    toss_decision = st.selectbox("Toss decision", ["field", "bat"])
    venue = st.selectbox("Venue", venues)
    city = st.selectbox("City", cities)

def rate(team):
    x = feat[(feat.team1 == team) | (feat.team2 == team)]
    return (x.winner == team).mean() if len(x) else .5

def recent(team):
    x = feat[(feat.team1 == team) | (feat.team2 == team)].tail(5)
    return (x.winner == team).mean() if len(x) else .5

def h2h(t1, t2):
    x = feat[((feat.team1 == t1) & (feat.team2 == t2)) | ((feat.team1 == t2) & (feat.team2 == t1))]
    return (x.winner == t1).mean() if len(x) else .5

def venue_rate(team):
    x = feat[(feat.venue == venue) & ((feat.team1 == team) | (feat.team2 == team))]
    return (x.winner == team).mean() if len(x) else .5

def toss_rate(team):
    x = matches[matches.toss_winner == team]
    total = len(matches[(matches.team1 == team) | (matches.team2 == team)])
    return len(x) / total if total else .5

def toss_bat_win_rate(team):
    x = matches[(matches.toss_winner == team) & (matches.toss_decision == "bat")]
    wins = (x.winner == team).sum()
    return wins / len(x) if len(x) else .5

stats = {
    "team1_win_rate": rate(team1), "team2_win_rate": rate(team2),
    "team1_recent_form": recent(team1), "team2_recent_form": recent(team2),
    "h2h_team1_rate": h2h(team1, team2),
    "team1_venue_win_rate": venue_rate(team1), "team2_venue_win_rate": venue_rate(team2),
    "team1_toss_win_rate": toss_rate(team1), "team2_toss_win_rate": toss_rate(team2),
    "team1_toss_bat_win_rate": toss_bat_win_rate(team1), "team2_toss_bat_win_rate": toss_bat_win_rate(team2),
}

row = pd.DataFrame([{
    "team1": team1, "team2": team2, "toss_winner": toss_winner,
    "toss_decision": toss_decision, "venue": venue, "city": city,
    "season": int(feat.season.max()) + 1, **stats
}])[FEATURES]

probs = bundle["model"].predict_proba(row)[0]
classes = list(bundle["model"].classes_)
p1 = float(probs[classes.index(1)])
p2 = 1 - p1
winner = team1 if p1 >= .5 else team2

c1,c2,c3,c4 = st.columns(4)
c1.metric("🏆 Predicted Winner", winner)
c2.metric(f"{team1} probability", f"{p1:.1%}")
c3.metric(f"{team2} probability", f"{p2:.1%}")
c4.metric("Confidence", f"{max(p1,p2):.1%}")

st.divider()
left,right = st.columns([1.4,1])
with left:
    st.subheader("Win Probability")
    chart = pd.DataFrame({"Team":[team1,team2],"Probability":[p1,p2]})
    fig = px.bar(chart, x="Team", y="Probability", range_y=[0,1],
                 text=chart.Probability.map(lambda x:f"{x:.1%}"))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🔎 Pre-match signals")
    st.write(f"**{team1}** historical win rate: **{stats['team1_win_rate']:.1%}**")
    st.write(f"**{team2}** historical win rate: **{stats['team2_win_rate']:.1%}**")
    st.write(f"**{team1}** last-5 form: **{stats['team1_recent_form']:.1%}**")
    st.write(f"**{team2}** last-5 form: **{stats['team2_recent_form']:.1%}**")
    st.write(f"Head-to-head share for **{team1}**: **{stats['h2h_team1_rate']:.1%}**")

st.divider()
t1,t2,t3 = st.tabs(["📊 Model Performance","📈 IPL Insights","🧠 Methodology"])

with t1:
    if METRICS_FILE.exists():
        m = json.loads(METRICS_FILE.read_text())
        rows = [{"Model":n, "Accuracy":v["accuracy"], "Precision":v["precision"], "Recall":v["recall"], "F1":v["f1"]} for n,v in m["results"].items()]
        st.dataframe(pd.DataFrame(rows).sort_values("F1", ascending=False), hide_index=True, use_container_width=True)
        if CONFUSION_FILE.exists():
            st.subheader("Confusion Matrix")
            st.dataframe(pd.read_csv(CONFUSION_FILE, index_col=0), use_container_width=True)
        if FEATURE_IMPORTANCE_FILE.exists():
            st.subheader("Top Features")
            fi = pd.read_csv(FEATURE_IMPORTANCE_FILE).head(12).sort_values("importance")
            fig = px.bar(fi, x="importance", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

with t2:
    a,b,c = st.columns(3)
    a.metric("Matches", len(matches))
    b.metric("Seasons", matches.season.nunique())
    c.metric("Teams", len(set(matches.team1) | set(matches.team2)))
    wins = matches.winner.value_counts().reset_index()
    wins.columns = ["Team","Wins"]
    st.plotly_chart(px.bar(wins, x="Team", y="Wins", title="Historical Wins"), use_container_width=True)

with t3:
    st.markdown("""
**Used:** team identity, toss, venue/city, historical win rate, last-five form,
head-to-head, venue performance and toss history.

**Not used:** final score, winning margin, Player of the Match or any other
post-match information.

The validation is chronological when enough seasons are available. A model
probability is an estimate, not a guaranteed outcome.
""")

st.caption(f"Best model: {bundle['best_model']} • Historical matches: {bundle['data_rows']} • Through season: {bundle['last_season']}")
