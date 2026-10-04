# 🏏 IPL Winning Team Prediction

A pre-match machine-learning system that predicts whether **Team 1 or Team 2**
is more likely to win an IPL match.

## What was rebuilt

The uploaded Colab notebook was a **first-innings score regression** project:
it predicts a numeric score using in-match variables such as runs, wickets and
overs. That is different from the InternPe Week 3 requirement.

This project therefore uses binary classification:

**Target = Does Team 1 win?**

The result is displayed as Team 1 vs Team 2 win probability.

## Features

- Team 1 / Team 2
- Toss winner
- Toss decision
- Venue
- City
- Historical win rate
- Last-five-match form
- Head-to-head record
- Venue win rate
- Toss history

Historical statistics are generated sequentially so the current match result is
not used to calculate the current match's features.

## Models

- Logistic Regression
- Decision Tree
- Random Forest

The best model is selected using F1 score, with accuracy as the secondary
criterion.

## Dataset

The project is configured to download a public IPL match-level dataset covering
2008–2024 from the `avinashyadav16/ipl-analytics` GitHub repository. The
source contains season, city, date, venue, both teams, toss information and
winner fields.

Source:
https://github.com/avinashyadav16/ipl-analytics/blob/main/matches_2008-2024.csv

## Windows setup

```powershell
cd C:\Users\mdamaanali\Downloads\IPL_Winning_Team_Prediction

pip install -r requirements.txt

python scripts/download_data.py

python scripts/train.py

streamlit run app/app.py
```

## Project structure

```text
IPL_Winning_Team_Prediction/
├── app/
│   └── app.py
├── data/
│   └── matches.csv
├── models/
├── outputs/
├── scripts/
│   ├── download_data.py
│   └── train.py
├── src/
│   ├── config.py
│   ├── data_utils.py
│   └── features.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Leakage control

The model does not use final score, winning margin, Player of the Match, or
other post-match variables.

The validation is chronological when enough seasons are available, which is
more realistic for future-match prediction than randomly mixing past and future
matches.

## Limitations

Predictions are probabilities, not guarantees. Playing XI changes, injuries,
weather, pitch conditions and match-day performance can change the outcome.

## Demo video

Show:
1. Problem statement
2. Dataset
3. Feature engineering
4. Model comparison
5. Streamlit prediction
6. Probability chart
7. Limitations

8. # 🏏 IPL Winning Team Predictor

<p align="center">
  <a href="https://ipl-winning-team-prediction-8bhwcegmake72uftzxpzhl.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Live Demo">
  </a>
</p>

Pre-match ML • Historical form • Head-to-head • Toss • Venue
