import argparse, json, sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MATCHES_FILE, MODEL_FILE, METRICS_FILE, CONFUSION_FILE, PREDICTIONS_FILE, FEATURE_IMPORTANCE_FILE, MODEL_DIR, OUTPUT_DIR
from src.data_utils import clean_matches
from src.features import FEATURES, CATEGORICAL, NUMERIC, build_sequential_features

def prep(all_features):
    categories = [sorted(all_features[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL]
    return ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories=categories, handle_unknown="ignore"))
        ]), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC)
    ])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(MATCHES_FILE))
    args = p.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nRun: python scripts/download_data.py")

    matches = clean_matches(pd.read_csv(path))
    feat = build_sequential_features(matches)
    seasons = sorted(feat.season.unique())

    if len(seasons) >= 4:
        test_season = seasons[-1]
        train = feat[feat.season < test_season].copy()
        test = feat[feat.season == test_season].copy()
    else:
        cut = int(len(feat) * 0.8)
        train, test = feat.iloc[:cut].copy(), feat.iloc[cut:].copy()

    if len(test) < 20:
        cut = int(len(feat) * 0.8)
        train, test = feat.iloc[:cut].copy(), feat.iloc[cut:].copy()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    Xtr, ytr = train[FEATURES], train.target_team1_win
    Xte, yte = test[FEATURES], test.target_team1_win
    results, fitted = {}, {}

    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", prep(feat[FEATURES])), ("model", estimator)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = {
            "accuracy": float(accuracy_score(yte, pred)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "classification_report": classification_report(yte, pred, output_dict=True, zero_division=0)
        }
        fitted[name] = pipe

    best = max(results, key=lambda x: (results[x]["f1"], results[x]["accuracy"]))

    final = Pipeline([("preprocessor", prep(feat[FEATURES])), ("model", models[best])])
    final.fit(feat[FEATURES], feat.target_team1_win)

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": final,
        "feature_columns": FEATURES,
        "teams": sorted(set(feat.team1) | set(feat.team2)),
        "venues": sorted(feat.venue.dropna().unique()),
        "cities": sorted(feat.city.dropna().unique()),
        "best_model": best,
        "data_rows": len(matches),
        "last_season": int(feat.season.max())
    }, MODEL_FILE)

    pred = fitted[best].predict(Xte)
    pd.DataFrame(
        confusion_matrix(yte, pred, labels=[0, 1]),
        index=["Team 2 wins", "Team 1 wins"],
        columns=["Predicted Team 2", "Predicted Team 1"]
    ).to_csv(CONFUSION_FILE)

    pd.DataFrame({
        "date": test.date.astype(str), "team1": test.team1, "team2": test.team2,
        "actual_winner": test.winner, "predicted_team1_win": pred
    }).to_csv(PREDICTIONS_FILE, index=False)

    try:
        pre = fitted[best].named_steps["preprocessor"]
        model = fitted[best].named_steps["model"]
        names = pre.get_feature_names_out()
        imp = model.feature_importances_ if hasattr(model, "feature_importances_") else abs(model.coef_[0])
        pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    except Exception:
        pass

    METRICS_FILE.write_text(json.dumps({
        "best_model": best, "results": results,
        "train_rows": len(train), "test_rows": len(test),
        "test_season": int(test.season.max()),
        "seasons": [int(x) for x in seasons],
        "target": "Team 1 wins (1) vs Team 2 wins (0)"
    }, indent=2))

    print("\n=== IPL WINNER MODEL COMPLETE ===")
    print(f"Matches: {len(matches)} | Seasons: {seasons[0]}-{seasons[-1]}")
    print(f"Validation rows: {len(test)}")
    print(f"Best model: {best}\n")
    for n, m in results.items():
        print(f"{n:20} Accuracy={m['accuracy']:.2%}  F1={m['f1']:.2%}")

if __name__ == "__main__":
    main()
