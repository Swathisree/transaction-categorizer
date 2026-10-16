"""
Model tournament on the cleaned transaction CSV.

- Target  : minor_category (12 classes). Mapped to major_category (9) via lookup.
- Features: numeric (amount_value, fees, hour, day-of-week), low-cardinality
            one-hot categoricals, and target-encoded merchant_name.
- Split   : stratified 80/20, random_state=42.
- Rounds  : baseline vs LR, LR vs RF, RF vs HGB, RF vs MLP.
            Winner each round carries forward on minor macro F1.

Usage:
    python model_tournament.py                        # default input
    python model_tournament.py --input /path/to/cleaned.csv
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "data" / "transaction-dataset-cleaned.csv"
RNG = 42


def main(input_csv: Path) -> None:
    # Imported inside main so `--help` works even if xgboost's native lib
    # (libomp on macOS) isn't installed on the host.
    from xgboost import XGBClassifier

    df = pd.read_csv(input_csv, parse_dates=["created_at"])
    MINOR_TO_MAJOR = (df.drop_duplicates("minor_category")
                        .set_index("minor_category")["major_category"].to_dict())

    df["hour"] = df["created_at"].dt.hour
    df["dow"]  = df["created_at"].dt.dayofweek
    df["pfm_recurring_frequency"] = df["pfm_recurring_frequency"].fillna("none")

    NUM = ["amount_value", "fees", "hour", "dow"]
    LOW = ["card_name", "digitalization_type", "merchant_pan_entry_method",
           "status", "pfm_recurring_frequency"]
    HIG = ["merchant_name"]

    X = df[NUM + LOW + HIG].copy()
    y = df["minor_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RNG, stratify=y)
    le = LabelEncoder().fit(y_train)

    def preprocessor():
        return ColumnTransformer([
            ("num", StandardScaler(), NUM),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW),
            ("te",  TargetEncoder(target_type="multiclass", random_state=RNG), HIG),
        ])

    MODELS = {
        "Stratified random baseline": DummyClassifier(strategy="stratified", random_state=RNG),
        "Logistic Regression":        LogisticRegression(max_iter=1000, random_state=RNG, n_jobs=-1),
        "Random Forest":              RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                             random_state=RNG, n_jobs=-1),
        "HistGradientBoosting":       HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                                     random_state=RNG),
        "MLP (deep learning)":        MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                                                    random_state=RNG, early_stopping=False),
        "XGBoost":                    XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                                    objective="multi:softprob", tree_method="hist",
                                                    random_state=RNG, n_jobs=-1, eval_metric="mlogloss"),
    }

    def evaluate(y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        print(f"  {name:30s} acc={acc*100:5.2f}%  macro P/R/F1={p*100:5.2f}%/{r*100:5.2f}%/{f*100:5.2f}%")
        return f  # macro F1 for winner selection

    print("Training all models...\n")
    scores = {}
    for name, model in MODELS.items():
        if isinstance(model, DummyClassifier):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        elif isinstance(model, (MLPClassifier, XGBClassifier)):
            # Both need integer labels
            pipe = Pipeline([("pre", preprocessor()), ("clf", model)])
            pipe.fit(X_train, le.transform(y_train))
            pred = le.inverse_transform(pipe.predict(X_test))
        else:
            pipe = Pipeline([("pre", preprocessor()), ("clf", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
        pred_major = pd.Series(pred).map(MINOR_TO_MAJOR).values
        print(f"\n{name}")
        print("  --- minor_category ---")
        f_minor = evaluate(y_test.values, pred, "minor")
        print("  --- major_category ---")
        evaluate(y_test.map(MINOR_TO_MAJOR).values, pred_major, "major")
        scores[name] = f_minor

    print("\n\nTournament ranking (minor macro F1):")
    for n, f in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {n:30s}  {f*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                   help=f"Path to the cleaned CSV (default: {DEFAULT_INPUT})")
    args = p.parse_args()
    main(args.input)
