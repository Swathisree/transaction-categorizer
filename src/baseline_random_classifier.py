"""
Stratified-random multiclass baseline on the cleaned transaction CSV.

- Target  : minor_category (12 classes)
- Features: every column EXCEPT minor_category AND major_category
            (major_category is excluded because it is a deterministic function
             of the target, i.e. would be perfect leakage.)
- Model   : sklearn DummyClassifier(strategy='stratified', random_state=42)
- After predicting minor_category, a deterministic minor→major lookup
  (built from the data itself) is applied and the major level is evaluated too.
- Metrics are reported in percentage scale.

Usage:
    python baseline_random_classifier.py                       # default input
    python baseline_random_classifier.py --input /path/to/cleaned.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "data" / "transaction-dataset-cleaned.csv"
RNG = 42


def main(input_csv: Path) -> None:
    df = pd.read_csv(input_csv, parse_dates=["created_at"])

    # minor → major is deterministic in the cleaned data
    MINOR_TO_MAJOR = (df.drop_duplicates("minor_category")
                        .set_index("minor_category")["major_category"]
                        .to_dict())

    y = df["minor_category"]
    X = df.drop(columns=["minor_category", "major_category"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RNG, stratify=y)

    clf = DummyClassifier(strategy="stratified", random_state=RNG).fit(X_train, y_train)
    pred_minor = clf.predict(X_test)
    pred_major = pd.Series(pred_minor).map(MINOR_TO_MAJOR).values
    true_major = y_test.map(MINOR_TO_MAJOR).values

    def report(label, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        print(f"\n=== {label} ===")
        print(f"  accuracy                 {acc*100:6.2f}%")
        print(f"  precision/recall/F1 macro   {p*100:6.2f}% / {r*100:6.2f}% / {f*100:6.2f}%")
        print(f"  precision/recall/F1 weight  {pw*100:6.2f}% / {rw*100:6.2f}% / {fw*100:6.2f}%")
        print(classification_report(y_true, y_pred, zero_division=0, digits=4))

    report("minor_category (direct)",     y_test.values, pred_minor)
    report("major_category (via lookup)", true_major,    pred_major)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                   help=f"Path to the cleaned CSV (default: {DEFAULT_INPUT})")
    args = p.parse_args()
    main(args.input)
