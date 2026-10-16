# docs — reports, write-ups, and the test brief

Every HTML in this folder is a self-contained report (open directly in a
browser).

## The test

| File                                                   | What it is                                  |
|--------------------------------------------------------|---------------------------------------------|
| `data-scientist-technical-test-transaction-labeling.pdf` | Original take-home brief from Lydia       |

## Data exploration & cleaning

| File                                 | What it covers                                              |
|--------------------------------------|-------------------------------------------------------------|
| `transaction_analysis_report.html`   | EDA on the raw dataset: column stats, class balance, anomalies |
| `transaction_cleaning_report.html`   | Each of the 6 cleaning corrections, with before/after counts   |

## Modelling

| File                                              | What it covers                                         |
|---------------------------------------------------|--------------------------------------------------------|
| `baseline_random_classifier_report.html`          | Stratified-random `DummyClassifier` floor (~8% macro F1) |
| `model_round1_baseline_stratified_random_vs_logistic_regression.html` | Round 1: baseline vs Logistic Regression |
| `model_round2_logistic_regression_vs_random_forest.html`              | Round 2: LR vs Random Forest             |
| `model_round3_random_forest_vs_hist_gradient_boosting.html`           | Round 3: RF vs HistGradientBoosting      |
| `model_round4_random_forest_vs_mlp_deep_learning.html`                | Round 4: RF vs MLP                       |
| `model_round5_random_forest_vs_xgboost.html`                          | Round 5: RF vs XGBoost                   |
| `model_round6_random_forest_vs_random_forest_tuned.html`              | Round 6: RF vs grid-tuned RF             |
| `model_tournament_overview.html`                                      | Final leaderboard + headline numbers     |

## Reading order

If you want the shortest path through the story:

1. `data-scientist-technical-test-transaction-labeling.pdf` — the problem
2. `transaction_analysis_report.html` — what the data looks like
3. `transaction_cleaning_report.html` — what had to be fixed
4. `baseline_random_classifier_report.html` — the floor to beat
5. `model_tournament_overview.html` — the leaderboard and winner
6. Any individual `model_roundN_*.html` for round-by-round detail

The code that produced all of these is in [`../src/`](../src/README.md); the
production service that hosts the winning model is in [`../api/`](../api/README.md).
