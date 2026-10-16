# data — transaction CSVs

Input and output of the cleaning pipeline. **Both CSVs are gitignored** —
transaction data is private and is not be committed. Put your own copy
of the raw CSV at `transaction-dataset.csv` before running the training
scripts.

| File                                 | Produced by               | Consumed by                                                                                 |
|--------------------------------------|---------------------------|---------------------------------------------------------------------------------------------|
| `transaction-dataset.csv`            | provided externally       | [`../src/clean.py`](../src/clean.py)                                                        |
| `transaction-dataset-cleaned.csv`    | [`../src/clean.py`](../src/clean.py) | [`../src/baseline_random_classifier.py`](../src/baseline_random_classifier.py), [`../src/model_tournament.py`](../src/model_tournament.py), [`../api/train_model.py`](../api/train_model.py) |

Every script that reads the cleaned CSV defaults to the path
`data/transaction-dataset-cleaned.csv` resolved from the script's own
location — you can run them from any working directory. Pass `--input
/path/to/other.csv` on the command line to point them elsewhere.
