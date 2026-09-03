# Assignment 2 — k-Nearest Neighbours & Decision Trees

Multi-class network attack classification (`attack_cat`) on the UNSW network
traffic dataset, comparing a k-NN pipeline against a Decision Tree pipeline.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.12).

## 2. Data

Already included — nothing to download:

```
Code/data/networkTraffic.csv
Code/data/attack_category_map.csv
Code/data/features.csv
```

## 3. Important: where to run scripts from

The scripts use **relative** data paths, so you must `cd` into the folder a
script lives in before running it:

| Script            | Run from     |
| ----------------- | ------------ |
| `eda.py`          | `Code/`      |
| `tree/*.py`       | `Code/tree/` |
| `knn/*.py`        | `Code/knn/`  |

## 4. Exploratory data analysis (optional)

```bash
cd Code
python eda.py
```

`main()` calls one EDA function at a time — edit it to pick which check runs
(missing values, cardinality, outliers, correlation, duplicates).

## 5. Decision Tree

```bash
cd Code/tree

python stage1.py        # criterion / class_weight / max_depth
python stage2.py        # cost-complexity pruning (ccp_alpha)
python stage3.py        # refined ccp_alpha + min_samples_leaf
python tree_final.py    # final config, 5-fold CV, per-class + confusion matrix
```

Stages 1–3 are the hyper-parameter search; each writes its results table to
`Code/tree/output/`. `tree_final.py` is the one that produces the final
reported numbers.

## 6. k-NN

```bash
cd Code/knn

python stage1.py euclidean     # distance metric is a required argument
python stage1.py manhattan
python stage2.py               # refine k and weights
python stage3.py               # imbalance handling: none / SMOTE / TomekLinks
```

Then the final evaluation, **one fold at a time** (0 through 4):

```bash
python final_knn.py 0
python final_knn.py 1
python final_knn.py 2
python final_knn.py 3
python final_knn.py 4
```

Each run writes `fold_N.csv` plus `oof_idx_N.npy` / `oof_pred_N.npy`. Folds are
run separately because a full-dataset k-NN fold is slow — combine the five
`fold_*.csv` files to get the mean/std across folds.

Note: k-NN stages 1–3 tune on a stratified subsample (25k / 80k / 50k rows) for
speed; only `final_knn.py` uses the full dataset.

## 7. Outputs

```
Code/tree/output/    tree_stage1..3.csv, tree_final_folds.csv, tree_oof.npy
Code/knn/            stage1_*.csv, stage2.csv, stage3.csv, fold_*.csv, oof_*.npy
```

Results files from a previous run are already committed; re-running overwrites
them.

## 8. Sanity check

Both pipelines deduplicate once, upstream, and print their row count on start:

```
[shared_dedup] 257,673 -> 162,745 rows (94,928 exact duplicates removed)
```

Both the tree and k-NN must report **162,745 rows**. If you see 157,869,
deduplication has run twice and the two models are no longer being compared on
the same data.
