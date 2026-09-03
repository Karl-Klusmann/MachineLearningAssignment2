import numpy as np, pandas as pd, time
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                             classification_report, confusion_matrix)

from config import ATTACK_MAP_PATH, RANDOM_STATE
from preprocessing_tree import build_preprocessor
from tree_model import load, CV

Path("output").mkdir(exist_ok=True)

BEST = dict(
    criterion="entropy",
    class_weight="balanced",
    max_depth=None,
    ccp_alpha=1e-4,
    min_samples_leaf=5,
)
print("Configuration:", BEST)

X, y_series = load()
y = y_series.to_numpy()
print(f"[check] rows: {len(X):,}  (must match k-NN's 162,745)")

oof = np.empty_like(y); rows = []; sizes = []
t0 = time.time()
for i, (tr, te) in enumerate(CV.split(X, y), 1):
    pipe = Pipeline([("pre", build_preprocessor(list(X.columns))),
                     ("tree", DecisionTreeClassifier(random_state=RANDOM_STATE,
                                                     **BEST))])
    pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict(X.iloc[te]); oof[te] = p
    t_ = pipe.named_steps["tree"]
    sizes.append((t_.get_depth(), t_.get_n_leaves()))
    rows.append({"fold": i,
        "f1_macro": f1_score(y[te], p, average="macro"),
        "f1_weighted": f1_score(y[te], p, average="weighted"),
        "bal_acc": balanced_accuracy_score(y[te], p),
        "accuracy": accuracy_score(y[te], p)})
    print(f"  fold {i}: depth={sizes[-1][0]:3d} leaves={sizes[-1][1]:5d} "
          f"f1_macro={rows[-1]['f1_macro']:.4f} ({time.time()-t0:.0f}s)")

folds = pd.DataFrame(rows)
folds.to_csv("output/tree_final_folds.csv", index=False)
print("\n=== PER-FOLD ===")
print(folds.round(4).to_string(index=False))
print("\n=== MEAN +/- STD (5-fold, full dataset) ===")
for c in ["f1_macro", "f1_weighted", "bal_acc", "accuracy"]:
    print(f"  {c:12s} {folds[c].mean():.4f} +/- {folds[c].std():.4f}")
print(f"\nPruned tree size: depth {np.mean([s[0] for s in sizes]):.1f}, "
      f"leaves {np.mean([s[1] for s in sizes]):.0f} "
      f"(unpruned was depth 64, 16,984 leaves)")

amap = pd.read_csv(ATTACK_MAP_PATH)
nm = dict(zip(amap.iloc[:, 1], amap.iloc[:, 0]))
labels = sorted(np.unique(y)); tn = [str(nm.get(l, l)) for l in labels]
print("\n=== PER-CLASS (pooled out-of-fold) ===")
print(classification_report(y, oof, target_names=tn, digits=3, zero_division=0))
print("=== CONFUSION MATRIX (rows=true, cols=predicted) ===")
print(pd.DataFrame(confusion_matrix(y, oof), index=tn, columns=labels).to_string())
np.save("output/tree_oof.npy", oof)
