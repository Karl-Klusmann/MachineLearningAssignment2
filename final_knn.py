import sys, time, numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from knn_model import load, CV
from preprocess_knn import build_preprocessor

FOLD = int(sys.argv[1])
BEST = dict(n_neighbors=9, weights="uniform", metric="manhattan", n_jobs=-1)

X, y_series = load()
y = y_series.to_numpy()
print(f"[check] rows: {len(X):,}  (expect 162,745 - if 157,869, dedup ran twice)")

tr, te = list(CV.split(X, y))[FOLD]
t = time.time()
pipe = Pipeline([("pre", build_preprocessor(list(X.columns))),
                 ("knn", KNeighborsClassifier(**BEST))])
pipe.fit(X.iloc[tr], y[tr])
p = pipe.predict(X.iloc[te])

np.save(f"oof_idx_{FOLD}.npy", te)
np.save(f"oof_pred_{FOLD}.npy", p)
row = dict(fold=FOLD+1,
           f1_macro=f1_score(y[te], p, average="macro"),
           f1_weighted=f1_score(y[te], p, average="weighted"),
           bal_acc=balanced_accuracy_score(y[te], p),
           accuracy=accuracy_score(y[te], p))
pd.DataFrame([row]).to_csv(f"fold_{FOLD}.csv", index=False)
print(f"fold {FOLD+1}: {time.time()-t:.0f}s  " +
      "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "fold"))
