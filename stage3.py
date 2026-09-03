import time, pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from config import SCORING, RANDOM_STATE
from knn_model import load, CV
from preprocess_knn import build_preprocessor

SUB = 50000
X, y = load()
Xs, _, ys, _ = train_test_split(X, y, train_size=SUB, stratify=y,
                                random_state=RANDOM_STATE)
BEST = dict(n_neighbors=7, weights="uniform", metric="manhattan", n_jobs=-1)


def build(sampler=None):
    steps = [("pre", build_preprocessor(list(X.columns)))]
    if sampler is not None:
        steps.append(("sample", sampler))
    steps.append(("knn", KNeighborsClassifier(**BEST)))
    return ImbPipeline(steps)


strategies = {
    "none (uniform weighting only)": None,
    "SMOTE": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
    "TomekLinks": TomekLinks(sampling_strategy="majority"),
}
rows = []
for name, samp in strategies.items():
    t = time.time()
    r = cross_validate(build(samp), Xs, ys, cv=CV, scoring=SCORING, n_jobs=-1)
    rows.append({"strategy": name,
        "f1_macro": r["test_f1_macro"].mean(), "f1_macro_std": r["test_f1_macro"].std(),
        "bal_acc": r["test_balanced_accuracy"].mean(),
        "f1_weighted": r["test_f1_weighted"].mean(),
        "accuracy": r["test_accuracy"].mean(), "secs": time.time()-t})
    print(f"  done: {name} ({time.time()-t:.0f}s)")

out = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
out.to_csv("stage3.csv", index=False)
print(); print(out.round(4).to_string(index=False))
