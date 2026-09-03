import time, pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from config import SCORING, RANDOM_STATE
from knn_model import load, make_pipe, CV

SUB = 80000
X, y = load()
Xs, _, ys, _ = train_test_split(X, y, train_size=SUB, stratify=y,
                                random_state=RANDOM_STATE)

grid = {"knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["manhattan"]}
gs = GridSearchCV(make_pipe(list(X.columns)), grid, scoring=SCORING,
                  refit=False, cv=CV, n_jobs=-1)
t = time.time(); gs.fit(Xs, ys)
print(f"[stage2] {time.time()-t:.0f}s on {SUB:,} rows")

res = pd.DataFrame(gs.cv_results_)[["param_knn__n_neighbors","param_knn__weights",
    "mean_test_f1_macro","std_test_f1_macro","mean_test_balanced_accuracy",
    "mean_test_f1_weighted","mean_test_accuracy"]]
res.columns = ["k","weights","f1_macro","f1_macro_std","bal_acc","f1_weighted","accuracy"]
res.sort_values("f1_macro", ascending=False, inplace=True)
res.to_csv("stage2.csv", index=False)
print(res.round(4).to_string(index=False))
