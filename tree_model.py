import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from config import (load_raw, shared_dedup, TARGET, RANDOM_STATE,
                    N_FOLDS, SCORING, PRIMARY)
from preprocessing_tree import clean, build_preprocessor

CV = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def load():
    df = clean(shared_dedup(load_raw()))
    return df.drop(columns=[TARGET]), df[TARGET]


def make_pipe(columns, **tree_kwargs):
    return Pipeline([
        ("pre", build_preprocessor(columns)),
        ("tree", DecisionTreeClassifier(random_state=RANDOM_STATE,
                                        **tree_kwargs)),
    ])


def report(gs, params, path):
    cols = [f"param_tree__{p}" for p in params]
    res = pd.DataFrame(gs.cv_results_)[cols + [
        "mean_test_f1_macro", "std_test_f1_macro",
        "mean_test_balanced_accuracy", "mean_test_f1_weighted",
        "mean_test_accuracy"]]
    res.columns = params + ["f1_macro", "f1_macro_std", "bal_acc",
                            "f1_weighted", "accuracy"]
    res = res.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    res.to_csv(path, index=False)
    return res
