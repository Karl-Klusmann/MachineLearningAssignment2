import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from config import (load_raw, shared_dedup, TARGET, RANDOM_STATE,
                    N_FOLDS, SCORING, PRIMARY)
from preprocess_knn import clean, build_preprocessor

CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def load():
    df = clean(shared_dedup(load_raw()))
    return df.drop(columns=[TARGET]), df[TARGET]


def make_pipe(columns, **knn_kwargs):
    return Pipeline([
        ("pre", build_preprocessor(columns)),
        ("knn", KNeighborsClassifier(n_jobs=-1, **knn_kwargs)),
    ])
