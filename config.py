import os
from pathlib import Path
import pandas as pd


DATA_PATH = "../data/networkTraffic.csv"
ATTACK_MAP_PATH = "../data/attack_category_map.csv"

RANDOM_STATE = 42
TARGET = "attack_cat"
N_FOLDS = 5

SCORING = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}
PRIMARY = "f1_macro"


def load_raw():
    return pd.read_csv(DATA_PATH, na_values="?")


def shared_dedup(df):
    before = len(df)
    df = df.drop_duplicates(
        subset=[c for c in df.columns if c != "id"]
    ).reset_index(drop=True)
    print(f"[shared_dedup] {before:,} -> {len(df):,} rows "
          f"({before - len(df):,} exact duplicates removed)")
    return df
