import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from config import TARGET

CATEGORICAL = ["proto", "state", "service"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["id"], errors="ignore")
    return df


def build_preprocessor(feature_columns) -> ColumnTransformer:
    cat_cols = [c for c in CATEGORICAL if c in feature_columns]
    num_cols = [c for c in feature_columns if c not in cat_cols]

    ord_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    return ColumnTransformer(
        [
            ("categorical", ord_enc, cat_cols),
            ("numeric", "passthrough", num_cols),
        ], remainder="drop"
    )


if __name__ == "__main__":
    from config import load_raw, shared_dedup, RANDOM_STATE
    from sklearn.tree import DecisionTreeClassifier

    df = clean(shared_dedup(load_raw()))
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    print(f"[clean] feature matrix: {X.shape}  ({X.shape[1]} features retained)")

    pre = build_preprocessor(list(X.columns))
    Xt = pre.fit_transform(X)
    print(f"[transform] output shape: {Xt.shape}")

    expected_nans = int(df["service"].isna().sum())
    actual_nans = int(pd.DataFrame(Xt).isna().sum().sum())
    assert expected_nans == actual_nans, "NaN was NOT preserved - bug!"
    print(f"[check] NaNs preserved: {actual_nans:,} (all from `service`) - PASS")

    clf = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    clf.fit(Xt, y)
    print(f"[check] tree fits on NaN-containing output "
          f"(train acc @ depth 5: {clf.score(Xt, y):.4f})")
