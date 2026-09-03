import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer

from config import TARGET

DROP_ID = ["id"]

DROP_REDUNDANT = [
    "is_ftp_login",
    "dloss",
    "sloss",
    "dwin",
    "ct_srv_dst",
    "dpkts",
    "spkts",
    "ct_src_dport_ltm",
    "ct_dst_src_ltm",
    "tcprtt",
    "is_sm_ips_ports",
    "ct_dst_ltm",
]

LOG_TRANSFORM = [
    "dur", "sbytes", "dbytes", "rate", "sload", "dload", "sinpkt",
    "dinpkt", "sjit", "djit", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_flw_http_mthd", "ct_src_ltm",
]

CATEGORICAL = ["proto", "state", "service"]

MIN_CATEGORY_FREQUENCY = 0.005


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=DROP_ID + DROP_REDUNDANT, errors="ignore")

    df["service"] = df["service"].fillna("unknown")

    return df


def build_preprocessor(feature_columns) -> ColumnTransformer:
    log_cols = [c for c in LOG_TRANSFORM if c in feature_columns]
    cat_cols = [c for c in CATEGORICAL if c in feature_columns]
    scale_cols = [c for c in feature_columns
                  if c not in log_cols and c not in cat_cols]

    log_pipe = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", MinMaxScaler()),
    ])

    cat_pipe = OneHotEncoder(
        min_frequency=MIN_CATEGORY_FREQUENCY,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )

    return ColumnTransformer([
        ("log_scale", log_pipe, log_cols),
        ("scale_only", MinMaxScaler(), scale_cols),
        ("categorical", cat_pipe, cat_cols),
    ], remainder="drop")


if __name__ == "__main__":
    from config import load_raw, shared_dedup

    raw = load_raw()
    print(f"[load] raw shape: {raw.shape}")

    df = clean(shared_dedup(raw))
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    print(f"[clean] feature matrix: {X.shape}  ({X.shape[1]} features retained)")

    pre = build_preprocessor(list(X.columns))
    Xt = pre.fit_transform(X)

    print(f"[transform] output shape: {Xt.shape}")
    print(f"[transform] value range: [{Xt.min():.3f}, {Xt.max():.3f}]  (expected [0,1])")

    names = pre.get_feature_names_out()
    print(f"[transform] {len(names)} output dimensions")
    print(f"  from one-hot: {sum(n.startswith('categorical') for n in names)}")
    print(f"  from numeric: {sum(not n.startswith('categorical') for n in names)}")

    print("\n[check] class distribution after cleaning:")
    print((y.value_counts().sort_index() / len(y) * 100).round(2).to_string())
