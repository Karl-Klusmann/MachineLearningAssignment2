import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)


def load_data():
    df = pd.read_csv("data/networkTraffic.csv", na_values="?")
    attack_map = pd.read_csv("data/attack_category_map.csv")


    obj_cols = df.select_dtypes(include="object").columns.tolist()

    return df, attack_map

def find_missing_values(df):
    missing = df.isna().sum()
    percentage_missing = ((missing / len(df))*100).round(2)

    missing_report = pd.DataFrame({"Missing Count": missing, "Percentage Missing": percentage_missing})
    missing_report = missing_report[missing_report["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )
    print(missing_report if len(missing_report) > 0 else "No missing values found")

def check_cardinality(df):
    cardinality = df.nunique().sort_values()
    print(cardinality)

def check_numeric_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "id" in numeric_cols:
        numeric_cols.remove("id")

    if "attack_cat" in numeric_cols:
        numeric_cols.remove("attack_cat")


    desc = df[numeric_cols].describe().T

    desc["range"] = desc["max"] - desc["min"]

    non_negative_cols = [
        "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "sload", "dload", "spkts", "dpkts", "smean", "dmean",
    "trans_depth", "response_body_len", "sinpkt", "dinpkt",
    "sjit", "djit", "swin", "dwin", "tcprtt", "synack", "ackdat",
    "rate",
    ]

    for col in non_negative_cols:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()

            if n_neg > 0:
                print(f"  {col}: {n_neg} negative values (min={df[col].min()})")

def check_for_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rows = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        number_of_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        percentage_outliers = ((number_of_outliers/len(df))*100).round(2)
        outlier_rows.append((col, number_of_outliers, percentage_outliers, df[col].max()))

    outlier_df = pd.DataFrame(
        outlier_rows, columns=["feature", "number_of_outliers","percentage_outliers","max_value"]
    ).sort_values("percentage_outliers", ascending=False)
    print(outlier_df)

def check_feature_correlation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation = df[numeric_cols].corr()
    pairs = []
    cols = correlation.columns
    print(cols)
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            r = correlation.iloc[i,j]
            if (abs(r)) > 0.9:
                pairs.append((cols[i], cols[j], round(r,4)))
    pairs_df = pd.DataFrame(pairs, columns=["Feature_1", "Feature_2", "Correlation"])
    print(pairs_df)

def check_for_duplicates(df):
    cols = [c for c in df.columns if c != "id"]
    number_of_duplicates = df.duplicated(subset=cols).sum()
    print(number_of_duplicates)
    print(((number_of_duplicates/len(df))*100).round(3))

def main():
    df, attack_map = load_data()
    check_for_duplicates(df)


if __name__ == "__main__":
    main()
