import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

RAW_PATH       = Path("data/raw/transactions.csv")
PROCESSED_PATH = Path("data/processed/features.parquet")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. Temporal features
# ─────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"]   = df["timestamp"].dt.hour
    df["day_of_week"]   = df["timestamp"].dt.dayofweek      # 0=Mon, 6=Sun
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]      = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    return df


# ─────────────────────────────────────────────
# 3. Transaction velocity (rolling window)
#    Counts how many transactions a user made
#    in the last N hours — high velocity = risk
# ─────────────────────────────────────────────
def compute_velocity(df: pd.DataFrame, window_hours: int, col_name: str) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("timestamp")
    counts = (
        df.groupby("user_id")["transaction_id"]
        .rolling(f"{window_hours}h", closed="left")
        .count()
        .reset_index()
        .rename(columns={"transaction_id": col_name})
    )
    df = df.reset_index()
    df = df.merge(counts, on=["timestamp", "user_id"], how="left")
    df[col_name] = df[col_name].fillna(0).astype(int)
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_velocity(df, window_hours=1,  col_name="tx_velocity_1h")
    df = compute_velocity(df, window_hours=24, col_name="tx_velocity_24h")
    df = compute_velocity(df, window_hours=168, col_name="tx_velocity_7d")
    return df


# ─────────────────────────────────────────────
# 4. Amount deviation features
#    Compare current transaction amount to the
#    user's own historical spending pattern
# ─────────────────────────────────────────────
def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    # Rolling 30-day mean and std per user (excludes current row)
    df = df.set_index("timestamp")
    rolling = (
        df.groupby("user_id")["amount"]
        .rolling("30D", closed="left")
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "user_amt_mean_30d", "std": "user_amt_std_30d"})
    )
    df = df.reset_index()
    df = df.merge(rolling, on=["timestamp", "user_id"], how="left")

    df["user_amt_mean_30d"] = df["user_amt_mean_30d"].fillna(df["amount"])
    df["user_amt_std_30d"]  = df["user_amt_std_30d"].fillna(1.0).clip(lower=0.01)

    # Z-score: how many std devs above the user's normal
    df["amount_zscore"] = (
        (df["amount"] - df["user_amt_mean_30d"]) / df["user_amt_std_30d"]
    ).clip(-10, 10)

    # Log-amount (stabilises skewed distribution)
    df["log_amount"] = np.log1p(df["amount"])

    return df


# ─────────────────────────────────────────────
# 5. Merchant risk score
#    Historical fraud rate per merchant,
#    computed on the training split only
#    (target encoding with smoothing)
# ─────────────────────────────────────────────
def add_merchant_risk(df: pd.DataFrame, global_mean: float = None, k: int = 10) -> pd.DataFrame:
    if global_mean is None:
        global_mean = df["label"].mean()

    merchant_stats = (
        df.groupby("merchant_id")["label"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fraud_count", "count": "tx_count"})
        .reset_index()
    )

    # Smoothed target encoding: prevents noise from rare merchants
    # score = (fraud_count + k * global_mean) / (tx_count + k)
    merchant_stats["merchant_fraud_rate"] = (
        (merchant_stats["fraud_count"] + k * global_mean)
        / (merchant_stats["tx_count"] + k)
    )

    df = df.merge(
        merchant_stats[["merchant_id", "merchant_fraud_rate"]],
        on="merchant_id",
        how="left"
    )
    df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(global_mean)
    return df


# ─────────────────────────────────────────────
# 6. Geographic / card-not-present features
# ─────────────────────────────────────────────
def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    # Flag transactions in a different country from user's home country
    if "country" in df.columns and "user_home_country" in df.columns:
        df["is_foreign"] = (df["country"] != df["user_home_country"]).astype(int)
    else:
        df["is_foreign"] = 0

    # Card-not-present (online) transactions are higher risk
    if "is_online" in df.columns:
        df["is_card_not_present"] = df["is_online"].astype(int)
    else:
        df["is_card_not_present"] = 0

    return df


# ─────────────────────────────────────────────
# 7. User account age feature
#    New accounts have higher fraud risk
# ─────────────────────────────────────────────
def add_account_age(df: pd.DataFrame) -> pd.DataFrame:
    if "user_created_at" in df.columns:
        df["user_created_at"] = pd.to_datetime(df["user_created_at"])
        df["account_age_days"] = (
            df["timestamp"] - df["user_created_at"]
        ).dt.days.clip(lower=0)
        df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    else:
        df["account_age_days"] = 365
        df["is_new_account"]   = 0
    return df


# ─────────────────────────────────────────────
# 8. Final feature selection
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "transaction_id",
    "user_id",
    "timestamp",
    # temporal
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    # velocity
    "tx_velocity_1h",
    "tx_velocity_24h",
    "tx_velocity_7d",
    # amount
    "amount",
    "log_amount",
    "amount_zscore",
    "user_amt_mean_30d",
    # merchant
    "merchant_fraud_rate",
    # geo / channel
    "is_foreign",
    "is_card_not_present",
    # account
    "account_age_days",
    "is_new_account",
    # target
    "label",
]


# ─────────────────────────────────────────────
# 9. Main pipeline
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[features] Raw rows: {len(df):,}")

    df = add_temporal_features(df)
    print("[features] Temporal features done")

    df = add_velocity_features(df)
    print("[features] Velocity features done")

    df = add_amount_features(df)
    print("[features] Amount features done")

    df = add_merchant_risk(df)
    print("[features] Merchant risk scores done")

    df = add_geo_features(df)
    print("[features] Geo features done")

    df = add_account_age(df)
    print("[features] Account age features done")

    # Keep only defined columns that exist in the dataframe
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[cols]

    fraud_rate = df["label"].mean() * 100
    print(f"[features] Final shape: {df.shape} | Fraud rate: {fraud_rate:.2f}%")
    return df


if __name__ == "__main__":
    df_raw      = load_data(RAW_PATH)
    df_features = build_features(df_raw)
    df_features.to_parquet(PROCESSED_PATH, index=False)
    print(f"[features] Saved to {PROCESSED_PATH}")