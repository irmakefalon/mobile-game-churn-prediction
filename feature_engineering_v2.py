import pandas as pd
import os

# =========================
# 1. Load data
# =========================
path = r"C:\Users\irmak\.cache\kagglehub\datasets\debs2x\gamelytics-mobile-analytics-challenge\versions\2"

reg = pd.read_csv(os.path.join(path, "reg_data.csv"), sep=";")
auth = pd.read_csv(os.path.join(path, "auth_data.csv"), sep=";")
ab = pd.read_csv(os.path.join(path, "ab_test.csv"), sep=";")

# Make user ID column consistent
ab = ab.rename(columns={"user_id": "uid"})

# Convert timestamps
reg["reg_ts"] = pd.to_datetime(reg["reg_ts"], unit="s")
auth["auth_ts"] = pd.to_datetime(auth["auth_ts"], unit="s")

# =========================
# 2. Merge registration + activity
# =========================
full = auth.merge(reg, on="uid", how="left")
full["days_since_reg"] = (full["auth_ts"] - full["reg_ts"]).dt.days

# =========================
# 3. Feature windows
# =========================
early = full[full["days_since_reg"] <= 3].copy()
mid = full[(full["days_since_reg"] > 3) & (full["days_since_reg"] <= 7)].copy()
later = full[full["days_since_reg"] > 7].copy()

# =========================
# 4. Base feature table
# =========================
features = reg[["uid", "reg_ts"]].copy()

# =========================
# 5. Feature engineering
# =========================

# Feature 1: total activity in first 3 days
sessions_3d = (
    early.groupby("uid")
    .size()
    .reset_index(name="total_sessions_3d")
)
features = features.merge(sessions_3d, on="uid", how="left")

# Feature 2: activity span in first 3 days
activity_span = (
    early.groupby("uid")["auth_ts"]
    .agg(lambda x: (x.max() - x.min()).total_seconds())
    .reset_index(name="activity_span_seconds_3d")
)
features = features.merge(activity_span, on="uid", how="left")

# Feature 3: activity between day 3 and day 7
mid_activity = (
    mid.groupby("uid")
    .size()
    .reset_index(name="mid_activity_3_7d")
)
features = features.merge(mid_activity, on="uid", how="left")

# Fill missing feature values
features["total_sessions_3d"] = features["total_sessions_3d"].fillna(0)
features["activity_span_seconds_3d"] = features["activity_span_seconds_3d"].fillna(0)
features["mid_activity_3_7d"] = features["mid_activity_3_7d"].fillna(0)

# =========================
# 6. Churn label
# =========================
returned_after_7 = later["uid"].drop_duplicates()

features["returned_after_7"] = features["uid"].isin(returned_after_7).astype(int)
features["churn"] = 1 - features["returned_after_7"]

# =========================
# 7. Merge business variables
# =========================
features = features.merge(ab, on="uid", how="left")

# Encode test group
features["testgroup"] = features["testgroup"].map({"a": 0, "b": 1})

# Fill missing business values
features["revenue"] = features["revenue"].fillna(0)
features["testgroup"] = features["testgroup"].fillna(0)

# =========================
# 8. Checks
# =========================
print(features.head())

print("\nColumns:")
print(features.columns)

print("\nMissing values:")
print(features.isnull().sum())

print("\nChurn rate:")
print(features["churn"].mean())

print("\nFeature summary:")
print(
    features[
        [
            "total_sessions_3d",
            "activity_span_seconds_3d",
            "mid_activity_3_7d",
            "revenue",
            "testgroup",
        ]
    ].describe()
)

print("\nFeature correlation:")
print(
    features[
        [
            "total_sessions_3d",
            "activity_span_seconds_3d",
            "mid_activity_3_7d",
            "revenue",
            "testgroup",
        ]
    ].corr()
)

# =========================
# 9. Save dataset
# =========================
output_path = r"C:\Users\irmak\Desktop\gamelytics_features.csv"
features.to_csv(output_path, index=False)

print(f"\nSaved to {output_path}")