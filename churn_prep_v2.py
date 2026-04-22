import pandas as pd
import os

# Dataset path
path = r"C:\Users\irmak\.cache\kagglehub\datasets\debs2x\gamelytics-mobile-analytics-challenge\versions\2"

# Load data
reg = pd.read_csv(os.path.join(path, "reg_data.csv"), sep=';')
auth = pd.read_csv(os.path.join(path, "auth_data.csv"), sep=';')
ab = pd.read_csv(os.path.join(path, "ab_test.csv"), sep=';')

# Make column names consistent
ab = ab.rename(columns={"user_id": "uid"})

# Convert timestamps
reg["reg_ts"] = pd.to_datetime(reg["reg_ts"], unit="s")
auth["auth_ts"] = pd.to_datetime(auth["auth_ts"], unit="s")

# Merge auth with registration time
full = auth.merge(reg, on="uid", how="left")

# Days since registration
full["days_since_reg"] = (full["auth_ts"] - full["reg_ts"]).dt.days

# Users who returned after day 7
later_activity = full[full["days_since_reg"] > 7]
returned_after_7 = later_activity.groupby("uid").size().reset_index(name="returned_after_7")

# Build user-level dataset
df = reg.copy()
df["returned_after_7"] = df["uid"].isin(returned_after_7["uid"]).astype(int)

# Churn = user did not return after day 7
df["churn"] = 1 - df["returned_after_7"]

# Merge A/B test info
df = df.merge(ab, on="uid", how="left")

# Check output
print(df.head())
print("\nChurn distribution:")
print(df["churn"].value_counts())
print("\nChurn rate:")
print(df["churn"].mean())   