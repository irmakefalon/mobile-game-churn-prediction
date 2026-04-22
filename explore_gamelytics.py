import pandas as pd
import os

path = r"C:\Users\irmak\.cache\kagglehub\datasets\debs2x\gamelytics-mobile-analytics-challenge\versions\2"

#print("Files in folder:")
#print(os.listdir(path))

reg = pd.read_csv(os.path.join(path, "reg_data.csv"), sep=';')
auth = pd.read_csv(os.path.join(path, "auth_data.csv"), sep=';')
ab = pd.read_csv(os.path.join(path, "ab_test.csv"), sep=';')

#print("\nREG columns:")
#print(reg.columns)
#print(reg.head())

#print("\nAUTH columns:")
#print(auth.columns)
#print(auth.head())

#print("\nAB columns:")
#print(ab.columns)
#print(ab.head())

## Normalize schema

ab = ab.rename(columns={"user_id": "uid"})

## Convert timestamps

reg['reg_ts'] = pd.to_datetime(reg['reg_ts'], unit='s')
auth['auth_ts'] = pd.to_datetime(auth['auth_ts'], unit='s')

## Last activity 

last_activity = auth.groupby('uid', as_index=False)['auth_ts'].max()
last_activity = last_activity.rename(columns={'auth_ts': 'last_auth_ts'})

## Merge 

df = reg.merge(last_activity, on='uid', how='left')
df = df.merge(ab, on='uid', how='left')

## Define churn

# Churn = no activity for ≥ 7 days after last seen date

max_date = auth['auth_ts'].max()

df['days_inactive'] = (max_date - df['last_auth_ts']).dt.days
df['churn'] = (df['days_inactive'] >= 7).astype(int)

## Sanity check

print(df[['uid', 'reg_ts', 'last_auth_ts', 'days_inactive', 'churn']].head())
print("\nChurn rate:", df['churn'].mean())


