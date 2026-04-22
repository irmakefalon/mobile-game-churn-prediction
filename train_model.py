import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv(r"C:\Users\irmak\Desktop\gamelytics_features.csv")

# =========================
# 2. Sort by time
# =========================
df = df.sort_values("reg_ts")

# =========================
# 3. Drop leakage + useless columns
# =========================
df = df.drop(columns=["uid", "returned_after_7", "mid_activity_3_7d"])

# =========================
# 4. Train-test split (time-based)
# =========================
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train.drop(columns=["churn", "reg_ts"])
y_train = train["churn"]

X_test = test.drop(columns=["churn", "reg_ts"])
y_test = test["churn"]

# =========================
# 5. Train model
# =========================
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# =========================
# Model interpretation
# =========================
import pandas as pd

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_[0]
})

coef_df = coef_df.sort_values(by="coefficient", ascending=False)

print("\nFeature Importance (Logistic Regression Coefficients):")
print(coef_df)

# =========================
# 6. Evaluate
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#threshold = 0.7  # try 0.6, 0.7, 0.8
threshold = 0.8
y_pred_adjusted = (y_prob > threshold).astype(int)

print("\n=== Adjusted Threshold ===")
print(classification_report(y_test, y_pred_adjusted))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_adjusted))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_prob))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("\nROC AUC:", roc_auc_score(y_test, y_prob_rf))