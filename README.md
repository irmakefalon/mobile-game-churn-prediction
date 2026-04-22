# 🎮 Early Churn Prediction in Mobile Gaming

## 📊 Problem

Predict which users will churn in a mobile game using only **early behavioral data (first 3 days after registration)**.

Churn definition:

> A user who does not return after day 7.

---

## 📁 Dataset

* User registration timestamps
* User login/activity logs
* A/B test group
* Revenue

---

## ⚙️ Feature Engineering

To avoid data leakage, only early user behavior was used.

### Features:

* `total_sessions_3d`: number of sessions in first 3 days
* `activity_span_seconds_3d`: time span of activity
* `revenue`: early monetization signal
* `testgroup`: A/B test group

---

## 🧠 Modeling Approach

* Time-based train/test split
* Baseline: Logistic Regression
* Comparison: Random Forest
* Class imbalance handled using:

  * `class_weight="balanced"`
  * threshold tuning

---

## 📈 Results

| Model               | ROC AUC  |
| ------------------- | -------- |
| Logistic Regression | **0.73** |
| Random Forest       | 0.72     |

* Best threshold: **0.6**
* High recall for churners
* Trade-off: increased false positives

---

## 🔍 Key Insights

* Early engagement is the strongest predictor of retention
* Users with multiple sessions in the first 3 days are significantly less likely to churn
* Revenue and A/B test group had minimal predictive power

---

## 💡 Business Impact

* Identify at-risk users within first 3 days
* Trigger retention campaigns (notifications, rewards)
* Improve onboarding experience

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn

---

## 📌 Key Learnings

* Importance of avoiding data leakage
* Time-based validation in user behavior modeling
* Trade-off between recall and precision in imbalanced datasets

---

## 🚀 Future Improvements

* Add temporal features (session intervals, trends)
* Try gradient boosting models (XGBoost, LightGBM)
* Deploy as real-time churn scoring system
