"""ATM cash-demand forecasting (predictive analysis).

Predicts daily cash outflow across the ATM fleet using lag features,
calendar features, and gradient boosting — benchmarked against a
seasonal-naive baseline and linear regression.

Data: the 2,290 cleaned transactions from the sibling SQL audit project
(synthetic but patterned on real cash-management behaviour).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SRC = "/home/hatch/workspace/data-analyst-projects/atm-data-quality-sql/cleaned.csv"
OUT = "/home/hatch/workspace/data-analyst-projects/atm-cash-demand-forecasting/charts"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# ---------- 1. daily demand series ----------
df = pd.read_csv(SRC)
df["txn_date"] = pd.to_datetime(df["txn_date"])
wd = df[(df["txn_type"] == "CASH_WITHDRAWAL")
        & (df["status"].str.lower() == "success")].copy()
wd["amount"] = pd.to_numeric(wd["amount"], errors="coerce").fillna(0)
daily = (wd.groupby("txn_date")["amount"].sum()
           .asfreq("D", fill_value=0.0).rename("demand"))
print(f"days={len(daily)}  total=Rs {daily.sum():,.0f}  "
      f"mean=Rs {daily.mean():,.0f}/day")

# ---------- 2. feature engineering ----------
f = daily.to_frame()
f["lag_1"] = f["demand"].shift(1)
f["lag_7"] = f["demand"].shift(7)
f["lag_14"] = f["demand"].shift(14)
f["roll7"] = f["demand"].shift(1).rolling(7).mean()
f["roll14"] = f["demand"].shift(1).rolling(14).mean()
f["dow"] = f.index.dayofweek
f["weekend"] = (f["dow"] >= 5).astype(int)
f["month"] = f.index.month
f = f.dropna()
FEATURES = ["lag_1", "lag_7", "lag_14", "roll7", "roll14",
            "dow", "weekend", "month"]
X, y = f[FEATURES], f["demand"]

# time-based split: last 14 days = test
split = len(f) - 14
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"train={len(X_train)} test={len(X_test)}")

# ---------- 3. models ----------
def mape(a, p):
    return float(np.mean(np.abs((a - p) / np.maximum(a, 1e-9))) * 100)

results = {}

# baseline: seasonal naive (same weekday last week)
base_pred = X_test["lag_7"].values
results["Seasonal naive"] = base_pred

lr = LinearRegression().fit(X_train, y_train)
results["Linear regression"] = lr.predict(X_test)

gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                   max_leaf_nodes=15, random_state=42)
gbm.fit(X_train, y_train)
results["Gradient boosting"] = gbm.predict(X_test)

metrics = {}
for name, pred in results.items():
    metrics[name] = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAPE": mape(y_test.values, pred),
    }
m = pd.DataFrame(metrics).T
print(m.round(0).to_string())
m.to_csv(OUT.replace("charts", "") + "model_metrics.csv")

# ---------- 4. charts ----------
# 4a. actual vs predicted (test window)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(y_test.index, y_test.values, "o-", label="Actual", color="#1a1a1a")
for name, color in [("Seasonal naive", "#9e9e9e"),
                    ("Linear regression", "#42a5f5"),
                    ("Gradient boosting", "#e53935")]:
    ax.plot(y_test.index, results[name], "s--", ms=5, label=name, color=color)
ax.set_title("Daily cash demand — actual vs predicted (14-day test window)")
ax.set_ylabel("Rs"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/01_actual_vs_predicted.png"); plt.close(fig)

# 4b. model comparison
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(m)); w = 0.25
for i, col in enumerate(["MAE", "RMSE", "MAPE"]):
    ax.bar(x + i * w, m[col], w, label=col)
ax.set_xticks(x + w); ax.set_xticklabels(m.index, rotation=8, ha="right")
ax.set_title("Forecast error by model (lower is better)")
ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/02_model_comparison.png"); plt.close(fig)

# 4c. feature importance (gradient boosting)
imp = pd.Series(np.zeros(len(FEATURES)), index=FEATURES)
# HistGradientBoostingRegressor has no built-in importance; use permutation
from sklearn.inspection import permutation_importance
pi = permutation_importance(gbm, X_test, y_test, n_repeats=10,
                            random_state=42, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot.barh(ax=ax, color="#43a047")
ax.set_title("What drives the forecast? (permutation importance, MAE)")
ax.set_xlabel("Drop in MAE when shuffled (Rs)"); fig.tight_layout()
fig.savefig(f"{OUT}/03_feature_importance.png"); plt.close(fig)

# ---------- 5. recursive 7-day forecast (refit on all data) ----------
gbm_full = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=15, random_state=42)
gbm_full.fit(X, y)
hist = list(y.values)
future_dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=7)
forecast = []
for d in future_dates:
    row = {}
    row["lag_1"] = hist[-1]; row["lag_7"] = hist[-7]; row["lag_14"] = hist[-14]
    row["roll7"] = float(np.mean(hist[-7:])); row["roll14"] = float(np.mean(hist[-14:]))
    row["dow"] = d.dayofweek; row["weekend"] = int(d.dayofweek >= 5)
    row["month"] = d.month
    pred = float(gbm_full.predict(pd.DataFrame([row], columns=FEATURES))[0])
    forecast.append(max(pred, 0.0)); hist.append(forecast[-1])
fc = pd.Series(forecast, index=future_dates)
fc.to_csv(OUT.replace("charts", "") + "forecast_next7.csv", header=["forecast_rs"])

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(y.index[-21:], y.values[-21:], "o-", ms=4, label="Actual (last 21d)",
        color="#1a1a1a")
ax.plot(fc.index, fc.values, "s--", ms=6, label="Forecast (next 7d)",
        color="#e53935")
ax.axvline(y.index[-1], color="gray", ls=":", label="today")
ax.set_title("Cash-demand forecast — next 7 days")
ax.set_ylabel("Rs"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/04_seven_day_forecast.png"); plt.close(fig)

print("\nNext-7-day forecast (Rs):")
print(fc.round(0).to_string())
print("\ncharts written to", OUT)
