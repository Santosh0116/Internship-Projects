#!/usr/bin/env python3
"""
Weather Data Analysis and Prediction
====================================
AI Internship Project - Codec Technologies

Analyses a year of daily weather data and predicts NEXT-DAY temperature.

Pipeline
--------
1. Synthetic data : 365 days of Temperature, Humidity, Wind Speed and Pressure with realistic
                    seasonality and relationships, plus the target (next-day temperature).
2. EDA            : summary statistics, temperature trend plot, correlation heatmap.
3. Features       : lag features, rolling averages, daily changes, and cyclical day-of-year.
4. Models         : Linear Regression, Random Forest, and a Random Forest that predicts the
                    day-to-day CHANGE, all compared with a naive "tomorrow = today" baseline.
5. Evaluation     : MAE, MSE, RMSE and R2 on a chronological hold-out set, plus
                    time-series cross-validation.

IMPORTANT: because this is time-series data, the split is CHRONOLOGICAL (train on the first
80% of days, test on the last 20%). A random split would leak future information into training.

Setup / run
-----------
    pip install numpy pandas scikit-learn matplotlib seaborn
    python weather_analysis_prediction.py
    python weather_analysis_prediction.py --data my_weather.csv   # optional real data (see load_csv)

All plots are saved as PNG files in the 'weather_outputs' folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT_DIR = Path("weather_outputs")
TARGET = "next_day_temp"
BASE_FEATURES = ["temperature", "humidity", "wind_speed", "pressure"]
sns.set_theme(style="whitegrid")


# =============================================================================
# 1. DATA (synthetic generation, or load your own CSV)
# =============================================================================
def generate_weather_data(n_days: int = 365, seed: int = SEED) -> pd.DataFrame:
    """Create `n_days` of synthetic daily weather with realistic structure.

    Built-in relationships (so the models have something real to learn):
      - temperature follows a yearly seasonal cycle + day-to-day persistence
      - higher pressure the day before nudges the next day's temperature up
      - humidity is lower when the day is warmer than usual
      - wind is gusty (skewed) and stronger when pressure changes quickly
    Units: temperature in C, humidity in %, wind in m/s, pressure in hPa.
    """
    rng = np.random.default_rng(seed)
    n = n_days + 1  # one extra day so the final row still has a "next day" label
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    doy = dates.dayofyear.to_numpy()

    seasonal = 22 + 9 * np.sin(2 * np.pi * (doy - 110) / 365)  # warmest around mid-July

    pressure_anom = np.zeros(n)
    temp_anom = np.zeros(n)
    for t in range(1, n):
        pressure_anom[t] = 0.8 * pressure_anom[t - 1] + rng.normal(0, 2.5)
        temp_anom[t] = 0.7 * temp_anom[t - 1] + 0.18 * pressure_anom[t - 1] + rng.normal(0, 1.2)

    temperature = seasonal + temp_anom
    humidity = np.clip(68 - 1.6 * temp_anom - 0.25 * (seasonal - 22) + rng.normal(0, 5, n), 20, 100)
    wind = rng.gamma(2.5, 1.6, n) + 0.15 * np.abs(np.diff(pressure_anom, prepend=pressure_anom[0]))
    pressure = 1013 + pressure_anom

    df = pd.DataFrame({
        "date": dates,
        "temperature": temperature.round(2),
        "humidity": humidity.round(1),
        "wind_speed": np.clip(wind, 0.2, None).round(1),
        "pressure": pressure.round(1),
    })
    df[TARGET] = df["temperature"].shift(-1)          # target = tomorrow's temperature
    return df.iloc[:-1].reset_index(drop=True)        # drop the last day (it has no label)


def load_csv(path: str) -> pd.DataFrame:
    """Load real data. Needs columns: date, temperature, humidity, wind_speed, pressure
    (one row per day, in order). The next-day target is created automatically."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df[TARGET] = df["temperature"].shift(-1)
    return df.iloc[:-1].reset_index(drop=True)


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
def run_eda(df: pd.DataFrame, show: bool) -> None:
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Rows: {len(df)} | Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Missing values: {int(df.isna().sum().sum())}\n")
    print(df[BASE_FEATURES + [TARGET]].describe().round(2).to_string())

    # --- Temperature trend over time -----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df["temperature"], color="#90a4ae", linewidth=1, label="Daily temperature")
    ax.plot(df["date"], df["temperature"].rolling(7, center=True).mean(),
            color="#e53935", linewidth=2.2, label="7-day moving average")
    ax.set_title("Daily Temperature Trend")
    ax.set_xlabel("Date"); ax.set_ylabel("Temperature (C)"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "temperature_trend.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # --- Correlation heatmap -------------------------------------------------
    corr = df[BASE_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation Heatmap of Weather Features")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    print("\nCorrelation with next-day temperature:")
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string())


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that use ONLY information available on the prediction day (no leakage)."""
    out = df.copy()
    for lag in (1, 2, 7):                                    # recent history
        out[f"temp_lag{lag}"] = out["temperature"].shift(lag)
    out["temp_roll3"] = out["temperature"].rolling(3).mean()  # short-term trend
    out["temp_roll7"] = out["temperature"].rolling(7).mean()  # weekly trend
    out["temp_change"] = out["temperature"].diff()            # warming / cooling momentum
    out["pressure_change"] = out["pressure"].diff()           # falling pressure often signals change
    doy = out["date"].dt.dayofyear                            # season, encoded cyclically so
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)         # Dec 31 is "close to" Jan 1
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out.dropna().reset_index(drop=True)                # drop first rows lacking history


# =============================================================================
# 4. MODELLING AND EVALUATION
# =============================================================================
class ChangeRegressor(BaseEstimator, RegressorMixin):
    """Wraps a regressor so it predicts the CHANGE from today's temperature
    (tomorrow - today) instead of the temperature itself, then adds today's value back.

    Tree models cannot predict values outside the range seen in training, so on seasonal data
    they struggle when the temperature level shifts. Predicting the change avoids this.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.model_ = clone(self.estimator).fit(X, y - X["temperature"])
        return self

    def predict(self, X):
        return self.model_.predict(X) + X["temperature"].to_numpy()


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather data analysis and next-day temperature prediction")
    parser.add_argument("--data", help="Optional CSV with columns: date,temperature,humidity,wind_speed,pressure")
    parser.add_argument("--no_show", action="store_true", help="Do not open plot windows (PNGs are still saved)")
    # parse_known_args() ignores extra arguments injected by Jupyter/Colab,
    # so the script also works when pasted into a notebook cell.
    args, _ = parser.parse_known_args()
    show = not args.no_show
    OUT_DIR.mkdir(exist_ok=True)

    # ---- Data + EDA ---------------------------------------------------------
    df = load_csv(args.data) if args.data else generate_weather_data()
    print("First 5 rows:")
    print(df.head().to_string(index=False), "\n")
    run_eda(df, show)

    # ---- Features and chronological split -----------------------------------
    data = engineer_features(df)
    feature_cols = [c for c in data.columns if c not in ("date", TARGET)]
    X, y = data[feature_cols], data[TARGET]
    split = int(len(data) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    dates_test = data["date"].iloc[split:]
    print("\n" + "=" * 60)
    print("MODELLING")
    print("=" * 60)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Train: {len(X_train)} days ({data['date'].iloc[0].date()} to {data['date'].iloc[split - 1].date()})")
    print(f"Test : {len(X_test)} days ({dates_test.iloc[0].date()} to {dates_test.iloc[-1].date()})")

    def new_forest() -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": new_forest(),
        "Random Forest (change)": ChangeRegressor(new_forest()),
    }
    predictions = {"Baseline (tomorrow = today)": X_test["temperature"].to_numpy()}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    # ---- Metrics table ------------------------------------------------------
    results = pd.DataFrame({name: regression_metrics(y_test, pred) for name, pred in predictions.items()}).T
    print("\nTest-set performance (lower MAE/MSE/RMSE is better, higher R2 is better):")
    print(results.round(3).to_string())

    # ---- Time-series cross-validation (no shuffling, always train on the past) ----
    print("\n5-fold TimeSeriesSplit cross-validation (MAE, mean +/- std):")
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in models.items():
        scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        print(f"  {name:<24}: {scores.mean():.3f} +/- {scores.std():.3f}")

    # ---- Feature importance ---------------------------------------------------
    rf = models["Random Forest (change)"].model_
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances (Random Forest predicting the day-to-day change):")
    print(importance.head(6).round(3).to_string())
    lr_coef = pd.Series(models["Linear Regression"][-1].coef_, index=feature_cols)
    print("\nLargest standardised Linear Regression coefficients:")
    print(lr_coef.reindex(lr_coef.abs().sort_values(ascending=False).index).head(5).round(3).to_string())

    # ---- Plots ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(dates_test, y_test.to_numpy(), color="black", linewidth=2, label="Actual")
    axes[0].plot(dates_test, predictions["Linear Regression"], "--", color="#1e88e5", label="Linear Regression")
    axes[0].plot(dates_test, predictions["Random Forest"], "--", color="#43a047", label="Random Forest")
    axes[0].plot(dates_test, predictions["Random Forest (change)"], "--", color="#8e24aa",
                 label="Random Forest (change)")
    axes[0].set_title("Actual vs Predicted Next-Day Temperature (test period)")
    axes[0].set_xlabel("Date"); axes[0].set_ylabel("Temperature (C)"); axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)
    best = results.drop(index="Baseline (tomorrow = today)")["MAE"].idxmin()
    axes[1].scatter(y_test, predictions[best], alpha=0.7, color="#fb8c00")
    lims = [min(y_test.min(), predictions[best].min()), max(y_test.max(), predictions[best].max())]
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_title(f"{best}: predicted vs actual")
    axes[1].set_xlabel("Actual (C)"); axes[1].set_ylabel("Predicted (C)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "predictions.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.head(10).sort_values().plot(kind="barh", color="#26a69a", ax=ax)
    ax.set_title("Random Forest (change model) - Top 10 Feature Importances"); ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    print(f"\nBest model by MAE: {best}")
    print(f"Plots saved in '{OUT_DIR}/': temperature_trend.png, correlation_heatmap.png, "
          f"predictions.png, feature_importance.png")


if __name__ == "__main__":
    main()
