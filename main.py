"""
Driver Acceptance Prediction System
=====================================
Predicts the probability that a taxi driver will accept a ride request,
and ranks drivers by acceptance likelihood for a given time slot and area.

Modeling Assumptions:
- A driver with acceptance_rate > 0.6 is labeled as a likely acceptor (will_accept=1).
- Historical behavior (acceptance/rejection rates) is the strongest signal.
- Time slot and active area influence driver availability and willingness.
- Random Forest is expected to outperform Logistic Regression due to non-linear patterns.

Usage:
    python main.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
RANDOM_SEED      = 42
TEST_SIZE        = 0.20
N_SYNTHETIC_ROWS = 500
DATA_DIR         = "data"
OUTPUT_DIR       = "output"
DATA_FILE        = os.path.join(DATA_DIR, "drivers.csv")
OUTPUT_FILE      = os.path.join(OUTPUT_DIR, "ranked_drivers.csv")

# Prediction filters — change these to query different segments
FILTER_TIME_SLOT   = "evening"   # morning | afternoon | evening | night
FILTER_ACTIVE_AREA = "Downtown"  # must match a value in the dataset
TOP_N_DRIVERS      = 10


# ──────────────────────────────────────────────
# 1. DATA GENERATION
# ──────────────────────────────────────────────
def data_generation(n: int = N_SYNTHETIC_ROWS) -> pd.DataFrame:
    """
    Generate a synthetic driver dataset when no real CSV is available.

    Parameters
    ----------
    n : int
        Number of synthetic driver records to create.

    Returns
    -------
    pd.DataFrame
        Raw synthetic driver data.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    car_types    = ["sedan", "suv", "hatchback", "luxury"]
    active_areas = ["Downtown", "Suburb", "Airport", "Mall", "University"]
    time_slots   = ["morning", "afternoon", "evening", "night"]

    total_requests    = rng.integers(50, 500, size=n)
    accepted_requests = (total_requests * rng.uniform(0.3, 0.95, size=n)).astype(int)
    # Ensure accepted <= total
    accepted_requests = np.minimum(accepted_requests, total_requests)
    rejected_requests = total_requests - accepted_requests

    df = pd.DataFrame({
        "driver_id"               : [f"DRV_{i:04d}" for i in range(1, n + 1)],
        "car_type"                : rng.choice(car_types, size=n),
        "active_area"             : rng.choice(active_areas, size=n),
        "avg_active_hours_per_day": np.round(rng.uniform(2.0, 12.0, size=n), 2),
        "total_requests"          : total_requests,
        "accepted_requests"       : accepted_requests,
        "rejected_requests"       : rejected_requests,
        "time_slot"               : rng.choice(time_slots, size=n),
    })

    print(f"[data_generation] Generated {n} synthetic driver records.")
    return df


# ──────────────────────────────────────────────
# 2. DATA LOADER
# ──────────────────────────────────────────────
def data_loader(filepath: str) -> pd.DataFrame:
    """
    Load driver data from a CSV or Excel file.

    Parameters
    ----------
    filepath : str
        Path to the data file.

    Returns
    -------
    pd.DataFrame
        Raw loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV or Excel.")

    print(f"[data_loader] Loaded {len(df)} rows from '{filepath}'.")
    return df


# ──────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "driver_id", "car_type", "active_area",
    "avg_active_hours_per_day", "total_requests",
    "accepted_requests", "rejected_requests", "time_slot",
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the raw driver DataFrame.

    Steps:
    - Check required columns exist
    - Validate driver_id uniqueness
    - Drop rows with missing values in critical columns
    - Clip numeric fields to valid ranges
    - Ensure accepted + rejected <= total_requests (fix inconsistencies)

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    # ── Column check ──────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    original_len = len(df)

    # ── Duplicate driver IDs ──────────────────
    if df["driver_id"].duplicated().any():
        n_dupes = df["driver_id"].duplicated().sum()
        print(f"  [preprocess] WARNING: {n_dupes} duplicate driver_id(s) found — keeping first occurrence.")
        df = df.drop_duplicates(subset="driver_id", keep="first")

    # ── Drop rows missing critical values ─────
    critical = ["driver_id", "total_requests", "accepted_requests", "rejected_requests"]
    df = df.dropna(subset=critical)

    # ── Coerce numeric columns ─────────────────
    numeric_cols = ["avg_active_hours_per_day", "total_requests",
                    "accepted_requests", "rejected_requests"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # ── Clip to sensible ranges ────────────────
    df["avg_active_hours_per_day"] = df["avg_active_hours_per_day"].clip(0, 24)
    df["total_requests"]           = df["total_requests"].clip(lower=0)
    df["accepted_requests"]        = df["accepted_requests"].clip(lower=0)
    df["rejected_requests"]        = df["rejected_requests"].clip(lower=0)

    # ── Fix: accepted + rejected must not exceed total ──
    df["accepted_requests"] = np.minimum(df["accepted_requests"], df["total_requests"])
    df["rejected_requests"] = np.minimum(df["rejected_requests"],
                                          df["total_requests"] - df["accepted_requests"])

    # ── Fill missing categorical values ────────
    df["car_type"]    = df["car_type"].fillna("unknown")
    df["active_area"] = df["active_area"].fillna("unknown")
    df["time_slot"]   = df["time_slot"].fillna("unknown")

    # ── Standardise categorical case ──────────
    df["time_slot"]   = df["time_slot"].str.strip().str.lower()
    df["car_type"]    = df["car_type"].str.strip().str.lower()
    df["active_area"] = df["active_area"].str.strip().str.title()

    print(f"[preprocess] {original_len} → {len(df)} rows after cleaning.")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive model features from cleaned driver data.

    Features created:
    - acceptance_rate    : fraction of requests accepted
    - rejection_rate     : fraction of requests rejected
    - activity_score     : hours active × acceptance_rate (productivity proxy)
    - consistency_score  : acceptance_rate − rejection_rate (net tendency)
    - will_accept        : binary target (1 if acceptance_rate > 0.6)
    - One-hot encoding   : car_type, active_area, time_slot

    Division-by-zero is prevented by replacing zero total_requests with NaN,
    setting derived rates to 0, and dropping those rows.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed driver data.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame with encoded categoricals and target column.
    """
    # ── Safe rate computation (avoid /0) ──────
    total = df["total_requests"].replace(0, np.nan)

    df["acceptance_rate"]  = (df["accepted_requests"] / total).fillna(0).clip(0, 1)
    df["rejection_rate"]   = (df["rejected_requests"] / total).fillna(0).clip(0, 1)
    df["activity_score"]   = df["avg_active_hours_per_day"] * df["acceptance_rate"]
    df["consistency_score"]= df["acceptance_rate"] - df["rejection_rate"]

    # ── Target variable ────────────────────────
    df["will_accept"] = (df["acceptance_rate"] > 0.6).astype(int)

    # ── One-hot encode categoricals ───────────
    df = pd.get_dummies(df, columns=["car_type", "active_area", "time_slot"],
                        prefix=["car", "area", "slot"], drop_first=False)

    print(f"[feature_engineering] Feature matrix shape: {df.shape}")
    print(f"  Target distribution — will_accept=1: "
          f"{df['will_accept'].sum()} / {len(df)} "
          f"({df['will_accept'].mean()*100:.1f}%)")
    return df


# ──────────────────────────────────────────────
# 5. TRAIN MODELS
# ──────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    """
    Train Logistic Regression and Random Forest classifiers.

    The feature set excludes identifier columns and the target.
    Features are scaled for Logistic Regression (required) but passed
    raw to Random Forest (tree-based, scale-invariant).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    tuple:
        (lr_model, rf_model, scaler, X_test, y_test, feature_cols)
        - lr_model     : trained LogisticRegression
        - rf_model     : trained RandomForestClassifier
        - scaler       : fitted StandardScaler (for LR)
        - X_test       : unscaled test features (DataFrame)
        - y_test       : test labels (Series)
        - feature_cols : list of feature column names
    """
    # ── Identify feature columns ───────────────
    non_feature = {"driver_id", "will_accept",
                   "accepted_requests", "rejected_requests", "total_requests"}
    feature_cols = [c for c in df.columns if c not in non_feature
                    and df[c].dtype in [np.float64, np.int64, np.uint8, bool]]

    X = df[feature_cols]
    y = df["will_accept"]

    # ── Train / test split ─────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # ── Scale for Logistic Regression ─────────
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Logistic Regression ────────────────────
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr_model.fit(X_train_scaled, y_train)
    print("[train_models] Logistic Regression trained.")

    # ── Random Forest ──────────────────────────
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("[train_models] Random Forest trained.")

    return lr_model, rf_model, scaler, X_test, y_test, feature_cols


# ──────────────────────────────────────────────
# 6. EVALUATE MODELS
# ──────────────────────────────────────────────
def evaluate_models(lr_model, rf_model, scaler, X_test, y_test):
    """
    Print evaluation metrics for both models and save a comparison bar chart.

    Metrics reported: Accuracy, Precision, Recall, F1-score.

    Parameters
    ----------
    lr_model : LogisticRegression
    rf_model : RandomForestClassifier
    scaler   : StandardScaler (fitted on training data)
    X_test   : pd.DataFrame — unscaled test features
    y_test   : pd.Series    — true labels
    """
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name, model, X in [
        ("Logistic Regression", lr_model, X_test_scaled),
        ("Random Forest",       rf_model, X_test),
    ]:
        y_pred = model.predict(X)
        results[name] = {
            "Accuracy" : accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall"   : recall_score(y_test, y_pred, zero_division=0),
            "F1"       : f1_score(y_test, y_pred, zero_division=0),
        }
        print(f"\n{'='*45}")
        print(f"  {name}")
        print(f"{'='*45}")
        for metric, val in results[name].items():
            print(f"  {metric:<12}: {val:.4f}")
        print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # ── Bar chart comparison ───────────────────
    _plot_model_comparison(results)
    return results


def _plot_model_comparison(results: dict):
    """Save a grouped bar chart comparing model metrics to output/."""
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, [results[models[0]][m] for m in metrics], width,
                   label=models[0], color="#4C72B0")
    bars2 = ax.bar(x + width/2, [results[models[1]][m] for m in metrics], width,
                   label=models[1], color="#DD8452")

    ax.set_title("Model Comparison — Driver Acceptance Prediction", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chart_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate_models] Chart saved → {chart_path}")


# ──────────────────────────────────────────────
# 7. RANK DRIVERS
# ──────────────────────────────────────────────
def rank_drivers(
    df: pd.DataFrame,
    rf_model,
    feature_cols: list,
    time_slot: str   = FILTER_TIME_SLOT,
    active_area: str = FILTER_ACTIVE_AREA,
    top_n: int       = TOP_N_DRIVERS,
) -> pd.DataFrame:
    """
    Score all drivers with the best model (Random Forest), apply filters,
    and return the top-N ranked drivers.

    Parameters
    ----------
    df          : feature-engineered DataFrame (full dataset)
    rf_model    : trained RandomForestClassifier
    feature_cols: list of column names used during training
    time_slot   : filter — only drivers whose recorded time_slot matches
    active_area : filter — only drivers in this area
    top_n       : number of top drivers to return

    Returns
    -------
    pd.DataFrame
        Top-N drivers sorted by acceptance_probability descending.
    """
    # ── Predict acceptance probability ────────
    X_all = df[feature_cols]
    df = df.copy()
    df["acceptance_probability"] = rf_model.predict_proba(X_all)[:, 1]

    # ── Rebuild original categorical columns for filtering ──
    # (they were one-hot encoded; recover from encoded dummies)
    time_col = f"slot_{time_slot.lower()}"
    area_col = f"area_{active_area.title()}"

    # Filter by time slot
    if time_col in df.columns:
        mask_time = df[time_col] == 1
    else:
        print(f"  [rank_drivers] WARNING: time_slot '{time_slot}' not found in data — skipping filter.")
        mask_time = pd.Series(True, index=df.index)

    # Filter by active area
    if area_col in df.columns:
        mask_area = df[area_col] == 1
    else:
        print(f"  [rank_drivers] WARNING: active_area '{active_area}' not found in data — skipping filter.")
        mask_area = pd.Series(True, index=df.index)

    filtered = df[mask_time & mask_area].copy()
    print(f"[rank_drivers] Drivers matching time_slot='{time_slot}' "
          f"& active_area='{active_area}': {len(filtered)}")

    if filtered.empty:
        print("  [rank_drivers] No drivers matched filters — returning top drivers from full dataset.")
        filtered = df.copy()

    # ── Sort & select output columns ──────────
    output_cols = [
        "driver_id", "acceptance_probability",
        "acceptance_rate", "rejection_rate",
        "activity_score", "consistency_score",
        "avg_active_hours_per_day", "will_accept",
    ]
    available = [c for c in output_cols if c in filtered.columns]
    ranked = (
        filtered[available]
        .sort_values("acceptance_probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.index += 1  # Rank starts from 1
    ranked.index.name = "rank"

    # ── Save to CSV ────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ranked.to_csv(OUTPUT_FILE)
    print(f"[rank_drivers] Ranked drivers saved → {OUTPUT_FILE}")

    return ranked


# ──────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────
def main():
    """
    End-to-end pipeline:
    1. Load or generate data
    2. Preprocess
    3. Feature engineering
    4. Train models
    5. Evaluate models
    6. Rank & export drivers
    """
    print("\n" + "═" * 55)
    print("   DRIVER ACCEPTANCE PREDICTION SYSTEM")
    print("═" * 55 + "\n")

    # ── Step 1: Load or generate data ─────────
    try:
        raw_df = data_loader(DATA_FILE)
    except FileNotFoundError:
        print(f"[main] No data file found at '{DATA_FILE}'.")
        print("[main] Generating synthetic dataset instead...\n")
        raw_df = data_generation(N_SYNTHETIC_ROWS)

        # Persist synthetic data for inspection / reuse
        os.makedirs(DATA_DIR, exist_ok=True)
        raw_df.to_csv(DATA_FILE, index=False)
        print(f"[main] Synthetic data saved → {DATA_FILE}\n")

    # ── Step 2: Preprocess ─────────────────────
    clean_df = preprocess(raw_df)

    # ── Step 3: Feature engineering ───────────
    feature_df = feature_engineering(clean_df)

    # ── Step 4: Train ─────────────────────────
    lr_model, rf_model, scaler, X_test, y_test, feature_cols = train_models(feature_df)

    # ── Step 5: Evaluate ──────────────────────
    evaluate_models(lr_model, rf_model, scaler, X_test, y_test)

    # ── Step 6: Rank drivers ───────────────────
    print(f"\n[main] Ranking top {TOP_N_DRIVERS} drivers for "
          f"time_slot='{FILTER_TIME_SLOT}', active_area='{FILTER_ACTIVE_AREA}'...\n")
    ranked = rank_drivers(
        feature_df, rf_model, feature_cols,
        time_slot=FILTER_TIME_SLOT,
        active_area=FILTER_ACTIVE_AREA,
        top_n=TOP_N_DRIVERS,
    )

    print("\n" + "─" * 55)
    print(f"  TOP {TOP_N_DRIVERS} DRIVERS  |  {FILTER_TIME_SLOT.upper()} · {FILTER_ACTIVE_AREA.upper()}")
    print("─" * 55)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(ranked.to_string())

    print("\n[main] ✅  Pipeline complete.")
    print(f"  → Ranked drivers : {OUTPUT_FILE}")
    print(f"  → Model chart    : {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()