"""
src/data/preprocessing.py
─────────────────────────
Reusable preprocessing pipeline for ScreenMind (Milestone 3).

The notebook 02_preprocessing.ipynb runs this interactively with explanations.
The Milestone 4 training script imports these functions directly.

Pipeline:
  1. load_and_clean(path)       → cleaned DataFrame
  2. make_features(df)          → X, y_reg, y_clf
  3. split_and_scale(X, y_clf)  → dict with all splits + fitted scaler + imputer
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Feature columns ───────────────────────────────────────────────────────────
# See notebook Section 3 for the reasoning behind each inclusion/exclusion.
FEATURE_COLS: list[str] = [
    "PHYSHLTH",
    "GENHLTH",
    "ADDEPEV3",
    "LSATISFY",
    "EMTSUPRT",
    "SDLONELY",
    "SDHBILLS",
    "EXERANY2",
    "SMOKE100",
    "SEXVAR",
    "EMPLOY1",
    "INCOME3",
    "EDUCA",
    "_AGEG5YR",
    "_BMI5CAT",
]

RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_days(s: pd.Series) -> pd.Series:
    """For 1-30 day columns: 88 → 0 (none), 77/99 → NaN."""
    return s.replace(88, 0).replace([77, 99], np.nan)


def _clean_binary(s: pd.Series) -> pd.Series:
    """Yes/No columns: 2 → 0 (No), 7/9 → NaN."""
    return s.replace(2, 0).replace([7, 9], np.nan)


def _clean_scale(s: pd.Series) -> pd.Series:
    """Likert-scale columns: invalidate don't-know (7) and refused (9)."""
    return s.replace([7, 9], np.nan)


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load the raw BRFSS XPT file and apply sentinel-code cleaning.

    Args:
        path: Path to the raw LLCP2024.XPT file.

    Returns:
        Cleaned DataFrame with FEATURE_COLS + MENTHLTH + high_risk columns.
    """
    need_cols = FEATURE_COLS + ["MENTHLTH", "POORHLTH", "_MENT14D"]

    df_raw = pd.read_sas(str(path), encoding="latin-1")
    available = [c for c in need_cols if c in df_raw.columns]
    df = df_raw[available].copy()

    # Days columns (88 = none/zero, 77/99 = unknown)
    for col in ["MENTHLTH", "PHYSHLTH", "POORHLTH"]:
        if col in df.columns:
            df[col] = _clean_days(df[col])

    df["GENHLTH"] = _clean_scale(df["GENHLTH"])

    for col in ["ADDEPEV3", "EXERANY2", "SMOKE100", "SDHBILLS"]:
        if col in df.columns:
            df[col] = _clean_binary(df[col])

    if "SEXVAR" in df.columns:
        df["SEXVAR"] = df["SEXVAR"].replace(2, 0)  # 1=Male, 0=Female

    for col in ["LSATISFY", "EMTSUPRT", "SDLONELY"]:
        if col in df.columns:
            df[col] = _clean_scale(df[col])

    for col in ["EMPLOY1", "INCOME3", "EDUCA"]:
        if col in df.columns:
            df[col] = df[col].replace([77, 99, 7, 9], np.nan)

    if "_AGEG5YR" in df.columns:
        df["_AGEG5YR"] = df["_AGEG5YR"].replace(14, np.nan)

    if "_MENT14D" in df.columns:
        df["_MENT14D"] = df["_MENT14D"].replace(9, np.nan)

    # Derive classification target (>14 days = high risk)
    df["high_risk"] = np.where(
        df["MENTHLTH"].isna(), np.nan,
        (df["MENTHLTH"] > 14).astype(float),
    )

    return df


def make_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the feature matrix and both targets from a cleaned DataFrame.

    Drops rows with missing MENTHLTH (can't train on unknown target).

    Args:
        df: Cleaned DataFrame from load_and_clean().

    Returns:
        X        — float32 array of shape (n_samples, n_features)
        y_reg    — float32 array of shape (n_samples,), continuous 0-30
        y_clf    — float32 array of shape (n_samples,), binary 0/1
    """
    mask = df["MENTHLTH"].notna() & df["high_risk"].notna()
    df_clean = df[mask].reset_index(drop=True)

    # Verify all feature columns are present
    missing = [c for c in FEATURE_COLS if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X     = df_clean[FEATURE_COLS].values.astype(np.float32)
    y_reg = df_clean["MENTHLTH"].values.astype(np.float32)
    y_clf = df_clean["high_risk"].values.astype(np.float32)

    return X, y_reg, y_clf


def split_and_scale(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_clf: np.ndarray,
) -> dict:
    """Stratified 70/15/15 split + median imputation + StandardScaler.

    Imputer and scaler are fit ONLY on training data to prevent data leakage.

    Args:
        X:     Feature matrix (n_samples, n_features).
        y_reg: Regression target.
        y_clf: Classification target used for stratification.

    Returns:
        Dictionary with keys:
            X_train, X_val, X_test           — scaled, imputed float32 arrays
            y_reg_train, y_reg_val, y_reg_test
            y_clf_train, y_clf_val, y_clf_test
            scaler   — fitted StandardScaler
            imputer  — fitted SimpleImputer
    """
    # ── Step 1: 70% train / 30% temp ─────────────────────────────────────────
    X_train, X_temp, y_reg_train, y_reg_temp, y_clf_train, y_clf_temp = (
        train_test_split(
            X, y_reg, y_clf,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=y_clf,
        )
    )

    # ── Step 2: split temp → 15% val / 15% test ───────────────────────────────
    X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = (
        train_test_split(
            X_temp, y_reg_temp, y_clf_temp,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=y_clf_temp,
        )
    )

    # ── Step 3: median imputation (fit on train only) ─────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    # ── Step 4: StandardScaler (fit on train only) ────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "X_val":   X_val.astype(np.float32),
        "X_test":  X_test.astype(np.float32),
        "y_reg_train": y_reg_train,
        "y_reg_val":   y_reg_val,
        "y_reg_test":  y_reg_test,
        "y_clf_train": y_clf_train,
        "y_clf_val":   y_clf_val,
        "y_clf_test":  y_clf_test,
        "scaler":  scaler,
        "imputer": imputer,
    }


def save_processed(splits: dict, out_dir: str | Path) -> None:
    """Save all arrays and sklearn objects produced by split_and_scale().

    Args:
        splits:  Return value of split_and_scale().
        out_dir: Directory to write into (created if absent).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    array_keys = [k for k in splits if k not in ("scaler", "imputer")]
    for key in array_keys:
        np.save(out_dir / f"{key}.npy", splits[key])

    joblib.dump(splits["scaler"],  out_dir / "scaler.pkl")
    joblib.dump(splits["imputer"], out_dir / "imputer.pkl")

    (out_dir / "feature_cols.json").write_text(
        json.dumps(FEATURE_COLS, indent=2)
    )


def load_processed(out_dir: str | Path) -> dict:
    """Load the artifacts saved by save_processed() back into a dict.

    Args:
        out_dir: Directory written by save_processed().

    Returns:
        Same dictionary structure as split_and_scale().
    """
    out_dir = Path(out_dir)

    array_names = [
        "X_train", "X_val", "X_test",
        "y_reg_train", "y_reg_val", "y_reg_test",
        "y_clf_train", "y_clf_val", "y_clf_test",
    ]

    data = {name: np.load(out_dir / f"{name}.npy") for name in array_names}
    data["scaler"]  = joblib.load(out_dir / "scaler.pkl")
    data["imputer"] = joblib.load(out_dir / "imputer.pkl")
    return data
