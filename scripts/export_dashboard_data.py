"""
scripts/export_dashboard_data.py
─────────────────────────────────
Export the test split from numpy format to a CSV that the web dashboard can load.

Run from project root:
    python scripts/export_dashboard_data.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT  = ROOT / "docs" / "data"

# ── Load ──────────────────────────────────────────────────────────────────────

scaler       = joblib.load(PROC / "scaler.pkl")
feature_cols = json.loads((PROC / "feature_cols.json").read_text())

X_orig  = scaler.inverse_transform(np.load(PROC / "X_test.npy"))
y_reg   = np.load(PROC / "y_reg_test.npy")
y_clf   = np.load(PROC / "y_clf_test.npy")

# ── Build DataFrame ───────────────────────────────────────────────────────────

READABLE = {
    "PHYSHLTH": "physical_health_days",
    "GENHLTH":  "general_health",
    "ADDEPEV3": "depression_diagnosis",
    "LSATISFY": "life_satisfaction",
    "EMTSUPRT": "emotional_support",
    "SDLONELY": "loneliness",
    "SDHBILLS": "bills_difficulty",
    "EXERANY2": "exercises",
    "SMOKE100": "ever_smoked",
    "SEXVAR":   "sex",
    "EMPLOY1":  "employment",
    "INCOME3":  "income",
    "EDUCA":    "education",
    "_AGEG5YR": "age_group",
    "_BMI5CAT": "bmi_category",
}

df = pd.DataFrame(X_orig, columns=feature_cols).rename(columns=READABLE)

# Round all feature columns to integers (inverse_transform can leave small float errors)
# menthlth_days is already integer-valued but keep as int too for clean CSV
for col in df.columns:
    df[col] = df[col].round().astype(int)

df["menthlth_days"] = y_reg.astype(int)
df["high_risk"]     = y_clf.astype(int)

# Clip physical_health_days and menthlth_days to valid range [0, 30]
df["physical_health_days"] = df["physical_health_days"].clip(0, 30)
df["menthlth_days"]        = df["menthlth_days"].clip(0, 30)

# ── Export ────────────────────────────────────────────────────────────────────

OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "brfss_test.csv"
df.to_csv(out_path, index=False)

print(f"Exported {len(df):,} rows × {len(df.columns)} columns")
print(f"Output:  {out_path}")
print(f"Size:    {out_path.stat().st_size / 1_000_000:.1f} MB")
print()
print("Columns:", list(df.columns))
print()
print("Sample value ranges:")
for col in df.columns:
    print(f"  {col:<25} {df[col].min()} – {df[col].max()}")
