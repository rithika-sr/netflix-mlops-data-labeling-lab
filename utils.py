"""
utils.py — Helper functions for the Netflix Data Labeling Lab
Author: Rithika Sankar Rajeswari
Purpose: Shared utility functions used across all Netflix lab notebooks
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# ==========================================================
# 📂 File Operations
# ==========================================================

def load_netflix_data(filepath: str):
    """
    Safely load the Netflix dataset.
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    Returns
    -------
    pd.DataFrame or None
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ Successfully loaded: {filepath}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def save_csv(df: pd.DataFrame, filepath: str):
    """Save a DataFrame safely to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"💾 Saved file to: {filepath}")


def ensure_dirs(*dirs):
    """Ensure that directories exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("📁 Verified directories:", ", ".join(dirs))


# ==========================================================
# 🧹 Data Cleaning
# ==========================================================

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names (lowercase, underscores)."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def parse_duration(value: str):
    """
    Parse Netflix 'duration' column like '90 min' or '3 Seasons'.
    Returns (num, unit) as tuple.
    """
    if pd.isna(value):
        return np.nan, None
    try:
        parts = str(value).split()
        num = int(parts[0])
        unit = parts[1].lower() if len(parts) > 1 else None
        if unit.endswith("s"):
            unit = unit[:-1]
        if unit == "minute":
            unit = "min"
        return num, unit
    except Exception:
        return np.nan, None


# ==========================================================
# 🔢 Feature Engineering Helpers
# ==========================================================

def split_multi(cell: str, sep: str = ","):
    """Split a comma-separated string into list elements."""
    if pd.isna(cell) or cell == "":
        return []
    return [x.strip() for x in str(cell).split(sep) if x.strip()]


def count_multi(series: pd.Series, sep: str = ",") -> Counter:
    """Count frequency of multi-value column entries."""
    c = Counter()
    for cell in series.fillna(""):
        c.update(split_multi(cell, sep))
    return c


def top_n_counter(series: pd.Series, n: int = 20, sep: str = ",") -> pd.DataFrame:
    """Return top-n most frequent items in a multi-value column."""
    c = count_multi(series, sep)
    df = pd.DataFrame(c.most_common(n), columns=["item", "count"])
    return df


def add_genre_flags(df: pd.DataFrame, top_genres):
    """Add one-hot genre indicator columns for the given top genres."""
    df = df.copy()
    li = df["listed_in"].fillna("")
    for g in top_genres:
        col = f"genre_{g.lower().replace(' ', '_').replace('&', 'and')}"
        df[col] = li.str.contains(rf"\b{g}\b", case=False, na=False).astype(int)
    return df


# ==========================================================
# 📊 Visualization Helpers
# ==========================================================

def save_fig(filepath: str, tight: bool = True, dpi: int = 150):
    """Save current matplotlib figure with consistent formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(filepath, dpi=dpi)
    print(f"🖼️ Saved figure to: {filepath}")
