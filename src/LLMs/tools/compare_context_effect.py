import os
import sys
import pandas as pd
from typing import Tuple


def load_metrics_csv(base_results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(base_results_dir, 'llm_models_metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}. Run generate_llm_metrics first.")
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    required = {'model', 'context', 'subset', 'f1_macro'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Normalize types
    df['subset'] = df['subset'].astype(str).str.lower()
    df['context'] = df['context'].astype(str).str.lower()
    df['model'] = df['model'].astype(str)
    df['f1_macro'] = pd.to_numeric(df['f1_macro'], errors='coerce')
    return df


def compute_context_delta(df: pd.DataFrame, subset: str) -> Tuple[pd.DataFrame, float, float, int]:
    """
    Returns:
      - table: DataFrame with columns [model, f1_context, f1_nocontext, delta, pct_change]
      - avg_delta: mean of deltas across matched models
      - avg_pct: mean of percent changes across matched models (only where baseline > 0)
      - n_models: number of matched models used
    """
    sub = subset.strip().lower()
    sub_df = df[df['subset'] == sub].copy()

    # Pivot to have one row per model with columns for context and nocontext
    pivot = sub_df.pivot_table(index='model', columns='context', values='f1_macro', aggfunc='max')

    # Keep only models that have both context and nocontext
    for col in ['context', 'nocontext']:
        if col not in pivot.columns:
            pivot[col] = float('nan')
    both = pivot.dropna(subset=['context', 'nocontext']).reset_index()
    both = both.rename(columns={'context': 'f1_context', 'nocontext': 'f1_nocontext'})

    if both.empty:
        return both.assign(delta=pd.NA, pct_change=pd.NA), 0.0, 0.0, 0

    both['delta'] = both['f1_context'] - both['f1_nocontext']
    # Percent change relative to no-context baseline
    def pct(row):
        base = row['f1_nocontext']
        if base and base != 0:
            return (row['delta'] / base) * 100.0
        return float('nan')
    both['pct_change'] = both.apply(pct, axis=1)

    avg_delta = float(both['delta'].mean())
    avg_pct = float(both['pct_change'].dropna().mean()) if both['pct_change'].notna().any() else 0.0
    n_models = int(len(both))

    # Sort by improvement descending
    both = both.sort_values(by='delta', ascending=False)
    return both[['model', 'f1_context', 'f1_nocontext', 'delta', 'pct_change']], avg_delta, avg_pct, n_models


def print_report(df: pd.DataFrame, subset: str) -> None:
    table, avg_delta, avg_pct, n = compute_context_delta(df, subset)
    print(f"\n=== Context effect for subset='{subset}' ===")
    if n == 0:
        print("No models with both context and nocontext found.")
        return
    print(f"Models matched: {n}")
    print(f"Average delta (context - nocontext) in F1_macro: {avg_delta:.2f}")
    if pd.notna(avg_pct):
        print(f"Average percent change relative to no-context: {avg_pct:.2f}%")

    # Pretty print per-model
    print("\nPer-model F1_macro (sorted by delta desc):")
    for _, row in table.iterrows():
        model = row['model']
        f1c = row['f1_context']
        f1n = row['f1_nocontext']
        delta = row['delta']
        pct = row['pct_change']
        pct_s = f"{pct:.2f}%" if pd.notna(pct) else "NA"
    print(f"- {model}: context={f1c:.2f} | nocontext={f1n:.2f} | delta={delta:+.2f} ({pct_s})")


def main():
    # Base results folder relative to this script
    base_results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    df = load_metrics_csv(base_results_dir)

    # Default: report for both subsets if available
    subsets = []
    if 'full' in df['subset'].unique():
        subsets.append('full')
    if 'test' in df['subset'].unique():
        subsets.append('test')
    if not subsets:
        print("No supported subsets found in CSV (expected 'full' and/or 'test').")
        sys.exit(0)

    for sub in subsets:
        print_report(df, sub)


if __name__ == '__main__':
    main()
