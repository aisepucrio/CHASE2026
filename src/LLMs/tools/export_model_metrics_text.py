import os
import sys
import pandas as pd
from typing import Dict, Any


def load_metrics(base_results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(base_results_dir, 'llm_models_metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}. Run generate_llm_metrics first.")
    df = pd.read_csv(csv_path)

    required = {
        'model', 'context', 'subset',
        'precision_macro', 'recall_macro', 'f1_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted'
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize fields for robust filtering
    df['model'] = df['model'].astype(str)
    df['context'] = df['context'].astype(str).str.lower()
    df['subset'] = df['subset'].astype(str).str.lower()
    return df


essential_fields = {
    'macro': ['precision_macro', 'recall_macro', 'f1_macro'],
    'weighted': ['precision_weighted', 'recall_weighted', 'f1_weighted']
}


def sel(df: pd.DataFrame, model: str, context: str, subset: str) -> Dict[str, Any]:
    row = df[(df['model'] == model) & (df['context'] == context) & (df['subset'] == subset)]
    if row.empty:
        return {}
    r = row.iloc[0].to_dict()
    return r


def fmt(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "NA"


def fmt_pct(v):
    try:
        return f"{float(v) * 100.0:.2f}%"
    except Exception:
        return "NA%"


def build_block(df: pd.DataFrame, model: str) -> str:
    parts = [model]
    # Full + context
    rc = sel(df, model, 'context', 'full')
    parts.append(f"precision macro full dataset com contexto: {fmt(rc.get('precision_macro'))}")
    parts.append(f"recall macro full dataset com contexto: {fmt(rc.get('recall_macro'))}")
    parts.append(f"f1 macro full dataset com contexto: {fmt(rc.get('f1_macro'))}")
    parts.append(f"coverage full dataset com contexto: {fmt_pct(rc.get('coverage_inclusive'))}")
    parts.append("-----")
    parts.append(f"precision weighted full dataset com contexto: {fmt(rc.get('precision_weighted'))}")
    parts.append(f"recall weighted full dataset com contexto: {fmt(rc.get('recall_weighted'))}")
    parts.append(f"f1 weighted full dataset com contexto: {fmt(rc.get('f1_weighted'))}")
    parts.append("-----")

    # Full + nocontext
    rn = sel(df, model, 'nocontext', 'full')
    parts.append(f"precision macro full dataset sem contexto: {fmt(rn.get('precision_macro'))}")
    parts.append(f"recall macro full dataset sem contexto: {fmt(rn.get('recall_macro'))}")
    parts.append(f"f1 macro full dataset sem contexto: {fmt(rn.get('f1_macro'))}")
    parts.append(f"coverage full dataset sem contexto: {fmt_pct(rn.get('coverage_inclusive'))}")
    parts.append("-----")
    parts.append(f"precision weighted full dataset sem contexto: {fmt(rn.get('precision_weighted'))}")
    parts.append(f"recall weighted full dataset sem contexto: {fmt(rn.get('recall_weighted'))}")
    parts.append(f"f1 weighted full dataset sem contexto: {fmt(rn.get('f1_weighted'))}")
    parts.append("-----")

    # Test + context
    rtc = sel(df, model, 'context', 'test')
    parts.append(f"precision macro test split com contexto: {fmt(rtc.get('precision_macro'))}")
    parts.append(f"recall macro test split com contexto: {fmt(rtc.get('recall_macro'))}")
    parts.append(f"f1 macro test split com contexto: {fmt(rtc.get('f1_macro'))}")
    parts.append(f"coverage test split com contexto: {fmt_pct(rtc.get('coverage_inclusive'))}")
    parts.append("-----")
    parts.append(f"precision weighted test split com contexto: {fmt(rtc.get('precision_weighted'))}")
    parts.append(f"recall weighted test split com contexto: {fmt(rtc.get('recall_weighted'))}")
    parts.append(f"f1 weighted test split com contexto: {fmt(rtc.get('f1_weighted'))}")
    parts.append("-----")

    # Test + nocontext
    rtn = sel(df, model, 'nocontext', 'test')
    parts.append(f"precision macro test split sem contexto: {fmt(rtn.get('precision_macro'))}")
    parts.append(f"recall macro test split sem contexto: {fmt(rtn.get('recall_macro'))}")
    parts.append(f"f1 macro test split sem contexto: {fmt(rtn.get('f1_macro'))}")
    parts.append(f"coverage test split sem contexto: {fmt_pct(rtn.get('coverage_inclusive'))}")
    parts.append("-----")
    parts.append(f"precision weighted test split sem contexto: {fmt(rtn.get('precision_weighted'))}")
    parts.append(f"recall weighted test split sem contexto: {fmt(rtn.get('recall_weighted'))}")
    parts.append(f"f1 weighted test split sem contexto: {fmt(rtn.get('f1_weighted'))}")

    return "\n".join(parts)


def main():
    base_results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    df = load_metrics(base_results_dir)

    report_path = os.path.join(base_results_dir, 'model_metrics_report.txt')
    models = sorted(df['model'].unique().tolist())

    with open(report_path, 'w', encoding='utf-8') as f:
        for i, model in enumerate(models):
            block = build_block(df, model)
            f.write(block)
            if i < len(models) - 1:
                f.write("\n\n\n")

    print(f"Saved: {report_path}")


if __name__ == '__main__':
    main()
