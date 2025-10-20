import os
import json
import glob
import sys
from typing import Dict, Tuple, List, Optional

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Canonical emotions mapping (same as notebook)
EMOTION_MAPPING_PT_TO_EN = {
    'felicidade': 'Happiness',
    'tristeza': 'Sadness',
    'medo': 'Fear',
    'raiva': 'Anger',
    'surpresa': 'Surprise',
    'desgosto': 'Disgust',
    'neutro': 'Neutral',
}
CANONICAL_LABELS = set(EMOTION_MAPPING_PT_TO_EN.values())
LABELS_ORDER = sorted(CANONICAL_LABELS)


def load_ground_truth(gt_path: str) -> Dict[str, Dict[str, str]]:
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt: Dict[str, Dict[str, str]] = {}
    for output_name, segments in data.items():
        gt[output_name] = {}
        for seg_id, seg_data in segments.items():
            label = seg_data.get('principal_emocao_detectada')
            gt[output_name][seg_id] = label
    return gt


def normalize_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip().lower()

    if raw in EMOTION_MAPPING_PT_TO_EN:
        return EMOTION_MAPPING_PT_TO_EN[raw]

    alias = {
        'happy': 'Happiness', 'happiness': 'Happiness',
        'sad': 'Sadness', 'sadness': 'Sadness',
        'fear': 'Fear', 'afraid': 'Fear',
        'angry': 'Anger', 'anger': 'Anger',
        'surprise': 'Surprise', 'surprised': 'Surprise',
        'disgust': 'Disgust', 'disgusted': 'Disgust',
        'neutral': 'Neutral',
    }
    if raw in alias:
        return alias[raw]

    cap = raw.capitalize()
    if cap in CANONICAL_LABELS:
        return cap
    return None


def load_splits(csv_path: str) -> Dict[Tuple[str, str], str]:
    df = pd.read_csv(csv_path)
    mapping: Dict[Tuple[str, str], str] = {}
    for _, row in df.iterrows():
        file_path = str(row['File Path'])
        split = str(row['Split']).strip().lower()
        file_path = file_path.replace('/', '\\').replace('\\', os.sep)
        parts = file_path.split(os.sep)
        if len(parts) >= 2:
            output_name = parts[0]
            seg_with_ext = parts[1]
            seg = os.path.splitext(seg_with_ext)[0]
            mapping[(output_name, seg)] = split
    return mapping


def extract_pred_label(seg_obj: dict) -> Optional[str]:
    for key in (
        'principal_emocao_detectada', 'emoção', 'emocao', 'emotion', 'label', 'pred'
    ):
        if key in seg_obj and seg_obj[key]:
            return str(seg_obj[key])
    return None


def load_model_predictions(all_json_path: str) -> Dict[str, Dict[str, Optional[str]]]:
    with open(all_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    preds: Dict[str, Dict[str, Optional[str]]] = {}
    for output_name, segments in data.items():
        if not isinstance(segments, dict):
            continue
        preds[output_name] = {}
        for seg_id, seg_obj in segments.items():
            if not isinstance(seg_obj, dict):
                continue
            raw_label = extract_pred_label(seg_obj)
            preds[output_name][seg_id] = normalize_label(raw_label)
    return preds


def collect_pairs(gt: Dict[str, Dict[str, str]],
                  preds: Dict[str, Dict[str, Optional[str]]],
                  subset: Optional[str],
                  split_map: Dict[Tuple[str, str], str]) -> Tuple[List[str], List[str]]:
    y_true: List[str] = []
    y_pred: List[str] = []
    for output_name, segs_gt in gt.items():
        if output_name not in preds:
            continue
        segs_pred = preds[output_name]
        for seg_id, true_label in segs_gt.items():
            if subset == 'test' and split_map.get((output_name, seg_id), None) != 'test':
                continue
            if seg_id not in segs_pred:
                continue
            pred_label = segs_pred.get(seg_id)
            if pred_label is None:
                continue
            y_true.append(true_label)
            y_pred.append(pred_label)
    return y_true, y_pred


def per_emotion_metrics(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    if not y_true:
        # empty frame with standard columns
        return pd.DataFrame({
            'emotion': LABELS_ORDER,
            'precision': [0.0]*len(LABELS_ORDER),
            'recall': [0.0]*len(LABELS_ORDER),
            'f1': [0.0]*len(LABELS_ORDER),
            'support': [0]*len(LABELS_ORDER),
        })
    labels = LABELS_ORDER
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    df = pd.DataFrame({
        'emotion': labels,
        'precision': p,
        'recall': r,
        'f1': f1,
        'support': support,
    })
    return df


def main():
    # Base results dir relative to this script
    base_results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    gt_path = os.path.join(base_results_dir, 'resultado_manual.json')
    splits_csv = os.path.join(base_results_dir, 'data_splits.csv')

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found at {gt_path}")
    if not os.path.exists(splits_csv):
        raise FileNotFoundError(f"Splits CSV not found at {splits_csv}")

    gt = load_ground_truth(gt_path)
    split_map = load_splits(splits_csv)

    # Where to look for predictions
    specs = [
        ('resultados_com_contexto', 'context'),
        ('resultados_sem_contexto', 'nocontext'),
    ]

    rows = []  # base rows per emotion

    for subdir, ctx in specs:
        ctx_dir = os.path.join(base_results_dir, subdir)
        if not os.path.isdir(ctx_dir):
            continue
        for name in sorted(os.listdir(ctx_dir)):
            model_dir = os.path.join(ctx_dir, name)
            if not os.path.isdir(model_dir):
                continue
            json_candidates = glob.glob(os.path.join(model_dir, 'all_*.json'))
            if not json_candidates:
                continue
            all_json = json_candidates[0]

            preds = load_model_predictions(all_json)

            for subset in ('full', 'test'):
                subset_arg = None if subset == 'full' else 'test'
                y_true, y_pred = collect_pairs(gt, preds, subset_arg, split_map)
                df_em = per_emotion_metrics(y_true, y_pred)
                for _, r in df_em.iterrows():
                    rows.append({
                        'model': name,
                        'context': ctx,
                        'subset': subset,
                        'emotion': r['emotion'],
                        'precision': float(r['precision']),
                        'recall': float(r['recall']),
                        'f1': float(r['f1']),
                        'support': int(r['support']),
                    })

    if not rows:
        print('No per-emotion results found.')
        return

    df_base = pd.DataFrame(rows)

    # Save base per-emotion metrics
    out_dir = base_results_dir
    base_csv = os.path.join(out_dir, 'per_emotion_metrics.csv')
    df_base.to_csv(base_csv, index=False, encoding='utf-8')
    print(f"Saved: {base_csv}")

    # Build comparison: join context vs nocontext per model/subset/emotion
    piv_cols = ['model', 'subset', 'emotion']
    ctx_df = df_base[df_base['context'] == 'context'][piv_cols + ['precision', 'recall', 'f1']].copy()
    nctx_df = df_base[df_base['context'] == 'nocontext'][piv_cols + ['precision', 'recall', 'f1']].copy()

    ctx_df = ctx_df.rename(columns={
        'precision': 'precision_context',
        'recall': 'recall_context',
        'f1': 'f1_context',
    })
    nctx_df = nctx_df.rename(columns={
        'precision': 'precision_nocontext',
        'recall': 'recall_nocontext',
        'f1': 'f1_nocontext',
    })

    cmp_df = pd.merge(ctx_df, nctx_df, on=piv_cols, how='inner')
    if not cmp_df.empty:
        cmp_df['precision_delta'] = cmp_df['precision_context'] - cmp_df['precision_nocontext']
        cmp_df['recall_delta'] = cmp_df['recall_context'] - cmp_df['recall_nocontext']
        cmp_df['f1_delta'] = cmp_df['f1_context'] - cmp_df['f1_nocontext']

        # Optional percent changes (relative to no-context)
        def pct(a, b):
            if b and b != 0:
                return (a - b) / b * 100.0
            return float('nan')
        cmp_df['f1_pct_change'] = cmp_df.apply(lambda r: pct(r['f1_context'], r['f1_nocontext']), axis=1)
        cmp_df['precision_pct_change'] = cmp_df.apply(lambda r: pct(r['precision_context'], r['precision_nocontext']), axis=1)
        cmp_df['recall_pct_change'] = cmp_df.apply(lambda r: pct(r['recall_context'], r['recall_nocontext']), axis=1)

    cmp_csv = os.path.join(out_dir, 'per_emotion_metrics_comparison.csv')
    cmp_df.to_csv(cmp_csv, index=False, encoding='utf-8')
    print(f"Saved: {cmp_csv}")

    # Additionally, create a single wide CSV per model/emotion with
    # precision/recall/f1 for (full/test) x (context/nocontext)
    if not cmp_df.empty:
        # Pivot to wide format
        wide = cmp_df.copy()
        # Build a MultiIndex for columns: metric_context/nocontext by subset
        wide = wide.set_index(['model', 'emotion', 'subset'])
        keep_cols = [
            'precision_context', 'precision_nocontext',
            'recall_context', 'recall_nocontext',
            'f1_context', 'f1_nocontext',
        ]
        wide = wide[keep_cols]
        # Move subset to columns
        wide = wide.unstack('subset')
        # Flatten columns like ('precision_context','full') -> precision_full_context
        wide.columns = [f"{metric}_{subset}_{'context' if 'context' in metric else 'nocontext'}" for metric, subset in wide.columns]
        # Reorder columns in a friendly order
        desired = []
        for m in ('precision', 'recall', 'f1'):
            for subset in ('full', 'test'):
                for ctx in ('context', 'nocontext'):
                    desired.append(f"{m}_{subset}_{ctx}")
        # Some combinations may be missing if data absent; keep those that exist
        cols = [c for c in desired if c in wide.columns]
        other_cols = [c for c in wide.columns if c not in cols]
        wide = wide[cols + other_cols]
        wide = wide.reset_index()

        wide_csv = os.path.join(out_dir, 'per_emotion_metrics_wide.csv')
        wide.to_csv(wide_csv, index=False, encoding='utf-8')
        print(f"Saved: {wide_csv}")

    # Print quick readable summary (top 5 improvements by f1 per subset and emotion)
    for subset in ('full', 'test'):
        print(f"\n=== Top F1 improvements per emotion (subset={subset}) ===")
        sub = cmp_df[cmp_df['subset'] == subset]
        if sub.empty:
            print("No data")
            continue
        for emo in LABELS_ORDER:
            emo_df = sub[sub['emotion'] == emo]
            if emo_df.empty:
                continue
            top = emo_df.sort_values(by='f1_delta', ascending=False).head(5)
            if top.empty:
                continue
            print(f"\nEmotion: {emo}")
            for _, row in top.iterrows():
                print(
                    f"- {row['model']}: f1_ctx={row['f1_context']:.2f} | f1_nctx={row['f1_nocontext']:.2f} | "
                    f"delta={row['f1_delta']:+.2f} ({row['f1_pct_change']:.2f}%)"
                )

    # Optional: print a full per-emotion table for a specific model
    if '--print-model' in sys.argv:
        try:
            idx = sys.argv.index('--print-model')
            model_name = sys.argv[idx + 1]
        except Exception:
            model_name = None
        if model_name:
            print(f"\n=== Per-emotion metrics for model: {model_name} ===")
            mdf = cmp_df[cmp_df['model'] == model_name]
            if mdf.empty:
                print("No data for this model (ensure it exists in both context and nocontext).")
            else:
                for subset in ('full', 'test'):
                    ms = mdf[mdf['subset'] == subset]
                    if ms.empty:
                        continue
                    print(f"\nSubset: {subset}")
                    # Ensure consistent emotion order
                    ms = ms.copy()
                    ms['emotion'] = pd.Categorical(ms['emotion'], categories=LABELS_ORDER, ordered=True)
                    ms = ms.sort_values('emotion')
                    for _, row in ms.iterrows():
                        print(
                            f"{row['emotion']}: "
                            f"P/R/F1 ctx={row['precision_context']:.2f}/{row['recall_context']:.2f}/{row['f1_context']:.2f} | "
                            f"nctx={row['precision_nocontext']:.2f}/{row['recall_nocontext']:.2f}/{row['f1_nocontext']:.2f} | "
                            f"ΔF1={row['f1_delta']:+.2f} ({row['f1_pct_change']:.2f}%)"
                        )


if __name__ == '__main__':
    main()
