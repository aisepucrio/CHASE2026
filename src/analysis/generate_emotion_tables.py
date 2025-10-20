#!/usr/bin/env python3
"""
Generate LaTeX tables showing per-emotion metrics (Precision, Recall, F1).

Creates tables with:
- Rows: Models (Classical ML + Sequential)
- Columns: Emotions (7 total: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise)
- Sub-columns: Embedding types (HuBERT-base, Wav2Vec2-large, Wav2Vec2-XLSR-PT)
- Sub-sub-columns: Metrics (Pr, Re, F1)

Due to width constraints, the table is split into multiple row groups:
- Group 1: First 2 emotions (Anger, Disgust)
- Group 2: Next 2 emotions (Fear, Happiness)
- Group 3: Next 2 emotions (Neutral, Sadness)
- Group 4: Last emotion (Surprise, with empty columns for visual consistency)

Finally, an average row is added showing aggregated metrics across all models.

Usage:
    # Generate table for interno dataset
    python generate_emotion_tables.py --dataset interno

    # Generate table for externo dataset
    python generate_emotion_tables.py --dataset externo

    # Generate table for all datasets (non-transfer experiments)
    python generate_emotion_tables.py

Output:
    - results/emotion_table_{dataset}.tex (or emotion_table_all.tex)
    - Table consists of 3 parts (one for each emotion group)
    - Each part is a separate LaTeX table* environment
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def calculate_per_class_metrics(confusion_matrix: List[List[int]], class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 for each class from confusion matrix.

    Args:
        confusion_matrix: Confusion matrix as list of lists (note: may be 6x6 missing first row/col)
        class_names: List of class names (7 emotions)

    Returns:
        Dictionary mapping class_name -> {'precision': float, 'recall': float, 'f1': float}
    """
    cm = np.array(confusion_matrix)
    n_classes_in_matrix = cm.shape[0]
    n_classes_expected = len(class_names)

    # Handle case where confusion matrix is smaller (e.g., 6x6 instead of 7x7)
    # This happens when first class (Anger) has no samples
    if n_classes_in_matrix < n_classes_expected:
        # Pad with zeros at the beginning (assuming missing class is first)
        padded_cm = np.zeros((n_classes_expected, n_classes_expected), dtype=int)
        padded_cm[1:, 1:] = cm
        cm = padded_cm

    metrics = {}

    for i, class_name in enumerate(class_names):
        # True positives: diagonal element
        tp = cm[i, i]

        # False positives: sum of column excluding diagonal
        fp = cm[:, i].sum() - tp

        # False negatives: sum of row excluding diagonal
        fn = cm[i, :].sum() - tp

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return metrics


def parse_experiment_name(json_data: dict, filename: str) -> str:
    """Extract experiment/model name from JSON data."""
    if 'model_type' in json_data:
        model_type = json_data['model_type']
        model_name_map = {
            'LightGBM': 'LightGBM',
            'XGBoost': 'XGBoost',
            'RandomForest': 'Random Forest',
            'ExtraTrees': 'Extra Trees',
            'GradientBoosting': 'Gradient Boosting',
            'LogisticRegression': 'Logistic Regression',
            'Extra Trees': 'Extra Trees',
            'Gradient Boosting': 'Gradient Boosting',
            'Random Forest': 'Random Forest'
        }

        if model_type in model_name_map:
            return model_name_map[model_type]
        elif model_type == 'SVM':
            kernel = json_data.get('hyperparameters', {}).get('kernel', 'unknown')
            if kernel == 'linear':
                return 'SVM (Linear)'
            elif kernel == 'rbf':
                return 'SVM (RBF)'
            return f'SVM ({kernel.capitalize()})'
        else:
            return model_type

    # RNN model
    hyperparams = json_data.get('hyperparameters', {})
    rnn_type = hyperparams.get('rnn_type', '').upper()
    strategy = hyperparams.get('strategy', '')

    if strategy == 'full_context':
        return f'Bi-{rnn_type}'
    else:
        return rnn_type


def extract_embedding_type(json_data: dict, filename: str) -> str:
    """Extract embedding type from JSON data."""
    # For RNN models
    if 'hyperparameters' in json_data and 'model_name' in json_data['hyperparameters']:
        model_name = json_data['hyperparameters']['model_name']
        if 'hubert' in model_name:
            return 'HuBERT-base'
        elif 'wav2vec2_large_960h_lv60_self' in model_name:
            return 'Wav2Vec2-large'
        elif 'wav2vec2_large_xlsr_53_portuguese' in model_name:
            return 'Wav2Vec2-XLSR-PT'

    # For classic ML models
    if 'data_info' in json_data and 'csv_path' in json_data['data_info']:
        csv_path = json_data['data_info']['csv_path']
        if 'hubert' in csv_path:
            return 'HuBERT-base'
        elif 'wav2vec2_large_960h_lv60_self' in csv_path:
            return 'Wav2Vec2-large'
        elif 'wav2vec2_large_xlsr_53_portuguese' in csv_path:
            return 'Wav2Vec2-XLSR-PT'

    return 'Unknown'


def extract_dataset_info(json_data: dict) -> Tuple[str, str]:
    """Extract train and test dataset information from JSON data."""
    # For RNN models
    if 'hyperparameters' in json_data:
        hyperparams = json_data['hyperparameters']
        train_dataset = hyperparams.get('train_dataset', 'interno')
        test_dataset = hyperparams.get('test_dataset', 'interno')
        return train_dataset, test_dataset

    # For classic ML models
    if 'data_info' in json_data:
        data_info = json_data['data_info']
        train_dataset = data_info.get('train_dataset', 'interno')
        test_dataset = data_info.get('test_dataset', 'interno')
        return train_dataset, test_dataset

    return 'interno', 'interno'


def format_metric(value: float) -> str:
    """Format metric with 2 decimal places."""
    if pd.isna(value) or value is None:
        return '-'
    return f'{value:.2f}'


def generate_emotion_table_latex(df: pd.DataFrame, emotions: List[str], caption: str, label: str) -> str:
    """
    Generate LaTeX table with emotions as columns and embeddings/metrics as sub-columns.

    Table structure:
    - Main columns: Emotions (up to 3 per group)
    - Sub-columns: Embedding types (HuBERT, Wav2Vec2-large, Wav2Vec2-XLSR-PT)
    - Sub-sub-columns: Metrics (Pr, Re, F1)
    """
    lines = []

    # Split emotions into groups of 2
    emotion_groups = [emotions[i:i+2] for i in range(0, len(emotions), 2)]

    for group_idx, emotion_group in enumerate(emotion_groups):
        # Start table
        lines.append("\\begin{table*}[tbp]")
        lines.append("\\centering")

        # Caption and label (add part number if multiple groups)
        if len(emotion_groups) > 1:
            lines.append(f"\\caption{{{caption} (Part {group_idx + 1}/{len(emotion_groups)})}}")
            lines.append(f"\\label{{{label}_part{group_idx + 1}}}")
        else:
            lines.append(f"\\caption{{{caption}}}")
            lines.append(f"\\label{{{label}}}")

        # Calculate number of columns
        # Always use 2 emotions worth of columns for consistent table width
        # 1 (model name) + 2 emotions * 3 embeddings * 3 metrics = 1 + 18 = 19 columns
        n_emotions_in_group = len(emotion_group)
        n_data_cols = 18  # Always 2 emotions * 3 embeddings * 3 metrics
        col_spec = 'l' + 'c' * n_data_cols

        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row 1: Emotion names spanning 9 columns each (3 embeddings * 3 metrics)
        header1 = "\\textbf{Model}"
        for emotion in emotion_group:
            header1 += f" & \\multicolumn{{9}}{{c}}{{\\textbf{{{emotion}}}}}"
        # Pad with empty columns if last group has fewer than 2 emotions
        if n_emotions_in_group < 2:
            for _ in range(2 - n_emotions_in_group):
                header1 += " & \\multicolumn{9}{c}{}"
        lines.append(header1 + " \\\\")

        # Header row 2: Embedding names spanning 3 columns each (3 metrics)
        header2 = ""
        for emotion in emotion_group:
            header2 += " & \\multicolumn{3}{c}{\\textbf{HuBERT}} & \\multicolumn{3}{c}{\\textbf{W2V2-L}} & \\multicolumn{3}{c}{\\textbf{W2V2-PT}}"
        # Pad with empty columns if needed
        if n_emotions_in_group < 2:
            for _ in range(2 - n_emotions_in_group):
                header2 += " & \\multicolumn{3}{c}{} & \\multicolumn{3}{c}{} & \\multicolumn{3}{c}{}"
        lines.append(header2 + " \\\\")

        # Header row 3: Metrics (Pr, Re, F1) repeated for each embedding
        header3 = ""
        n_embedding_repetitions = n_emotions_in_group if n_emotions_in_group == 2 else 2  # Pad to 2
        for _ in range(n_embedding_repetitions):
            for _ in range(3):  # 3 embeddings per emotion
                header3 += " & \\textbf{Pr} & \\textbf{Re} & \\textbf{F1}"
        lines.append(header3 + " \\\\")
        lines.append("\\midrule")

        # Define model groups
        classic_ml_models = [
            'Extra Trees', 'Gradient Boosting', 'LightGBM',
            'Logistic Regression', 'Random Forest',
            'SVM (Linear)', 'SVM (RBF)', 'XGBoost'
        ]
        sequential_models = ['Bi-GRU', 'Bi-LSTM', 'GRU', 'LSTM']
        embedding_order = ['HuBERT-base', 'Wav2Vec2-large', 'Wav2Vec2-XLSR-PT']

        # Get unique models from dataframe
        all_models = df['Model'].unique()
        classic_in_df = [m for m in classic_ml_models if m in all_models]
        sequential_in_df = [m for m in sequential_models if m in all_models]

        # Add Classical ML Models section
        if classic_in_df:
            n_cols_with_model = 1 + (2 * 9)  # 1 model col + 2 emotion groups * 9 cols each
            lines.append(f"\\multicolumn{{{n_cols_with_model}}}{{c}}{{\\textbf{{Classical ML Models}}}} \\\\")
            lines.append("\\midrule")

            for model in classic_in_df:
                row_data = [model]

                # For each emotion in this group
                for emotion in emotion_group:
                    # For each embedding
                    for embedding in embedding_order:
                        # Find the metrics for this model/embedding/emotion
                        mask = (df['Model'] == model) & (df['Embedding'] == embedding) & (df['Emotion'] == emotion)
                        model_data = df[mask]

                        if not model_data.empty:
                            row = model_data.iloc[0]
                            row_data.extend([
                                row['Precision'],
                                row['Recall'],
                                row['F1']
                            ])
                        else:
                            row_data.extend(['-', '-', '-'])

                # Pad if fewer than 2 emotions
                if n_emotions_in_group < 2:
                    for _ in range((2 - n_emotions_in_group) * 9):
                        row_data.append('-')

                lines.append(' & '.join(row_data) + ' \\\\')

        # Add Sequential Models section
        if sequential_in_df:
            lines.append("\\midrule")
            n_cols_with_model = 1 + (2 * 9)  # 1 model col + 2 emotion groups * 9 cols each
            lines.append(f"\\multicolumn{{{n_cols_with_model}}}{{c}}{{\\textbf{{Sequential Models}}}} \\\\")
            lines.append("\\midrule")

            for model in sequential_in_df:
                row_data = [model]

                # For each emotion in this group
                for emotion in emotion_group:
                    # For each embedding
                    for embedding in embedding_order:
                        # Find the metrics for this model/embedding/emotion
                        mask = (df['Model'] == model) & (df['Embedding'] == embedding) & (df['Emotion'] == emotion)
                        model_data = df[mask]

                        if not model_data.empty:
                            row = model_data.iloc[0]
                            row_data.extend([
                                row['Precision'],
                                row['Recall'],
                                row['F1']
                            ])
                        else:
                            row_data.extend(['-', '-', '-'])

                # Pad if fewer than 2 emotions
                if n_emotions_in_group < 2:
                    for _ in range((2 - n_emotions_in_group) * 9):
                        row_data.append('-')

                lines.append(' & '.join(row_data) + ' \\\\')

        # Add average row (aggregated across all models)
        lines.append("\\midrule")
        n_cols_with_model = 1 + (2 * 9)
        lines.append(f"\\multicolumn{{{n_cols_with_model}}}{{c}}{{\\textbf{{All Models}}}} \\\\")
        lines.append("\\midrule")

        avg_row = ['Average']
        for emotion in emotion_group:
            for embedding in embedding_order:
                # Calculate average across all models for this emotion/embedding
                mask = (df['Embedding'] == embedding) & (df['Emotion'] == emotion)
                emotion_data = df[mask]

                if not emotion_data.empty:
                    # Simple mean across all models
                    avg_precision = emotion_data['Precision'].apply(lambda x: float(x) if x != '-' else np.nan).mean()
                    avg_recall = emotion_data['Recall'].apply(lambda x: float(x) if x != '-' else np.nan).mean()
                    avg_f1 = emotion_data['F1'].apply(lambda x: float(x) if x != '-' else np.nan).mean()

                    avg_row.extend([
                        format_metric(avg_precision),
                        format_metric(avg_recall),
                        format_metric(avg_f1)
                    ])
                else:
                    avg_row.extend(['-', '-', '-'])

        # Pad if fewer than 2 emotions
        if n_emotions_in_group < 2:
            for _ in range((2 - n_emotions_in_group) * 9):
                avg_row.append('-')

        lines.append(' & '.join(avg_row) + ' \\\\')

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")

        # Add spacing between table groups
        if group_idx < len(emotion_groups) - 1:
            lines.append("\n")

    return '\n'.join(lines)


def main(dataset_filter: str = None):
    """
    Generate emotion-wise tables from experiment results.

    Args:
        dataset_filter: Filter experiments by dataset ('interno', 'externo', or None for all)
    """
    results_dir = Path('results/static')
    output_dir = Path('results')
    json_files = sorted(results_dir.glob('experiment_*.json'))

    if not json_files:
        print(f"No experiment JSON files found in {results_dir}/")
        return

    print(f"Processing {len(json_files)} experiment files...")

    # Collect data: list of dicts with model, embedding, emotion, and metrics
    data_rows = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Extract dataset info and filter if needed
        train_dataset, test_dataset = extract_dataset_info(json_data)

        # Skip if filtering by dataset and this experiment doesn't match
        if dataset_filter:
            if train_dataset != dataset_filter or test_dataset != dataset_filter:
                continue

        # Skip transfer learning experiments (train != test)
        if train_dataset != test_dataset:
            continue

        # Extract confusion matrix
        if 'confusion_matrix' not in json_data:
            print(f"Warning: No confusion matrix in {json_file.name}")
            continue

        cm_data = json_data['confusion_matrix']
        confusion_matrix = cm_data.get('matrix', [])
        class_names = cm_data.get('class_names', [])

        if not confusion_matrix or not class_names:
            print(f"Warning: Invalid confusion matrix in {json_file.name}")
            continue

        # Calculate per-class metrics
        per_class_metrics = calculate_per_class_metrics(confusion_matrix, class_names)

        # Extract model and embedding info
        model_name = parse_experiment_name(json_data, json_file.name)
        embedding_type = extract_embedding_type(json_data, json_file.name)

        # Create a row for each emotion
        for emotion, metrics in per_class_metrics.items():
            row = {
                'Model': model_name,
                'Embedding': embedding_type,
                'Emotion': emotion,
                'Precision': format_metric(metrics['precision']),
                'Recall': format_metric(metrics['recall']),
                'F1': format_metric(metrics['f1'])
            }
            data_rows.append(row)

    if not data_rows:
        print("No matching experiments found!")
        return

    print(f"Found {len(data_rows)} emotion-level results")

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Define emotion order
    emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    # Generate caption and label
    if dataset_filter:
        caption = f"Per-emotion Precision, Recall, and F1 scores for {dataset_filter.capitalize()} dataset"
        label = f"tab:emotion_metrics_{dataset_filter}"
        output_suffix = f"_{dataset_filter}"
    else:
        caption = "Per-emotion Precision, Recall, and F1 scores for all experiments"
        label = "tab:emotion_metrics_all"
        output_suffix = "_all"

    # Generate LaTeX table
    latex_table = generate_emotion_table_latex(df, emotions, caption, label)
    output_tex = output_dir / f'emotion_table{output_suffix}.tex'

    with open(output_tex, 'w') as f:
        f.write(latex_table)

    print(f"\n✓ Emotion table saved to: {output_tex}")

    # Print summary
    print("\nTable summary:")
    print(f"  Dataset filter: {dataset_filter if dataset_filter else 'all'}")
    print(f"  Total emotion results: {len(data_rows)}")
    print(f"  Unique models: {len(df['Model'].unique())}")
    print(f"  Unique embeddings: {len(df['Embedding'].unique())}")
    print(f"  Emotions: {len(emotions)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate LaTeX tables with per-emotion metrics from experiment JSON files.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['interno', 'externo'],
        help='Filter experiments by dataset (interno or externo). If not specified, includes all non-transfer experiments.'
    )

    args = parser.parse_args()
    main(dataset_filter=args.dataset)
