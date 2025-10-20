#!/usr/bin/env python3
"""
Generate a comprehensive results table from all experiment JSON files.

Creates tables showing test set F1 scores (Weighted and Macro) grouped by:
- Classical ML Models (Extra Trees, Gradient Boosting, etc.)
- Sequential Models (LSTM, GRU, Bi-LSTM, Bi-GRU)
- Three embedding types (HuBERT-base, Wav2Vec2-large, Wav2Vec2-XLSR-PT)

Supports filtering by dataset (interno/externo) to generate separate tables.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np


def parse_experiment_name(json_data, filename):
    """Extract experiment name from JSON data."""
    # Check if it's a classic ML model
    if 'model_type' in json_data:
        model_type = json_data['model_type']
        # Clean up model names to match the expected format
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
            # Check hyperparameters for kernel type
            kernel = json_data.get('hyperparameters', {}).get('kernel', 'unknown')
            if kernel == 'linear':
                return 'SVM (Linear)'
            elif kernel == 'rbf':
                return 'SVM (RBF)'
            return f'SVM ({kernel.capitalize()})'
        else:
            return model_type

    # Otherwise it's an RNN model
    hyperparams = json_data.get('hyperparameters', {})
    rnn_type = hyperparams.get('rnn_type', '').upper()
    strategy = hyperparams.get('strategy', '')

    if strategy == 'full_context':
        return f'Bi-{rnn_type}'
    else:
        return rnn_type


def extract_embedding_type(json_data, filename):
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


def extract_metrics(json_data):
    """Extract Precision, Recall, and F1 scores from test results.

    Args:
        json_data: Experiment JSON data

    Returns:
        Dictionary with precision, recall, and f1 for both weighted and macro
    """
    if 'test_results' not in json_data:
        return None

    test_metrics = json_data['test_results']

    # Extract all metrics for weighted and macro
    return {
        'precision_weighted': test_metrics.get('precision_weighted_mean', np.nan),
        'recall_weighted': test_metrics.get('recall_weighted_mean', np.nan),
        'f1_weighted': test_metrics.get('f1_weighted_mean', np.nan),
        'precision_macro': test_metrics.get('precision_macro_mean', np.nan),
        'recall_macro': test_metrics.get('recall_macro_mean', np.nan),
        'f1_macro': test_metrics.get('f1_macro_mean', np.nan)
    }


def format_metric(value):
    """Format metric with 2 decimal places."""
    if pd.isna(value):
        return '-'
    return f'{value:.2f}'


def extract_dataset_info(json_data):
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


def generate_comprehensive_latex_table(df, caption, label):
    """Generate comprehensive LaTeX table with P/R/F1 for weighted and macro metrics."""
    lines = []
    lines.append("\\begin{table*}[tbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lcccccccccccccccccc}")
    lines.append("\\toprule")

    # Header row 1: Embedding names spanning 6 columns each
    lines.append("\\textbf{Model} & \\multicolumn{6}{c}{\\textbf{HuBERT-base}} & \\multicolumn{6}{c}{\\textbf{Wav2Vec2-large}} & \\multicolumn{6}{c}{\\textbf{Wav2Vec2-XLSR-PT}} \\\\")

    # Header row 2: Weighted and Macro spanning 3 columns each
    lines.append(" & \\multicolumn{3}{c}{\\textbf{Weighted}} & \\multicolumn{3}{c}{\\textbf{Macro}} & \\multicolumn{3}{c}{\\textbf{Weighted}} & \\multicolumn{3}{c}{\\textbf{Macro}} & \\multicolumn{3}{c}{\\textbf{Weighted}} & \\multicolumn{3}{c}{\\textbf{Macro}} \\\\")

    # Header row 3: P, R, F1 for each
    header3 = ""
    for _ in range(3):  # 3 embeddings
        for _ in range(2):  # Weighted and Macro
            header3 += " & \\textbf{F1} & \\textbf{P} & \\textbf{R}"
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
        lines.append("\\multicolumn{19}{c}{\\textbf{Classical ML Models}} \\\\")
        lines.append("\\midrule")

        for model in classic_in_df:
            row_data = [model]

            # For each embedding
            for embedding in embedding_order:
                model_data = df[(df['Model'] == model) & (df['Embedding'] == embedding)]

                if not model_data.empty:
                    row = model_data.iloc[0]
                    # Weighted: F1, P, R
                    row_data.append(row['F1_Weighted'])
                    row_data.append(row['Precision_Weighted'])
                    row_data.append(row['Recall_Weighted'])
                    # Macro: F1, P, R
                    row_data.append(row['F1_Macro'])
                    row_data.append(row['Precision_Macro'])
                    row_data.append(row['Recall_Macro'])
                else:
                    row_data.extend(['-', '-', '-', '-', '-', '-'])

            lines.append(' & '.join(row_data) + ' \\\\')

    # Add Sequential Models section
    if sequential_in_df:
        lines.append("\\midrule")
        lines.append("\\multicolumn{19}{c}{\\textbf{Sequential Models}} \\\\")
        lines.append("\\midrule")

        for model in sequential_in_df:
            row_data = [model]

            # For each embedding
            for embedding in embedding_order:
                model_data = df[(df['Model'] == model) & (df['Embedding'] == embedding)]

                if not model_data.empty:
                    row = model_data.iloc[0]
                    # Weighted: F1, P, R
                    row_data.append(row['F1_Weighted'])
                    row_data.append(row['Precision_Weighted'])
                    row_data.append(row['Recall_Weighted'])
                    # Macro: F1, P, R
                    row_data.append(row['F1_Macro'])
                    row_data.append(row['Precision_Macro'])
                    row_data.append(row['Recall_Macro'])
                else:
                    row_data.extend(['-', '-', '-', '-', '-', '-'])

            lines.append(' & '.join(row_data) + ' \\\\')

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return '\n'.join(lines)


def generate_latex_table(df, caption, label):
    """Generate LaTeX table from DataFrame with embeddings as main columns."""
    lines = []
    lines.append("\\begin{table*}[tbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")

    # Header: Model name + 3 embedding columns with 2 F1 score subcolumns each
    lines.append("\\textbf{Model} & \\multicolumn{2}{c}{\\textbf{HuBERT-base}} & \\multicolumn{2}{c}{\\textbf{Wav2Vec2-large}} & \\multicolumn{2}{c}{\\textbf{Wav2Vec2-XLSR-PT}} \\\\")
    lines.append(" & \\textbf{F1 (Weighted)} & \\textbf{F1 (Macro)} & \\textbf{F1 (Weighted)} & \\textbf{F1 (Macro)} & \\textbf{F1 (Weighted)} & \\textbf{F1 (Macro)} \\\\")
    lines.append("\\midrule")

    # Define model groups
    classic_ml_models = [
        'Extra Trees', 'Gradient Boosting', 'LightGBM',
        'Logistic Regression', 'Random Forest',
        'SVM (Linear)', 'SVM (RBF)', 'XGBoost'
    ]

    sequential_models = ['Bi-GRU', 'Bi-LSTM', 'GRU', 'LSTM']

    # Define embedding order
    embedding_order = ['HuBERT-base', 'Wav2Vec2-large', 'Wav2Vec2-XLSR-PT']

    # Get unique models from dataframe
    all_models = df['Model'].unique()

    # Separate models into groups
    classic_in_df = [m for m in classic_ml_models if m in all_models]
    sequential_in_df = [m for m in sequential_models if m in all_models]

    # Add Classical ML Models section
    if classic_in_df:
        lines.append("\\multicolumn{7}{c}{\\textbf{Classical ML Models}} \\\\")
        lines.append("\\midrule")

        for model in classic_in_df:
            row_data = [model]

            # For each embedding, get the F1 scores
            for embedding in embedding_order:
                model_data = df[(df['Model'] == model) & (df['Embedding'] == embedding)]

                if not model_data.empty:
                    row = model_data.iloc[0]
                    row_data.append(row['F1_Weighted'])
                    row_data.append(row['F1_Macro'])
                else:
                    row_data.extend(['-', '-'])

            lines.append(' & '.join(row_data) + ' \\\\')

    # Add Sequential Models section
    if sequential_in_df:
        lines.append("\\midrule")
        lines.append("\\multicolumn{7}{c}{\\textbf{Sequential Models}} \\\\")
        lines.append("\\midrule")

        for model in sequential_in_df:
            row_data = [model]

            # For each embedding, get the F1 scores
            for embedding in embedding_order:
                model_data = df[(df['Model'] == model) & (df['Embedding'] == embedding)]

                if not model_data.empty:
                    row = model_data.iloc[0]
                    row_data.append(row['F1_Weighted'])
                    row_data.append(row['F1_Macro'])
                else:
                    row_data.extend(['-', '-'])

            lines.append(' & '.join(row_data) + ' \\\\')

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return '\n'.join(lines)


def main(dataset_filter=None):
    """Generate results tables.

    Args:
        dataset_filter: Filter experiments by dataset ('interno', 'externo', or None for all)
    """
    results_dir = Path('results/static')
    output_dir = Path('results')
    json_files = sorted(results_dir.glob('experiment_*.json'))

    if not json_files:
        print(f"No experiment JSON files found in {results_dir}/")
        return

    # Dictionary to store data: {(experiment_name, embedding_type): metrics}
    data_dict = {}

    print(f"Processing {len(json_files)} experiment files...")
    for json_file in json_files:
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Extract dataset info and filter if needed
        train_dataset, test_dataset = extract_dataset_info(json_data)

        # Skip if filtering by dataset and this experiment doesn't match
        if dataset_filter:
            # Only include experiments where train_dataset == test_dataset == filter
            if train_dataset != dataset_filter or test_dataset != dataset_filter:
                continue

        # Skip transfer learning experiments (train != test)
        if train_dataset != test_dataset:
            continue

        experiment_name = parse_experiment_name(json_data, json_file.name)
        embedding_type = extract_embedding_type(json_data, json_file.name)
        metrics = extract_metrics(json_data)

        if metrics is None:
            print(f"Warning: No test results found in {json_file.name}")
            continue

        # Store in dictionary with (experiment, embedding) as key
        key = (experiment_name, embedding_type)
        data_dict[key] = metrics

    if not data_dict:
        print("No matching experiments found!")
        return

    print(f"Found {len(data_dict)} experiments")

    # Build rows for DataFrame
    rows = []
    for (exp_name, emb_type), metrics in data_dict.items():
        row = {
            'Model': exp_name,
            'Embedding': emb_type,
            'F1_Weighted': format_metric(metrics['f1_weighted']),
            'F1_Macro': format_metric(metrics['f1_macro']),
            'Precision_Weighted': format_metric(metrics['precision_weighted']),
            'Precision_Macro': format_metric(metrics['precision_macro']),
            'Recall_Weighted': format_metric(metrics['recall_weighted']),
            'Recall_Macro': format_metric(metrics['recall_macro'])
        }
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Generate captions and labels
    if dataset_filter:
        caption_f1 = f"Test set F1 scores for {dataset_filter.capitalize()} dataset"
        label_f1 = f"tab:results_f1_{dataset_filter}"
        caption_comprehensive = f"Test set Precision, Recall, and F1 scores for {dataset_filter.capitalize()} dataset"
        label_comprehensive = f"tab:results_comprehensive_{dataset_filter}"
        output_suffix = f"_{dataset_filter}"
    else:
        caption_f1 = "Test set F1 scores for all experiments"
        label_f1 = "tab:results_f1_all"
        caption_comprehensive = "Test set Precision, Recall, and F1 scores for all experiments"
        label_comprehensive = "tab:results_comprehensive_all"
        output_suffix = "_all"

    # Generate F1-only LaTeX table
    latex_table_f1 = generate_latex_table(df, caption_f1, label_f1)
    output_tex_f1 = output_dir / f'results_table_test_f1{output_suffix}.tex'
    with open(output_tex_f1, 'w') as f:
        f.write(latex_table_f1)
    print(f"\n✓ F1 table saved to: {output_tex_f1}")

    # Generate comprehensive LaTeX table with P/R/F1
    latex_table_comprehensive = generate_comprehensive_latex_table(df, caption_comprehensive, label_comprehensive)
    output_tex_comprehensive = output_dir / f'results_table_comprehensive{output_suffix}.tex'
    with open(output_tex_comprehensive, 'w') as f:
        f.write(latex_table_comprehensive)
    print(f"✓ Comprehensive table saved to: {output_tex_comprehensive}")

    # Also print summary
    print("\nTable summary:")
    print(f"  Dataset filter: {dataset_filter if dataset_filter else 'all'}")
    print(f"  Total experiments: {len(data_dict)}")
    print(f"  Unique models: {len(df['Model'].unique())}")
    print(f"  Unique embeddings: {len(df['Embedding'].unique())}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate LaTeX results tables from experiment JSON files using test set F1 scores.'
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
