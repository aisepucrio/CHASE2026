#!/usr/bin/env python3
"""
Generate LaTeX tables for transfer learning experiments.

Creates 2 tables showing test set F1 scores (Weighted and Macro):
1. Externo → Interno (Train on Externo, Test on Interno)
2. Interno → Externo (Train on Interno, Test on Externo)

Each table shows both Weighted F1 and Macro F1 scores for:
- Classical ML Models (Extra Trees, Gradient Boosting, etc.)
- Sequential Models (LSTM, GRU, Bi-LSTM, Bi-GRU)
- Three embedding types (HuBERT-base, Wav2Vec2-large, Wav2Vec2-XLSR-PT)
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np


def extract_metrics(json_data, average_type='weighted'):
    """Extract F1 scores from test results (without std since all std are 0)."""
    # Extract test results
    if 'test_results' not in json_data:
        return None

    test_metrics = json_data['test_results']

    # Extract F1 score based on average_type
    f1_weighted = test_metrics.get('f1_weighted_mean', np.nan)
    f1_macro = test_metrics.get('f1_macro_mean', np.nan)

    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }


def format_metric(value):
    """Format metric with 2 decimal places."""
    if pd.isna(value):
        return '-'
    return f'{value:.2f}'


def parse_experiment_info(json_data, filename):
    """Extract experiment configuration from JSON."""
    # Check if it's a classic ML model or RNN model
    model_type = json_data.get('model_type', '')

    if model_type:
        # Classic ML model
        model_name = model_type
        train_dataset = json_data.get('data_info', {}).get('train_dataset', 'interno')
        test_dataset = json_data.get('data_info', {}).get('test_dataset', 'interno')

        # Determine embedding from csv_path
        csv_path = json_data.get('data_info', {}).get('csv_path', '')
        if 'hubert' in csv_path:
            embedding = 'HuBERT-base'
        elif 'wav2vec2_large_960h_lv60_self' in csv_path:
            embedding = 'Wav2Vec2-large'
        elif 'wav2vec2_large_xlsr_53_portuguese' in csv_path:
            embedding = 'Wav2Vec2-XLSR-PT'
        else:
            embedding = 'Unknown'
    else:
        # RNN model
        hyperparams = json_data.get('hyperparameters', {})

        rnn_type = hyperparams.get('rnn_type', '').upper()
        strategy = hyperparams.get('strategy', '')
        train_dataset = hyperparams.get('train_dataset', 'interno')
        test_dataset = hyperparams.get('test_dataset', 'interno')

        # Determine model name
        if strategy == 'full_context':
            model_name = f'Bi-{rnn_type}'
        else:
            model_name = rnn_type

        # Determine embedding type
        model_name_param = hyperparams.get('model_name', '')
        if 'hubert' in model_name_param:
            embedding = 'HuBERT-base'
        elif 'wav2vec2_large_960h_lv60_self' in model_name_param:
            embedding = 'Wav2Vec2-large'
        elif 'wav2vec2_large_xlsr_53_portuguese' in model_name_param:
            embedding = 'Wav2Vec2-XLSR-PT'
        else:
            embedding = 'Unknown'

    return {
        'model_name': model_name,
        'embedding': embedding,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset
    }


def generate_latex_table(df, caption, label, average_type):
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


def main():
    # Look for transfer learning results in a dedicated folder
    transfer_results_dir = Path('results/transfer_learning')

    # If transfer_learning folder doesn't exist, look in main results folder
    if not transfer_results_dir.exists():
        print(f"Transfer learning results folder not found: {transfer_results_dir}")
        print("Looking in main results/ folder...")
        results_dir = Path('results')
    else:
        results_dir = transfer_results_dir

    json_files = sorted(results_dir.glob('experiment_*.json'))

    if not json_files:
        print(f"No experiment JSON files found in {results_dir}/")
        return

    print(f"Reading experiment files from: {results_dir}")
    print(f"Found {len(json_files)} JSON files")

    # Dictionary to store data by configuration
    # Key: (train_dataset, test_dataset, model_name, embedding)
    data_dict = {}

    print("Processing experiment files...")
    for json_file in json_files:
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Skip if no test results
        if 'test_results' not in json_data:
            continue

        # Extract experiment info
        info = parse_experiment_info(json_data, json_file.name)

        # Only process transfer learning experiments
        if info['train_dataset'] == info['test_dataset']:
            continue

        # Extract metrics from test results
        metrics = extract_metrics(json_data)

        if metrics is None:
            continue

        key = (info['train_dataset'], info['test_dataset'], info['model_name'], info['embedding'])
        data_dict[key] = metrics

    if not data_dict:
        print("No transfer learning experiments found!")
        return

    print(f"Found {len(data_dict)} transfer learning experiments")

    # Generate tables for each configuration
    configurations = [
        ('externo', 'interno', 'externo_to_interno'),
        ('interno', 'externo', 'interno_to_externo')
    ]

    output_dir = results_dir / 'transfer_learning_tables'
    output_dir.mkdir(exist_ok=True)

    all_tables = []

    for train_ds, test_ds, config_name in configurations:
        # Filter data for this configuration
        rows = []
        for (tr_ds, te_ds, model, embedding), metrics in data_dict.items():
            if tr_ds == train_ds and te_ds == test_ds:
                row = {
                    'Model': model,
                    'Embedding': embedding,
                    'F1_Weighted': format_metric(metrics['f1_weighted']),
                    'F1_Macro': format_metric(metrics['f1_macro'])
                }
                rows.append(row)

        if not rows:
            print(f"No data for {config_name}")
            continue

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Generate caption and label
        transfer_label = f"Train on {train_ds.capitalize()}, Test on {test_ds.capitalize()}"
        caption = f"Transfer learning results: {transfer_label} (Test set F1 scores)"
        label = f"tab:{config_name}_test_f1"

        # Generate LaTeX table
        latex_table = generate_latex_table(df, caption, label, None)
        all_tables.append(latex_table)

        # Save individual table
        output_file = output_dir / f"{config_name}_test_f1.tex"
        with open(output_file, 'w') as f:
            f.write(latex_table)

        print(f"✓ Generated: {output_file}")

    # Save all tables to a single file
    all_tables_file = output_dir / 'all_transfer_learning_tables.tex'
    with open(all_tables_file, 'w') as f:
        f.write('\n\n'.join(all_tables))

    print(f"\n✓ All tables saved to: {all_tables_file}")
    print(f"✓ Individual tables saved to: {output_dir}/")

    # Print summary
    print("\nGenerated tables:")
    print("  1. Externo → Interno (Test F1 scores)")
    print("  2. Interno → Externo (Test F1 scores)")


if __name__ == '__main__':
    main()
