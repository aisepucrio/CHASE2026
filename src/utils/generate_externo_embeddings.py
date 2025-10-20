#!/usr/bin/env python3
"""
Generate embeddings for the externo dataset.

This script:
1. Reads audio_labels.csv from dataExternal/
2. Maps external emotions to internal emotions
3. Filters out samples with emotions not in the internal dataset
4. Generates embeddings using audio_embedder.py for each embedding model
5. Saves embeddings in the same format as interno dataset
"""

import os
import pandas as pd
from pathlib import Path
from src.embedders.audio_embedder import AudioEmbedder

# Emotion mapping from externo to interno
# Externo has: Angry, Disgust, Fear, Happy, Neutral, Pleasant_surprise, Pleasant_surprised, Sad
# Interno has: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
EMOTION_MAPPING = {
    'Angry': 'Anger',
    'Disgust': 'Disgust',
    'Fear': 'Fear',
    'Happy': 'Happiness',
    'Neutral': 'Neutral',
    'Sad': 'Sadness',
    'Pleasant_surprise': 'Surprise',
    'Pleasant_surprised': 'Surprise',
}

# Embedding models to process
EMBEDDING_MODELS = {
    'hubert_base_ls960': 'facebook/hubert-base-ls960',
    'wav2vec2_large_960h_lv60_self': 'facebook/wav2vec2-large-960h-lv60-self',
    'wav2vec2_large_xlsr_53_portuguese': 'facebook/wav2vec2-large-xlsr-53-portuguese'
}

def prepare_labels_csv():
    """
    Prepare the labels CSV for embedding generation.

    Returns:
        Path to the prepared CSV file
    """
    print("📋 Preparing labels CSV...")

    # Read the original labels file
    df = pd.read_csv('dataExternal/audio_labels.csv')

    print(f"  - Original samples: {len(df)}")
    print(f"  - Unique emotions: {sorted(df['Emotion'].unique())}")

    # Map emotions
    df['Emotion_Mapped'] = df['Emotion'].map(EMOTION_MAPPING)

    # Filter out unmapped emotions
    unmapped = df[df['Emotion_Mapped'].isna()]
    if len(unmapped) > 0:
        print(f"  - Warning: {len(unmapped)} samples with unmapped emotions will be removed")
        print(f"    Unmapped emotions: {unmapped['Emotion'].unique()}")

    df = df[df['Emotion_Mapped'].notna()].copy()

    # Convert file paths from Windows format to Unix format and make them absolute
    df['File Path'] = df['File Path'].str.replace('\\', '/')
    df['File Path'] = df['File Path'].apply(
        lambda x: str(Path('dataExternal') / x)
    )

    # Check if all files exist
    missing_files = []
    for file_path in df['File Path']:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"  - Warning: {len(missing_files)} files not found!")
        if len(missing_files) <= 10:
            for f in missing_files:
                print(f"    Missing: {f}")
        else:
            print(f"    (showing first 10)")
            for f in missing_files[:10]:
                print(f"    Missing: {f}")

        # Remove missing files
        df = df[df['File Path'].apply(os.path.exists)].copy()

    # Use the mapped emotion
    df['Emotion'] = df['Emotion_Mapped']
    df = df[['File Path', 'Emotion', 'Split']]

    # Save prepared CSV
    output_path = 'dataExternal/audio_labels_prepared.csv'
    df.to_csv(output_path, index=False)

    print(f"  ✓ Prepared {len(df)} samples")
    print(f"  - Train: {len(df[df['Split'] == 'train'])}")
    print(f"  - Test: {len(df[df['Split'] == 'test'])}")
    print(f"  - Emotions: {sorted(df['Emotion'].unique())}")
    print(f"  ✓ Saved to: {output_path}\n")

    return output_path

def generate_embeddings(labels_csv, model_name, model_path, batch_size=4):
    """
    Generate embeddings for a specific model.

    Args:
        labels_csv: Path to the labels CSV file
        model_name: Name of the model (for output filename)
        model_path: HuggingFace model path
        batch_size: Batch size for processing
    """
    print(f"🚀 Generating embeddings for {model_name}...")
    print(f"  - Model: {model_path}")
    print(f"  - Batch size: {batch_size}")

    # Load the prepared CSV
    df = pd.read_csv(labels_csv)
    print(f"  - Processing {len(df)} samples")

    # Initialize the embedder
    embedder = AudioEmbedder(model_name=model_path, device=None)

    # Generate embeddings
    df_embeddings = embedder.build_embeddings_dataframe(
        df,
        batch_size=batch_size,
        path_col="File Path"
    )

    # Save embeddings to externo folder
    dest_path = f'embeddings/externo/embeddings_{model_name}.csv'
    os.makedirs('embeddings/externo', exist_ok=True)

    df_embeddings.to_csv(dest_path, index=False)
    print(f"  ✓ Embeddings saved to: {dest_path}")
    print(f"  ✓ Shape: {df_embeddings.shape}")
    print()

def main():
    print("="*80)
    print("EXTERNO DATASET EMBEDDINGS GENERATION")
    print("="*80)
    print()

    # Step 1: Prepare labels CSV with emotion mapping
    labels_csv = prepare_labels_csv()

    # Step 2: Generate embeddings for each model
    print("="*80)
    print("GENERATING EMBEDDINGS")
    print("="*80)
    print()

    for model_name, model_path in EMBEDDING_MODELS.items():
        try:
            generate_embeddings(labels_csv, model_name, model_path)
        except Exception as e:
            print(f"Failed to generate embeddings for {model_name}: {e}")
            continue

    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print()
    print("Generated embeddings:")
    for model_name in EMBEDDING_MODELS.keys():
        dest_path = f'embeddings/externo/embeddings_{model_name}.csv'
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  ✓ {dest_path} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {dest_path} (not found)")

if __name__ == '__main__':
    main()
