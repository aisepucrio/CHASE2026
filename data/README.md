# Data Directory

This directory contains datasets and embeddings for the audio emotion classification system.

## Structure

```
data/
├── embeddings/          # Pre-computed audio embeddings
│   ├── interno/        # Internal dataset embeddings
│   └── externo/        # External dataset embeddings
└── raw/                # Raw audio files (not included in repo)
```

## Embeddings

Embeddings are generated using pre-trained transformer models:
- HuBERT-base (`facebook/hubert-base-ls960`)
- Wav2Vec2-large (`facebook/wav2vec2-large-960h-lv60-self`)
- Wav2Vec2-XLSR-PT (`facebook/wav2vec2-large-xlsr-53-portuguese`)

### Generating Embeddings

To generate embeddings from audio files:

```bash
python src/embedders/run_embedder.py \
    --model facebook/hubert-base-ls960 \
    --batch_size 4
```

See `docs/MULTI_MODEL_USAGE.md` for more details.

## Dataset Information

- **Interno**: Internal dataset for emotion classification
- **Externo**: External dataset for cross-dataset validation

Each dataset contains 7 emotion classes:
- Anger
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

## Note

Due to size constraints, CSV embedding files are not included in this repository.
To obtain the embeddings:
1. Generate them using the embedding scripts
2. Or download from [provide link when available]
