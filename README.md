# Emotion Recognition in Agile Software Meetings

This repository accompanies the paper **"Emotion Recognition in Agile Software Meetings: A Comparative Study of ML, DL, and Text-based LLM Approaches"**. It provides complementary materials and scripts to evaluate Machine Learning (ML), Deep Learning (DL), and text-based Large Language Models (LLMs) for emotion classification in audio segments from real agile software development meetings.

---

## 🧩 Overview

The project implements a complete pipeline for **audio emotion recognition**.

### 🎯 Evaluated Approaches

1. **Classical Machine Learning:** Random Forest, SVM, XGBoost, LightGBM, Extra Trees, Gradient Boosting, Logistic Regression.
2. **Sequential Deep Learning:** LSTM and GRU (unidirectional and bidirectional).
3. **Large Language Models (LLMs):** Text-based models with and without conversation context.

### 🎵 Audio Embeddings

* **HuBERT-base-ls960** (768d)
* **Wav2Vec2-large-960h** (1024d)
* **Wav2Vec2-XLSR-Portuguese** (1024d, multilingual)

### 😊 Target Emotions

Seven emotion classes: *Happiness, Sadness, Fear, Anger, Surprise, Disgust, Neutral*

---

## 🗂️ Project Structure

```
├── data/                  # Audio data and metadata
│   ├── internal/          # Internal dataset (PUC-Rio meetings)
│   ├── external/          # OAF/YAF datasets
│   └── transcriptions/    # Text transcriptions with/without context
├── src/                   # Main source code
│   ├── data/              # Data loaders and splits
│   ├── embedders/         # Audio embedding extraction
│   ├── models/            # Deep learning models (LSTM/GRU)
│   ├── training/          # Training scripts for ML/DL
│   ├── LLMs/              # Text-based LLM evaluation tools and results
│   ├── analysis/          # Metrics and LaTeX report generation
│   └── scripts/           # Experiment automation
└── results/               # Output results (ML, DL, LLM)
```

---

## ⚙️ Installation

### Requirements

```bash
pip install torch pytorch-lightning transformers librosa
pip install scikit-learn lightgbm xgboost optuna
pip install matplotlib seaborn pandas numpy tqdm
```

or install the specific requirements for LLM analysis:

```bash
pip install -r src/LLMs/requirements.txt
```

### Environment Setup

```bash
git clone <repository-url>
cd CHASE2026
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

---

## 🚀 Complete Pipeline

### 1. 📊 Embedding Extraction

```bash
python src/embedders/run_embedder.py \
  --dataset data/internal/data_splits.csv \
  --model facebook/hubert-base-ls960 \
  --batch_size 4
```

Features:

* Resampling to 16kHz.
* Mean pooling of final-layer features.
* Saves CSVs with embeddings and metadata.

---

### 2. 🤖 Classical ML Training

```bash
python src/training/train_classic_ml.py \
  --algorithm lightgbm \
  --dataset internal \
  --optuna_trials 50
```

Features:

* Stratified K-Fold cross-validation.
* Optuna hyperparameter search.
* Macro/micro/weighted metrics.

---

### 3. 🧠 Deep Learning (RNNs)

```bash
python src/training/train_rnn.py \
  --rnn_type lstm \
  --model_name hubert_base_ls960 \
  --epochs 20 --batch_size 8
```

Model Example:

```python
EmotionClassifierRNN(
  input_size=768,
  num_classes=7,
  rnn_type='lstm',
  hidden_size=256,
  num_layers=2,
  bidirectional=True,
  dropout=0.3,
  learning_rate=1e-4
)
```

---

### 4. 🔄 Transfer Learning

Evaluate cross-dataset generalization:

```bash
python src/training/train_rnn.py \
  --train_dataset internal \
  --test_dataset external
```

---

## 🧠 Text-based LLM Analysis

This section refers to the LLM emotion recognition experiments.

### Ground Truth and Splits

* `src/LLMs/results/manual_labelling.json`: Ground truth labels.
* `src/LLMs/results/data_splits.csv`: Train/test mapping for each audio segment.

### Model Predictions

* `src/LLMs/results/results_ctxt/<model>/all_*.json`: Model outputs *with context*.
* `src/LLMs/results/results_no_ctxt/<model>/all_*.json`: Model outputs *without context*.

### Consolidate Metrics and Rankings

Open the notebook below in VS Code or Jupyter:

```
src/LLMs/tools/generate_llm_metrics.ipynb
```

It will:

* Load ground truth and splits.
* Traverse results (context/no-context).
* Compute metrics for subsets: `full` (all) and `test` (only test set).
* Save outputs:

  * `src/LLMs/results/llm_models_metrics.csv`
  * Rankings:

    * `llm_models_metrics_ranking_context_full.csv`
    * `llm_models_metrics_ranking_context_test.csv`
    * `llm_models_metrics_ranking_nocontext_full.csv`
    * `llm_models_metrics_ranking_nocontext_test.csv`
    * Combined: `llm_models_metrics_rankings_all.csv`

The notebook also prints the rankings sorted by **f1_macro**.

---

### Reporting Scripts

All commands assume the repo root as current working directory.

#### 1️⃣ Average Context Effect per Model

File: `src/LLMs/tools/compare_context_effect.py`

```bash
python src/LLMs/tools/compare_context_effect.py
```

Outputs:

* Average delta of `f1_macro` (context − nocontext) per subset (full/test).
* Per-model list: F1 with/without context, delta, and percentage.

#### 2️⃣ Text Report per Model

File: `src/LLMs/tools/export_model_metrics_text.py`

```bash
python src/LLMs/tools/export_model_metrics_text.py
```

Generates `src/LLMs/results/model_metrics_report.txt` with:

* Full and test datasets (with/without context).
* Macro/weighted precision, recall, F1, and coverage (inclusive).

#### 3️⃣ Per‑Emotion Metrics

File: `src/LLMs/tools/compare_per_emotion.py`

```bash
python src/LLMs/tools/compare_per_emotion.py
# Optional: print table for a specific model
python src/LLMs/tools/compare_per_emotion.py --print-model qwen3_14b
```

Outputs:

* `per_emotion_metrics.csv`: metrics per emotion.
* `per_emotion_metrics_comparison.csv`: deltas with/without context.
* `per_emotion_metrics_wide.csv`: wide-format summary.

---

### Metric Definitions

* **total_expected:** number of GT segments in subset.
* **total_seen:** number of segments with predictions.
* **missing_predictions:** expected segments without predictions.
* **hallucinations:** predictions with invalid/missing labels.
* **samples:** valid (y_true, y_pred) pairs.
* **coverage:** `samples / total_seen`.
* **coverage_inclusive:** `samples / total_expected` (reported coverage).
* **hallucination_rate:** `hallucinations / total_seen`.
* **hallucination_rate_inclusive:** `(missing + hallucinations) / total_expected`.

Metrics: macro/weighted F1, precision, recall, accuracy — computed via scikit‑learn with `zero_division=0`.

Canonical labels: `Happiness, Sadness, Fear, Anger, Surprise, Disgust, Neutral`.

---

## 📊 Metrics Overview

| Metric                             | Description                          |
| ---------------------------------- | ------------------------------------ |
| **Accuracy**                       | Correct predictions / total          |
| **Macro / Weighted F1**            | Class-level balance                  |
| **Coverage**                       | Usable samples / total_seen          |
| **Coverage (inclusive)**           | Usable samples / total_expected      |
| **Hallucination Rate**             | Invalid predictions / total_seen     |
| **Hallucination Rate (inclusive)** | (missing + invalid) / total_expected |

---

## 🧩 Advanced Options

### Custom Hyperparameters

```bash
python src/training/train_rnn.py --hidden_size 512 --num_layers 3 --epochs 50
python src/training/train_classic_ml.py --optuna_trials 100 --k_folds 5
```

### Add New Models

1. Add entry in `MODEL_REGISTRY`.
2. Implement optimization logic.
3. Test with a small dataset.

### Add New Embeddings

1. Register in `src/data/dataset.py`.
2. Extract using `run_embedder.py`.

---

## 🤝 Contributing

1. Fork repository.
2. Create branch: `git checkout -b feature/new-feature`.
3. Commit and push.
4. Open a Pull Request.

---

## 📜 License & Contact

This project is under [license TBD]. See `LICENSE` file.

**Contact:**

* Repository: [https://github.com/aisepucrio/CHASE2026](https://github.com/aisepucrio/CHASE2026)
* Issues: GitHub Issues
* Documentation: README and code comments.

---

**Note:** This project is part of academic research on emotion recognition in agile development meetings. Results and methodologies are detailed in the associated academic paper.
