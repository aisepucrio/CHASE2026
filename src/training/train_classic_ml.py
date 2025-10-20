"""
Classic ML Training Pipeline for Audio Emotion Classification

Trains multiple ML models on segment-level embeddings using same k-fold splits as LSTM pipeline.
Supports LightGBM, XGBoost, Random Forest, SVM, Logistic Regression, Extra Trees, Gradient Boosting.
Includes Optuna-based hyperparameter optimization.
Outputs comparable metrics (macro/micro/weighted precision/recall/F1) with mean and std.
"""

import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import load_for_classic_ml
from src.data.dataset import EMBEDDING_FILES


# Model registry
MODEL_REGISTRY = {
    'lightgbm': 'LightGBM',
    'xgboost': 'XGBoost',
    'random_forest': 'Random Forest',
    'extra_trees': 'Extra Trees',
    'gradient_boosting': 'Gradient Boosting',
    'svm_linear': 'SVM (Linear)',
    'svm_rbf': 'SVM (RBF)',
    'logistic_regression': 'Logistic Regression',
}


def create_model(model_type: str, num_classes: int, random_state: int, **model_params):
    """
    Factory function to create ML model instances.

    Args:
        model_type: Model type from MODEL_REGISTRY keys
        num_classes: Number of classes
        random_state: Random seed
        **model_params: Model-specific hyperparameters

    Returns:
        Model instance (returns None for lightgbm/xgboost as they use custom training)
    """
    if model_type == 'lightgbm':
        return None  # LightGBM uses lgb.train() instead of sklearn interface

    elif model_type == 'xgboost':
        return None  # XGBoost uses xgb.train() instead of sklearn interface

    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            max_features=model_params.get('max_features', 'sqrt'),
            random_state=random_state,
            n_jobs=-1
        )

    elif model_type == 'extra_trees':
        return ExtraTreesClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', None),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            max_features=model_params.get('max_features', 'sqrt'),
            random_state=random_state,
            n_jobs=-1
        )

    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            learning_rate=model_params.get('learning_rate', 0.1),
            max_depth=model_params.get('max_depth', 3),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            subsample=model_params.get('subsample', 1.0),
            random_state=random_state
        )

    elif model_type == 'svm_linear':
        return SVC(
            kernel='linear',
            C=model_params.get('C', 1.0),
            probability=True,
            random_state=random_state
        )

    elif model_type == 'svm_rbf':
        return SVC(
            kernel='rbf',
            C=model_params.get('C', 1.0),
            gamma=model_params.get('gamma', 'scale'),
            probability=True,
            random_state=random_state
        )

    elif model_type == 'logistic_regression':
        return LogisticRegression(
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 1000),
            solver=model_params.get('solver', 'lbfgs'),
            multi_class='multinomial',
            random_state=random_state,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_default_params(model_type: str, num_classes: int, random_state: int) -> Dict[str, Any]:
    """Get default hyperparameters for each model type."""

    if model_type == 'lightgbm':
        return {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state,
        }

    elif model_type == 'xgboost':
        return {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': random_state,
        }

    elif model_type == 'random_forest':
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
        }

    elif model_type == 'extra_trees':
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
        }

    elif model_type == 'gradient_boosting':
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
        }

    elif model_type == 'svm_linear':
        return {'C': 1.0}

    elif model_type == 'svm_rbf':
        return {'C': 1.0, 'gamma': 'scale'}

    elif model_type == 'logistic_regression':
        return {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'}

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_fold(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    random_state: int,
    model_params: Dict[str, Any]
) -> Tuple[np.ndarray, Any]:
    """
    Train a model on one fold and return predictions.

    Args:
        model_type: Model type from MODEL_REGISTRY
        X_train, y_train: Training data
        X_val, y_val: Validation data
        num_classes: Number of classes
        random_state: Random seed
        model_params: Model hyperparameters

    Returns:
        y_pred: Predictions on validation set
        model: Trained model
    """
    if model_type == 'lightgbm':
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)

    elif model_type == 'xgboost':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            model_params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        y_pred_proba = model.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1)

    else:
        # Sklearn-based models
        model = create_model(model_type, num_classes, random_state, **model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    return y_pred, model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics (micro/macro/weighted)."""
    return {
        'acc_micro': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'acc_macro': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'acc_weighted': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def optimize_hyperparameters(
    model_type: str,
    csv_path: str,
    n_splits: int,
    random_state: int,
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.

    Args:
        model_type: Model type from MODEL_REGISTRY
        csv_path: Path to embeddings CSV
        n_splits: Number of k-folds
        random_state: Random seed
        n_trials: Number of Optuna trials

    Returns:
        Best hyperparameters
    """
    loader = load_for_classic_ml(csv_path)
    info = loader.get_info()

    def objective(trial):
        # Define hyperparameter search space per model
        if model_type == 'lightgbm':
            params = {
                'objective': 'multiclass',
                'num_class': info['num_classes'],
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'verbose': -1,
                'random_state': random_state,
            }

        elif model_type == 'xgboost':
            params = {
                'objective': 'multi:softprob',
                'num_class': info['num_classes'],
                'eval_metric': 'mlogloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'seed': random_state,
            }

        elif model_type in ['random_forest', 'extra_trees']:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }

        elif model_type == 'gradient_boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }

        elif model_type in ['svm_linear', 'svm_rbf']:
            params = {'C': trial.suggest_float('C', 0.01, 100, log=True)}
            if model_type == 'svm_rbf':
                params['gamma'] = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)

        elif model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                'max_iter': 1000,
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluate on k-fold CV
        f1_scores = []
        for X_train, y_train, X_val, y_val in loader.get_kfold_splits(n_splits=n_splits, random_state=random_state):
            y_pred, _ = train_fold(
                model_type, X_train, y_train, X_val, y_val,
                info['num_classes'], random_state, params
            )
            metrics = compute_metrics(y_val, y_pred)
            f1_scores.append(metrics['f1_macro'])

        return np.mean(f1_scores)

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n🎯 Best hyperparameters found:")
    print(f"  Best F1 (macro): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Merge with default params
    best_params = get_default_params(model_type, info['num_classes'], random_state)
    best_params.update(study.best_params)

    return best_params


def train_model_kfold(
    model_type: str,
    csv_path: str,
    n_splits: int = 3,
    random_state: int = 42,
    optimize: bool = False,
    n_trials: int = 50,
    train_dataset: str = "interno",
    test_dataset: str = "interno",
    test_csv_path: Optional[str] = None,
    **model_params
):
    """
    Train a model with k-fold cross-validation.

    Args:
        model_type: Model type from MODEL_REGISTRY keys
        csv_path: Path to embeddings CSV
        n_splits: Number of k-folds
        random_state: Random seed for reproducibility
        optimize: Whether to optimize hyperparameters with Optuna
        n_trials: Number of Optuna trials if optimize=True
        **model_params: Model hyperparameters (ignored if optimize=True)

    Returns:
        Dictionary with fold results and aggregated metrics
    """
    # 1. Load data for classic ML (segments flattened)
    loader = load_for_classic_ml(csv_path)
    info = loader.get_info()

    print(f"\n{'='*60}")
    print(f"🤖 Training {MODEL_REGISTRY[model_type]}")
    print(f"{'='*60}")
    print("📊 Dataset Information:")
    print(f"  Embedding dim: {info['embedding_dim']}")
    print(f"  Num classes: {info['num_classes']}")
    print(f"  Label mapping: {info['label_to_idx']}")
    print(f"  Train/val audios: {info['train_val_audios']}")
    print(f"  Test audios: {info['test_audios']}")
    print(f"  Train/val segments: {info['train_val_segments']}")
    print(f"  Test segments: {info['test_segments']}")

    # 2. Optimize hyperparameters if requested
    if optimize:
        print(f"\n🔍 Optimizing hyperparameters with Optuna ({n_trials} trials)...")
        params = optimize_hyperparameters(model_type, csv_path, n_splits, random_state, n_trials)
    else:
        # Use default or provided params
        params = get_default_params(model_type, info['num_classes'], random_state)
        params.update(model_params)

    print(f"\n📝 Hyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 3. K-fold cross-validation
    print(f"\n🔄 Starting {n_splits}-Fold Cross-Validation...")

    fold_results = []

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(
        loader.get_kfold_splits(n_splits=n_splits, random_state=random_state)
    ):
        print(f"\n{'='*60}")
        print(f"📁 FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")
        print(f"  Train segments: {len(X_train)}")
        print(f"  Val segments: {len(X_val)}")

        # Train model
        print(f"\n  🏋️ Training {MODEL_REGISTRY[model_type]}...")
        y_pred, _ = train_fold(
            model_type, X_train, y_train, X_val, y_val,
            info['num_classes'], random_state, params
        )

        # Calculate metrics
        metrics = compute_metrics(y_val, y_pred)
        metrics['fold'] = fold_idx + 1

        print(f"\n  ✅ Fold {fold_idx + 1} Results:")
        print(f"    Accuracy (micro): {metrics['acc_micro']:.4f}")
        print(f"    Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"    Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"    F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")

        fold_results.append(metrics)

    # 4. Aggregate metrics across folds (mean and std)
    metrics_list = ['acc_micro', 'precision_micro', 'recall_micro', 'f1_micro',
                    'acc_macro', 'precision_macro', 'recall_macro', 'f1_macro',
                    'acc_weighted', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    aggregated_metrics = {}
    for metric in metrics_list:
        values = [fold[metric] for fold in fold_results]
        aggregated_metrics[f'{metric}_mean'] = np.mean(values)
        aggregated_metrics[f'{metric}_std'] = np.std(values)

    print(f"\n{'='*60}")
    print(f"📊 K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Micro Averaging:")
    print(f"  Accuracy:  {aggregated_metrics['acc_micro_mean']:.4f} ± {aggregated_metrics['acc_micro_std']:.4f}")
    print(f"  Precision: {aggregated_metrics['precision_micro_mean']:.4f} ± {aggregated_metrics['precision_micro_std']:.4f}")
    print(f"  Recall:    {aggregated_metrics['recall_micro_mean']:.4f} ± {aggregated_metrics['recall_micro_std']:.4f}")
    print(f"  F1:        {aggregated_metrics['f1_micro_mean']:.4f} ± {aggregated_metrics['f1_micro_std']:.4f}")
    print(f"\nMacro Averaging:")
    print(f"  Accuracy:  {aggregated_metrics['acc_macro_mean']:.4f} ± {aggregated_metrics['acc_macro_std']:.4f}")
    print(f"  Precision: {aggregated_metrics['precision_macro_mean']:.4f} ± {aggregated_metrics['precision_macro_std']:.4f}")
    print(f"  Recall:    {aggregated_metrics['recall_macro_mean']:.4f} ± {aggregated_metrics['recall_macro_std']:.4f}")
    print(f"  F1:        {aggregated_metrics['f1_macro_mean']:.4f} ± {aggregated_metrics['f1_macro_std']:.4f}")
    print(f"\nWeighted Averaging:")
    print(f"  Accuracy:  {aggregated_metrics['acc_weighted_mean']:.4f} ± {aggregated_metrics['acc_weighted_std']:.4f}")
    print(f"  Precision: {aggregated_metrics['precision_weighted_mean']:.4f} ± {aggregated_metrics['precision_weighted_std']:.4f}")
    print(f"  Recall:    {aggregated_metrics['recall_weighted_mean']:.4f} ± {aggregated_metrics['recall_weighted_std']:.4f}")
    print(f"  F1:        {aggregated_metrics['f1_weighted_mean']:.4f} ± {aggregated_metrics['f1_weighted_std']:.4f}")

    # 5. Train final model on full train/val and evaluate on test
    print(f"\n{'='*60}")
    print("🏆 Training final model on full train/val data...")
    print(f"{'='*60}")

    # Get test data (may be from different dataset for transfer learning)
    if test_dataset != train_dataset and test_csv_path:
        # Transfer learning: load test data from different dataset
        print(f"📊 Transfer Learning: Testing on {test_dataset} dataset")
        test_loader = load_for_classic_ml(test_csv_path)
        # Get ALL segments from test dataset (not just test split)
        test_seqs, test_lbls, _ = test_loader.get_all_sequences()
        X_test = np.vstack(test_seqs)
        y_test = np.concatenate(test_lbls)
    else:
        # Same dataset: use test split
        X_test, y_test = loader.get_test_data()

    # Get all train/val data (not split into folds)
    train_val_seqs, train_val_lbls, _, _, _, _ = loader.get_train_val_test_sequences()
    X_train_full = np.vstack(train_val_seqs)
    y_train_full = np.concatenate(train_val_lbls)

    print(f"  Training on {len(X_train_full)} segments...")
    print(f"  Testing on {len(X_test)} segments...")

    # Train final model
    y_test_pred, final_model = train_fold(
        model_type, X_train_full, y_train_full, X_test, y_test,
        info['num_classes'], random_state, params
    )

    # Calculate test metrics (add _mean and _std suffix for consistency with k-fold CV)
    test_metrics_raw = compute_metrics(y_test, y_test_pred)
    test_metrics = {}
    for metric, value in test_metrics_raw.items():
        test_metrics[f'{metric}_mean'] = value
        test_metrics[f'{metric}_std'] = 0.0  # Test set evaluated once, so std=0

    print(f"\n{'='*60}")
    print("📊 TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Micro Averaging:")
    print(f"  Accuracy:  {test_metrics['acc_micro_mean']:.4f} ± {test_metrics['acc_micro_std']:.4f}")
    print(f"  Precision: {test_metrics['precision_micro_mean']:.4f} ± {test_metrics['precision_micro_std']:.4f}")
    print(f"  Recall:    {test_metrics['recall_micro_mean']:.4f} ± {test_metrics['recall_micro_std']:.4f}")
    print(f"  F1:        {test_metrics['f1_micro_mean']:.4f} ± {test_metrics['f1_micro_std']:.4f}")
    print(f"\nMacro Averaging:")
    print(f"  Accuracy:  {test_metrics['acc_macro_mean']:.4f} ± {test_metrics['acc_macro_std']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro_mean']:.4f} ± {test_metrics['precision_macro_std']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro_mean']:.4f} ± {test_metrics['recall_macro_std']:.4f}")
    print(f"  F1:        {test_metrics['f1_macro_mean']:.4f} ± {test_metrics['f1_macro_std']:.4f}")
    print(f"\nWeighted Averaging:")
    print(f"  Accuracy:  {test_metrics['acc_weighted_mean']:.4f} ± {test_metrics['acc_weighted_std']:.4f}")
    print(f"  Precision: {test_metrics['precision_weighted_mean']:.4f} ± {test_metrics['precision_weighted_std']:.4f}")
    print(f"  Recall:    {test_metrics['recall_weighted_mean']:.4f} ± {test_metrics['recall_weighted_std']:.4f}")
    print(f"  F1:        {test_metrics['f1_weighted_mean']:.4f} ± {test_metrics['f1_weighted_std']:.4f}")

    # 6. Generate confusion matrix
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_test_pred)
    class_names = [loader.idx_to_label[i] for i in range(loader.num_classes)]

    # Create results directory (use transfer_learning subfolder if applicable)
    if train_dataset != test_dataset:
        results_dir = 'results/transfer_learning'
    else:
        results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Test Set\n{MODEL_REGISTRY[model_type]} Classifier',
              fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_filename = f"{results_dir}/confusion_matrix_{model_type}_{timestamp_str}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to: {cm_filename}")

    # 7. Save results to JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_type": MODEL_REGISTRY[model_type],
        "hyperparameters": params,
        "hyperparameter_optimization": {
            "enabled": optimize,
            "n_trials": n_trials if optimize else None,
        },
        "data_info": {
            "csv_path": csv_path,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "embedding_dim": info['embedding_dim'],
            "num_classes": info['num_classes'],
            "label_to_idx": info['label_to_idx'],
            "train_val_segments": info['train_val_segments'],
            "test_segments": len(X_test),  # Actual test segments used
        },
        "k_fold_cross_validation": {
            "n_folds": n_splits,
            "random_state": random_state,
            "individual_folds": fold_results,
            "aggregated_metrics": aggregated_metrics,
        },
        "test_results": test_metrics,
        "confusion_matrix": {
            "image_path": cm_filename,
            "class_names": class_names,
            "matrix": cm.tolist()
        }
    }

    results_filename = f"{results_dir}/experiment_{model_type}_{timestamp_str}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_filename}")
    print("✅ Training complete!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train classic ML models for audio emotion classification.")
    parser.add_argument('--model_name', type=str, required=True, choices=EMBEDDING_FILES.keys(),
                        help="Name of the embedding model to use.")
    parser.add_argument('--model_type', type=str, required=True, choices=MODEL_REGISTRY.keys(),
                        help="Type of ML model to train.")
    parser.add_argument('--k_folds', type=int, default=3,
                        help="Number of folds for k-fold cross-validation.")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random state for reproducibility.")
    parser.add_argument('--optimize', action='store_true',
                        help="Enable hyperparameter optimization with Optuna.")
    parser.add_argument('--n_trials', type=int, default=50,
                        help="Number of Optuna trials for hyperparameter optimization.")
    parser.add_argument('--train_dataset', type=str, default='interno', choices=['interno', 'externo'],
                        help="Dataset to use for training (default: interno)")
    parser.add_argument('--test_dataset', type=str, default='interno', choices=['interno', 'externo'],
                        help="Dataset to use for testing (default: interno)")

    args = parser.parse_args()

    print("🚀 Initializing classic ML training with the following configuration:")
    print(f"  - Embedding Model: {args.model_name}")
    print(f"  - ML Model: {MODEL_REGISTRY[args.model_type]}")
    print(f"  - Train Dataset: {args.train_dataset}")
    print(f"  - Test Dataset: {args.test_dataset}")
    print(f"  - K-Folds: {args.k_folds}")
    print(f"  - Random State: {args.random_state}")
    print(f"  - Hyperparameter Optimization: {'Yes' if args.optimize else 'No'}")
    if args.optimize:
        print(f"  - Optuna Trials: {args.n_trials}")

    # Resolve embeddings paths
    from src.data.dataset import resolve_embeddings_path
    csv_path = resolve_embeddings_path(EMBEDDING_FILES[args.model_name], args.train_dataset)

    # Resolve test embeddings path if different dataset
    test_csv_path = None
    if args.test_dataset != args.train_dataset:
        test_csv_path = resolve_embeddings_path(EMBEDDING_FILES[args.model_name], args.test_dataset)

    # Train model
    train_model_kfold(
        model_type=args.model_type,
        csv_path=csv_path,
        n_splits=args.k_folds,
        random_state=args.random_state,
        optimize=args.optimize,
        n_trials=args.n_trials,
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        test_csv_path=test_csv_path
    )


if __name__ == '__main__':
    main()
