import os
import argparse
import json
from datetime import datetime
from collections import Counter

import torch
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Project modules
from src.data.kfold_splitter import create_kfold_splitter
from src.data.dataset import AudioDataModule, EMBEDDING_FILES, LABEL_PAD_VALUE
from src.models.rnn_model import EmotionClassifierRNN


def main():
    parser = argparse.ArgumentParser(description="Train an RNN for audio emotion classification.")
    parser.add_argument('--model_name', type=str, required=True, choices=EMBEDDING_FILES.keys(), help="Name of the embedding model to use.")
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help="Type of RNN cell.")
    parser.add_argument('--strategy', type=str, default='instant', choices=['instant', 'full_context'], help="'instant' for unidirectional, 'full_context' for bidirectional.")

    # Dataset selection
    parser.add_argument('--train_dataset', type=str, default='interno', choices=['interno', 'externo'], help="Dataset to use for training.")
    parser.add_argument('--test_dataset', type=str, default='interno', choices=['interno', 'externo'], help="Dataset to use for testing.")

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=3e-3, help="Learning rate for the optimizer.")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden size of the RNN.")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of RNN layers.")
    parser.add_argument('--k_folds', type=int, default=3, help="Number of folds for k-fold cross-validation.")
    parser.add_argument('--use_class_weights', action='store_true', help="Use class weights to handle imbalanced data.")

    args = parser.parse_args()

    print("🚀 Initializing training with the following configuration:")
    print(f"  - Embedding Model: {args.model_name}")
    print(f"  - RNN Type: {args.rnn_type.upper()}")
    print(f"  - Strategy: {args.strategy} ({'Bidirectional' if args.strategy == 'full_context' else 'Unidirectional'})")
    print(f"  - Train Dataset: {args.train_dataset}")
    print(f"  - Test Dataset: {args.test_dataset}")
    print(f"  - Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"  - K-Folds: {args.k_folds}, Class Weights: {args.use_class_weights}")

    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)

    # 1. Initialize DataModule to get metadata (without k-fold splits yet)
    data_module_init = AudioDataModule(
        model_name=args.model_name,
        batch_size=args.batch_size,
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset
    )
    data_module_init.setup()

    # 2. Prepare data for K-Fold Cross-Validation
    train_val_sequences = data_module_init.train_val_sequences
    train_val_labels = data_module_init.train_val_labels

    # Determine k-fold strategy based on dataset
    # 'interno' uses grouped splitting (StratifiedGroupKFold) to keep sequences together
    # 'externo' uses independent splitting (StratifiedKFold) as samples are independent
    kfold_strategy = 'grouped' if args.train_dataset == 'interno' else 'independent'
    print(f"✅ K-Fold strategy: {kfold_strategy} ({'StratifiedGroupKFold' if kfold_strategy == 'grouped' else 'StratifiedKFold'})")

    # Create k-fold splitter
    kfold_splitter = create_kfold_splitter(
        n_splits=args.k_folds,
        shuffle=True,
        random_state=42,
        strategy=kfold_strategy
    )

    # 3. Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        all_labels_flat = []
        for label_seq in train_val_labels:
            all_labels_flat.extend(label_seq.tolist())

        class_weights_np = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(all_labels_flat),
            y=all_labels_flat
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
        print(f"✅ Computed class weights: {class_weights}")

    # 4. Stratified Group K-Fold Cross-Validation
    fold_results = []
    all_fold_test_preds = []
    all_fold_test_labels = []

    print(f"\n🔄 Starting {args.k_folds}-Fold Cross-Validation with Stratified Group K-Fold...")

    for fold_idx, (train_indices, val_indices) in enumerate(kfold_splitter.split(train_val_sequences, train_val_labels)):
        print(f"\n{'='*60}")
        print(f"📁 FOLD {fold_idx + 1}/{args.k_folds}")
        print(f"{'='*60}")

        # Create data module for this fold
        data_module = AudioDataModule(
            model_name=args.model_name,
            batch_size=args.batch_size,
            train_indices=train_indices.tolist(),
            val_indices=val_indices.tolist(),
            train_dataset=args.train_dataset,
            test_dataset=args.test_dataset
        )
        data_module.setup()

        # Initialize model for this fold
        model = EmotionClassifierRNN(
            input_size=data_module.embedding_dim,
            num_classes=data_module.num_classes,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bidirectional=(args.strategy == 'full_context'),
            learning_rate=args.learning_rate,
            class_weights=class_weights
        )

        # Configure callbacks for this fold
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/fold_{fold_idx + 1}/',
            filename=f'{args.model_name}-{args.rnn_type}-{args.strategy}-{{epoch:02d}}-{{val_loss:.2f}}',
            save_top_k=1,
            mode='min',
        )
        progress_bar = TQDMProgressBar(refresh_rate=10)

        # Initialize trainer for this fold
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator='auto',
            callbacks=[checkpoint_callback, progress_bar],
            deterministic=True,
            enable_model_summary=False
        )

        # Train the model
        print(f"\n🏋️ Training fold {fold_idx + 1}...")
        trainer.fit(model, datamodule=data_module)

        # Validate on this fold
        val_results = trainer.validate(model, datamodule=data_module, ckpt_path='best', verbose=False)

        fold_results.append({
            'fold': fold_idx + 1,
            'val_loss': val_results[0]['val_loss'],
            # Micro averaging
            'val_acc_micro': val_results[0].get('val_acc_micro', 0),
            'val_precision_micro': val_results[0].get('val_precision_micro', 0),
            'val_recall_micro': val_results[0].get('val_recall_micro', 0),
            'val_f1_micro': val_results[0].get('val_f1_micro', 0),
            # Macro averaging
            'val_acc_macro': val_results[0].get('val_acc_macro', 0),
            'val_precision_macro': val_results[0].get('val_precision_macro', 0),
            'val_recall_macro': val_results[0].get('val_recall_macro', 0),
            'val_f1_macro': val_results[0].get('val_f1_macro', 0),
            # Weighted averaging
            'val_acc_weighted': val_results[0].get('val_acc_weighted', 0),
            'val_precision_weighted': val_results[0].get('val_precision_weighted', 0),
            'val_recall_weighted': val_results[0].get('val_recall_weighted', 0),
            'val_f1_weighted': val_results[0].get('val_f1_weighted', 0),
            'checkpoint': checkpoint_callback.best_model_path
        })

        print(f"\n  ✅ Fold {fold_idx + 1} Results:")
        print(f"    Val Loss: {val_results[0]['val_loss']:.4f}")
        print(f"    Accuracy (micro): {val_results[0].get('val_acc_micro', 0):.4f}")
        print(f"    Precision (macro): {val_results[0].get('val_precision_macro', 0):.4f}")
        print(f"    Recall (macro): {val_results[0].get('val_recall_macro', 0):.4f}")
        print(f"    F1 (macro): {val_results[0].get('val_f1_macro', 0):.4f}")
        print(f"    F1 (weighted): {val_results[0].get('val_f1_weighted', 0):.4f}")

    # 5. Compute average and std metrics across folds
    metrics_list = ['val_loss', 'val_acc_micro', 'val_acc_macro', 'val_acc_weighted',
                    'val_f1_micro', 'val_f1_macro', 'val_f1_weighted',
                    'val_precision_micro', 'val_precision_macro', 'val_precision_weighted',
                    'val_recall_micro', 'val_recall_macro', 'val_recall_weighted']

    avg_metrics = {}
    for metric in metrics_list:
        values = [f.get(metric, 0) for f in fold_results]
        avg_metrics[f'{metric}_mean'] = np.mean(values)
        avg_metrics[f'{metric}_std'] = np.std(values)

    print(f"\n{'='*60}")
    print(f"📊 K-FOLD CROSS-VALIDATION RESULTS (Mean ± Std across {args.k_folds} folds)")
    print(f"{'='*60}")
    print(f"Micro Averaging:")
    print(f"  Accuracy:  {avg_metrics['val_acc_micro_mean']:.4f} ± {avg_metrics['val_acc_micro_std']:.4f}")
    print(f"  Precision: {avg_metrics['val_precision_micro_mean']:.4f} ± {avg_metrics['val_precision_micro_std']:.4f}")
    print(f"  Recall:    {avg_metrics['val_recall_micro_mean']:.4f} ± {avg_metrics['val_recall_micro_std']:.4f}")
    print(f"  F1:        {avg_metrics['val_f1_micro_mean']:.4f} ± {avg_metrics['val_f1_micro_std']:.4f}")
    print(f"\nMacro Averaging:")
    print(f"  Accuracy:  {avg_metrics['val_acc_macro_mean']:.4f} ± {avg_metrics['val_acc_macro_std']:.4f}")
    print(f"  Precision: {avg_metrics['val_precision_macro_mean']:.4f} ± {avg_metrics['val_precision_macro_std']:.4f}")
    print(f"  Recall:    {avg_metrics['val_recall_macro_mean']:.4f} ± {avg_metrics['val_recall_macro_std']:.4f}")
    print(f"  F1:        {avg_metrics['val_f1_macro_mean']:.4f} ± {avg_metrics['val_f1_macro_std']:.4f}")
    print(f"\nWeighted Averaging:")
    print(f"  Accuracy:  {avg_metrics['val_acc_weighted_mean']:.4f} ± {avg_metrics['val_acc_weighted_std']:.4f}")
    print(f"  Precision: {avg_metrics['val_precision_weighted_mean']:.4f} ± {avg_metrics['val_precision_weighted_std']:.4f}")
    print(f"  Recall:    {avg_metrics['val_recall_weighted_mean']:.4f} ± {avg_metrics['val_recall_weighted_std']:.4f}")
    print(f"  F1:        {avg_metrics['val_f1_weighted_mean']:.4f} ± {avg_metrics['val_f1_weighted_std']:.4f}")

    # 6. Train final model on full training data and evaluate on test set
    print(f"\n{'='*60}")
    print("🏆 Training final model on full training data...")
    print(f"{'='*60}")

    final_data_module = AudioDataModule(
        model_name=args.model_name,
        batch_size=args.batch_size,
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset
    )
    final_data_module.setup()

    final_model = EmotionClassifierRNN(
        input_size=final_data_module.embedding_dim,
        num_classes=final_data_module.num_classes,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=(args.strategy == 'full_context'),
        learning_rate=args.learning_rate,
        class_weights=class_weights
    )

    final_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/final/',
        filename=f'{args.model_name}-{args.rnn_type}-{args.strategy}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        mode='min',
    )
    final_progress_bar = TQDMProgressBar(refresh_rate=10)

    final_trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        callbacks=[final_checkpoint_callback, final_progress_bar],
        deterministic=True
    )

    final_trainer.fit(final_model, datamodule=final_data_module)

    # 7. Test the final model on the test set
    print("\n🧪 Evaluating final model on the test set...")
    test_results = final_trainer.test(final_model, datamodule=final_data_module, ckpt_path='best')
    print("\n--- Test Results ---")
    for key, value in test_results[0].items():
        print(f"  - {key}: {value:.4f}")
    print("--------------------")

    # 7.5. Generate predictions for confusion matrix
    print("\n📊 Generating confusion matrix...")
    final_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in final_data_module.test_dataloader():
            embeddings = batch['embeddings'].to(final_model.device)
            labels = batch['labels'].to(final_model.device)
            lengths = batch['lengths'].to(final_model.device)

            logits = final_model(embeddings, lengths)
            preds = torch.argmax(logits, dim=-1)

            # Flatten and filter out padding
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)

            # Remove padding tokens
            mask = labels_flat != LABEL_PAD_VALUE
            preds_flat = preds_flat[mask]
            labels_flat = labels_flat[mask]

            all_preds.extend(preds_flat.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [final_data_module.idx_to_label[i] for i in range(final_data_module.num_classes)]

    # Create results directory if it doesn't exist
    # Use separate folder for transfer learning experiments
    if args.train_dataset != args.test_dataset:
        results_dir = 'results/transfer_learning'
    else:
        results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp for consistent naming
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot confusion matrix with high resolution
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Test Set\n{args.model_name} | {args.rnn_type.upper()} | {args.strategy}',
              fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save with high DPI
    cm_filename = f"{results_dir}/confusion_matrix_{args.model_name}_{args.rnn_type}_{args.strategy}_{timestamp_str}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to: {cm_filename}")

    # 8. Save experiment metadata and results to JSON
    experiment_data = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "model_name": args.model_name,
            "rnn_type": args.rnn_type,
            "strategy": args.strategy,
            "bidirectional": args.strategy == 'full_context',
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "k_folds": args.k_folds,
            "use_class_weights": args.use_class_weights,
            "embedding_dim": final_data_module.embedding_dim,
            "num_classes": final_data_module.num_classes,
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset
        },
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "data_splits": {
            "train_size": len(final_data_module.train_sequences),
            "val_size": len(final_data_module.val_sequences),
            "test_size": len(final_data_module.test_sequences)
        },
        "label_mapping": {
            "label_to_idx": final_data_module.label_to_idx,
            "idx_to_label": final_data_module.idx_to_label
        },
        "k_fold_cross_validation": {
            "n_folds": args.k_folds,
            "strategy": "StratifiedGroupKFold" if kfold_strategy == 'grouped' else "StratifiedKFold",
            "individual_folds": fold_results,
            "aggregated_metrics": avg_metrics
        },
        "test_results": {
            "loss_mean": float(test_results[0].get('test_loss', 0)),
            "loss_std": 0.0,
            # Micro averaging
            "acc_micro_mean": float(test_results[0].get('test_acc_micro', 0)),
            "acc_micro_std": 0.0,
            "precision_micro_mean": float(test_results[0].get('test_precision_micro', 0)),
            "precision_micro_std": 0.0,
            "recall_micro_mean": float(test_results[0].get('test_recall_micro', 0)),
            "recall_micro_std": 0.0,
            "f1_micro_mean": float(test_results[0].get('test_f1_micro', 0)),
            "f1_micro_std": 0.0,
            # Macro averaging
            "acc_macro_mean": float(test_results[0].get('test_acc_macro', 0)),
            "acc_macro_std": 0.0,
            "precision_macro_mean": float(test_results[0].get('test_precision_macro', 0)),
            "precision_macro_std": 0.0,
            "recall_macro_mean": float(test_results[0].get('test_recall_macro', 0)),
            "recall_macro_std": 0.0,
            "f1_macro_mean": float(test_results[0].get('test_f1_macro', 0)),
            "f1_macro_std": 0.0,
            # Weighted averaging
            "acc_weighted_mean": float(test_results[0].get('test_acc_weighted', 0)),
            "acc_weighted_std": 0.0,
            "precision_weighted_mean": float(test_results[0].get('test_precision_weighted', 0)),
            "precision_weighted_std": 0.0,
            "recall_weighted_mean": float(test_results[0].get('test_recall_weighted', 0)),
            "recall_weighted_std": 0.0,
            "f1_weighted_mean": float(test_results[0].get('test_f1_weighted', 0)),
            "f1_weighted_std": 0.0
        },
        "confusion_matrix": {
            "image_path": cm_filename,
            "class_names": class_names,
            "matrix": cm.tolist()
        },
        "best_checkpoint_path": final_checkpoint_callback.best_model_path
    }

    # Generate filename with timestamp and config
    results_filename = f"{results_dir}/experiment_{args.model_name}_{args.rnn_type}_{args.strategy}_{timestamp_str}.json"

    # Save to JSON
    with open(results_filename, 'w') as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\n💾 Experiment results saved to: {results_filename}")
    print("✅ Training and evaluation complete.")


if __name__ == '__main__':
    main()