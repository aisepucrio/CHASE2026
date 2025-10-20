"""
K-Fold Cross-Validation Splitter for Sequential Audio Data

Provides two splitting strategies:
1. StratifiedGroupKFold: For grouped data (e.g., interno dataset)
   - Stratification based on dominant emotion per sequence
   - Groups defined by sequence IDs to prevent data leakage
2. StratifiedKFold: For independent samples (e.g., externo dataset)
   - Simple stratified splitting without grouping
"""

import numpy as np
from typing import List, Tuple
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import torch


class AudioSequenceKFoldSplitter:
    """
    Handles k-fold cross-validation for sequential audio data.

    Supports two strategies:
    - 'grouped': StratifiedGroupKFold (sequences kept together)
    - 'independent': StratifiedKFold (no grouping constraint)
    """

    def __init__(
        self,
        n_splits: int = 3,
        shuffle: bool = True,
        random_state: int = 42,
        strategy: str = 'grouped'
    ):
        """
        Initialize the k-fold splitter.

        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            strategy: 'grouped' for StratifiedGroupKFold or 'independent' for StratifiedKFold
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.strategy = strategy

        if strategy == 'grouped':
            self.splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        elif strategy == 'independent':
            self.splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        else:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'grouped' or 'independent'.")

    def prepare_sequences_for_split(
        self,
        sequences: List[torch.Tensor],
        labels: List[torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for stratified group k-fold splitting.

        Args:
            sequences: List of sequence tensors (each sequence can have variable length)
            labels: List of label tensors (same structure as sequences)

        Returns:
            Tuple of (sequences_array, dominant_labels, group_ids):
                - sequences_array: Input sequences as array for indexing
                - dominant_labels: Most common label per sequence (for stratification)
                - group_ids: Sequential IDs for grouping (prevents sequence splitting)
        """
        # Get dominant emotion per sequence (most common label)
        dominant_labels = []
        for label_seq in labels:
            label_counts = Counter(label_seq.tolist())
            dominant_label = label_counts.most_common(1)[0][0]
            dominant_labels.append(dominant_label)

        # Create group IDs (each sequence is its own group)
        group_ids = np.arange(len(sequences))

        # Convert to numpy arrays
        dominant_labels = np.array(dominant_labels)
        sequences_array = np.array(sequences, dtype=object)

        return sequences_array, dominant_labels, group_ids

    def split(
        self,
        sequences: List[torch.Tensor],
        labels: List[torch.Tensor]
    ):
        """
        Generate k-fold train/val indices for sequences.

        Args:
            sequences: List of sequence tensors
            labels: List of label tensors

        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        sequences_array, dominant_labels, group_ids = self.prepare_sequences_for_split(
            sequences, labels
        )

        # Generate splits based on strategy
        if self.strategy == 'grouped':
            for train_idx, val_idx in self.splitter.split(sequences_array, dominant_labels, group_ids):
                yield train_idx, val_idx
        else:  # independent
            for train_idx, val_idx in self.splitter.split(sequences_array, dominant_labels):
                yield train_idx, val_idx

    def get_fold_info(
        self,
        sequences: List[torch.Tensor],
        labels: List[torch.Tensor],
        train_idx: np.ndarray,
        val_idx: np.ndarray
    ) -> dict:
        """
        Get information about a specific fold split.

        Args:
            sequences: Full list of sequences
            labels: Full list of labels
            train_idx: Training indices for this fold
            val_idx: Validation indices for this fold

        Returns:
            Dictionary with fold statistics
        """
        _, dominant_labels, _ = self.prepare_sequences_for_split(sequences, labels)

        train_label_counts = Counter(dominant_labels[train_idx])
        val_label_counts = Counter(dominant_labels[val_idx])

        return {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_label_distribution': dict(train_label_counts),
            'val_label_distribution': dict(val_label_counts)
        }


def create_kfold_splitter(
    n_splits: int = 3,
    shuffle: bool = True,
    random_state: int = 42,
    strategy: str = 'grouped'
) -> AudioSequenceKFoldSplitter:
    """
    Factory function to create a k-fold splitter.

    Args:
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
        strategy: 'grouped' for StratifiedGroupKFold or 'independent' for StratifiedKFold

    Returns:
        AudioSequenceKFoldSplitter instance
    """
    return AudioSequenceKFoldSplitter(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        strategy=strategy
    )
