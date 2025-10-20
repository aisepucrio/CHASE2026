"""
Unified Data Loader for Audio Emotion Classification

Loads embeddings and provides data in format suitable for:
1. LSTM/RNN pipelines (sequences grouped by AudioID)
2. Classic ML pipelines (flattened segments)

Supports two dataset types:
- interno: Sequences grouped by AudioID (extracted from file path structure)
- externo: Independent samples (each gets unique AudioID if not provided in CSV)

Both formats use identical k-fold splits for fair comparison.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from src.data.kfold_splitter import create_kfold_splitter

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AudioEmbeddingDataLoader:
    """
    Loads audio embeddings and metadata from CSV files.

    Provides data in two formats:
    - sequence: List of arrays grouped by AudioID (for RNN/LSTM)
    - segment: Flattened arrays (for classic ML)
    """

    def __init__(
        self,
        csv_path: str,
        format: Literal["sequence", "segment"] = "sequence",
        return_type: Literal["torch", "numpy"] = "numpy"
    ):
        """
        Initialize data loader.

        Args:
            csv_path: Path to embeddings CSV file
            format: "sequence" (grouped by AudioID) or "segment" (flattened)
            return_type: "torch" or "numpy" for tensor/array format
        """
        self.csv_path = csv_path
        self.format = format
        self.return_type = return_type

        if return_type == "torch" and not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Use return_type='numpy'")

        # Will be populated by load_data()
        self.df: Optional[pd.DataFrame] = None
        self.embedding_cols: List[str] = []
        self.metadata_cols: List[str] = []
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.num_classes: int = 0
        self.embedding_dim: int = 0

        # Sequences (grouped by AudioID)
        self.sequences: List = []
        self.labels: List = []
        self.audio_ids: List[int] = []
        self.splits: List[str] = []

    def load_data(self) -> 'AudioEmbeddingDataLoader':
        """
        Load and preprocess data from CSV.

        Returns:
            self (for method chaining)
        """
        # 1. Load CSV
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # 2. Extract metadata from file paths or use existing columns
        # Check if SegmentID and AudioID already exist in CSV (e.g., for externo dataset)
        if "SegmentID" not in self.df.columns:
            # For interno dataset: extract from file path structure
            try:
                self.df["SegmentID"] = self.df["File Path"].apply(
                    lambda x: int(os.path.splitext(os.path.basename(x))[0][-3:])
                )
            except (ValueError, IndexError):
                # If extraction fails, create sequential IDs
                self.df["SegmentID"] = range(len(self.df))

        if "AudioID" not in self.df.columns:
            # For interno dataset: extract from file path structure
            try:
                self.df["AudioID"] = self.df["File Path"].apply(
                    lambda x: int(os.path.basename(os.path.dirname(x))[-1])
                )
            except (ValueError, IndexError):
                # For externo dataset: each sample is independent, create unique IDs
                self.df["AudioID"] = range(len(self.df))

        # 3. Identify columns
        self.metadata_cols = ["File Path", "Emotion", "Split", "SegmentID", "AudioID"]
        self.embedding_cols = [col for col in self.df.columns if col not in self.metadata_cols]
        self.embedding_dim = len(self.embedding_cols)

        # 4. Create label mapping
        unique_labels = sorted(self.df["Emotion"].unique())
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        self.df["EmotionIdx"] = self.df["Emotion"].map(self.label_to_idx)

        # 5. Group by AudioID and create sequences
        grouped = self.df.groupby("AudioID")

        for audio_id, group in grouped:
            sorted_group = group.sort_values("SegmentID")

            # Extract embeddings
            emb_values = sorted_group[self.embedding_cols].values.astype(np.float32)

            # Extract labels
            label_values = sorted_group["EmotionIdx"].values.astype(np.int64)

            # Store as torch or numpy
            if self.return_type == "torch":
                seq = torch.tensor(emb_values, dtype=torch.float32)
                lbl = torch.tensor(label_values, dtype=torch.long)
            else:
                seq = emb_values
                lbl = label_values

            self.sequences.append(seq)
            self.labels.append(lbl)
            self.audio_ids.append(audio_id)
            self.splits.append(sorted_group["Split"].iloc[0])

        return self

    def get_all_sequences(self) -> Tuple[List, List, List]:
        """
        Get all sequences regardless of split.

        Useful for transfer learning where we want to evaluate on the entire target dataset.

        Returns:
            Tuple of (sequences, labels, audio_ids)
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first")

        return (self.sequences, self.labels, self.audio_ids)

    def get_train_val_test_sequences(
        self
    ) -> Tuple[List, List, List, List, List, List]:
        """
        Get sequences split by train/val/test (grouped by AudioID).

        Note: train_val contains all training data (use k-fold to split further).

        Returns:
            Tuple of (train_val_sequences, train_val_labels,
                     test_sequences, test_labels,
                     train_val_audio_ids, test_audio_ids)
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first")

        train_val_sequences = [seq for i, seq in enumerate(self.sequences) if self.splits[i] == 'train']
        train_val_labels = [lbl for i, lbl in enumerate(self.labels) if self.splits[i] == 'train']
        train_val_audio_ids = [aid for i, aid in enumerate(self.audio_ids) if self.splits[i] == 'train']

        test_sequences = [seq for i, seq in enumerate(self.sequences) if self.splits[i] == 'test']
        test_labels = [lbl for i, lbl in enumerate(self.labels) if self.splits[i] == 'test']
        test_audio_ids = [aid for i, aid in enumerate(self.audio_ids) if self.splits[i] == 'test']

        return (train_val_sequences, train_val_labels,
                test_sequences, test_labels,
                train_val_audio_ids, test_audio_ids)

    def get_kfold_splits(
        self,
        n_splits: int = 3,
        random_state: int = 42
    ):
        """
        Generate k-fold train/val splits.

        Args:
            n_splits: Number of folds
            random_state: Random seed for reproducibility

        Yields:
            For sequence format:
                (train_sequences, train_labels, val_sequences, val_labels)
            For segment format:
                (X_train, y_train, X_val, y_val) - flattened numpy arrays
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first")

        # Get train/val data (test is held out)
        train_val_seqs, train_val_lbls, _, _, _, _ = self.get_train_val_test_sequences()

        # Create k-fold splitter
        splitter = create_kfold_splitter(n_splits=n_splits, random_state=random_state)

        # Generate splits
        for train_idx, val_idx in splitter.split(train_val_seqs, train_val_lbls):
            train_sequences = [train_val_seqs[i] for i in train_idx]
            train_labels = [train_val_lbls[i] for i in train_idx]
            val_sequences = [train_val_seqs[i] for i in val_idx]
            val_labels = [train_val_lbls[i] for i in val_idx]

            if self.format == "sequence":
                # Return sequences grouped by AudioID
                yield train_sequences, train_labels, val_sequences, val_labels

            elif self.format == "segment":
                # Flatten to segment-level (for classic ML)
                if self.return_type == "torch":
                    # Stack torch tensors
                    X_train = torch.cat(train_sequences, dim=0)
                    y_train = torch.cat(train_labels, dim=0)
                    X_val = torch.cat(val_sequences, dim=0)
                    y_val = torch.cat(val_labels, dim=0)
                else:
                    # Stack numpy arrays
                    X_train = np.vstack(train_sequences)
                    y_train = np.concatenate(train_labels)
                    X_val = np.vstack(val_sequences)
                    y_val = np.concatenate(val_labels)

                yield X_train, y_train, X_val, y_val

    def get_test_data(self):
        """
        Get test data.

        Returns:
            For sequence format: (test_sequences, test_labels)
            For segment format: (X_test, y_test) - flattened
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first")

        _, _, test_seqs, test_lbls, _, _ = self.get_train_val_test_sequences()

        if self.format == "sequence":
            return test_seqs, test_lbls

        elif self.format == "segment":
            if self.return_type == "torch":
                X_test = torch.cat(test_seqs, dim=0)
                y_test = torch.cat(test_lbls, dim=0)
            else:
                X_test = np.vstack(test_seqs)
                y_test = np.concatenate(test_lbls)

            return X_test, y_test

    def get_info(self) -> Dict:
        """
        Get dataset information.

        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first")

        train_val_seqs, train_val_lbls, test_seqs, test_lbls, _, _ = self.get_train_val_test_sequences()

        total_train_segments = sum(len(seq) for seq in train_val_seqs)
        total_test_segments = sum(len(seq) for seq in test_seqs)

        return {
            "csv_path": self.csv_path,
            "format": self.format,
            "return_type": self.return_type,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "total_audios": len(self.sequences),
            "train_val_audios": len(train_val_seqs),
            "test_audios": len(test_seqs),
            "train_val_segments": total_train_segments,
            "test_segments": total_test_segments,
        }


# Convenience functions

def load_for_lstm(
    csv_path: str,
    return_type: Literal["torch", "numpy"] = "torch"
) -> AudioEmbeddingDataLoader:
    """
    Load data for LSTM/RNN pipeline (sequences grouped by AudioID).

    Args:
        csv_path: Path to embeddings CSV
        return_type: "torch" or "numpy"

    Returns:
        AudioEmbeddingDataLoader configured for sequences
    """
    return AudioEmbeddingDataLoader(
        csv_path=csv_path,
        format="sequence",
        return_type=return_type
    ).load_data()


def load_for_classic_ml(csv_path: str) -> AudioEmbeddingDataLoader:
    """
    Load data for classic ML pipeline (flattened segments).

    Args:
        csv_path: Path to embeddings CSV

    Returns:
        AudioEmbeddingDataLoader configured for segments (numpy)
    """
    return AudioEmbeddingDataLoader(
        csv_path=csv_path,
        format="segment",
        return_type="numpy"
    ).load_data()
