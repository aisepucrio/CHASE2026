"""
PyTorch Dataset and DataModule for Audio Emotion Classification

Contains PyTorch-specific components for LSTM/RNN training:
- AudioSequenceDataset: Dataset wrapper for sequences
- AudioDataModule: PyTorch Lightning DataModule with k-fold support
"""

import os
import sys
from typing import List, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from src.data.data_loader import load_for_lstm


# --- Configuration ---
# Mapping from the argument name to the actual filename
EMBEDDING_FILES = {
    "hubert_base_ls960": "embeddings_hubert_base_ls960.csv",
    "wav2vec2_large_960h_lv60_self": "embeddings_wav2vec2_large_960h_lv60_self.csv",
    "wav2vec2_large_xlsr_53_portuguese": "embeddings_wav2vec2_large_xlsr_53_portuguese.csv"
}

# Use -1 as the padding value for labels, which will be ignored by the loss function
LABEL_PAD_VALUE = -1


# --- Helper Functions ---
def resolve_embeddings_path(file_name: str, dataset: str = "interno") -> str:
    """Resolve embeddings file path relative to script directory.

    Args:
        file_name: Name of the embeddings file
        dataset: Dataset folder name ('interno' or 'externo')

    Returns:
        Full path to embeddings file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "embeddings", dataset, file_name)


def check_embeddings_path(path: str) -> str:
    """Check if embeddings file exists."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


# --- Custom PyTorch Dataset ---
class AudioSequenceDataset(Dataset):
    """
    Custom Dataset for handling sequences of audio segment embeddings.
    Each item in this dataset is a complete audio sequence (grouped by AudioID).
    """
    def __init__(self, sequences: List[torch.Tensor], labels: List[torch.Tensor]):
        """
        Initialize dataset.

        Args:
            sequences: List of sequence tensors [n_segments, embedding_dim]
            labels: List of label tensors [n_segments]
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# --- PyTorch Lightning Data Module ---
class AudioDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for audio emotion classification.

    Handles data loading, k-fold splitting, and DataLoader creation.
    Uses data_loader.py for unified data loading logic.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        train_indices: Optional[List[int]] = None,
        val_indices: Optional[List[int]] = None,
        train_dataset: str = "interno",
        test_dataset: str = "interno"
    ):
        """
        Initialize AudioDataModule.

        Args:
            model_name: Name of embedding model (key in EMBEDDING_FILES)
            batch_size: Batch size for DataLoaders
            train_indices: Indices for training split (for k-fold)
            val_indices: Indices for validation split (for k-fold)
            train_dataset: Dataset to use for training ('interno' or 'externo')
            test_dataset: Dataset to use for testing ('interno' or 'externo')
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.embedding_file = EMBEDDING_FILES.get(model_name)
        if not self.embedding_file:
            print(f"Error: Invalid model name '{model_name}'.", file=sys.stderr)
            sys.exit(1)

        # Metadata (populated by setup())
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.num_classes: int = 0
        self.embedding_dim: int = 0
        self.class_weights: Optional[torch.Tensor] = None

        # Sequences (populated by setup())
        self.train_val_sequences: List[torch.Tensor] = []
        self.train_val_labels: List[torch.Tensor] = []
        self.test_sequences: List[torch.Tensor] = []
        self.test_labels: List[torch.Tensor] = []
        self.train_sequences: List[torch.Tensor] = []
        self.train_labels: List[torch.Tensor] = []
        self.val_sequences: List[torch.Tensor] = []
        self.val_labels: List[torch.Tensor] = []

    def prepare_data(self):
        """Verify embeddings file exists."""
        train_path = resolve_embeddings_path(self.embedding_file, self.train_dataset)
        check_embeddings_path(train_path)

        # Only check test path if it's different from train
        if self.test_dataset != self.train_dataset:
            test_path = resolve_embeddings_path(self.embedding_file, self.test_dataset)
            check_embeddings_path(test_path)

    def setup(self, stage: str = None):
        """Load and prepare data for training/validation/testing."""
        # 1. Load training data
        train_full_path = resolve_embeddings_path(self.embedding_file, self.train_dataset)
        train_loader = load_for_lstm(train_full_path, return_type="torch")

        # 2. Get metadata from training loader
        info = train_loader.get_info()
        self.embedding_dim = info['embedding_dim']
        self.num_classes = info['num_classes']
        self.label_to_idx = info['label_to_idx']
        self.idx_to_label = info['idx_to_label']

        print(f"✅ Training dataset: {self.train_dataset}")
        print(f"✅ Test dataset: {self.test_dataset}")
        print(f"✅ Found {self.embedding_dim} embedding dimensions.")
        print(f"✅ Found {self.num_classes} emotion classes: {self.label_to_idx}")

        # 3. Get train/val sequences from training dataset
        (train_val_sequences, train_val_labels,
         _, _, _, _) = train_loader.get_train_val_test_sequences()

        # Store full train/val data for k-fold
        self.train_val_sequences = train_val_sequences
        self.train_val_labels = train_val_labels

        # 4. Load test data (may be from different dataset)
        if self.test_dataset == self.train_dataset:
            # Use test split from same dataset
            (_, _, test_sequences, test_labels, _, _) = train_loader.get_train_val_test_sequences()
        else:
            # Transfer learning: Load ALL sequences from different dataset for evaluation
            test_full_path = resolve_embeddings_path(self.embedding_file, self.test_dataset)
            test_loader = load_for_lstm(test_full_path, return_type="torch")
            (test_sequences, test_labels, _) = test_loader.get_all_sequences()

        self.test_sequences = test_sequences
        self.test_labels = test_labels

        # If k-fold indices provided, use them; otherwise use simple split
        if self.train_indices is not None and self.val_indices is not None:
            self.train_sequences = [train_val_sequences[i] for i in self.train_indices]
            self.train_labels = [train_val_labels[i] for i in self.train_indices]
            self.val_sequences = [train_val_sequences[i] for i in self.val_indices]
            self.val_labels = [train_val_labels[i] for i in self.val_indices]
        else:
            # Fallback to simple train/test split
            self.train_sequences, self.val_sequences, self.train_labels, self.val_labels = train_test_split(
                train_val_sequences, train_val_labels, test_size=0.1, random_state=42
            )

        print(f"✅ Data split: Train={len(self.train_sequences)}, Val={len(self.val_sequences)}, Test={len(self.test_sequences)}")

        # 4. Create PyTorch Datasets (use different names to avoid conflict with train_dataset/test_dataset strings)
        self.train_data = AudioSequenceDataset(self.train_sequences, self.train_labels)
        self.val_data = AudioSequenceDataset(self.val_sequences, self.val_labels)
        self.test_data = AudioSequenceDataset(self.test_sequences, self.test_labels)

    def _collate_fn(self, batch):
        """Custom collate function to handle padding for variable length sequences."""
        sequences, labels = zip(*batch)

        # Get sequence lengths before padding
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

        # Pad sequences and labels
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=LABEL_PAD_VALUE)

        return {
            "embeddings": padded_sequences,
            "labels": padded_labels,
            "lengths": lengths
        }

    def train_dataloader(self):
        """Create training DataLoader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        """Create validation DataLoader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        """Create test DataLoader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
