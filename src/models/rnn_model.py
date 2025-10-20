"""
RNN-based Emotion Classifier Model

PyTorch Lightning module for sequence-level emotion classification using LSTM/GRU.
"""

from typing import Optional

import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score
)

from src.data.dataset import LABEL_PAD_VALUE


class EmotionClassifierRNN(pl.LightningModule):
    """
    RNN-based emotion classifier using PyTorch Lightning.

    Supports LSTM and GRU with optional bidirectionality.
    Handles variable-length sequences via packing/unpacking.
    Computes metrics with micro/macro/weighted averaging.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        rnn_type: str = 'lstm',
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize EmotionClassifierRNN.

        Args:
            input_size: Embedding dimension
            num_classes: Number of emotion classes
            rnn_type: 'lstm' or 'gru'
            hidden_size: RNN hidden dimension
            num_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate (applied if num_layers > 1)
            learning_rate: Learning rate for optimizer
            class_weights: Optional class weights for loss function
        """
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])

        # Select RNN type
        RNN = torch.nn.LSTM if rnn_type.lower() == 'lstm' else torch.nn.GRU

        # Create RNN
        self.rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Classifier head
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = torch.nn.Linear(rnn_output_size, num_classes)

        # Loss function with optional class weights
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=LABEL_PAD_VALUE,
            weight=class_weights
        )

        # Initialize metrics with ignore_index
        # Micro averaging
        self.accuracy_micro = MulticlassAccuracy(
            num_classes=num_classes, average='micro', ignore_index=LABEL_PAD_VALUE
        )
        self.precision_micro = MulticlassPrecision(
            num_classes=num_classes, average='micro', ignore_index=LABEL_PAD_VALUE
        )
        self.recall_micro = MulticlassRecall(
            num_classes=num_classes, average='micro', ignore_index=LABEL_PAD_VALUE
        )
        self.f1_micro = MulticlassF1Score(
            num_classes=num_classes, average='micro', ignore_index=LABEL_PAD_VALUE
        )

        # Macro averaging
        self.accuracy_macro = MulticlassAccuracy(
            num_classes=num_classes, average='macro', ignore_index=LABEL_PAD_VALUE
        )
        self.precision_macro = MulticlassPrecision(
            num_classes=num_classes, average='macro', ignore_index=LABEL_PAD_VALUE
        )
        self.recall_macro = MulticlassRecall(
            num_classes=num_classes, average='macro', ignore_index=LABEL_PAD_VALUE
        )
        self.f1_macro = MulticlassF1Score(
            num_classes=num_classes, average='macro', ignore_index=LABEL_PAD_VALUE
        )

        # Weighted averaging
        self.accuracy_weighted = MulticlassAccuracy(
            num_classes=num_classes, average='weighted', ignore_index=LABEL_PAD_VALUE
        )
        self.precision_weighted = MulticlassPrecision(
            num_classes=num_classes, average='weighted', ignore_index=LABEL_PAD_VALUE
        )
        self.recall_weighted = MulticlassRecall(
            num_classes=num_classes, average='weighted', ignore_index=LABEL_PAD_VALUE
        )
        self.f1_weighted = MulticlassF1Score(
            num_classes=num_classes, average='weighted', ignore_index=LABEL_PAD_VALUE
        )

    def forward(self, embeddings, lengths):
        """
        Forward pass.

        Args:
            embeddings: Padded sequences [batch, max_seq_len, input_size]
            lengths: Actual sequence lengths [batch]

        Returns:
            logits: [batch, max_seq_len, num_classes]
        """
        # Pack sequence to handle variable lengths efficiently
        packed_input = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.rnn(packed_input)

        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Pass RNN output through the classifier
        logits = self.classifier(output)
        return logits

    def _shared_step(self, batch, batch_idx):
        """Shared logic for train/val/test steps."""
        embeddings, labels, lengths = batch['embeddings'], batch['labels'], batch['lengths']

        logits = self(embeddings, lengths)

        # Reshape for loss calculation: (Batch * SeqLen, NumClasses)
        logits_flat = logits.view(-1, self.hparams.num_classes)
        labels_flat = labels.view(-1)

        loss = self.criterion(logits_flat, labels_flat)

        # Get predictions by taking argmax
        preds_flat = torch.argmax(logits_flat, dim=1)

        return loss, preds_flat, labels_flat

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, preds_flat, labels_flat = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

        # Log metrics - micro
        self.log('val_acc_micro', self.accuracy_micro(preds_flat, labels_flat), on_epoch=True)
        self.log('val_precision_micro', self.precision_micro(preds_flat, labels_flat), on_epoch=True)
        self.log('val_recall_micro', self.recall_micro(preds_flat, labels_flat), on_epoch=True)
        self.log('val_f1_micro', self.f1_micro(preds_flat, labels_flat), on_epoch=True)

        # Log metrics - macro
        self.log('val_acc_macro', self.accuracy_macro(preds_flat, labels_flat), on_epoch=True)
        self.log('val_precision_macro', self.precision_macro(preds_flat, labels_flat), on_epoch=True)
        self.log('val_recall_macro', self.recall_macro(preds_flat, labels_flat), on_epoch=True)
        self.log('val_f1_macro', self.f1_macro(preds_flat, labels_flat), on_epoch=True)

        # Log metrics - weighted
        self.log('val_acc_weighted', self.accuracy_weighted(preds_flat, labels_flat), on_epoch=True, prog_bar=True)
        self.log('val_precision_weighted', self.precision_weighted(preds_flat, labels_flat), on_epoch=True)
        self.log('val_recall_weighted', self.recall_weighted(preds_flat, labels_flat), on_epoch=True)
        self.log('val_f1_weighted', self.f1_weighted(preds_flat, labels_flat), on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, preds_flat, labels_flat = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss)

        # Log metrics - micro
        self.log('test_acc_micro', self.accuracy_micro(preds_flat, labels_flat), on_epoch=True)
        self.log('test_precision_micro', self.precision_micro(preds_flat, labels_flat), on_epoch=True)
        self.log('test_recall_micro', self.recall_micro(preds_flat, labels_flat), on_epoch=True)
        self.log('test_f1_micro', self.f1_micro(preds_flat, labels_flat), on_epoch=True)

        # Log metrics - macro
        self.log('test_acc_macro', self.accuracy_macro(preds_flat, labels_flat), on_epoch=True)
        self.log('test_precision_macro', self.precision_macro(preds_flat, labels_flat), on_epoch=True)
        self.log('test_recall_macro', self.recall_macro(preds_flat, labels_flat), on_epoch=True)
        self.log('test_f1_macro', self.f1_macro(preds_flat, labels_flat), on_epoch=True)

        # Log metrics - weighted
        self.log('test_acc_weighted', self.accuracy_weighted(preds_flat, labels_flat), on_epoch=True)
        self.log('test_precision_weighted', self.precision_weighted(preds_flat, labels_flat), on_epoch=True)
        self.log('test_recall_weighted', self.recall_weighted(preds_flat, labels_flat), on_epoch=True)
        self.log('test_f1_weighted', self.f1_weighted(preds_flat, labels_flat), on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
