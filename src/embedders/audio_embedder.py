import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    HubertModel,
)


class AudioEmbedder:
    """Extract audio embeddings using Wav2Vec2 or HuBERT models."""

    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        """Initialize the audio embedder.

        Args:
            model_name (str): HuggingFace model name. 
                Examples: "facebook/wav2vec2-base-960h", "facebook/hubert-base-ls960".
            device (str, optional): "cuda" or "cpu". If None, automatically detects.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print("Using device:", self.device)

        # Load model and feature extractor
        if "hubert" in model_name.lower():
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = HubertModel.from_pretrained(model_name).to(self.device)
        else:
            self.feature_extractor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_embeddings(self, file_paths):
        """Extract embeddings from a list of audio files.

        Args:
            file_paths (list[str]): List of audio file paths.

        Returns:
            tuple:
                - embeddings (np.ndarray): Array of shape [N, hidden_dim] with audio embeddings.
                - valid_paths (list[str]): List of successfully processed file paths.
        """
        audios = []
        for fp in file_paths:
            try:
                audio, _ = librosa.load(fp, sr=16000, mono=True)
                audios.append(audio)
            except Exception as e:
                print(f"Error with file {fp}: {e}")
                audios.append(None)

        # Filter invalid files
        valid_idx = [i for i, a in enumerate(audios) if a is not None]
        valid_paths = [file_paths[i] for i in valid_idx]
        audios = [a for a in audios if a is not None]

        if len(audios) == 0:
            return [], []

        # Prepare batch
        inputs = self.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state  # [batch, seq_len, hidden_dim]

        # Mean pooling -> [batch, hidden_dim]
        embeddings = outputs.mean(dim=1).cpu().numpy()

        return embeddings, valid_paths

    def build_embeddings_dataframe(self, df_splits, batch_size=4, path_col="File Path"):
        """Process dataset in batches and generate a DataFrame with embeddings.

        Args:
            df_splits (pd.DataFrame): DataFrame with columns ["File Path", "Emotion", "Split"].
            batch_size (int): Batch size for processing.

        Returns:
            pd.DataFrame: DataFrame with embeddings and metadata.
        """
        embeddings_list = []
        labels = []
        splits = []
        paths = []

        for i in tqdm(range(0, len(df_splits), batch_size)):
            batch_paths = df_splits[path_col].iloc[i : i + batch_size].tolist()
            embs, valid_paths = self.extract_embeddings(batch_paths)

            # if len(embs) > 0:
            #     embeddings_list.extend(embs)
            #     labels.extend(
            #         df_splits[df_splits["File Path"].isin(valid_paths)]["Emotion"].tolist()
            #     )
            #     splits.extend(
            #         df_splits[df_splits["File Path"].isin(valid_paths)]["Split"].tolist()
            #     )
            #     paths.extend(valid_paths)

            if len(embs) > 0:
                valid_rows = df_splits[df_splits[path_col].isin(valid_paths)]
                embeddings_list.extend(embs)
                labels.extend(valid_rows["Emotion"].tolist())
                splits.extend(valid_rows["Split"].tolist())
                paths.extend(valid_rows[path_col].tolist())

        # Build final DataFrame
        df_embeddings = pd.DataFrame(embeddings_list)
        df_embeddings["Emotion"] = labels
        df_embeddings["Split"] = splits
        df_embeddings["File Path"] = paths

        return df_embeddings