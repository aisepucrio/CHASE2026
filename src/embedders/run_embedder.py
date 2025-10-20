import os
import argparse
import pandas as pd
from core.audio_embedder import AudioEmbedder

def main(args):
    # Carrega o CSV com os caminhos dos áudios e metadados
    df_splits = pd.read_csv(args.dataset)
    df_splits["File Path"] = df_splits["File Path"].str.replace("\\", "/", regex=False)

    # Garante caminho absoluto para cada arquivo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_splits["abs_path"] = df_splits["File Path"].apply(
        lambda x: os.path.abspath(os.path.join(script_dir, "..", "data", x))
    )

    # Extrai embeddings dos áudios (usando abs_path)
    embedder = AudioEmbedder(model_name=args.model, device=args.device)
    df_embeddings = embedder.build_embeddings_dataframe(
        df_splits,
        batch_size=args.batch_size,
        path_col="abs_path",   # Nova opção
    )

    # Salva os embeddings em uma nova pasta
    base_path = "results_embedder"
    os.makedirs(base_path, exist_ok=True)
    emb_path = f"{base_path}/embeddings.csv"
    df_embeddings.to_csv(emb_path, index=False)
    print(f"Embeddings salvos em {emb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai embeddings dos áudios e salva para ML futuro")
    parser.add_argument(
        "--dataset", type=str, default="../data/data_splits.csv",
        help="Caminho para o CSV com os áudios"
    )
    parser.add_argument(
        "--model", type=str, default="facebook/hubert-base-ls960",
        help="Nome do modelo HuggingFace"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Tamanho do batch para extração"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Dispositivo: 'cuda' ou 'cpu'. Se None, detecta automaticamente"
    )
    args = parser.parse_args()
    main(args)