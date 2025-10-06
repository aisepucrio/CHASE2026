import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main(args):
    try:
        df_embeddings = pd.read_csv(args.embeddings_file)
        print(f"Arquivo de embeddings '{args.embeddings_file}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo de embeddings '{args.embeddings_file}' não foi encontrado.")
        print("Por favor, execute o script 'runEmbedder.py' primeiro para gerar os embeddings.")
        return

    # Seleciona colunas de features e target
    feature_cols = [col for col in df_embeddings.columns if col not in ["Emotion", "Split", "File Path"]]
    X = df_embeddings[feature_cols]
    y = df_embeddings["Emotion"]

    # Split estratificado
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Shape treino: {X_train.shape}")
    print("Distribuição no treino:\n", pd.Series(y_train_str).value_counts())
    print("Distribuição no teste:\n", pd.Series(y_test_str).value_counts())

    # Codifica os rótulos
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test  = le.transform(y_test_str)

    # Treina modelo
    print("\nTreinando o modelo RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Treinamento concluído.")

    # Faz predições e avalia
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n--- Resultados da Avaliação no Conjunto de Teste ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Ponderada): {precision:.4f}")
    print(f"Recall (Ponderado): {recall:.4f}")

    # Aviso de classes ausentes no teste/previsão
    present = np.union1d(np.unique(y_test), np.unique(y_pred))
    all_lbls = np.arange(len(le.classes_))
    missing = sorted(set(all_lbls) - set(present))
    if missing:
        ausentes = ", ".join(le.classes_[missing])
        print(f"\n[AVISO] As seguintes classes não aparecem no teste e/ou previsão: {ausentes}")

    # Relatório completo
    print("\nRelatório de Classificação Detalhado (todas as classes):")
    print(classification_report(
        y_test,
        y_pred,
        labels=all_lbls,
        target_names=le.classes_,
        zero_division=0
    ))
    print("Shape treino:", X_train.shape)
    print("Distribuição:", pd.Series(y_train_str).value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo de ML clássico com embeddings de áudio.")
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="results_embedder_audio_externo/embeddings_hubert_base_ls960.csv",
        help="Caminho para o arquivo CSV de embeddings gerado pelo runEmbedder.py"
    )
    args = parser.parse_args()
    main(args)
