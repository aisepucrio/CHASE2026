import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Importa modelos extras, se disponíveis
try:
    from lightgbm import LGBMClassifier
    has_lightgbm = True
except ImportError:
    has_lightgbm = False

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False

try:
    from catboost import CatBoostClassifier
    has_catboost = True
except ImportError:
    has_catboost = False

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, le):
    print(f"\nTreinando o modelo {model_name}...")
    model.fit(X_train, y_train)
    print(f"Treinamento do {model_name} concluído.")

    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n--- Resultados ({model_name}) ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (Ponderada): {precision:.4f}")
    print(f"Recall (Ponderado): {recall:.4f}")

    present = np.union1d(np.unique(y_test), np.unique(y_pred))
    all_lbls = np.arange(len(le.classes_))
    missing = sorted(set(all_lbls) - set(present))
    if missing:
        ausentes = ", ".join(le.classes_[missing])
        print(f"[AVISO] As seguintes classes não aparecem no teste e/ou previsão: {ausentes}")

    print("\nRelatório de Classificação Detalhado (todas as classes):")
    print(classification_report(
        y_test, y_pred,
        labels=all_lbls,
        target_names=le.classes_,
        zero_division=0
    ))

def main(args):
    try:
        df_embeddings = pd.read_csv(args.embeddings_file)
        print(f"Arquivo de embeddings '{args.embeddings_file}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo de embeddings '{args.embeddings_file}' não foi encontrado.")
        print("Por favor, execute o script 'runEmbedder.py' primeiro para gerar os embeddings.")
        return

    feature_cols = [col for col in df_embeddings.columns if col not in ["Emotion", "Split", "File Path"]]
    X = df_embeddings[feature_cols]
    y = df_embeddings["Emotion"]

    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Shape treino: {X_train.shape}")
    print("Distribuição no treino:\n", pd.Series(y_train_str).value_counts())
    print("Distribuição no teste:\n", pd.Series(y_test_str).value_counts())

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test  = le.transform(y_test_str)

    # Modelos a testar
    models = [
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ]
    if has_lightgbm:
        models.append(("LightGBM", LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced')))
    else:
        print("LightGBM não instalado, pulando.")
    if has_xgboost:
        models.append(("XGBoost", XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss')))
    else:
        print("XGBoost não instalado, pulando.")
    if has_catboost:
        models.append(("CatBoost", CatBoostClassifier(iterations=100, random_seed=42, verbose=0, class_weights='Balanced')))
    else:
        print("CatBoost não instalado, pulando.")

    # Treina e avalia todos
    for model_name, model in models:
        train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, le)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelos de ML clássicos com embeddings de áudio.")
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="results_embedder/embeddings.csv",
        help="Caminho para o arquivo CSV de embeddings gerado pelo runEmbedder.py"
    )
    args = parser.parse_args()
    main(args)