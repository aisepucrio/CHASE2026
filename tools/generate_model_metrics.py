import json
import glob
import pandas as pd
import os
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from typing import Dict, List, Tuple, Any
import numpy as np

EMOTION_MAPPING = {
    'felicidade': 'Happiness',
    'tristeza': 'Sadness', 
    'medo': 'Fear',
    'raiva': 'Anger',
    'surpresa': 'Surprise',
    'desgosto': 'Disgust',
    'neutro': 'Neutral'
}

VALID_EMOTIONS_EN = set(EMOTION_MAPPING.values())

def normalize_emotion(emotion: str) -> str:

    if not emotion:
        return None
    
    emotion = emotion.strip().lower()
    
    if emotion in EMOTION_MAPPING:
        return EMOTION_MAPPING[emotion]
    
    emotion_capitalized = emotion.capitalize()
    if emotion_capitalized in VALID_EMOTIONS_EN:
        return emotion_capitalized
    
    variations = {
        'happy': 'Happiness',
        'happiness': 'Happiness',
        'sad': 'Sadness',
        'sadness': 'Sadness',
        'angry': 'Anger',
        'anger': 'Anger',
        'fear': 'Fear',
        'afraid': 'Fear',
        'surprised': 'Surprise',
        'surprise': 'Surprise',
        'disgust': 'Disgust',
        'disgusted': 'Disgust',
        'neutral': 'Neutral',
        'expectativa': 'Surprise',
        'ansiedade': 'Fear',
    }
    
    if emotion in variations:
        return variations[emotion]
    
    return None

def load_ground_truth(file_path: str) -> Dict[str, Dict[str, str]]:

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flattened = {}
    for output_name, segments in data.items():
        for segment_id, segment_data in segments.items():
            key = f"{output_name}_{segment_id}"
            flattened[key] = segment_data['principal_emocao_detectada']
    
    return data, flattened

def load_model_results(file_path: str) -> Dict[str, Dict[str, str]]:

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flattened = {}
    for output_name, segments in data.items():
        for segment_id, segment_data in segments.items():
            key = f"{output_name}_{segment_id}"
            emotion = segment_data.get('principal_emocao_detectada', '')
            normalized_emotion = normalize_emotion(emotion)
            flattened[key] = {
                'original': emotion,
                'normalized': normalized_emotion
            }
    
    return data, flattened

def calculate_metrics_for_output(ground_truth_output: Dict, model_output: Dict, output_name: str) -> Dict[str, float]:

    y_true = []
    y_pred = []
    hallucinations = 0
    total_predictions = 0
    
    for segment_id in ground_truth_output.keys():
        if segment_id in model_output:
            gt_emotion = ground_truth_output[segment_id]['principal_emocao_detectada']
            model_data = model_output[segment_id]
            
            if 'principal_emocao_detectada' in model_data:
                original_emotion = model_data['principal_emocao_detectada']
                normalized_emotion = normalize_emotion(original_emotion)
                
                total_predictions += 1
                
                if normalized_emotion is None:
                    hallucinations += 1
                    y_true.append(gt_emotion)
                    y_pred.append('HALLUCINATION')
                else:
                    y_true.append(gt_emotion)
                    y_pred.append(normalized_emotion)
    
    if len(y_true) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0, 
            'f1score': 0.0,
            'accuracy': 0.0,
            'hallucination_rate': 0.0,
            'total_samples': 0
        }
    
    y_true_clean = []
    y_pred_clean = []
    
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label != 'HALLUCINATION':
            y_true_clean.append(true_label)
            y_pred_clean.append(pred_label)
    
    if len(y_true_clean) == 0:
        precision = recall = f1score = accuracy = 0.0
    else:
        precision, recall, f1score, _ = precision_recall_fscore_support(
            y_true_clean, y_pred_clean, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
    
    hallucination_rate = hallucinations / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1score': f1score, 
        'accuracy': accuracy,
        'hallucination_rate': hallucination_rate,
        'total_samples': len(y_true)
    }

def calculate_overall_metrics(ground_truth_data: Dict, model_data: Dict) -> Dict[str, float]:
    """
    Calcula métricas overall considerando todos os outputs.
    """
    y_true = []
    y_pred = []
    hallucinations = 0
    total_predictions = 0
    
    for output_name in ground_truth_data.keys():
        if output_name in model_data:
            gt_output = ground_truth_data[output_name]
            model_output = model_data[output_name]
            
            for segment_id in gt_output.keys():
                if segment_id in model_output:
                    gt_emotion = gt_output[segment_id]['principal_emocao_detectada']
                    model_segment = model_output[segment_id]
                    
                    if 'principal_emocao_detectada' in model_segment:
                        original_emotion = model_segment['principal_emocao_detectada']
                        normalized_emotion = normalize_emotion(original_emotion)
                        
                        total_predictions += 1
                        
                        if normalized_emotion is None:
                            hallucinations += 1
                            y_true.append(gt_emotion)
                            y_pred.append('HALLUCINATION')
                        else:
                            y_true.append(gt_emotion)
                            y_pred.append(normalized_emotion)
    
    if len(y_true) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1score': 0.0,
            'accuracy': 0.0,
            'hallucination_rate': 0.0,
            'total_samples': 0
        }
    
    y_true_clean = []
    y_pred_clean = []
    
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label != 'HALLUCINATION':
            y_true_clean.append(true_label)
            y_pred_clean.append(pred_label)
    
    if len(y_true_clean) == 0:
        precision = recall = f1score = accuracy = 0.0
    else:
        precision, recall, f1score, _ = precision_recall_fscore_support(
            y_true_clean, y_pred_clean, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
    
    hallucination_rate = hallucinations / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1score': f1score,
        'accuracy': accuracy,
        'hallucination_rate': hallucination_rate,
        'total_samples': len(y_true)
    }

def parse_filename(filename: str) -> Tuple[str, str, str]:

    basename = os.path.basename(filename).replace('.json', '')
    
    pattern = r'(alldata|testdata)-(.+?)-(context|nocontext)'
    match = re.match(pattern, basename)
    
    if match:
        data_type = match.group(1)
        model_name = match.group(2)
        context_type = match.group(3)
        return model_name, context_type, data_type
    else:
        parts = basename.split('-')
        if len(parts) >= 3:
            return parts[1], parts[2], parts[0]
        else:
            return basename, 'unknown', 'unknown'

def main():
    results_dir = 'data/results'
    ground_truth_file = os.path.join(results_dir, 'resultado_manual.json')
    
    if not os.path.exists(ground_truth_file):
        print(f"Erro: Arquivo gabarito não encontrado: {ground_truth_file}")
        return
    
    print("Carregando arquivo gabarito...")
    ground_truth_data, ground_truth_flat = load_ground_truth(ground_truth_file)
    
    model_patterns = [
        'alldata-*-context.json',
        'alldata-*-nocontext.json', 
        'testdata-*-context.json',
        'testdata-*-nocontext.json'
    ]
    
    model_files = []
    for pattern in model_patterns:
        files = glob.glob(os.path.join(results_dir, pattern))
        model_files.extend(files)
    
    if not model_files:
        print(f"Nenhum arquivo de resultado encontrado em {results_dir}")
        print("Formatos esperados: alldata-modelo-context.json, testdata-modelo-nocontext.json, etc.")
        return
    
    print(f"Encontrados {len(model_files)} arquivos de resultado de modelos")
    
    results = []
    
    available_outputs = list(ground_truth_data.keys())
    print(f"Outputs disponíveis: {available_outputs}")
    
    for model_file in model_files:
        print(f"\nProcessando: {os.path.basename(model_file)}")
        
        try:
            model_name, context_type, data_type = parse_filename(model_file)
            
            model_data, model_flat = load_model_results(model_file)
            
            row = {
                'model_name': model_name,
                'context': context_type,
                'data_type': data_type
            }
            
            for output_name in available_outputs:
                if output_name in ground_truth_data and output_name in model_data:
                    metrics = calculate_metrics_for_output(
                        ground_truth_data[output_name], 
                        model_data[output_name], 
                        output_name
                    )
                    
                    for metric_name, value in metrics.items():
                        if metric_name != 'total_samples':
                            col_name = f"{metric_name}-{output_name}"
                            row[col_name] = round(value, 4)
                else:
                    for metric_name in ['precision', 'recall', 'f1score', 'accuracy', 'hallucination_rate']:
                        col_name = f"{metric_name}-{output_name}"
                        row[col_name] = 0.0
            
            overall_metrics = calculate_overall_metrics(ground_truth_data, model_data)
            for metric_name, value in overall_metrics.items():
                if metric_name != 'total_samples':
                    col_name = f"{metric_name}-overall"
                    row[col_name] = round(value, 4)
            
            results.append(row)
            print(f"  ✓ Processado com sucesso")
            
        except Exception as e:
            print(f"  ✗ Erro ao processar {model_file}: {str(e)}")
            continue
    
    if not results:
        print("Nenhum resultado foi processado com sucesso.")
        return
    
    df = pd.DataFrame(results)
    base_cols = ['model_name', 'context', 'data_type']
    overall_cols = [col for col in df.columns if col.endswith('-overall')]
    overall_cols.sort()
    output_cols = [col for col in df.columns if not col.endswith('-overall') and col not in base_cols]
    output_cols.sort()
    final_cols = base_cols + overall_cols + output_cols
    df = df[final_cols]
    output_file = os.path.join(results_dir, 'model_evaluation_metrics.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Métricas salvas em: {output_file}")
    print(f"Total de modelos avaliados: {len(results)}")
    print("\nPreview dos resultados:")
    print(df[base_cols + [col for col in overall_cols]].to_string())

if __name__ == "__main__":
    main()