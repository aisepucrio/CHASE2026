import json
import time
import glob
from pathlib import Path
from tqdm import tqdm
import ollama

OLLAMA_MODEL = "qwen3:32b"
DATASET_FOLDER = "data"
RESULTS_FOLDER = "emotion_results"

ollama_client = ollama.Client()

def get_system_prompt() -> str:
    return """
    Analise o sentimento e as emoções no texto fornecido.

    Responda APENAS com um objeto JSON no seguinte formato exato:
    {
        "principal_emocao_detectada": "felicidade|tristeza|medo|raiva|surpresa|desgosto|neutro",
        "nivel_confianca": "0-100%",
        "explicacao_breve": "explicação em português brasileiro"
    }
    
    Regras:
    - Responda APENAS o JSON, sem texto adicional
    - Use apenas as emoções listadas acima
    - Nível de confiança deve ser um número de 0 a 100 seguido de %
    - Explicação deve ser breve e em português brasileiro
    - Não use aspas duplas dentro dos valores
    """

def load_text_files(folder: str) -> list:
    dataset = []
    
    output_folders = sorted(glob.glob(f"{folder}/**/output_v*"))
    
    for output_folder in output_folders:
        output_name = Path(output_folder).name
        txt_files = sorted(glob.glob(f"{output_folder}/*.txt"))
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    dataset.append({
                        "file_id": Path(txt_file).stem,
                        "output_version": output_name,
                        "content": content,
                        "file_path": txt_file
                    })
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
    
    return dataset

def ask_ollama(text: str, max_retries: int = 3) -> str | None:

    system_prompt = get_system_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    for attempt in range(max_retries):
        try:
            response = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                format="json",
                stream=False,
                options={
                    "temperature": 0,
                    "num_predict": 200,
                },
            )
            return response['message']['content']
        except Exception as e:
            print(f"Ollama request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f"Ollama failed after {max_retries} attempts.")
    return None

def main():

    print("Loading text files...")
    dataset = load_text_files(DATASET_FOLDER)
    
    if not dataset:
        print("no text files found")
        return
    
    print(f"found {len(dataset)}/782 text files")
    
    Path(RESULTS_FOLDER).mkdir(exist_ok=True)
    
    results_by_version = {}
    all_results = {}
    
    for entry in tqdm(dataset, desc="Analyzing emotions"):
        print(f"Processing: {entry['output_version']}/{entry['file_id']}")
        
        classification = ask_ollama(entry['content'])
        
        if classification is None:
            print(f"no response for {entry['file_id']}, skipping.")
            continue
        
        try:
            emotion_data = json.loads(classification)
            emotion_data['text_content'] = entry['content']
            emotion_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

            if entry['output_version'] not in all_results:
                all_results[entry['output_version']] = {}

            all_results[entry['output_version']][entry['file_id']] = emotion_data

        except json.JSONDecodeError as e:
            print(f"failed to parse JSON for {entry['file_id']}: {e}")
            print(f"raw response: {classification[:100]}...")
            continue
    
    for version, version_results in results_by_version.items():
        version_file = Path(RESULTS_FOLDER) / f"{version}.json"
        with version_file.open("w", encoding='utf-8') as f:
            json.dump(version_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {version}: {len(version_results)} files")
    
    for version, version_results in results_by_version.items():
        version_num = version.replace('output_', '')
        all_results[version_num] = version_results
    
    all_file = Path(RESULTS_FOLDER) / f"all_{OLLAMA_MODEL.replace(":", "_")}.json"
    with all_file.open("w", encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nanalysis complete")
    
    total_files = sum(len(version_results) for version_results in all_results.values())

    if all_results:
        print("\nemotion status:")
        emotions = {}
        
        for version_data in all_results.values():
            for segment_data in version_data.values():
                emotion = segment_data.get('principal_emocao_detectada', 'unknown')
                emotions[emotion] = emotions.get(emotion, 0) + 1
        
        for emotion, count in sorted(emotions.items()):
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            print(f"   {emotion}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
