import json
import os
import time
import glob
from pathlib import Path
from tqdm import tqdm
import ollama

OLLAMA_MODEL = "mixtral:8x7b"
DATASET_FOLDER = "dataset"
RESULTS_FOLDER = f"results_no_context/{OLLAMA_MODEL.replace(':', '_')}/"

ollama_client = ollama.Client()

def get_system_prompt() -> str:
    return \
"""
Você é um robô construído para identificar emoções em trechos transcritos de reuniões. 

Esses trechos estão escritos em português brasileiro.

Cada trecho pode ser classificado como apenas uma dessas opções:
	- felicidade
	- tristeza
	- medo
	- raiva
	- surpresa
	- desgosto
	- neutro

Classifique a mensagem como "neutro" quando houver ausência de emoção.

Para cada trecho, dê também uma curta justificativa para sua classificação. Sua justificativa também deverá ser escrita em português brasileiro.

A resposta deve ser feita no seguinte formato, e deve conter apenas o JSON da resposta:

{
    "emoção": "<principal emoção>",
    "justificativa": "explicação curta em português brasileiro"
}
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
        {"role": "user", "content": \
f"""
Classifique o seguinte trecho:

Trecho da mensagem: {text}
JSON de resposta: 
""" 
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                format="json", # Comment if using gpt-oss
                stream=False,
                think=False, # True if using gpt-oss
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
    
    os.makedirs(os.path.dirname(RESULTS_FOLDER), exist_ok=True)
    
    all_results = {}
    
    for entry in tqdm(dataset, desc="Analyzing emotions"):
        print(f"Processing: {entry['output_version']}/{entry['file_id']}")
        
        classification = ask_ollama(entry['content'])

        print(classification)
        
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
                emotion = segment_data.get('emoção', 'unknown')
                emotions[emotion] = emotions.get(emotion, 0) + 1
        
        for emotion, count in sorted(emotions.items()):
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            print(f"   {emotion}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
