import json
import time
import glob
from pathlib import Path
from tqdm import tqdm
from bisect import bisect_left
import ollama

OLLAMA_MODEL = "qwen3:32b"
TARGET_DATASET_FOLDER = "data/transcriptions/target"
CONTEXT_DATASET_FOLDER = "data/transcriptions/context"
RESULTS_FOLDER = f"emotion_results/{OLLAMA_MODEL}"
AMOUNT_CONTEXT = 15 

ollama_client = ollama.Client()

def get_system_prompt() -> str:
     
    prompt = f"""Você é um robô que classifica trechos transcritos de reuniões.
Para isso, você receberá um trecho alvo para classificação e um contexto anterior.
Utilize o contexto anterior da reunião para compreender melhor as nuances do trecho alvo.
Em seguida, Classifique o trecho da reunião em um dos seis emoções: felicidade, tristeza, medo, raiva, surpresa ou desgosto.
No caso de dúvida ou ausência de emoções, classifique como neutro.

Responda com um objeto JSON no seguinte formato:
    {
        "principal_emocao_detectada": "felicidade|tristeza|medo|raiva|surpresa|desgosto|neutro",
        "nivel_confianca": "0-100%",
        "explicacao_breve": "breve explicação em português brasileiro"
    }

    Regras:
    - Responda APENAS o JSON, sem texto adicional
    - Use apenas as emoções listadas acima
    - Nível de confiança deve ser um número de 0 a 100 seguido de %
    - Explicação deve ser breve e em português brasileiro
    - Não use aspas duplas dentro dos valores
    
    """
     
    return prompt

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

def build_user_message(target_id: str, target_text: str, context_segments: list[dict]) -> str:

    parts = []
    if context_segments:
        parts.append(f"Contexto anterior (últimos {len(context_segments)} segmentos):")
        for seg in context_segments:
            parts.append(f"[{seg['file_id']}] {seg['content']}")
    else:
        parts.append("(Sem contexto anterior disponível)")
    parts.append("")
    parts.append(f"Trecho alvo [{target_id}]:")
    parts.append(target_text)
    parts.append("")
    parts.append("Classifique SOMENTE o trecho alvo conforme as instruções.")
    return "\n".join(parts)


def ask_ollama(target_id: str, target_text: str, context_segments: list[dict], max_retries: int = 3) -> str | None:
    system_prompt = get_system_prompt()
    user_content = build_user_message(target_id, target_text, context_segments)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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

def _extract_num(file_id: str) -> int:
    try:
        return int(file_id.split('_')[-1])
    except Exception:
        return -1


def main():

    print("Loading text files...")
    dataset_target = load_text_files(TARGET_DATASET_FOLDER)
    dataset_context = load_text_files(CONTEXT_DATASET_FOLDER)
    
    if not dataset_target:
        print("no target text files found")
        return
    
    print(f"found target={len(dataset_target)} | context={len(dataset_context)} text files")
    
    Path(RESULTS_FOLDER).mkdir(exist_ok=True)
    
    grouped_target: dict[str, list[dict]] = {}
    for entry in dataset_target:
        grouped_target.setdefault(entry['output_version'], []).append(entry)

    grouped_context: dict[str, list[dict]] = {}
    for entry in dataset_context:
        grouped_context.setdefault(entry['output_version'], []).append(entry)

    for entries in grouped_target.values():
        entries.sort(key=lambda x: _extract_num(x['file_id']))
    for entries in grouped_context.values():
        entries.sort(key=lambda x: _extract_num(x['file_id']))

    context_nums_by_version: dict[str, list[int]] = {}
    for version, entries in grouped_context.items():
        context_nums_by_version[version] = [_extract_num(e['file_id']) for e in entries]

    results_by_version: dict[str, dict] = {}

    for version, target_entries in grouped_target.items():
        print(f"\nProcessando versão: {version} ({len(target_entries)} segmentos-alvo)")
        results_by_version[version] = {}
        context_entries = grouped_context.get(version, [])
        context_nums = context_nums_by_version.get(version, [])

        for entry in tqdm(target_entries, desc=f"Analyzing {version}"):
            target_id = entry['file_id']
            target_num = _extract_num(target_id)

            if context_nums:
                pos = bisect_left(context_nums, target_num)
            else:
                pos = 0

            if AMOUNT_CONTEXT > 0 and pos > 0:
                start = max(0, pos - AMOUNT_CONTEXT)
                context_segments = context_entries[start:pos]
            else:
                context_segments = []

            print(f"Processing: {version}/{target_id} (context={len(context_segments)})")
            classification = ask_ollama(target_id, entry['content'], context_segments)

            if classification is None:
                print(f"no response for {target_id}, skipping.")
                continue

            try:
                emotion_data = json.loads(classification)
                emotion_data['text_content'] = entry['content']
                emotion_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                emotion_data['context_size'] = len(context_segments)
                emotion_data['context_ids'] = [c['file_id'] for c in context_segments]
                results_by_version[version][target_id] = emotion_data
            except json.JSONDecodeError as e:
                print(f"failed to parse JSON for {target_id}: {e}")
                print(f"raw response: {classification[:120]}...")
                continue

        version_file = Path(RESULTS_FOLDER) / f"{version}.json"
        with version_file.open("w", encoding='utf-8') as f:
            json.dump(results_by_version[version], f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {version}: {len(results_by_version[version])} files")

    all_results = results_by_version

    all_file = Path(RESULTS_FOLDER) / f'all_{OLLAMA_MODEL.replace(":", "_")}.json'
    with all_file.open("w", encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nanalysis complete")

    total_files = sum(len(v) for v in all_results.values())
    if total_files:
        print("\nemotion status:")
        emotions: dict[str, int] = {}
        for version_data in all_results.values():
            for seg_data in version_data.values():
                emotion = seg_data.get('principal_emocao_detectada', 'unknown')
                emotions[emotion] = emotions.get(emotion, 0) + 1
        for emotion, count in sorted(emotions.items()):
            perc = (count / total_files) * 100
            print(f"   {emotion}: {count} ({perc:.1f}%)")
        print(f"Total arquivos: {total_files}")
    else:
        print("Nenhum resultado processado.")

if __name__ == "__main__":
    main()
