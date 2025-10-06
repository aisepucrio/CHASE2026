import os
import pandas as pd
import random
from collections import defaultdict

# Diretório base
base_dir = "../dataExternal"

def get_emotion_from_folder(folder_name):
    parts = folder_name.split('_', 1)
    if len(parts) > 1:
        return parts[1].capitalize()
    else:
        return folder_name.capitalize()

# Buscar todos os áudios por emoção
files_by_emotion = defaultdict(list)
for root, dirs, files in os.walk(base_dir):
    for fname in files:
        if fname.lower().endswith(('.wav', '.m4a', '.mp3')):
            folder = os.path.basename(root)
            emotion = get_emotion_from_folder(folder)
            rel_path = os.path.join(folder, fname).replace("/", "\\")
            files_by_emotion[emotion].append(rel_path)

# Split estratificado 70/30
rows = []
for emotion, files in files_by_emotion.items():
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * 0.7)
    for i, path in enumerate(files):
        split = "train" if i < n_train else "test"
        rows.append([path, emotion, split])

df = pd.DataFrame(rows, columns=["File Path", "Emotion", "Split"])
df.to_csv("audio_labels.csv", index=False, encoding="utf-8")
print("Arquivo salvo como audio_labels.csv")
print(df["Split"].value_counts())
print(df.head())
