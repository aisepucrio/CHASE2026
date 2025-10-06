"""Transcrição de áudios segmentados usando Google Gemini.

Percorre todos os diretórios output_v*/ em data/recordings/by_15s_segment
e gera transcrições texto em data/transcriptions/by_15s_segment/<mesmo_output>/

Uso rápido (Windows PowerShell):
  pip install -r requirements.txt
  python gemini_transcriptor.py

Requisitos: Definir a chave em gemini_config.py (GEMINI_API_KEY)
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Iterable

try:
	import google.generativeai as genai
except ImportError:  # Mensagem amigável caso dependência falte
	print("Dependência 'google-generativeai' não instalada. Rode: pip install -r requirements.txt", file=sys.stderr)
	sys.exit(1)

try:
	from gemini_config import GEMINI_API_KEY
except Exception as e:  # noqa
	print("Erro ao importar GEMINI_API_KEY de gemini_config.py", file=sys.stderr)
	raise


BASE_DIR = Path(__file__).parent
SEGMENTS_DIR = BASE_DIR / "data" / "recordings" / "by_15s_segment"
TRANSCRIPT_BASE_DIR = BASE_DIR / "data" / "transcriptions" / "by_15s_segment"

MODEL_NAME = "gemini-1.5-flash-latest"  # simples e rápido para transcrição


def listar_subdirs_output(root: Path) -> Iterable[Path]:
	"""Retorna subdiretórios tipo output_vX/ contendo segmentos .m4a"""
	if not root.exists():
		return []
	for p in sorted(root.iterdir()):
		if p.is_dir() and p.name.startswith("output_v"):
			yield p


def inicializar_modelo():
	genai.configure(api_key=GEMINI_API_KEY)
	return genai.GenerativeModel(MODEL_NAME)


def transcrever_arquivo(model, audio_path: Path, idioma: str = "pt") -> str:
	"""Envia o áudio para o modelo e retorna texto transcrito.

	Estratégia simples: prompt sucinto em português + arquivo de áudio.
	"""
	# A API aceita media (file) na chamada generate_content
	prompt = (
		"Você é um transcritor. Transcreva fielmente o áudio em português "
		"(pt-BR), mantendo pontuação natural e sem inventar conteúdo."\
	)
	try:
		# upload do arquivo (cacheia para reutilizar se quiser, mas aqui simples)
		uploaded = genai.upload_file(path=str(audio_path))
		resposta = model.generate_content([
			prompt,
			uploaded,
		])
		texto = resposta.text or ""
		return texto.strip()
	finally:
		# Opcional: poderia deletar arquivo remoto, porém a SDK ainda não expõe fácil.
		pass


def salvar_texto(destino: Path, conteudo: str):
	destino.parent.mkdir(parents=True, exist_ok=True)
	destino.write_text(conteudo, encoding="utf-8")


def main():
	print("== Transcrição Gemini (segmentos 15s) ==")
	if not SEGMENTS_DIR.exists():
		print(f"Diretório de segmentos não encontrado: {SEGMENTS_DIR}")
		return

	model = inicializar_modelo()
	relatorio = []

	for output_dir in listar_subdirs_output(SEGMENTS_DIR):
		rel_out_name = output_dir.name  # ex: output_v1
		destino_dir = TRANSCRIPT_BASE_DIR / rel_out_name
		print(f"Processando {rel_out_name} ...")

		segmentos = sorted(output_dir.glob("segment_*.m4a"))
		if not segmentos:
			print("  (sem arquivos .m4a)")
			continue

		for idx, seg_path in enumerate(segmentos, start=1):
			txt_path = destino_dir / (seg_path.stem + ".txt")
			if txt_path.exists():
				# pular se já existe (idempotência)
				continue
			try:
				texto = transcrever_arquivo(model, seg_path)
				if not texto:
					texto = "[VAZIO]"
				salvar_texto(txt_path, texto)
				relatorio.append({
					"arquivo": str(seg_path.relative_to(BASE_DIR)),
					"transcricao": str(txt_path.relative_to(BASE_DIR)),
					"ok": True,
				})
				print(f"  [{idx}/{len(segmentos)}] OK {seg_path.name}")
				# pequeno intervalo para não saturar limites
				time.sleep(0.3)
			except Exception as e:  # noqa
				relatorio.append({
					"arquivo": str(seg_path.relative_to(BASE_DIR)),
					"erro": str(e),
					"ok": False,
				})
				print(f"  ERRO {seg_path.name}: {e}")
				# Espera breve antes de seguir
				time.sleep(1)

	# salvar log JSON
	if relatorio:
		log_path = TRANSCRIPT_BASE_DIR / "relatorio_transcricao.json"
		log_path.parent.mkdir(parents=True, exist_ok=True)
		log_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"Relatório salvo em: {log_path}")
	print("Concluído.")


if __name__ == "__main__":
	main()

