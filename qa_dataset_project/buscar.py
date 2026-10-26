#!/usr/bin/env python3
"""Busca preguntas y respuestas en las exportaciones JSONL/CSV del proyecto."""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "resultados"
RESULT_FIELDS = (
    "dataset", "context_id", "question_id", "answer_id", "score_id", "paragraph_id",
    "question_text", "answer_text", "reference_answer", "score", "score_scale",
    "is_unanswerable", "difficulty_questioner", "difficulty_answerer",
    "source_file", "source_record_id", "source_url", "context_text",
)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def rows_from(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def save_results(results: list[dict], args: argparse.Namespace) -> tuple[Path, Path]:
    """Guarda cada búsqueda en JSONL y CSV dentro de la carpeta de resultados."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    term = args.query or args.record_id or "busqueda"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", term).strip("_")[:40] or "busqueda"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = args.output_dir / f"busqueda_{timestamp}_{slug}"
    jsonl_path = base.with_suffix(".jsonl")
    csv_path = base.with_suffix(".csv")

    with jsonl_path.open("w", encoding="utf-8", newline="") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return jsonl_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Texto a buscar; ignora mayúsculas y tildes")
    parser.add_argument("--id", dest="record_id", help="Busca coincidencia exacta en context_id, question_id o answer_id")
    parser.add_argument("--dataset", help="Filtra por dataset, por ejemplo squad2_train o mohler_v1.0")
    parser.add_argument("--field", choices=("all", "question", "answer", "reference", "context"),
                        default="all", help="Campo de texto en el que buscar (default: all)")
    parser.add_argument("--limit", type=int, default=10, help="Máximo de resultados (default: 10)")
    parser.add_argument("--records", type=Path, default=DATA_DIR / "registros.jsonl")
    parser.add_argument("--contexts", type=Path, default=DATA_DIR / "contextos.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "busquedas",
                        help="Carpeta para guardar cada búsqueda (default: resultados/busquedas)")
    parser.add_argument("--context-chars", type=int, default=500,
                        help="Máximo de caracteres de contexto mostrados (default: 500)")
    args = parser.parse_args()
    if not args.query and not args.record_id:
        parser.error("Indica una frase o --id")
    if args.limit < 1:
        parser.error("--limit debe ser mayor que cero")

    contexts = {}
    if args.contexts.exists():
        for context in rows_from(args.contexts):
            contexts[context["context_id"]] = context

    query = fold(args.query) if args.query else None
    fields = {"question": "question_text", "answer": "answer_text",
              "reference": "reference_answer", "context": "context_text"}
    selected = list(fields) if args.field == "all" else [args.field]
    results = []
    for row in rows_from(args.records):
        if args.dataset and row.get("dataset", "").casefold() != args.dataset.casefold():
            continue
        if args.record_id:
            matched = args.record_id in (row.get("context_id"), row.get("question_id"), row.get("answer_id"))
        else:
            context = contexts.get(row.get("context_id"), {})
            matched = any(query in fold(str(value)) for name in selected
                          for value in [row.get(fields[name]), context.get("context_text") if name == "context" else None]
                          if value is not None)
        if not matched:
            continue
        context = contexts.get(row.get("context_id"), {})
        result = {key: row.get(key) for key in (
            "dataset", "context_id", "question_id", "answer_id", "score_id", "paragraph_id",
            "question_text", "answer_text", "reference_answer", "score", "score_scale",
            "is_unanswerable", "difficulty_questioner", "difficulty_answerer",
            "source_file", "source_record_id", "source_url")}
        result["context_text"] = context.get("context_text")
        if result["context_text"] and len(result["context_text"]) > args.context_chars:
            result["context_text"] = result["context_text"][:args.context_chars] + "…"
        results.append(result)
        if len(results) >= args.limit:
            break
    for result in results:
        # Escaped JSON remains printable in Windows consoles using legacy code pages.
        print(json.dumps(result, ensure_ascii=True))
    print(f"Resultados: {len(results)}")
    jsonl_path, csv_path = save_results(results, args)
    print(f"Guardados en:\n- {jsonl_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
