#!/usr/bin/env python3
"""Build per-answer embedding and cosine-similarity views from normalized QA data."""
from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parent
DEFAULT_NORMALIZED = ROOT.parent / "qa_dataset_project" / "resultados"
DEFAULT_OUTPUT = ROOT / "resultados"
MODEL_DEFAULT = "text-embedding-3-small"
DIMENSIONS_DEFAULT = 1536
BATCH_SIZE = 256
MAX_BATCH_CHARS = 150_000
MAX_CHARS = 24000
CSV_FIELDS = [
    "dataset", "context_id", "question_id", "answer_id", "score_id",
    "question_embedding_id", "answer_embedding_id", "context_centroid_embedding_id",
    "paragraph_nearest_question_id", "paragraph_nearest_question_embedding_id",
    "paragraph_nearest_answer_id", "paragraph_nearest_answer_embedding_id",
    "paragraph_nearest_question_text", "paragraph_nearest_answer_text",
    "sim_question_answer", "sim_question_context_centroid", "sim_answer_context_centroid",
    "sim_question_nearest_question_paragraph", "sim_question_nearest_answer_paragraph",
    "sim_answer_nearest_question_paragraph", "sim_answer_nearest_answer_paragraph",
    "target_answer", "target_score", "score_scale", "is_unanswerable",
    "source_file", "source_record_id", "source_answer_id",
]


def digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL inválido en {path.name}, línea {line_number}") from exc


def normalized_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "por_base").is_dir():
        return path
    if (path / "resultados" / "por_base").is_dir():
        return path / "resultados"
    raise FileNotFoundError(f"No encuentro resultados/por_base dentro de: {path}")


def paragraphs_from_context(text: str | None) -> list[str]:
    if not text or not text.strip():
        return []
    blocks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    result: list[str] = []
    for block in blocks:
        if len(block) <= MAX_CHARS:
            result.append(block)
            continue
        # Keep long passages within the embedding model's input limit while retaining text.
        sentences = re.split(r"(?<=[.!?])\s+", block)
        chunk = ""
        for sentence in sentences:
            if len(sentence) > MAX_CHARS:
                if chunk:
                    result.append(chunk)
                    chunk = ""
                for start in range(0, len(sentence), MAX_CHARS):
                    result.append(sentence[start:start + MAX_CHARS])
            elif not chunk:
                chunk = sentence
            elif len(chunk) + len(sentence) + 1 <= MAX_CHARS:
                chunk += " " + sentence
            else:
                result.append(chunk)
                chunk = sentence
        if chunk:
            result.append(chunk)
    return result


def load_api_key(api_key_file: Path | None) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if api_key_file is None:
        raise ValueError("Define OPENAI_API_KEY o indica --api-key-file con la ruta del archivo de clave.")
    # The file is treated only as secret material. Its contents are never logged or copied.
    for line in api_key_file.read_text(encoding="utf-8-sig").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        assignment = re.search(r"(?:OPENAI_API_KEY|API_KEY|OPENAI API KEY)\s*[:=]\s*(sk-[A-Za-z0-9_-]+)", candidate, re.I)
        if assignment:
            candidate = assignment.group(1)
        else:
            candidate = candidate.strip("\"'")
        if candidate.startswith("sk-"):
            return candidate
    raise ValueError("El archivo de clave está vacío.")


def request_embeddings(api_key: str, model: str, dimensions: int,
                       texts: list[str], retries: int = 6) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts, "dimensions": dimensions,
                          "encoding_format": "float"}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings", data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            data = sorted(body["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]
            if len(vectors) != len(texts):
                raise ValueError("La API devolvió una cantidad inesperada de embeddings.")
            return vectors
        except urllib.error.HTTPError as exc:
            # Never include request headers or the API key in diagnostics.
            if exc.code == 429 or exc.code >= 500:
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
            raise RuntimeError(f"La API de embeddings respondió HTTP {exc.code}.") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError("No se pudo conectar con la API de embeddings tras varios intentos.") from None
    raise RuntimeError("Falló la solicitud de embeddings.")


def ensure_text_embeddings(store: ChromaVectorStore, api_key: str, texts: dict[str, str]) -> int:
    missing = []
    existing = store.existing_ids(list(texts))
    for embedding_id, text in texts.items():
        if embedding_id not in existing:
            if not text.strip():
                continue
            if len(text) > MAX_CHARS:
                raise ValueError(f"Texto demasiado largo para una entrada de embedding ({len(text)} caracteres).")
            missing.append((embedding_id, text))
    tokens = 0
    batches = []
    start = 0
    while start < len(missing):
        end = start
        batch_chars = 0
        while end < len(missing) and end - start < BATCH_SIZE:
            text_chars = len(missing[end][1])
            if end > start and batch_chars + text_chars > MAX_BATCH_CHARS:
                break
            batch_chars += text_chars
            end += 1
        batches.append(missing[start:end])
        start = end
    completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for wave_start in range(0, len(batches), 4):
            wave = batches[wave_start:wave_start + 4]
            futures = [pool.submit(request_embeddings, api_key, store.model,
                                   store.dimensions, [text for _, text in batch])
                       for batch in wave]
            for batch, future in zip(wave, futures):
                vectors = future.result()
                store.put_many([embedding_id for embedding_id, _ in batch], vectors,
                               "text", [text for _, text in batch])
                store.commit()
                completed += len(batch)
                print(f"  embeddings: {completed}/{len(missing)} nuevos (lote guardado)",
                      file=sys.stderr, flush=True)
    return tokens


def cosine(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return None
    return dot / (norm_left * norm_right)


def make_centroid(store: ChromaVectorStore, context_id: str, paragraphs: list[dict],
                  vector_lookup: dict[str, list[float]] | None = None,
                  pending_centroids: dict[str, list[float]] | None = None):
    if not paragraphs:
        return None, None
    paragraph_embedding_ids = [store.text_id(p["paragraph_text"]) for p in paragraphs]
    centroid_id = store.centroid_id(context_id, paragraph_embedding_ids)
    cached = (vector_lookup or {}).get(centroid_id)
    if cached is None:
        cached = store.get(centroid_id)
    if cached is not None:
        if vector_lookup is not None:
            vector_lookup[centroid_id] = cached
        return centroid_id, cached
    vectors = [((vector_lookup or {}).get(embedding_id) or store.get(embedding_id))
               for embedding_id in paragraph_embedding_ids]
    if any(vector is None for vector in vectors):
        return None, None
    centroid = [sum(vector[i] for vector in vectors) / len(vectors)
                for i in range(store.dimensions)]
    if vector_lookup is not None:
        vector_lookup[centroid_id] = centroid
    if pending_centroids is not None:
        pending_centroids[centroid_id] = centroid
    else:
        store.put(centroid_id, centroid, "context_centroid",
                  document=f"Centroide del contexto {context_id}")
    return centroid_id, centroid


def process_dataset(dataset: str, normalized: Path, output: Path, store: ChromaVectorStore,
                    api_key: str, limit: int | None) -> tuple[int, int]:
    records_path = normalized / "por_base" / f"{dataset}.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"No se encontró la exportación normalizada: {records_path}")
    rows = list(read_jsonl(records_path))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return 0, 0

    context_ids = {row.get("context_id") for row in rows if row.get("context_id")}
    contexts = {item["context_id"]: item for item in read_jsonl(normalized / "contextos.jsonl")
                if item.get("context_id") in context_ids}
    paragraph_map: dict[str, list[dict]] = {context_id: [] for context_id in context_ids}
    paragraphs_file = normalized / "parrafos.jsonl"
    if paragraphs_file.exists():
        for paragraph in read_jsonl(paragraphs_file):
            if paragraph.get("context_id") in paragraph_map:
                paragraph_map[paragraph["context_id"]].append(paragraph)
    for context_id in context_ids:
        if not paragraph_map[context_id]:
            text = contexts.get(context_id, {}).get("context_text")
            paragraph_map[context_id] = [
                {"paragraph_id": "PAR-" + digest(f"{context_id}:{i}:{text}"), "paragraph_text": part}
                for i, part in enumerate(paragraphs_from_context(text), 1)
            ]

    texts: dict[str, str] = {}
    for row in rows:
        for value in (row.get("question_text"), row.get("answer_text")):
            if value is not None and str(value).strip():
                value = str(value)
                texts[store.text_id(value)] = value
    for paragraphs in paragraph_map.values():
        for paragraph in paragraphs:
            value = paragraph.get("paragraph_text") or ""
            if value.strip():
                texts[store.text_id(value)] = value
    ensure_text_embeddings(store, api_key, texts)

    dataset_dir = output / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    view_path = dataset_dir / "vista_respuestas.jsonl"
    answers_path = dataset_dir / "entrenamiento_respuestas.jsonl"
    grades_path = dataset_dir / "entrenamiento_calificaciones.jsonl"
    csv_path = dataset_dir / "vista_respuestas.csv"
    with view_path.open("w", encoding="utf-8", newline="\n") as view_file, \
         answers_path.open("w", encoding="utf-8", newline="\n") as answer_file, \
         grades_path.open("w", encoding="utf-8", newline="\n") as grade_file, \
         csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for batch_start in range(0, len(rows), 256):
          batch_rows = rows[batch_start:batch_start + 256]
          pending_centroids = {}
          needed_ids = []
          for batch_row in batch_rows:
              qtext = str(batch_row.get("question_text") or "")
              atext = str(batch_row.get("answer_text") or "")
              if qtext.strip():
                  needed_ids.append(store.text_id(qtext))
              if atext.strip():
                  needed_ids.append(store.text_id(atext))
              context_paragraphs = paragraph_map.get(batch_row.get("context_id"), [])
              paragraph_ids = [store.text_id(p.get("paragraph_text") or "")
                               for p in context_paragraphs if (p.get("paragraph_text") or "").strip()]
              needed_ids.extend(paragraph_ids)
              if paragraph_ids:
                  needed_ids.append(store.centroid_id(batch_row.get("context_id", ""), paragraph_ids))
          batch_vectors = store.get_many(needed_ids)
          for row in batch_rows:
            question = str(row.get("question_text") or "")
            answer = str(row.get("answer_text") or "")
            qid = store.text_id(question) if question else None
            aid = store.text_id(answer) if answer else None
            qvec = batch_vectors.get(qid) if qid else None
            avec = batch_vectors.get(aid) if aid else None
            paragraphs = paragraph_map.get(row.get("context_id"), [])
            valid_paragraphs = []
            for paragraph in paragraphs:
                text = paragraph.get("paragraph_text") or ""
                embedding_id = store.text_id(text) if text.strip() else None
                vector = batch_vectors.get(embedding_id) if embedding_id else None
                if vector is not None:
                    valid_paragraphs.append((paragraph, embedding_id, vector))

            nearest_q, nearest_a = None, None
            if qvec is not None and valid_paragraphs:
                nearest_q = max(valid_paragraphs, key=lambda item: cosine(qvec, item[2]))
            if avec is not None and valid_paragraphs:
                nearest_a = max(valid_paragraphs, key=lambda item: cosine(avec, item[2]))
            centroid_id, centroid = make_centroid(store, row.get("context_id", ""),
                                                  paragraphs, batch_vectors, pending_centroids)

            q_near_vec = nearest_q[2] if nearest_q else None
            a_near_vec = nearest_a[2] if nearest_a else None
            cosines = {
                "question_answer": cosine(qvec, avec),
                "question_context_centroid": cosine(qvec, centroid),
                "answer_context_centroid": cosine(avec, centroid),
                "question_nearest_question_paragraph": cosine(qvec, q_near_vec),
                "question_nearest_answer_paragraph": cosine(qvec, a_near_vec),
                "answer_nearest_question_paragraph": cosine(avec, q_near_vec),
                "answer_nearest_answer_paragraph": cosine(avec, a_near_vec),
            }
            nearest_q_obj = ({"paragraph_id": nearest_q[0].get("paragraph_id"),
                              "embedding_id": nearest_q[1], "paragraph_text": nearest_q[0]["paragraph_text"]}
                             if nearest_q else None)
            nearest_a_obj = ({"paragraph_id": nearest_a[0].get("paragraph_id"),
                              "embedding_id": nearest_a[1], "paragraph_text": nearest_a[0]["paragraph_text"]}
                             if nearest_a else None)
            ids = {key: row.get(key) for key in
                   ("context_id", "question_id", "answer_id", "score_id", "paragraph_id")}
            vector_refs = {
                "question_embedding_id": qid,
                "answer_embedding_id": aid,
                "context_centroid_embedding_id": centroid_id,
                "paragraph_nearest_question": nearest_q_obj,
                "paragraph_nearest_answer": nearest_a_obj,
            }
            # Candidate-answer features are for grading. Generation inputs exclude answer-derived data.
            grading_input = {"embedding_refs": vector_refs, "cosine_similarities": cosines}
            answer_input = {
                "question_embedding_id": qid,
                "context_centroid_embedding_id": centroid_id,
                "paragraph_nearest_question": nearest_q_obj,
            }
            target_answer = row.get("reference_answer") or row.get("answer_text")
            targets = {"answer_text": target_answer, "score": row.get("score"),
                       "score_scale": row.get("score_scale"),
                       "is_unanswerable": row.get("is_unanswerable")}
            view = {
                "dataset": dataset, "ids": ids,
                "source": {key: row.get(key) for key in ("source_file", "source_record_id", "source_answer_id", "source_url")},
                "texts": {"question": question, "candidate_answer": answer,
                          "paragraph_nearest_question": nearest_q_obj,
                          "paragraph_nearest_answer": nearest_a_obj},
                "embedding_refs": vector_refs,
                "cosine_similarities": cosines,
                "model_inputs": {"grading": grading_input, "answer_generation": answer_input},
                "targets": targets,
                "embedding_model": store.model,
                "embedding_dimensions": store.dimensions,
            }
            view_file.write(json.dumps(view, ensure_ascii=False, separators=(",", ":")) + "\n")
            if target_answer is not None:
                answer_file.write(json.dumps({"dataset": dataset, "ids": ids,
                    "input": answer_input, "target": target_answer}, ensure_ascii=False,
                    separators=(",", ":")) + "\n")
            if row.get("score") is not None:
                grade_file.write(json.dumps({"dataset": dataset, "ids": ids,
                    "input": grading_input, "target": row.get("score"),
                    "score_scale": row.get("score_scale")}, ensure_ascii=False,
                    separators=(",", ":")) + "\n")
            writer.writerow({
                "dataset": dataset, **ids,
                "question_embedding_id": qid, "answer_embedding_id": aid,
                "context_centroid_embedding_id": centroid_id,
                "paragraph_nearest_question_id": nearest_q_obj.get("paragraph_id") if nearest_q_obj else None,
                "paragraph_nearest_question_embedding_id": nearest_q_obj.get("embedding_id") if nearest_q_obj else None,
                "paragraph_nearest_answer_id": nearest_a_obj.get("paragraph_id") if nearest_a_obj else None,
                "paragraph_nearest_answer_embedding_id": nearest_a_obj.get("embedding_id") if nearest_a_obj else None,
                "paragraph_nearest_question_text": nearest_q_obj.get("paragraph_text") if nearest_q_obj else None,
                "paragraph_nearest_answer_text": nearest_a_obj.get("paragraph_text") if nearest_a_obj else None,
                **{f"sim_{key}": value for key, value in cosines.items()},
                "target_answer": target_answer, "target_score": row.get("score"),
                "score_scale": row.get("score_scale"), "is_unanswerable": row.get("is_unanswerable"),
                "source_file": row.get("source_file"), "source_record_id": row.get("source_record_id"),
                "source_answer_id": row.get("source_answer_id"),
            })
          if pending_centroids:
              centroid_ids = list(pending_centroids)
              store.put_many(centroid_ids,
                             [pending_centroids[key] for key in centroid_ids],
                             "context_centroid",
                             [f"Centroide {key}" for key in centroid_ids])
          print(f"  vista: {min(batch_start + len(batch_rows), len(rows))}/{len(rows)} filas",
                file=sys.stderr, flush=True)
    store.commit()
    return len(rows), sum(row.get("score") is not None for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED,
                        help="Proyecto normalizador o su carpeta resultados/ (default: proyecto QA previo)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="all", help="Nombre de base en por_base/ o all")
    parser.add_argument("--limit", type=int, help="Máximo de respuestas por base; útil para una ejecución de muestra")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS_DEFAULT)
    parser.add_argument("--api-key-file", type=Path, help="Archivo local que contiene la clave API; no se copia al proyecto")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit debe ser mayor que cero")
    if args.dimensions < 1:
        parser.error("--dimensions debe ser mayor que cero")
    try:
        normalized = normalized_root(args.normalized_dir)
        base_dir = normalized / "por_base"
        if args.dataset == "all":
            datasets = sorted(path.stem for path in base_dir.glob("*.jsonl"))
        else:
            datasets = [args.dataset]
        if not datasets:
            raise FileNotFoundError(f"No hay archivos JSONL por base en {base_dir}")
        api_key = load_api_key(args.api_key_file)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        store = ChromaVectorStore(args.output_dir / "chroma_db", args.model, args.dimensions)
        summary = {}
        try:
            for dataset in datasets:
                count, scored = process_dataset(dataset, normalized, args.output_dir, store,
                                                api_key, args.limit)
                summary[dataset] = {"responses": count, "with_score_target": scored,
                                    "model": args.model, "dimensions": args.dimensions}
                print(f"{dataset}: {count} respuestas procesadas; {scored} con calificación")
        finally:
            store.close()
        manifest_path = args.output_dir / "manifest.json"
        previous = {}
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
        datasets_manifest = dict(previous.get("datasets", {}))
        datasets_manifest.update(summary)
        manifest = {"source_normalized_dir": str(normalized), "datasets": datasets_manifest,
                    "vector_store": "ChromaDB PersistentClient",
                    "vector_store_path": "chroma_db",
                    "collection": ChromaVectorStore._collection_name(args.model, args.dimensions)}
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
