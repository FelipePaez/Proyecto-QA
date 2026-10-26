#!/usr/bin/env python3
"""Read one vector from the persistent ChromaDB store."""
import argparse
import json
from pathlib import Path

from vector_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_id", help="Vector reference from vista_respuestas.jsonl")
    parser.add_argument("--store", type=Path, default=Path(__file__).resolve().parent / "resultados" / "chroma_db",
                        help="Carpeta local de ChromaDB (default: resultados/chroma_db)")
    parser.add_argument("--output", type=Path, help="Write JSON to a file instead of the terminal")
    args = parser.parse_args()
    store = ChromaVectorStore(args.store)
    result = store.get_record(args.embedding_id)
    store.close()
    if result is None:
        parser.error(f"No se encontró el vector {args.embedding_id}")
    output = json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
