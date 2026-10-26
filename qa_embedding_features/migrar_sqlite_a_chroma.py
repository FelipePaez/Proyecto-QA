#!/usr/bin/env python3
"""Import existing SQLite embedding records into the project's ChromaDB store."""
import argparse
import sqlite3
import struct
from pathlib import Path

from vector_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parent / "resultados" / "embeddings.sqlite3")
    parser.add_argument("--destination", type=Path,
                        default=Path(__file__).resolve().parent / "resultados" / "chroma_db")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"No existe el archivo de respaldo SQLite: {args.source}")
    inserted = 0
    with sqlite3.connect(args.source) as db:
        records = db.execute("SELECT embedding_id, model, dimensions, kind, vector FROM embeddings ORDER BY model, dimensions")
        group = None
        store = None
        try:
            for embedding_id, model, dimensions, kind, blob in records:
                dimensions = int(dimensions)
                current_group = (model, dimensions)
                if current_group != group:
                    if store:
                        store.close()
                    store = ChromaVectorStore(args.destination, model, dimensions)
                    group = current_group
                vector = list(struct.unpack(f"<{dimensions}f", blob))
                store.put(embedding_id, vector, kind)
                inserted += 1
                if inserted % 256 == 0:
                    print(f"Migrados {inserted} embeddings...")
        finally:
            if store:
                store.close()
    print(f"Migración terminada: {inserted} embeddings en {args.destination}")


if __name__ == "__main__":
    main()
