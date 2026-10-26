#!/usr/bin/env python3
"""Prepare anonymized answer rows from the ROARS source CSV."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "originales" / "roars_dataset" / "ROARS.csv"
DEFAULT_OUTPUT = ROOT / "originales" / "adicionales" / "ROARS_answered.csv"
FIELDS = ["source_record_id", "question_source_id", "story", "question_num",
          "story_text", "passage_text", "question_text", "student_answer",
          "2class_human_groundtruth", "3class_human_groundtruth"]

def prepare(input_path: Path = DEFAULT_SOURCE, output_path: Path = DEFAULT_OUTPUT) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with input_path.open(encoding="utf-8-sig", newline="") as src, output_path.open("w", encoding="utf-8-sig", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(reader, start=1):
            question = (row.get("question_text") or "").strip()
            answer = (row.get("student_answer") or "").strip()
            passage = (row.get("passage_text") or "").strip()
            if not question or not answer:
                skipped += 1
                continue
            question_key = ":".join((row.get("story", ""), row.get("question_num", ""), question))
            writer.writerow({
                "source_record_id": f"ROARS-row-{index:04d}",
                "question_source_id": question_key,
                "story": row.get("story", ""),
                "question_num": row.get("question_num", ""),
                "story_text": row.get("story_text", ""),
                "passage_text": passage,
                "question_text": question,
                "student_answer": answer,
                "2class_human_groundtruth": (row.get("2class_human_groundtruth") or "").strip(),
                "3class_human_groundtruth": (row.get("3class_human_groundtruth") or "").strip(),
            })
            kept += 1
    print(f"Fuente: {input_path}")
    print(f"Salida derivada: {output_path}")
    print(f"Respuestas conservadas: {kept}; filas vacías/ inválidas excluidas: {skipped}")
    return kept, skipped

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.input, args.output)

if __name__ == "__main__":
    main()
