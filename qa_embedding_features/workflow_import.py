"""Import arbitrary tabular/JSON QA files, create feature views, and export training sets."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NORMALIZER_DIR = ROOT.parent / "qa_dataset_project"
if str(NORMALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(NORMALIZER_DIR))
import normalizar_datasets as norm  # noqa: E402


def _flatten(value, prefix=""):
    fields = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, (dict, list)):
                fields.update(_flatten(child, name))
            else:
                fields[name] = child
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        fields.update(_flatten(value[0], prefix + ".*"))
    return fields


def _field_coverage(rows, fields):
    """Inspect every record so the researcher can review field coverage and examples."""
    if not rows or not fields:
        return []
    present = {field: 0 for field in fields}
    examples = {}
    types = {field: {} for field in fields}
    for row in rows:
        values = _flatten(row)
        for field in fields:
            value = values.get(field)
            if value is not None and (not isinstance(value, str) or value.strip()):
                present[field] += 1
                kind = _value_type(value)
                types[field][kind] = types[field].get(kind, 0) + 1
                if field not in examples:
                    examples[field] = str(value).replace("\n", " ")[:180]
    total = len(rows)
    return [{"field": field, "present": present[field], "total": total,
             "coverage": round(present[field] / total, 4),
             "type": max(types[field], key=types[field].get) if types[field] else "vacío",
             "example": examples.get(field, "")} for field in fields]


def _value_type(value):
    if isinstance(value, bool):
        return "booleano"
    if isinstance(value, int):
        return "entero"
    if isinstance(value, float):
        return "decimal"
    if isinstance(value, (dict, list)):
        return "estructura"
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"true", "false"}:
            return "booleano"
        try:
            int(text)
            return "entero"
        except ValueError:
            try:
                float(text)
                return "decimal"
            except ValueError:
                return "texto"
    return type(value).__name__


def inspect_file(path: Path, records_path=""):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        fmt = "tsv" if suffix == ".tsv" else "csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t" if fmt == "tsv" else ",")
            rows = list(reader)
        fields = reader.fieldnames or []
        return {"format": fmt, "records_path": "", "fields": fields,
                "count": len(rows), "preview": rows[:5],
                "field_coverage": _field_coverage(rows, fields)}
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8-sig") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        fields = sorted(set().union(*(_flatten(row).keys() for row in rows))) if rows else []
        return {"format": "jsonl", "records_path": "", "fields": fields,
                "count": len(rows), "preview": rows[:5],
                "field_coverage": _field_coverage(rows, fields)}
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    candidates = []
    def walk(value, path_parts=()):
        if isinstance(value, dict):
            for k, v in value.items(): walk(v, path_parts + (str(k),))
        elif isinstance(value, list):
            selector = ".".join(path_parts + ("*",))
            if value and all(isinstance(item, dict) for item in value):
                candidates.append(selector)
            if value:
                walk(value[0], path_parts + ("*",))
    walk(root)
    selector = records_path or (candidates[0] if candidates else "*")
    matches = norm.json_matches(root, selector)
    sample = [value for value, _ in matches[:5]]
    fields = sorted(set().union(*(_flatten(item).keys() for item, _ in matches))) if matches else []
    return {"format": "json", "records_path": selector, "record_paths": candidates,
            "fields": fields, "count": len(matches), "preview": sample,
            "field_coverage": _field_coverage([item for item, _ in matches], fields)}


def read_file_page(path: Path, offset=0, limit=50, records_path=""):
    """Read a single input-file page without returning or rendering the full dataset."""
    offset = max(0, int(offset))
    limit = max(1, min(200, int(limit)))
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            for index, row in enumerate(reader):
                if index >= offset:
                    rows.append(row)
                    if len(rows) >= limit:
                        break
        return rows
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        index = 0
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                if index >= offset:
                    rows.append(json.loads(line))
                    if len(rows) >= limit:
                        break
                index += 1
        return rows
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    selector = records_path or "*"
    matches = norm.json_matches(root, selector)
    return [record for record, _ in matches[offset:offset + limit]]


def write_normalized(input_path: Path, config: dict, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, contexts = norm.load_custom(input_path, config_path)
    if not rows:
        raise ValueError("No se encontraron filas con pregunta y respuesta. Revisa el mapeo de campos.")
    per_base = destination / "por_base"
    per_base.mkdir(exist_ok=True)
    dataset = config["dataset"]
    with (per_base / f"{dataset}.jsonl").open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (destination / "contextos.jsonl").open("w", encoding="utf-8") as f:
        for row in contexts.values(): f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (destination / "parrafos.jsonl").open("w", encoding="utf-8") as f:
        for ctx in contexts.values():
            for index, paragraph in enumerate(norm.paras(ctx.get("context_text") or "")):
                item = {"paragraph_id": norm.uid("para", f"{ctx['context_id']}:{index}", dataset),
                        "context_id": ctx["context_id"], "paragraph_index": index,
                        "paragraph_text": paragraph, "dataset": dataset,
                        "source_file": ctx.get("source_file"), "source_url": ctx.get("source_url")}
                f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"dataset": dataset, "records": len(rows), "contexts": len(contexts),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(), "config": config}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def safe_name(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_-").lower()
    if not value: raise ValueError("Indica un nombre válido para la base.")
    return value[:60]
