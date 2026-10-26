#!/usr/bin/env python3
"""Normaliza SQuAD 2.0, CMU QA, Mohler y ROARS.

Solo usa la biblioteca estándar. Descarga los archivos oficiales si faltan,
conserva los originales en originales/, extrae a trabajo/ y escribe registros
JSONL/CSV con IDs deterministas y procedencia.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import tarfile
import urllib.request
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "resultados"
SOURCES = {
    "squad2_train.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "squad2_dev.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "cmu_qa_v1.2.tar.gz": "https://www.cs.cmu.edu/~ark/QA-data/data/Question_Answer_Dataset_v1.2.tar.gz",
    "mohler_short_answer_v1.0.tar.gz": "https://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v1.0.tar.gz",
    "mohler_short_answer_v2.0.zip": "https://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v2.0.zip",
    "roars_dataset/ROARS.csv": "https://raw.githubusercontent.com/owenhenkel/ROARS-dataset/main/ROARS.csv",
    "roars_dataset/ROARS_datadictionary.csv": "https://raw.githubusercontent.com/owenhenkel/ROARS-dataset/main/ROARS_datadictionary.csv",
}
DATASET_SOURCES = {
    "squad2_train": ("squad2_train.json",),
    "squad2_dev": ("squad2_dev.json",),
    "cmu_qa_v1.2": ("cmu_qa_v1.2.tar.gz",),
    "mohler_v1.0": ("mohler_short_answer_v1.0.tar.gz",),
    "mohler_v2.0": ("mohler_short_answer_v2.0.zip",),
    "roars_2class": ("roars_dataset/ROARS.csv", "roars_dataset/ROARS_datadictionary.csv"),
    "roars_3class": ("roars_dataset/ROARS.csv", "roars_dataset/ROARS_datadictionary.csv"),
}
BUILTIN_DATASETS = tuple(DATASET_SOURCES)
DATASET_TAGS = {
    "squad2_train": "SQTR", "squad2_dev": "SQDV",
    "cmu_qa_v1.2": "CMU12", "mohler_v1.0": "MH1", "mohler_v2.0": "MH2",
    "boolq_train": "BQTR", "boolq_dev": "BQDV",
    "roars_2class": "R2C", "roars_3class": "R3C",
}
FIELDS = ["dataset", "context_id", "question_id", "answer_id", "paragraph_id",
          "context_text", "context_kind", "question_text", "answer_text",
          "reference_answer", "score_id", "score", "score_scale", "is_unanswerable",
          "difficulty_questioner", "difficulty_answerer", "source_file",
          "source_record_id", "source_url", "source_answer_id"]


def id_hash(source: str) -> str:
    """Stable, compact hash; foreign-key columns carry the relationships."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:14]


def dataset_tag(dataset: str) -> str:
    if dataset in DATASET_TAGS:
        return DATASET_TAGS[dataset]
    slug = re.sub(r"[^A-Za-z0-9]+", "", dataset).upper()[:6] or "CUSTOM"
    # Keep custom dataset tags compact while separating similarly named datasets.
    return f"{slug}-{hashlib.sha256(dataset.encode('utf-8')).hexdigest()[:4].upper()}"


def context_uid(dataset: str, source: str) -> str:
    return f"C-{dataset_tag(dataset)}-{id_hash(dataset + ':' + source)}"


def question_uid(dataset: str, context_id: str, source: str) -> str:
    return f"P-{dataset_tag(dataset)}-{id_hash(context_id + ':' + source)}"


def answer_uid(dataset: str, question_id: str, source: str) -> str:
    return f"R-{dataset_tag(dataset)}-{id_hash(question_id + ':' + source)}"


def uid(kind: str, source: str, dataset: str) -> str:
    prefix = {"para": "PAR"}.get(kind, kind.upper())
    return f"{prefix}-{dataset_tag(dataset)}-{id_hash(source)}"


def clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def paras(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]


def fetch_sources(originals: Path, download: bool, source_names=None) -> None:
    originals.mkdir(parents=True, exist_ok=True)
    selected_names = source_names if source_names is not None else SOURCES
    for name in selected_names:
        url = SOURCES[name]
        target = originals / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size:
            continue
        if not download:
            raise FileNotFoundError(
                f"Falta el archivo local {target}. Añádelo a originales/ o ejecuta con --download."
            )
        request = urllib.request.Request(url, headers={"User-Agent": "dataset-normalizer/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
        if not target.stat().st_size:
            raise ValueError(f"Descarga vacía: {url}")


def download_custom_source(url: str, filename: str | None, originals: Path) -> Path:
    """Download one user-specified data file into originales/adicionales/."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("La URL debe comenzar con http:// o https://")
    raw_name = filename or Path(urllib.parse.unquote(parsed.path)).name
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(raw_name).name).strip("._")
    if not safe_name or Path(safe_name).suffix.lower() not in {".csv", ".tsv", ".json", ".jsonl"}:
        raise ValueError("Indica --filename con extensión .csv, .tsv, .json o .jsonl")
    destination_dir = originals / "adicionales"
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / safe_name
    temporary = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-normalizer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if not temporary.stat().st_size:
            raise ValueError(f"La descarga está vacía: {url}")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def extract(originals: Path, work: Path, datasets=None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    selected = set(BUILTIN_DATASETS if datasets is None else datasets)
    for name, dataset in (("cmu_qa_v1.2.tar.gz", "cmu_qa_v1.2"),
                          ("mohler_short_answer_v1.0.tar.gz", "mohler_v1.0")):
        if dataset not in selected:
            continue
        destination = work / Path(name).stem.replace(".tar", "")
        if destination.exists():
            continue
        destination.mkdir(parents=True)
        with tarfile.open(originals / name, "r:gz") as archive:
            # Refuse path traversal in archives.
            base = destination.resolve()
            for member in archive.getmembers():
                if not (destination / member.name).resolve().is_relative_to(base):
                    raise ValueError(f"Ruta insegura en el archivo: {member.name}")
            archive.extractall(destination)
    destination = work / "mohler_short_answer_v2.0"
    if "mohler_v2.0" in selected and not destination.exists():
        destination.mkdir(parents=True)
        with zipfile.ZipFile(originals / "mohler_short_answer_v2.0.zip") as archive:
            base = destination.resolve()
            for name in archive.namelist():
                if not (destination / name).resolve().is_relative_to(base):
                    raise ValueError(f"Ruta insegura en el archivo: {name}")
            archive.extractall(destination)


def record(dataset: str, source_id: str, qtext: str, atext: str | None,
           context: str | None, context_kind: str, source_file: str,
           source_url: str, answer_source_id: str = "0", score=None,
           score_scale=None, reference=None, paragraph_id=None,
           is_unanswerable=False, dq=None, da=None, context_source_id=None,
           question_source_id=None) -> dict:
    cid = context_uid(dataset, context_source_id or source_id)
    qid = question_uid(dataset, cid, dataset + ":" + (question_source_id or source_id))
    aid = answer_uid(dataset, qid, dataset + ":" + source_id + ":" + answer_source_id)
    return dict(dataset=dataset, context_id=cid, question_id=qid, answer_id=aid,
                paragraph_id=paragraph_id, context_text=context,
                context_kind=context_kind, question_text=clean(qtext), answer_text=clean(atext),
                reference_answer=clean(reference), score=score, score_scale=score_scale,
                is_unanswerable=bool(is_unanswerable), difficulty_questioner=dq,
                difficulty_answerer=da, source_file=source_file, source_record_id=source_id,
                source_url=source_url, source_answer_id=str(answer_source_id))


def load_squad(originals: Path, splits=None) -> list[dict]:
    rows = []
    for split in (splits or ("train", "dev")):
        name = f"squad2_{split}.json"
        data = json.loads((originals / name).read_text(encoding="utf-8"))
        for article in data["data"]:
            title = article.get("title", "")
            for pi, para in enumerate(article.get("paragraphs", [])):
                context = clean(para.get("context", "")) or ""
                # SQuAD provides a single paragraph per context; preserve its boundaries.
                pid = f"{split}:{title}:{pi}"
                for qa in para.get("qas", []):
                    answers = qa.get("answers", [])
                    common = dict(dataset=f"squad2_{split}", source_id=f"{pid}:{qa['id']}",
                                  qtext=qa.get("question", ""), context=context,
                                  context_kind="wikipedia_paragraph", source_file=name,
                                  source_url=SOURCES[name],
                                  paragraph_id=uid("para", f"{context_uid(f'squad2_{split}', pid)}:0", f"squad2_{split}"),
                                  context_source_id=pid)
                    if answers:
                        for i, ans in enumerate(answers):
                            rows.append(record(**common, atext=ans.get("text"),
                                               answer_source_id=f"gold{i}", score=None,
                                               reference=ans.get("text")))
                    else:
                        rows.append(record(**common, atext=None, answer_source_id="no_answer",
                                           score=None, reference=None,
                                           is_unanswerable=True))
    return rows


def load_cmu(work: Path) -> list[dict]:
    base = next(work.glob("cmu_qa_v1.2/Question_Answer_Dataset_v1.2"))
    rows, article_cache, context_index = [], {}, {}
    for table in sorted(base.glob("S*/question_answer_pairs.txt")):
        with table.open(encoding="latin-1", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, item in enumerate(reader):
                title, question, answer = (item.get(k, "") for k in ("ArticleTitle", "Question", "Answer"))
                if not question or not answer:
                    continue
                article_rel = item.get("ArticleFile") or ""
                source_path = f"{table.parent.name}/{article_rel}" if article_rel else ""
                article = base / table.parent.name / article_rel if article_rel else None
                if article is not None and article.suffix == "":
                    article = article.with_suffix(".txt.clean")
                if article is not None and not article.exists():
                    alt = article.with_suffix(".txt")
                    article = alt if alt.exists() else article
                cache_key = str(article.resolve()) if article is not None else ""
                if cache_key not in article_cache:
                    if article is not None and article.exists():
                        raw_bytes = article.read_bytes()
                        try: article_cache[cache_key] = raw_bytes.decode("utf-8-sig")
                        except UnicodeDecodeError: article_cache[cache_key] = raw_bytes.decode("cp1252")
                    else:
                        article_cache[cache_key] = ""
                raw_context = article_cache[cache_key]
                if raw_context:
                    source_id = f"{table.parent.name}:{i}:{title}"
                    ctx_id = context_uid("cmu_qa_v1.2", source_path)
                    context_index.setdefault(ctx_id, {"context_id": ctx_id, "dataset": "cmu_qa_v1.2",
                        "context_kind": "wikipedia_article", "context_text": clean(raw_context),
                        "source_file": source_path, "source_url": SOURCES["cmu_qa_v1.2.tar.gz"]})
                    rows.append(record("cmu_qa_v1.2", source_id, question, answer,
                                       None, "wikipedia_article", source_path,
                                       SOURCES["cmu_qa_v1.2.tar.gz"], answer_source_id=str(i),
                                       reference=answer,
                                       dq=item.get("DifficultyFromQuestioner"),
                                       da=item.get("DifficultyFromAnswerer"),
                                       context_source_id=source_path,
                                       question_source_id=f"{table.parent.name}:{title}:{question}"))
                else:
                    source_id = f"{table.parent.name}:{i}:{title}:missing_context"
                    rows.append(record("cmu_qa_v1.2", source_id, question, answer, None,
                                       "article_missing", source_path,
                                       SOURCES["cmu_qa_v1.2.tar.gz"], answer_source_id=str(i),
                                       reference=answer, dq=item.get("DifficultyFromQuestioner"),
                                       da=item.get("DifficultyFromAnswerer")))
    return rows, context_index


def parse_mohler_v1(work: Path) -> list[dict]:
    base = next(work.glob("mohler_short_answer_v1.0/ShortAnswerGrading_v1.0"))
    rows = []
    for file in sorted(base.glob("assign*.txt")):
        content = file.read_text(encoding="latin-1")
        blocks = re.split(r"(?m)^\s*#{5,}\s*$", content)
        for qi, block in enumerate(blocks):
            qm = re.search(r"(?m)^\s*Question:\s*(.*)$", block)
            am = re.search(r"(?m)^\s*Answer:\s*(.*)$", block)
            if not qm or not am:
                continue
            q = clean(qm.group(1)); ref = clean(am.group(1))
            for line in block[am.end():].splitlines():
                m = re.match(r"\s*([0-5](?:\.\d+)?)\s+\[(\d+)\]\s+(.*)\s*$", line)
                if not m:
                    continue
                score, student, answer = m.groups()
                source_id = f"{file.name}:{qi}:{student}"
                rows.append(record("mohler_v1.0", source_id, q, answer, None,
                                   "course_question", file.name, SOURCES["mohler_short_answer_v1.0.tar.gz"],
                                   answer_source_id=student, score=float(score), score_scale="0..5",
                                   reference=ref, question_source_id=f"{file.name}:{qi}",
                                   context_source_id=f"{file.name}:{qi}"))
    return rows


def indexed_lines(path: Path, strip_id=True) -> dict[str, str]:
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="latin-1").splitlines():
        m = re.match(r"^\s*(\d+\.\d+)\s+(.*)$", line)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def parse_mohler_v2(work: Path) -> list[dict]:
    base = next(work.glob("mohler_short_answer_v2.0/**/data/docs/files"))
    data = base.parent.parent
    questions = indexed_lines(data / "raw/questions")
    references = indexed_lines(data / "raw/answers")
    rows = []
    for raw_qid in base.read_text(encoding="latin-1").splitlines():
        qid = raw_qid.strip()
        if not qid or qid.startswith("#"):
            continue
        q = questions.get(qid)
        if not q:
            continue
        ref = references.get(qid)
        raw_file = data / "raw" / qid
        sent_file = data / "sent" / qid
        scores_file = data / "scores" / qid / "ave"
        raw_answers = raw_file.read_text(encoding="latin-1").splitlines() if raw_file.exists() else []
        sent_answers = sent_file.read_text(encoding="latin-1").splitlines() if sent_file.exists() else []
        scores = scores_file.read_text(encoding="latin-1").splitlines() if scores_file.exists() else []
        for i, raw in enumerate(raw_answers):
            # Files contain a repeated x.y prefix; retain the original student index for lineage.
            text = re.sub(r"^\s*\d+\.\d+\s+", "", raw)
            score = None
            if i < len(scores):
                try: score = float(scores[i].strip())
                except ValueError: pass
            source_id = f"{qid}:{i}"
            rows.append(record("mohler_v2.0", source_id, q, text, None,
                               "course_question", str(raw_file.relative_to(data)),
                               SOURCES["mohler_short_answer_v2.0.zip"], answer_source_id=str(i),
                               score=score, score_scale="0..5 (promedio normalizado)",
                               reference=ref, question_source_id=qid, context_source_id=qid))
    return rows


def csv_cell(value):
    """Flatten embedded line breaks for spreadsheet-friendly one-record-per-line CSV."""
    if isinstance(value, str):
        return " ".join(part.strip() for part in value.splitlines() if part.strip())
    return value


def csv_rows(rows):
    for row in rows:
        yield {key: csv_cell(value) for key, value in row.items()}


def write_jsonl(path: Path, values) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in values:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate(rows: list[dict], contexts: dict) -> dict:
    answer_ids = [r["answer_id"] for r in rows]
    question_ids = [r["question_id"] for r in rows]
    score_ids = [r.get("score_id") for r in rows if r.get("score_id")]
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("Hay answer_id duplicados")
    if len(score_ids) != len(set(score_ids)):
        raise ValueError("Hay score_id duplicados")
    if any((r.get("score") is not None) != bool(r.get("score_id")) for r in rows):
        raise ValueError("score_id debe estar presente exactamente cuando existe score")
    if any(not r["context_id"] or not r["question_id"] or not r["answer_id"] for r in rows):
        raise ValueError("Registro sin uno de los identificadores requeridos")
    if any(r["context_id"] not in contexts for r in rows):
        raise ValueError("Hay registros que apuntan a context_id inexistentes")
    if any(r["answer_text"] is None and not r["is_unanswerable"] for r in rows):
        raise ValueError("Respuesta vacía sin marca is_unanswerable")
    for r in rows:
        if r["score"] is not None and not isinstance(r["score"], (int, float)):
            raise ValueError(f"Puntuación no numérica: {r['answer_id']}")
    return {"unique_answer_ids": len(set(answer_ids)),
            "unique_question_ids": len(set(question_ids)),
            "unique_score_ids": len(set(score_ids)),
            "all_context_references_resolve": True,
            "all_required_ids_present": True,
            "score_ids_match_scores": True}


def json_matches(root, selector: str):
    """Return (value, path) pairs for a dotted JSON selector; * expands arrays/maps."""
    if selector in ("", "$", "."):
        return [(root, ())]
    states = [(root, ())]
    for part in selector.strip(".").split("."):
        next_states = []
        for value, path in states:
            if part == "*":
                children = enumerate(value) if isinstance(value, list) else value.items() if isinstance(value, dict) else []
                next_states.extend((child, path + (key,)) for key, child in children)
            elif isinstance(value, dict) and part in value:
                next_states.append((value[part], path + (part,)))
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                next_states.append((value[int(part)], path + (int(part),)))
        states = next_states
    return states


def json_relative(root, current_path: tuple, selector: str):
    """Resolve a field path from current JSON node; ../ walks to its parent node."""
    selector = str(selector)
    parent_match = re.match(r"^(?:\.\./)+", selector)
    parent_count = parent_match.group(0).count("../") if parent_match else 0
    selector = selector[parent_match.end():] if parent_match else selector
    pieces = selector.split(".") if selector else []
    path = list(current_path)
    for _ in range(parent_count):
        if path:
            path.pop()
    value = root
    for key in path:
        value = value[key]
    matches = json_matches(value, ".".join(pieces))
    if not matches:
        return None
    values = [item for item, _ in matches]
    return values if "*" in pieces else values[0]


def read_custom_rows(input_path: Path, config: dict):
    fmt = config.get("format", "auto").lower()
    if fmt == "auto":
        fmt = {".csv": "csv", ".tsv": "tsv", ".jsonl": "jsonl", ".ndjson": "jsonl"}.get(input_path.suffix.lower(), "json")
    if fmt in ("csv", "tsv"):
        delimiter = config.get("delimiter", "\t" if fmt == "tsv" else ",")
        with input_path.open("r", encoding=config.get("encoding", "utf-8-sig"), newline="") as handle:
            for line_no, value in enumerate(csv.DictReader(handle, delimiter=delimiter), start=2):
                yield value, (None, ()), line_no
        return
    if fmt == "jsonl":
        with input_path.open("r", encoding=config.get("encoding", "utf-8-sig")) as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    yield json.loads(line), (None, ()), line_no
        return
    if fmt != "json":
        raise ValueError("format debe ser auto, csv, tsv, jsonl o json")
    root = json.loads(input_path.read_text(encoding=config.get("encoding", "utf-8-sig")))
    selector = config.get("records_path", "*")
    for index, (value, path) in enumerate(json_matches(root, selector), start=1):
        yield (value, (root, path), index)


def custom_field(row, root_and_path, selector):
    if selector is None:
        return None
    root, path = root_and_path
    if root is not None:
        return json_relative(root, path, str(selector))
    value = row
    for part in str(selector).split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def load_custom(input_path: Path, config_path: Path) -> tuple[list[dict], dict]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    dataset = config.get("dataset")
    fields = config.get("fields", {})
    keys = config.get("keys", {})
    if not dataset or not fields.get("question_text") or not fields.get("answer_text"):
        raise ValueError("La configuración requiere dataset, fields.question_text y fields.answer_text")
    rows, context_index = [], {}
    for row, root_and_path, line_no in read_custom_rows(input_path, config):
        values = {name: custom_field(row, root_and_path, selector)
                  for name, selector in fields.items()}
        question = values.get("question_text")
        answers = values.get("answer_text")
        unanswerable = str(values.get("is_unanswerable", "")).lower() in ("true", "1", "yes")
        if question is None or (answers is None and not unanswerable):
            continue
        answer_values = answers if isinstance(answers, list) else [answers]
        if not answer_values:
            answer_values = [None]
        source_id = str(custom_field(row, root_and_path, keys.get("source_id")) or line_no)
        qkey = str(custom_field(row, root_and_path, keys.get("question_id")) or source_id)
        ckey = str(custom_field(row, root_and_path, keys.get("context_id")) or qkey)
        akey = str(custom_field(row, root_and_path, keys.get("answer_id")) or line_no)
        score = values.get("score")
        if score is not None and str(score).strip() != "":
            try:
                score = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Puntuación inválida en registro {line_no}: {score!r}") from exc
        else:
            score = None
        source_url = config.get("source_url", "")
        source_file = str(input_path.resolve())
        context_text = values.get("context_text")
        for answer_index, answer in enumerate(answer_values):
            ref = values.get("reference_answer")
            if isinstance(ref, list):
                ref = ref[answer_index] if answer_index < len(ref) else None
            row_record = record(dataset, f"{input_path.name}:{source_id}:{line_no}",
                                str(question), None if answer in (None, "") else str(answer),
                                None if context_text is None else str(context_text),
                                config.get("context_kind", "custom_context"), source_file,
                                source_url, answer_source_id=f"{akey}:{answer_index}", score=score,
                                score_scale=config.get("score_scale"), reference=ref,
                                is_unanswerable=unanswerable,
                                context_source_id=ckey, question_source_id=qkey)
            rows.append(row_record)
            context_index.setdefault(row_record["context_id"], {
                "context_id": row_record["context_id"], "dataset": dataset,
                "context_kind": row_record["context_kind"], "context_text": row_record["context_text"],
                "source_file": source_file, "source_url": source_url})
    return rows, context_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument("--download", dest="download", action="store_true",
                                help="Descargar de las fuentes oficiales los archivos que falten")
    download_group.add_argument("--no-download", dest="download", action="store_false",
                                help="Usar solo originales/ (predeterminado; opción compatible)")
    parser.set_defaults(download=False)
    parser.add_argument("--dataset", default="all",
                        help="Base que se normaliza: all, squad2_train, squad2_dev, cmu_qa_v1.2, mohler_v1.0, mohler_v2.0, roars_2class o roars_3class; para una base propia, indica su nombre junto con --input o --url y --config")
    parser.add_argument("--input", type=Path, help="JSON, JSONL, CSV o TSV adicional que se normalizará")
    parser.add_argument("--url", help="URL directa de un archivo de datos adicional; se descarga en originales/adicionales/")
    parser.add_argument("--filename", help="Nombre/extensión para la descarga, si la URL no termina en una extensión reconocida")
    parser.add_argument("--config", type=Path, help="Configuración JSON de mapeo del archivo adicional")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Carpeta de salida para las bases normalizadas (default: resultados)")
    args = parser.parse_args()
    if args.input and args.url:
        parser.error("Usa --input para un archivo local o --url para descargarlo, no ambos")
    custom_source = args.input or args.url
    if bool(custom_source) != bool(args.config):
        parser.error("--input o --url se deben usar junto con --config")
    if args.filename and not args.url:
        parser.error("--filename solo se usa junto con --url")
    if args.dataset in BUILTIN_DATASETS and custom_source:
        parser.error("--input/--url/--config se usan con --dataset all o con el nombre de la base propia")
    if args.dataset not in (*BUILTIN_DATASETS, "all") and not custom_source:
        parser.error("Para normalizar una base propia, indica --input o --url junto con --config")

    if args.dataset == "all":
        selected_datasets = set(BUILTIN_DATASETS)
        replace_datasets = set(BUILTIN_DATASETS)
    elif args.dataset in BUILTIN_DATASETS:
        selected_datasets = {args.dataset}
        replace_datasets = {args.dataset}
    else:
        selected_datasets = set()
        replace_datasets = {args.dataset}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    originals, work = ROOT / "originales", ROOT / "trabajo"
    source_names = {source for dataset in selected_datasets
                    for source in DATASET_SOURCES[dataset]}
    fetch_sources(originals, args.download, sorted(source_names))
    extract(originals, work, selected_datasets)

    new_rows = []
    new_contexts = {}
    squad_splits = [split for dataset, split in (("squad2_train", "train"),
                                                  ("squad2_dev", "dev"))
                    if dataset in selected_datasets]
    if squad_splits:
        new_rows.extend(load_squad(originals, squad_splits))
    if "cmu_qa_v1.2" in selected_datasets:
        cmu_rows, cmu_contexts = load_cmu(work)
        new_rows.extend(cmu_rows)
        new_contexts.update(cmu_contexts)
    if "mohler_v1.0" in selected_datasets:
        new_rows.extend(parse_mohler_v1(work))
    if "mohler_v2.0" in selected_datasets:
        new_rows.extend(parse_mohler_v2(work))
    roars_datasets = selected_datasets.intersection({"roars_2class", "roars_3class"})
    if roars_datasets:
        from preparar_roars import prepare as prepare_roars

        roars_source = originals / "roars_dataset" / "ROARS.csv"
        roars_prepared = originals / "adicionales" / "ROARS_answered.csv"
        prepare_roars(roars_source, roars_prepared)
        for name in sorted(roars_datasets):
            config_path = ROOT / f"config_{name}.json"
            roars_rows, roars_contexts = load_custom(roars_prepared, config_path)
            new_rows.extend(roars_rows)
            new_contexts.update(roars_contexts)

    custom_hash = None
    custom_manifest = None
    custom_dataset_names = set()
    input_path = args.input
    if args.url:
        input_path = download_custom_source(args.url, args.filename, originals)
    if input_path:
        custom_rows, custom_contexts = load_custom(input_path, args.config)
        custom_names = {row["dataset"] for row in custom_rows}
        if args.dataset != "all" and custom_names != {args.dataset}:
            parser.error(f"El dataset de la configuración debe ser {args.dataset!r}")
        replace_datasets.update(custom_names)
        custom_dataset_names = custom_names
        new_rows.extend(custom_rows)
        new_contexts.update(custom_contexts)
        custom_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()

    old_rows = []
    old_records_path = output_dir / "registros.jsonl"
    if old_records_path.exists():
        with old_records_path.open(encoding="utf-8") as stream:
            old_rows = [json.loads(line) for line in stream if line.strip()]
    kept_rows = [row for row in old_rows if row["dataset"] not in replace_datasets]
    rows = kept_rows + new_rows
    rows.sort(key=lambda r: (r["dataset"], r["source_record_id"], r["source_answer_id"]))
    for row in rows:
        # A score belongs to a specific answer and gets its own stable ID only
        # when the source provides a comparable score.
        row["score_id"] = (uid("cal", row["answer_id"], row["dataset"])
                            if row.get("score") is not None else None)
    # Preserve contexts for datasets not selected and replace contexts for the
    # datasets just regenerated.
    contexts = {}
    kept_context_ids = {row["context_id"] for row in kept_rows}
    old_contexts_path = output_dir / "contextos.jsonl"
    if old_contexts_path.exists():
        with old_contexts_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    if item["context_id"] in kept_context_ids:
                        contexts[item["context_id"]] = item
    contexts.update(new_contexts)
    for row in rows:
        if row["context_id"] in contexts:
            continue
        contexts.setdefault(row["context_id"], {
            "context_id": row["context_id"],
            "dataset": row["dataset"], "context_kind": row["context_kind"],
            "context_text": row["context_text"], "source_file": row["source_file"],
            "source_url": row["source_url"]})
    validation = validate(rows, contexts)
    write_jsonl(output_dir / "registros.jsonl", rows)
    write_jsonl(output_dir / "contextos.jsonl", contexts.values())
    paragraph_rows = []
    for ctx in contexts.values():
        for index, paragraph in enumerate(paras(ctx["context_text"] or "")):
            paragraph_rows.append({"paragraph_id": uid("para", f"{ctx['context_id']}:{index}", ctx["dataset"]),
                                   "context_id": ctx["context_id"], "paragraph_index": index,
                                   "paragraph_text": paragraph, "dataset": ctx["dataset"],
                                   "source_file": ctx["source_file"], "source_url": ctx["source_url"]})
    write_jsonl(output_dir / "parrafos.jsonl", paragraph_rows)
    with (output_dir / "registros.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(csv_rows(rows))
    # A separate export per dataset makes independent analysis straightforward.
    per_dataset_dir = output_dir / "por_base"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)
    filenames = {}
    for name in sorted({row["dataset"] for row in rows}):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "base"
        if safe_name in filenames.values():
            raise ValueError(f"Dos nombres de base generan el mismo nombre de archivo: {safe_name}")
        filenames[name] = safe_name
        jsonl_path = per_dataset_dir / f"{safe_name}.jsonl"
        csv_path = per_dataset_dir / f"{safe_name}.csv"
        # A single-dataset run must not rewrite unrelated exports (they may be
        # open in Excel, and their contents have not changed).
        if name not in replace_datasets and jsonl_path.exists() and csv_path.exists():
            continue
        base_rows = [row for row in rows if row["dataset"] == name]
        write_jsonl(jsonl_path, base_rows)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader(); writer.writerows(csv_rows(base_rows))
    old_manifest_path = output_dir / "manifest.json"
    custom_inputs = {}
    if old_manifest_path.exists():
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        custom_manifest = old_manifest.get("custom_input")
        custom_inputs = old_manifest.get("custom_inputs", {})
    for name in replace_datasets:
        custom_inputs.pop(name, None)
    if input_path:
        custom_manifest = {"file": str(input_path.resolve()), "sha256": custom_hash,
                           "config": str(args.config.resolve())}
        if args.url:
            custom_manifest["url"] = args.url
        for name in custom_dataset_names:
            custom_inputs[name] = dict(custom_manifest)
    # Manifest with file checksums and per-corpus row counts for reproducibility.
    manifest = {"schema_version": "1.0", "normalizer": Path(__file__).name,
                "sources": SOURCES, "record_count": len(rows),
                "context_count": len(contexts),
                "paragraph_count": len(paragraph_rows),
                "validation": validation,
                "counts_by_dataset": {name: sum(r["dataset"] == name for r in rows)
                                      for name in sorted({r["dataset"] for r in rows})},
                "files_by_dataset": {name: {"jsonl": f"por_base/{safe}.jsonl",
                                             "csv": f"por_base/{safe}.csv"}
                                     for name, safe in filenames.items()},
                "original_sha256": {p.relative_to(originals).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(originals.rglob("*")) if p.is_file()},
                "custom_input": custom_manifest,
                "custom_inputs": custom_inputs,
                "null_semantics": {"score": "null cuando la fuente no publica una nota comparable",
                                   "context_text": "null si la fuente no incluye un pasaje"}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "contexts": len(contexts),
                      "datasets": manifest["counts_by_dataset"],
                      "normalized": sorted(replace_datasets), "output": str(output_dir)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
