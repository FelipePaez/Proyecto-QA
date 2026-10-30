"""Local, reproducible scikit-learn training for QA response features."""
from __future__ import annotations

import json
import math
import random
import re
import tempfile
import time
import zipfile
import uuid
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


NUMERIC_FEATURES = [
    "cosine_similarities.question_answer",
    "cosine_similarities.question_context_centroid",
    "cosine_similarities.answer_context_centroid",
    "cosine_similarities.question_nearest_question_paragraph",
    "cosine_similarities.question_nearest_answer_paragraph",
    "cosine_similarities.answer_nearest_question_paragraph",
    "cosine_similarities.answer_nearest_answer_paragraph",
]
TEXT_FEATURES = ["texts.question", "texts.candidate_answer", "texts.paragraph_nearest_question.paragraph_text",
                 "texts.paragraph_nearest_answer.paragraph_text"]
ALLOWED_TARGETS = {"score", "is_unanswerable", "answer_text"}


def _value(row: dict, path: str):
    current = row
    for part in path.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return current


def _records(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL inválido en la línea {line_number}: {exc.msg}.") from exc
            if isinstance(item, dict):
                yield item


def _target_value(row, target):
    value = (row.get("targets") or {}).get(target)
    if target == "score":
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    if target == "is_unanswerable":
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        token = str(value).strip().lower()
        if token in {"true", "1", "yes", "y", "si", "sí"}:
            return "true"
        if token in {"false", "0", "no", "n"}:
            return "false"
        return None
    if target == "answer_text":
        value = value.strip() if isinstance(value, str) else value
        return value if value else None
    return None


def _feature_table(records, feature_keys, target_name, max_records=None):
    import pandas as pd
    rows, targets, ids = [], [], []
    missing_target = 0
    eligible = 0
    sampler = random.Random(42)
    for row in records:
        target = _target_value(row, target_name)
        if target is None:
            missing_target += 1
            continue
        features = {name: _value(row, name) for name in feature_keys}
        for name in NUMERIC_FEATURES:
            if name in features:
                try:
                    value = float(features[name])
                    features[name] = value if math.isfinite(value) else np.nan
                except (TypeError, ValueError):
                    features[name] = np.nan
        for name in TEXT_FEATURES:
            if name in features:
                features[name] = str(features[name] or "")
        item = (features, target, (row.get("ids") or {}).get("answer_id", ""), row.get("embedding_refs") or {})
        eligible += 1
        if max_records is None or len(rows) < max_records:
            rows.append(item)
        else:
            slot = sampler.randrange(eligible)
            if slot < max_records:
                rows[slot] = item
    frame_rows = [item[0] for item in rows]
    targets = np.asarray([item[1] for item in rows])
    ids = [item[2] for item in rows]
    refs = [item[3] for item in rows]
    return pd.DataFrame(frame_rows, columns=feature_keys), targets, ids, refs, missing_target, eligible


def _model_and_grid(task, model_name, feature_keys):
    scaling = StandardScaler()
    if task == "regression":
        models = {
            "ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
            "random_forest": (RandomForestRegressor(random_state=42, n_jobs=-1),
                              {"model__n_estimators": [100, 250], "model__max_depth": [None, 12]}),
            "extra_trees": (ExtraTreesRegressor(random_state=42, n_jobs=-1),
                            {"model__n_estimators": [100, 250], "model__max_depth": [None, 12]}),
            "svr": (SVR(), {"model__C": [0.1, 1, 10], "model__epsilon": [0.05, 0.2]}),
        }
    else:
        models = {
            "logistic_regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                                    {"model__C": [0.1, 1, 10]}),
            "random_forest": (RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                              {"model__n_estimators": [100, 250], "model__max_depth": [None, 12]}),
            "extra_trees": (ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                            {"model__n_estimators": [100, 250], "model__max_depth": [None, 12]}),
            "svc": (SVC(class_weight="balanced"), {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}),
        }
    if model_name not in models:
        raise ValueError(f"Modelo no válido: {model_name}.")
    model, grid = models[model_name]
    numeric = [name for name in feature_keys if name in NUMERIC_FEATURES or name.startswith("embedding_")]
    categorical = []
    text = [name for name in TEXT_FEATURES if name in feature_keys]
    transformers = []
    if numeric:
        transformers.append(("numerical", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                                                       ("scale", scaling)]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                         ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    if text:
        from sklearn.feature_extraction.text import TfidfVectorizer
        for index, name in enumerate(text):
            transformers.append((f"text_{index}", TfidfVectorizer(max_features=2500, ngram_range=(1, 2),
                                                                    strip_accents="unicode", sublinear_tf=True), name))
    preprocess = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    pipeline = Pipeline([("features", preprocess), ("model", model)])
    return pipeline, grid


def train_model(results_dir: Path, dataset: str, payload: dict) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", dataset):
        raise ValueError("Nombre de base no válido.")
    target = str(payload.get("target") or "")
    if target not in ALLOWED_TARGETS:
        raise ValueError("Elige un objetivo disponible: calificación, no respondible o respuesta esperada.")
    task = str(payload.get("task") or "")
    if task not in {"regression", "classification"}:
        raise ValueError("El tipo debe ser regresión o clasificación.")
    if task == "regression" and target != "score":
        raise ValueError("La regresión requiere una calificación numérica. Para clasificar, elige una etiqueta disponible.")
    model_name = str(payload.get("model") or "")
    feature_mode = str(payload.get("features") or "similarities")
    if feature_mode not in {"similarities", "similarities_and_text", "text",
                            "embeddings", "embeddings_and_similarities"}:
        raise ValueError("Configuración de características no válida.")
    optimize = bool(payload.get("optimize"))
    test_size = float(payload.get("test_size", 0.2))
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("La proporción de prueba debe estar entre 10% y 40%.")
    minimum_per_class = max(1, int(payload.get("minimum_per_class", 2)))
    source = results_dir / dataset / "vista_respuestas.jsonl"
    if not source.is_file():
        raise ValueError("No se encontró la vista procesada. Genera primero características para esta base.")
    feature_keys = []
    if feature_mode in {"similarities", "similarities_and_text"}:
        feature_keys.extend(NUMERIC_FEATURES)
    if feature_mode in {"text", "similarities_and_text"}:
        feature_keys.extend(TEXT_FEATURES)
    if feature_mode == "embeddings_and_similarities":
        feature_keys.extend(NUMERIC_FEATURES)
    leakage_guard = target in {"is_unanswerable", "answer_text"}
    if leakage_guard:
        excluded = {"cosine_similarities.question_answer", "cosine_similarities.answer_context_centroid",
                    "cosine_similarities.question_nearest_answer_paragraph",
                    "cosine_similarities.answer_nearest_question_paragraph", "cosine_similarities.answer_nearest_answer_paragraph",
                    "texts.candidate_answer", "texts.paragraph_nearest_answer.paragraph_text"}
        feature_keys = [key for key in feature_keys if key not in excluded]
    has_vectors = feature_mode in {"embeddings", "embeddings_and_similarities"}
    if not feature_keys and not has_vectors:
        raise ValueError("Selecciona al menos una familia de características.")
    vector_cap = 1000 if has_vectors else None
    frame, y, row_ids, refs, missing_target, eligible_count = _feature_table(
        _records(source), feature_keys, target, vector_cap)
    if len(y) < 8:
        raise ValueError(f"Solo hay {len(y)} registros utilizables con el objetivo '{target}'. Se necesitan al menos 8 para una evaluación básica.")
    vector_kinds = ["question_embedding_id", "context_centroid_embedding_id",
                    "paragraph_nearest_question"]
    if not leakage_guard:
        vector_kinds.extend(["answer_embedding_id", "paragraph_nearest_answer"])
    if has_vectors:
        import pandas as pd
        from vector_store import ChromaVectorStore
        ids_by_kind = {kind: [] for kind in vector_kinds}
        for reference in refs:
            for kind in vector_kinds:
                embedding_id = reference.get(kind) if kind.endswith("_embedding_id") else (reference.get(kind) or {}).get("embedding_id")
                ids_by_kind[kind].append(embedding_id)
        all_ids = [embedding_id for values in ids_by_kind.values() for embedding_id in values if embedding_id]
        if not all_ids:
            raise ValueError("Esta base no tiene referencias a embeddings. Genera primero los embeddings desde el flujo de datos.")
        store = ChromaVectorStore(results_dir / "chroma_db")
        try:
            vector_map = store.get_many(all_ids)
        finally:
            store.close()
        dimensions = {len(vector) for vector in vector_map.values()}
        if len(dimensions) != 1:
            raise ValueError("Los embeddings guardados tienen dimensiones diferentes. Regenera esta vista con un único modelo y dimensión.")
        dimension = next(iter(dimensions))
        roles = {"question_embedding_id": "question", "answer_embedding_id": "answer",
                 "context_centroid_embedding_id": "context_centroid",
                 "paragraph_nearest_question": "paragraph_nearest_question",
                 "paragraph_nearest_answer": "paragraph_nearest_answer"}
        vector_columns = [f"embedding_{roles[kind]}_{index:04d}" for kind in vector_kinds for index in range(dimension)]
        vector_rows = []
        for row_index in range(len(refs)):
            values = []
            for kind in vector_kinds:
                embedding_id = ids_by_kind[kind][row_index]
                vector = vector_map.get(embedding_id) if embedding_id else None
                values.extend(vector if vector is not None else [np.nan] * dimension)
            vector_rows.append(values)
        frame = frame.join(pd.DataFrame(vector_rows, columns=vector_columns))
        vector_requested = sum(1 for values in ids_by_kind.values() for embedding_id in values if embedding_id)
        vector_coverage = sum(1 for values in ids_by_kind.values() for embedding_id in values if embedding_id in vector_map)
        if vector_coverage < max(1, math.ceil(vector_requested * 0.8)):
            raise ValueError(f"Solo se recuperó {vector_coverage}/{vector_requested} vectores referenciados desde ChromaDB. Revisa que la base siga disponible y tenga la misma configuración de embeddings.")
        feature_keys.extend(vector_columns)
    if task == "classification":
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            raise ValueError("La clasificación necesita al menos dos clases distintas en el objetivo seleccionado.")
        if len(labels) > min(50, max(10, len(y) // 3)):
            raise ValueError(f"El objetivo contiene {len(labels)} clases distintas para {len(y)} registros. La clasificación de respuestas requiere clases repetidas; elige otro objetivo o agrupa las etiquetas antes de entrenar.")
        rare = {str(label): int(count) for label, count in zip(labels, counts) if count < minimum_per_class}
        if rare:
            raise ValueError("Hay clases con muy pocos ejemplos para separar entrenamiento y prueba: " + ", ".join(f"{label} ({count})" for label, count in rare.items()) + ". Añade datos o reduce el mínimo por clase.")
        stratify = y
    else:
        stratify = None
    test_count = max(2, math.ceil(len(y) * test_size))
    if task == "classification" and test_count < len(np.unique(y)):
        raise ValueError("El conjunto de prueba sería menor que el número de clases. Aumenta el porcentaje de prueba o añade registros.")
    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(
        frame, y, row_ids, test_size=test_size, random_state=42, stratify=stratify)
    pipeline, grid = _model_and_grid(task, model_name, feature_keys)
    started = time.time()
    search = None
    if optimize:
        min_train_class = min(np.unique(y_train, return_counts=True)[1]) if task == "classification" else len(y_train)
        folds = min(5, int(min_train_class))
        if folds < 2:
            raise ValueError("No hay suficientes ejemplos por clase para optimizar con validación cruzada.")
        search = GridSearchCV(pipeline, grid, cv=folds, n_jobs=1,
                              scoring="neg_mean_absolute_error" if task == "regression" else "f1_macro",
                              refit=True, error_score="raise")
        search.fit(x_train, y_train)
        trained = search.best_estimator_
    else:
        trained = pipeline.fit(x_train, y_train)
    pred = trained.predict(x_test)
    if task == "regression":
        r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else None
        metrics = {"mae": float(mean_absolute_error(y_test, pred)),
                   "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
                   "r2": r2 if r2 is not None and math.isfinite(r2) else None}
    else:
        labels = np.unique(y)
        metrics = {"accuracy": float(accuracy_score(y_test, pred)),
                   "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                   "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                   "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
                   "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
                   "classes": [str(label) for label in labels]}
    model_dir = results_dir / "modelos_entrenados"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_id = f"{dataset}_{task}_{target}_{uuid.uuid4().hex[:8]}"
    model_path = model_dir / f"{model_id}.joblib"
    manifest_path = model_dir / f"{model_id}.json"
    metadata = {"model_id": model_id, "dataset": dataset, "task": task, "target": target,
                "model": model_name, "feature_mode": feature_mode, "features": feature_keys,
                "train_records": int(len(y_train)), "test_records": int(len(y_test)),
                "rows_without_target": int(missing_target), "eligible_records": int(eligible_count),
                "training_sampled": bool(eligible_count > len(y)),
                "sampling_limit": vector_cap, "leakage_guard_applied": leakage_guard,
                "metrics": metrics,
                "optimization": {"enabled": optimize, "best_params": search.best_params_ if search else {},
                                  "cv_score": float(search.best_score_) if search else None},
                "split": {"test_size": test_size, "random_state": 42},
                "created_at_unix": int(time.time()), "training_seconds": round(time.time() - started, 3),
                "note": "El entrenamiento es supervisado y sus métricas dependen de la calidad, distribución y tamaño del dataset."}
    bundle = {"pipeline": trained, "metadata": metadata}
    with tempfile.NamedTemporaryFile(prefix=model_id, suffix=".tmp", dir=model_dir, delete=False) as handle:
        temp_model = Path(handle.name)
    try:
        joblib.dump(bundle, temp_model, compress=3)
        temp_model.replace(model_path)
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        predictions_path = model_dir / f"{model_id}_predicciones.csv"
        import csv
        with predictions_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["answer_id", "actual", "predicted"])
            for answer_id, actual, predicted in zip(ids_test, y_test, pred):
                writer.writerow([answer_id, actual, predicted])
    finally:
        temp_model.unlink(missing_ok=True)
    return {**metadata, "download": f"/api/models/download?model_id={model_id}",
            "model_file": str(model_path.resolve()), "metadata_file": str(manifest_path.resolve()),
            "predictions_file": str(predictions_path.resolve())}


def archive_model(results_dir: Path, model_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", model_id):
        raise ValueError("Identificador de modelo no válido.")
    root = (results_dir / "modelos_entrenados").resolve()
    model = root / f"{model_id}.joblib"
    metadata = root / f"{model_id}.json"
    predictions = root / f"{model_id}_predicciones.csv"
    if root not in model.parents or not model.is_file() or not metadata.is_file():
        raise ValueError("No se encontró el paquete del modelo solicitado.")
    archive = root / f"{model_id}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(model, model.name)
        bundle.write(metadata, metadata.name)
        if predictions.is_file():
            bundle.write(predictions, predictions.name)
    return archive
