#!/usr/bin/env python3
"""Local interactive browser for per-answer embedding feature views."""
from __future__ import annotations

import argparse
from array import array
import base64
import csv
import heapq
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import uuid
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "resultados"
DEFAULT_NORMALIZED = ROOT.parent / "qa_dataset_project" / "resultados"
RAW_CATALOG = DEFAULT_NORMALIZED / "originales_catalog.json"
RAW_CATALOG_LOCK = threading.Lock()
NORMALIZER_DIR = ROOT.parent / "qa_dataset_project"
NORMALIZER_SCRIPT = NORMALIZER_DIR / "normalizar_datasets.py"
NORMALIZER_DATASETS = {
    "all", "squad2_train", "squad2_dev", "cmu_qa_v1.2", "mohler_v1.0",
    "mohler_v2.0", "roars_2class", "roars_3class", "boolq_train", "boolq_dev",
}
NORMALIZATION_LOCK = threading.Lock()
NORMALIZATION_STATE = {"job_id": None, "status": "idle", "dataset": None,
                       "message": "", "result": None, "error": None}


def load_raw_catalog() -> dict:
    try:
        catalog = json.loads(RAW_CATALOG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": "1.0", "datasets": {}}
    if not isinstance(catalog, dict) or not isinstance(catalog.get("datasets"), dict):
        raise ValueError(f"El catálogo de originales no tiene un formato válido: {RAW_CATALOG}")
    return catalog


def save_raw_catalog(catalog: dict) -> None:
    RAW_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_CATALOG.with_name(RAW_CATALOG.name + ".tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(RAW_CATALOG)
COSINE_FIELDS = {
    "question_answer": "Pregunta – respuesta",
    "question_context_centroid": "Pregunta – centroide del contexto",
    "answer_context_centroid": "Respuesta – centroide del contexto",
    "question_nearest_question_paragraph": "Pregunta – párrafo más cercano a la pregunta",
    "question_nearest_answer_paragraph": "Pregunta – párrafo más cercano a la respuesta",
    "answer_nearest_question_paragraph": "Respuesta – párrafo más cercano a la pregunta",
    "answer_nearest_answer_paragraph": "Respuesta – párrafo más cercano a la respuesta",
}

PAGE = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explorador QA</title><style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#152536;background:#f3f6fa}body{margin:0}header{background:#102b46;color:#fff;padding:22px 5vw}header h1{margin:0 0 5px;font-size:24px}header p{margin:0;color:#cfdeed}.wrap{max-width:1500px;margin:22px auto;padding:0 20px}.panel{background:white;border:1px solid #dce4ed;border-radius:12px;padding:18px;margin-bottom:16px;box-shadow:0 2px 8px #16344d0a}.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;align-items:end}label{display:block;font-size:13px;font-weight:600;margin:0 0 5px}input,select,button{font:inherit}input[type=text],input[type=number],select{width:100%;box-sizing:border-box;padding:10px;border:1px solid #bdcad8;border-radius:7px;background:#fff}button{border:0;border-radius:7px;padding:10px 15px;background:#1769aa;color:white;font-weight:600;cursor:pointer}button:hover{background:#0d538d}.danger{background:#a83232}.danger:hover{background:#842525}.secondary{background:#e9f1f8;color:#163b5a;padding:6px 9px}.secondary:hover{background:#d7e6f3}.muted{color:#607386;font-size:13px}.summary{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.pill{border-radius:20px;background:#eaf2f9;padding:6px 11px;font-size:13px}.table-wrap{overflow:auto;max-height:520px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e7edf3;vertical-align:top}th{position:sticky;top:0;background:#f7f9fb;z-index:1}.id{font-family:Consolas,monospace;font-size:12px;white-space:nowrap}.record-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.record-grid section{border:1px solid #e4ebf2;border-radius:9px;padding:12px}.record-grid h3{font-size:14px;margin:0 0 9px}.pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:250px;overflow:auto;background:#f7f9fb;padding:10px;border-radius:7px;font-size:13px}.sim-row{display:grid;grid-template-columns:minmax(180px,330px) 1fr 60px;gap:10px;align-items:center;margin:9px 0;font-size:13px}.bar{height:12px;border-radius:8px;background:#e8eef4;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#63b4cf,#168353)}.bar-negative{background:#c66}.refs{display:flex;flex-wrap:wrap;gap:7px;margin-top:7px}.err{color:#a22020}.small{font-size:12px;color:#718296}.spinner{color:#42627e}a{color:#1267a5}@media(max-width:650px){header{padding:18px}.wrap{padding:0 10px}.sim-row{grid-template-columns:1fr 1fr 45px}}
/* Dashboard and workflow refinements */
body{background:#f4f7fb;color:#172b3f}header{background:radial-gradient(ellipse at 82% 0%,#246c83 0%,transparent 38%),linear-gradient(120deg,#102b46,#173e59);padding:30px max(5vw,calc((100vw - 1460px)/2));}header h1{font-size:clamp(25px,3vw,34px);letter-spacing:-.5px}header p{margin-top:8px;max-width:780px;line-height:1.5}.wrap{max-width:1500px;margin:24px auto}.panel{border-radius:16px;padding:22px;box-shadow:0 8px 24px #1b38500a;border-color:#e3eaf1}.tabs{position:sticky;top:0;z-index:5;backdrop-filter:blur(12px);background:#f4f7fbeb;border:1px solid #e0e8ef;border-radius:14px;padding:8px;box-shadow:0 4px 16px #1b38500a}.tabs button{padding:11px 17px;border-radius:10px}.tabs button.active{background:#173e59}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.kpi{background:linear-gradient(145deg,#fff,#f8fbfd);border:1px solid #e4ebf2;border-radius:13px;padding:16px}.kpi strong{display:block;font-size:25px;color:#173e59;letter-spacing:-.5px}.kpi span{font-size:12px;color:#65798b}.dash-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(280px,.8fr);gap:16px}.chart-row{display:grid;grid-template-columns:minmax(120px,190px) 1fr 84px;gap:12px;align-items:center;margin:12px 0;font-size:13px}.chart-track{height:13px;background:#edf2f6;border-radius:9px;overflow:hidden}.chart-fill{height:100%;border-radius:9px;background:linear-gradient(90deg,#42a4bd,#1769aa)}.chart-fill.train{background:linear-gradient(90deg,#1769aa,#48a4c0)}.chart-fill.dev{background:linear-gradient(90deg,#54a98a,#86c889)}.callout{padding:13px 15px;border-radius:11px;background:#eef7f5;border:1px solid #d7eee8;color:#315b53;font-size:13px;line-height:1.5}.stepper{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0 20px}.step{padding:10px 12px;border-radius:10px;background:#f2f5f8;color:#708194;font-size:12px;font-weight:700}.step b{display:inline-grid;place-items:center;width:23px;height:23px;margin-right:6px;border-radius:50%;background:#dce5ed}.step.current{background:#e8f3f6;color:#15546a}.step.current b{background:#1769aa;color:#fff}.step.done{background:#edf7f1;color:#35684a}.step.done b{background:#37855c;color:#fff}.coverage{display:grid;grid-template-columns:minmax(130px,220px) 1fr 70px;gap:10px;align-items:center;margin:7px 0;font-size:12px}.panel h2,.panel h3{color:#173e59}.panel h2{letter-spacing:-.3px}.panel:focus-within{border-color:#c5d9e5}.empty-state{padding:20px;border:1px dashed #ccd8e2;border-radius:12px;color:#64798a}@media(max-width:800px){.dash-grid{grid-template-columns:1fr}.stepper{grid-template-columns:repeat(2,1fr)}}
.field-catalog{border:1px solid #e2eaf0;border-radius:12px;padding:14px;margin:14px 0;background:#fbfcfd}.field-catalog .table-wrap{max-height:360px;margin-top:10px}.field-catalog table{font-size:12px}.field-catalog td{max-width:320px;overflow-wrap:anywhere}.field-example{color:#617588}.field-presence{min-width:110px}.field-presence .chart-track{margin-bottom:4px}.field-role{color:#24644c;font-weight:700}.map-card{border:1px solid #dfe8ef;border-radius:12px;padding:14px;background:#fbfcfd}.map-card label{font-size:14px;color:#173e59}.map-help{font-size:12px;line-height:1.5;color:#62788a;margin:0 0 9px}.map-preview{margin-top:10px;padding:9px 10px;border-radius:8px;background:#f0f5f8;color:#526c7e;font-size:12px;line-height:1.45;overflow-wrap:anywhere}.map-preview strong{color:#173e59}.map-preview em{color:#273f50;font-style:normal}.map-card input{margin-top:8px}.wf-explain{margin:14px 0;padding:16px;border-radius:12px;background:#eff6fb;border:1px solid #d5e6f1;line-height:1.55}.wf-explain h4{margin:0 0 7px;color:#173e59}.wf-explain p{margin:5px 0;font-size:13px}.wf-summary{margin:14px 0}.wf-summary strong{color:#173e59}#wfSourceTableWrap th{white-space:nowrap}#wfSourceTableWrap td{min-width:150px;max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.state-normalized{background:#e8f5ee;color:#246447}.state-direct{background:#fff3e3;color:#865613}.state-original{background:#eef2f6;color:#536879}.drop-zone{display:grid;place-items:center;min-height:130px;border:2px dashed #a9bfce;border-radius:14px;background:#f7fafc;text-align:center;color:#496579;cursor:pointer;padding:20px;transition:.15s}.drop-zone:hover,.drop-zone.dragover{border-color:#1769aa;background:#edf6fb}.drop-zone strong{display:block;color:#173e59;font-size:15px;margin-bottom:6px}.norm-map-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.norm-map{border:1px solid #dfe8ef;border-radius:11px;padding:12px;background:#fbfcfd}.norm-map label{color:#173e59}.norm-map select,.norm-map input{margin-top:6px}.norm-help{font-size:12px;line-height:1.5;color:#62788a;margin:4px 0 10px}
.delete-backdrop{position:fixed;inset:0;z-index:30;background:#102b468c;display:grid;place-items:center;padding:18px}.delete-dialog{width:min(520px,100%);box-sizing:border-box;background:#fff;border-radius:16px;padding:24px;box-shadow:0 20px 70px #102b4655}.delete-dialog h2{margin:0 0 10px;color:#173e59}.delete-dialog p{line-height:1.55;color:#526779}.delete-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:20px}.delete-actions button{min-width:105px}[hidden]{display:none!important}
</style></head><body>
<header><h1>Explorador de vistas QA</h1><p>Busca registros normalizados, revisa similitudes y consulta los vectores asociados.</p></header>
<main class="wrap">
<section class="panel tabs"><div class="summary"><button id="tabExplore" class="active">Explorar bases</button><button id="tabWorkflow" class="secondary">Preparar datos de entrenamiento</button><button id="tabNormalize" class="secondary">Añadir / normalizar bases</button></div></section>
<section class="panel" id="dashboard"><h2 style="margin:0 0 14px">Panorama de las bases</h2><div class="kpis" id="kpis"></div><div class="dash-grid" style="margin-top:18px"><section><h3 style="margin-top:0">Registros por base</h3><div id="datasetChart"></div></section><section><h3 style="margin-top:0">Cobertura de calificaciones</h3><div id="scoreChart"></div><div class="callout" id="splitCallout" style="margin-top:14px"></div></section></div></section>
<div id="exploreView">
<section class="panel"><div class="filters">
<div><label for="dataset">Base de datos</label><select id="dataset"></select></div>
<div><label for="deleteDataset">Administrar</label><button id="deleteDataset" class="danger" disabled>Eliminar base</button></div>
<div><label for="query">Texto o ID</label><input id="query" type="text" placeholder="Pregunta, respuesta, contexto o ID"></div>
<div><label for="field">Buscar en</label><select id="field"><option value="all">Todo (incluye contexto)</option><option value="question">Pregunta</option><option value="answer">Respuesta</option><option value="paragraph">Párrafos</option><option value="context">Contexto</option><option value="ids">IDs</option></select></div>
<div><label for="sim">Filtrar por similitud</label><select id="sim"><option value="">Sin filtro</option></select></div>
<div><label for="minimum">Similitud mínima (−1 a 1)</label><input id="minimum" type="number" min="-1" max="1" step="0.01" placeholder="Sin mínimo"></div>
<div><label for="limit">Filas por página</label><select id="limit"><option>50</option><option selected>100</option><option>200</option></select></div>
<div><label><input id="scored" type="checkbox"> Solo registros con calificación</label><button id="search">Buscar</button></div>
</div><p class="muted" id="searchHelp">La búsqueda de texto ignora mayúsculas y tildes. El filtro de similitud solo devuelve filas que tienen ese valor calculado.</p><p class="callout">El selector etiqueta cada base como <strong>Normalizada</strong> (exportación de qa_dataset_project) o <strong>Importación directa</strong> (archivo procesado desde el flujo de entrenamiento). También indica si se encontró el archivo fuente local.</p></section>
<section class="panel"><div id="summary" class="summary"><span class="muted">Elige una base y pulsa Buscar.</span></div><div id="pager" class="summary" hidden style="margin:12px 0"><button id="prevPage" class="secondary">Anterior</button><span id="pageInfo" class="pill"></span><label style="display:flex;align-items:center;gap:8px">Ir a página <input id="pageNumber" type="number" min="1" value="1" style="width:90px"></label><button id="goPage" class="secondary">Ir</button><button id="nextPage" class="secondary">Siguiente</button><a id="downloadDataset" class="pill" style="text-decoration:none" download>Descargar base completa en JSONL</a></div><p id="status" class="small"></p><div class="table-wrap"><table><thead id="resultsHead"><tr><th>Respuesta ID</th><th>Pregunta</th><th>Respuesta</th><th>Similitud</th><th>Calificación</th><th></th></tr></thead><tbody id="results"></tbody></table></div></section>
<section class="panel" id="detailPanel" hidden><h2 style="margin-top:0">Detalle de la respuesta</h2><div id="detail"></div></section>
</div>
<div id="workflowView" hidden>
<section class="panel"><h2>Flujo de trabajo para una nueva base</h2><p class="muted">Carga CSV, TSV, JSON o JSONL. Revisa la muestra, asigna los campos y procesa la base; tus datasets existentes se conservan.</p><div class="stepper"><div class="step current"><b>1</b>Archivo y vista previa</div><div class="step"><b>2</b>Asignar campos</div><div class="step"><b>3</b>Procesar</div><div class="step"><b>4</b>Exportar</div></div>
<div class="filters"><div><label>Archivo de datos</label><input id="wfFile" type="file" accept=".csv,.tsv,.json,.jsonl,.ndjson"></div><div><label>Nombre para esta base</label><input id="wfName" type="text" placeholder="mi_dataset"></div><div><label>Ruta de registros (JSON anidado)</label><input id="wfPath" type="text" placeholder="data.*.items.*"><span class="small">Para CSV/JSONL se ignora.</span></div><div><button id="wfPreview">Cargar y previsualizar</button></div></div><div id="wfPreviewBox"></div><div id="wfSourcePager" class="summary" hidden style="margin:14px 0"><label>Filas por página <select id="wfSourcePageSize"><option>25</option><option selected>50</option><option>100</option><option>200</option></select></label><button id="wfSourcePrev" class="secondary">Anterior</button><span id="wfSourcePageInfo" class="pill"></span><label style="display:flex;align-items:center;gap:8px">Ir a página <input id="wfSourcePageNumber" type="number" min="1" value="1" style="width:90px"></label><button id="wfSourceGo" class="secondary">Ir</button><button id="wfSourceNext" class="secondary">Siguiente</button></div><div id="wfSourceTableWrap" class="table-wrap" hidden style="max-height:460px"><table><thead id="wfSourceHead"></thead><tbody id="wfSourceRows"></tbody></table></div></section>
<section class="panel" id="wfMapPanel" hidden><h3>Paso 2 · Indica qué significa cada columna</h3><p class="muted">El catálogo revisa todos los registros. Elige la columna que corresponde a cada concepto; los campos no asignados se conservan en el archivo original.</p><div class="wf-explain"><h4>¿Qué se prepara aquí?</h4><p><strong>Entradas y características:</strong> pregunta, respuesta del estudiante y, si existe, contexto. De estos textos se calculan embeddings y similitudes.</p><p><strong>Objetivos de evaluación:</strong> respuesta esperada y calificación. Sirven como referencia para entrenar o evaluar después; no son entradas del modelo, para evitar que este vea la respuesta correcta.</p><p>Este paso <strong>no entrena un modelo</strong>: crea una tabla organizada de características y objetivos que luego puedes exportar.</p></div><div class="field-catalog"><label for="wfFieldSearch">Catálogo de campos</label><input id="wfFieldSearch" type="search" placeholder="Filtrar por nombre, tipo o ejemplo"><div class="table-wrap"><table><thead><tr><th>Campo</th><th>Tipo detectado</th><th>Presencia</th><th>Ejemplo</th><th>Asignado como</th></tr></thead><tbody id="wfFieldRows"></tbody></table></div><p id="wfFieldCount" class="small"></p></div><h4>Asigna los campos</h4><p class="muted">Pregunta y respuesta son obligatorias. Contexto es muy recomendable para calcular las características basadas en pasajes. Los demás campos son opcionales. En JSON anidado puedes escribir una ruta relativa, por ejemplo <code>../../context</code> (sube dos niveles hasta encontrar context).</p><div class="filters" id="wfMappings"></div><div class="wf-explain wf-summary" id="wfMappingSummary" aria-live="polite"></div><div class="filters"><div><label>Modelo de embeddings</label><input id="wfModel" value="text-embedding-3-small"></div><div><label>Dimensiones</label><input id="wfDims" type="number" value="1536" min="1"></div><div><label>Ruta del archivo de clave API (opcional)</label><input id="wfKeyFile" type="text" placeholder="Usa OPENAI_API_KEY si está configurada"></div><div><button id="wfProcess">Crear características y similitudes</button><p id="wfStatus" class="small" role="status" aria-live="polite"></p></div></div><p class="small">Al iniciar: se asignan IDs relacionados por registro, se generan embeddings de pregunta/respuesta/contexto, se calculan similitudes coseno y se guarda la vista en la base de datos del proyecto. Se requiere conexión y una clave API válida. La clave no se muestra ni se guarda en el proyecto.</p></section>
<section class="panel" id="wfExportPanel" hidden><h3>Resultado del procesamiento</h3><div id="wfResult" class="summary"></div><h4>Revisar registros procesados</h4><p class="muted">Navega por páginas sin cargar todo el dataset a la vez. Para inspeccionar todos los atributos y vectores usa la exportación seleccionable de abajo.</p><div id="workflowPager" class="summary"><label>Filas por página <select id="wfPageSize"><option>50</option><option selected>100</option><option>200</option></select></label><button id="wfPrevPage" class="secondary">Anterior</button><span id="wfPageInfo" class="pill"></span><label style="display:flex;align-items:center;gap:8px">Ir a página <input id="wfPageNumber" type="number" min="1" value="1" style="width:90px"></label><button id="wfGoPage" class="secondary">Ir</button><button id="wfNextPage" class="secondary">Siguiente</button><a id="wfDownloadAll" class="pill" style="text-decoration:none" download>Descargar vista completa (JSONL)</a></div><div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>Respuesta ID</th><th>Pregunta</th><th>Respuesta</th><th>Similitud Q–R</th><th>Calificación</th></tr></thead><tbody id="wfResults"></tbody></table></div><h4>Exportar conjunto para entrenamiento</h4><p>Elige qué atributos exportar. Las referencias de embeddings son compactas; los vectores completos se pueden incluir opcionalmente.</p><div class="filters" id="wfColumns"></div><div class="filters"><div><label>Formato</label><select id="wfFormat"><option>jsonl</option><option>json</option><option>csv</option></select></div><div><label><input type="checkbox" id="wfVectors"> Incluir vectores completos (archivos grandes)</label></div><div><button id="wfExport">Guardar en resultados/exportaciones</button></div></div><p class="small">En CSV, los saltos de línea internos del texto se unen con espacios para que cada respuesta ocupe una fila visible. El JSON/JSONL conserva el texto original con sus saltos.</p><p id="wfExportStatus" class="small"></p></section>
</div>
<div id="normalizeView" hidden>
<section class="panel"><h2>Añadir o preparar una base de datos</h2>
<p class="muted">Esta pestaña ofrece dos acciones distintas para un archivo nuevo: guardarlo tal como está para consultarlo, o normalizarlo para convertirlo al formato común del proyecto. Guardar el original no inicia la normalización.</p>
<div class="stepper"><div class="step current"><b>1</b>Seleccionar archivo</div><div class="step"><b>2</b>Elegir: guardar original o normalizar</div><div class="step"><b>3</b>Explorar en “Explorar bases”</div><div class="step"><b>4</b>Embeddings opcionales</div></div>
<div class="dash-grid"><div class="callout"><strong>Guardar original:</strong> conserva el archivo sin cambiar sus campos ni crear IDs o embeddings. Queda disponible en “Explorar bases” como <strong>Original</strong>.</div><div class="callout"><strong>Normalizar:</strong> asigna pregunta, respuesta y otros campos para crear registros estandarizados e IDs relacionados. La versión normalizada queda disponible en “Explorar bases”. Los embeddings son un paso posterior opcional.</div></div>
</section>
<section class="panel" id="normCustomPanel"><h3>1. Selecciona y revisa el archivo</h3><p class="norm-help">Admite CSV, TSV, JSON y JSONL. Puedes seleccionar el archivo o arrastrarlo al área. La vista previa solo lee una muestra para que puedas decidir qué hacer; aún no guarda ni normaliza la base.</p>
<input id="normFile" type="file" accept=".csv,.tsv,.json,.jsonl,.ndjson" hidden>
<div class="drop-zone" id="normDropZone" role="button" tabindex="0"><div><strong>Arrastra aquí tu archivo</strong><span>o haz clic para buscarlo en el equipo · máximo 20 MB</span></div></div>
<div class="filters" style="margin-top:14px"><div><label for="normCustomName">Nombre de la base</label><input id="normCustomName" type="text" placeholder="ej. mi_dataset"></div><div><label for="normRecordsPath">Ruta de registros (solo JSON anidado)</label><input id="normRecordsPath" type="text" placeholder="Se detecta automáticamente"><span class="small">Si cambias esta ruta, vuelve a previsualizar para actualizar los campos.</span></div><div><button id="normPreview">Previsualizar campos</button></div></div>
<p id="normFileInfo" class="small" role="status" aria-live="polite">Aún no se ha seleccionado un archivo.</p><div class="summary" style="margin:10px 0"><button id="normSaveOriginal" class="secondary" disabled>Guardar original para explorar (sin normalizar)</button><button id="normExploreOriginal" class="secondary" hidden>Abrir en Explorar bases</button><span id="normSaveStatus" class="small"></span></div><div id="normCustomPreview" hidden></div>
<section id="normCustomMapping" class="field-catalog" hidden><h3>2. Opción de normalización: convertir al formato común</h3><p class="norm-help">Esta opción es independiente de “Guardar original”. Si solo quieres consultar el archivo sin transformarlo, usa el botón anterior y no continúes aquí.</p><h4>Campos detectados y cobertura</h4><p class="norm-help">La cobertura indica cuántas filas contienen un valor. Elige las columnas y analiza la calidad antes de normalizar; podrás revisar los problemas por fila y descargar el informe CSV. Para JSON anidado puedes escribir una ruta relativa como <code>../../context</code>.</p><div class="table-wrap"><table><thead><tr><th>Campo detectado</th><th>Tipo</th><th>Filas con valor</th><th>Ejemplo</th><th>Uso</th></tr></thead><tbody id="normFieldRows"></tbody></table></div><p id="normFieldCount" class="small"></p><h4>Asignar campos del archivo</h4><p class="norm-help">Pregunta y respuesta son obligatorias. Contexto es opcional, pero permite crear los embeddings y comparaciones con los párrafos más adelante. Calificación y respuesta esperada se guardan como objetivos de evaluación.</p><div id="normMappings" class="norm-map-grid"></div><div class="filters" style="margin-top:12px"><div><button id="normQualityRun">Analizar calidad para normalizar</button></div><div><button id="normCustomStart" disabled>Normalizar y guardar versión estandarizada</button></div></div><section id="normQualityPanel" class="field-catalog" hidden><h4>Informe de calidad previo</h4><p id="normQualitySummary" class="callout" role="status" aria-live="polite"></p><div class="summary" id="normQualityMetrics"></div><div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>Revisión</th><th>Filas</th><th>Interpretación</th></tr></thead><tbody id="normQualityCounts"></tbody></table></div><h4>Filas que requieren atención</h4><p class="norm-help">Las advertencias no siempre impiden normalizar. Las filas sin pregunta/respuesta se omiten; las calificaciones no numéricas y respuestas vacías dentro de una lista sí bloquean el proceso.</p><div class="table-wrap" style="max-height:300px"><table><thead><tr><th>Fila</th><th>Hallazgo</th><th>Detalle</th></tr></thead><tbody id="normQualityIssues"></tbody></table></div><div class="summary" style="margin-top:12px"><button id="normQualityDownload" class="secondary">Descargar hallazgos CSV</button><span id="normQualityReady" class="small"></span></div></section></section>
</section>
<section class="panel"><h3>Normalizar una base que ya está en el proyecto</h3><p class="norm-help">Usa esta opción únicamente si quieres convertir una fuente local al formato común. No es necesaria para explorar una base que guardaste como original. BoolQ requiere sus archivos JSONL y configuración locales.</p>
<div class="filters" style="margin-top:18px">
<div><label for="normDataset">¿Qué base quieres normalizar?</label><select id="normDataset">
<option value="all">Todas las bases integradas</option><option value="squad2_train">SQuAD 2.0 · train</option><option value="squad2_dev">SQuAD 2.0 · dev</option><option value="cmu_qa_v1.2">CMU QA</option><option value="mohler_v1.0">Mohler · versión 1.0</option><option value="mohler_v2.0">Mohler · versión 2.0</option><option value="roars_2class">ROARS · 2 clases</option><option value="roars_3class">ROARS · 3 clases</option><option value="boolq_train">BoolQ · train</option><option value="boolq_dev">BoolQ · dev</option>
</select></div>
<div><label><input id="normDownload" type="checkbox"> Descargar originales oficiales que falten</label><p class="small">Sin marcar, procesa solo archivos locales. Esta opción no aplica a BoolQ, que requiere sus archivos JSONL y config locales.</p></div>
<div><button id="normStart">Iniciar normalización</button></div>
</div><p id="normStatus" class="small" role="status" aria-live="polite">Elige una base y ejecuta el proceso.</p>
<section id="normSummary" hidden><h3>Resumen de la ejecución</h3><div class="summary" id="normMetrics"></div><div class="summary" style="margin-top:10px"><button id="normExploreNormalized" class="secondary" hidden>Explorar base normalizada</button></div><div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>Base normalizada</th><th>Respuestas</th></tr></thead><tbody id="normRows"></tbody></table></div><p class="small" id="normOutput"></p><div class="callout"><strong>La base ya está disponible en Explorar bases como Normalizada (sin embeddings).</strong> Si también quieres similitudes y vectores, ejecuta este comando desde la carpeta <code>qa_embedding_features</code> para calcular embeddings y similitudes; requiere que <code>OPENAI_API_KEY</code> esté configurada o que añadas <code>--api-key-file "C:\ruta\KEY.txt"</code>.<pre id="normNextCommand" class="pre"></pre></div></section>
<details class="panel"><summary style="cursor:pointer;font-weight:700;color:#173e59">¿Cómo convierte el archivo a un formato normalizado?</summary><ol><li><strong>Lee y revisa:</strong> detecta filas y campos, y presenta una muestra y la cobertura de cada campo.</li><li><strong>Mapea:</strong> indicas dónde están la pregunta y la respuesta; contexto, respuesta esperada, calificación e ID de origen son opcionales.</li><li><strong>Estandariza:</strong> cada fila válida se convierte a un registro común. El normalizador crea IDs de contexto, pregunta, respuesta y calificación relacionados; si hay varias respuestas en una fila, crea un registro por respuesta.</li><li><strong>Organiza la salida:</strong> separa el contexto en párrafos y actualiza JSONL, CSV y el manifiesto del proyecto. Las filas sin pregunta o sin respuesta se omiten; una calificación que no se pueda convertir a número detiene el proceso con un error.</li></ol><p class="small">El archivo subido se conserva en <code>qa_dataset_project/originales/adicionales/</code> con nombre derivado de su hash SHA-256; el mapeo queda en <code>config_&lt;nombre&gt;.json</code>. Las salidas quedan en <code>qa_dataset_project/resultados/</code>, incluyendo <code>por_base/&lt;nombre&gt;.jsonl</code>, <code>contextos.jsonl</code>, <code>parrafos.jsonl</code> y <code>manifest.json</code>. Esta etapa prepara los datos; el cálculo de embeddings es posterior.</p></details>
</section>
</div>
<div id="deleteDialog" class="delete-backdrop" role="dialog" aria-modal="true" aria-labelledby="deleteTitle" hidden><section class="delete-dialog"><h2 id="deleteTitle">Confirmar eliminación</h2><p id="deleteMessage"></p><div class="delete-actions"><button id="deleteCancel" class="secondary">Cancelar</button><button id="deleteConfirm" class="danger">Eliminar base</button></div></section></div></main><script>
const $=id=>document.getElementById(id), simLabels={
 question_answer:'Pregunta – respuesta',question_context_centroid:'Pregunta – centroide del contexto',answer_context_centroid:'Respuesta – centroide del contexto',
 question_nearest_question_paragraph:'Pregunta – párrafo más cercano a la pregunta',question_nearest_answer_paragraph:'Pregunta – párrafo más cercano a la respuesta',
 answer_nearest_question_paragraph:'Respuesta – párrafo más cercano a la pregunta',answer_nearest_answer_paragraph:'Respuesta – párrafo más cercano a la respuesta'};
let datasetCatalog=[],workflowDataset='',workflowFields=[],workflowCatalog=[],workflowOffset=0,workflowPageSize=100,workflowTotal=0,sourceOffset=0,sourcePageSize=50,sourceTotal=0,sourceFields=[];function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined&&text!==null)x.textContent=String(text);if(cls)x.className=cls;return x}
function add(parent,tag,text,cls){const x=el(tag,text,cls);parent.appendChild(x);return x}
function renderDashboard(datasets){const valid=datasets.filter(d=>Number(d.responses)>0),total=valid.reduce((n,d)=>n+Number(d.responses||0),0),processed=valid.filter(d=>d.provenance?.kind!=='original'),processedTotal=processed.reduce((n,d)=>n+Number(d.responses||0),0),scored=processed.reduce((n,d)=>n+Number(d.with_score_target||0),0);$('kpis').replaceChildren();for(const [value,label]of [[total.toLocaleString(),'registros disponibles para explorar'],[valid.length.toLocaleString(),'bases disponibles'],[scored.toLocaleString(),'registros de vistas con calificación'],[Math.max(0,processedTotal-scored).toLocaleString(),'vistas sin calificación numérica']]){const card=add($('kpis'),'div',undefined,'kpi');add(card,'strong',value);add(card,'span',label)}const chart=$('datasetChart');chart.replaceChildren();const max=Math.max(1,...valid.map(d=>Number(d.responses||0)));for(const d of [...valid].sort((a,b)=>Number(b.responses)-Number(a.responses))){const row=add(chart,'div',undefined,'chart-row');row.title='Pulsa para explorar esta base';row.style.cursor='pointer';add(row,'span',`${d.display_name||d.name} · ${d.provenance?.label||'Sin estado'}`);const track=add(row,'div',undefined,'chart-track');const fill=add(track,'div',undefined,'chart-fill');fill.style.width=(100*Number(d.responses||0)/max)+'%';add(row,'b',Number(d.responses||0).toLocaleString());row.onclick=()=>{const option=[...$('dataset').options].find(o=>o.value===d.name);if(option){$('dataset').value=d.name;configureExplorerMode();tab('explore');search();$('exploreView').scrollIntoView({behavior:'smooth'})}}}const score=$('scoreChart');score.replaceChildren();for(const [label,n,cls]of [['Con calificación',scored,'train'],['Sin calificación numérica',Math.max(0,processedTotal-scored),'dev']]){const row=add(score,'div',undefined,'chart-row');add(row,'span',label);const track=add(row,'div',undefined,'chart-track');const fill=add(track,'div',undefined,'chart-fill '+cls);fill.style.width=(processedTotal?100*n/processedTotal:0)+'%';add(row,'b',`${processedTotal?(100*n/processedTotal).toFixed(1):'0.0'}%`)}const train=valid.find(d=>d.name==='squad2_train'),dev=valid.find(d=>d.name==='squad2_dev');$('splitCallout').textContent=train&&dev?`SQuAD 2.0 está separado en ${Number(train.responses).toLocaleString()} ejemplos de train y ${Number(dev.responses).toLocaleString()} de dev. Conserva dev para evaluación y train para entrenamiento.`:`Las calificaciones solo se muestran cuando existen en la fuente. La similitud coseno es una característica y no equivale a una nota.`}
function selectedDatasetInfo(){return datasetCatalog.find(item=>item.name===$('dataset').value)}
function configureExplorerMode(){const info=selectedDatasetInfo(),kind=info?.provenance?.kind,sourceMode=kind==='original'||kind==='normalized_only',select=$('field'),previous=select.value;select.replaceChildren();const all=el('option',sourceMode?'Todos los campos':'Todo (incluye contexto)');all.value='all';select.appendChild(all);if(sourceMode){for(const field of info.provenance.fields||[]){const option=el('option',field);option.value=field;select.appendChild(option)}select.value=(info.provenance.fields||[]).includes(previous)?previous:'all';$('sim').disabled=true;$('minimum').disabled=true;$('scored').disabled=true;$('query').placeholder=kind==='original'?'Texto o campo original':'Texto o campo normalizado';$('searchHelp').textContent=kind==='original'?'Base original sin normalizar: busca texto en los campos de origen. No hay IDs normalizados, embeddings ni similitudes.':'Base normalizada lista para explorar. Conserva los IDs y campos comunes, pero todavía no tiene embeddings ni similitudes.';$('downloadDataset').textContent=kind==='original'?'Descargar archivo original':'Descargar JSONL normalizado'}else{for(const [value,label]of [['question','Pregunta'],['answer','Respuesta'],['paragraph','Párrafos'],['context','Contexto'],['ids','IDs']]){const option=el('option',label);option.value=value;select.appendChild(option)}select.value=['all','question','answer','paragraph','context','ids'].includes(previous)?previous:'all';$('sim').disabled=false;$('minimum').disabled=false;$('scored').disabled=false;$('query').placeholder='Pregunta, respuesta, contexto o ID';$('searchHelp').textContent='La búsqueda de texto ignora mayúsculas y tildes. El filtro de similitud solo devuelve filas que tienen ese valor calculado.';$('downloadDataset').textContent='Descargar vista completa en JSONL'}$('deleteDataset').disabled=!info||kind==='normalized_only'}
function renderOriginalHeaders(fields,label='Fila original'){const head=$('resultsHead');head.replaceChildren();const row=add(head,'tr');add(row,'th',label);for(const field of fields)add(row,'th',field);add(row,'th','Detalle')}
function renderNormalizedHeaders(){const head=$('resultsHead');head.replaceChildren();const row=add(head,'tr');for(const label of ['Respuesta ID','Pregunta','Respuesta','Similitud','Calificación',''])add(row,'th',label)}
function rawDisplayValue(value){if(value==null)return '';if(typeof value==='object')return JSON.stringify(value);return String(value)}
function showSourceRecord(dataset,rowNumber,record){$('detailPanel').hidden=false;$('detail').replaceChildren();$('detailPanel').scrollIntoView({behavior:'smooth'});const entry=selectedDatasetInfo(),original=entry?.provenance?.kind==='original';const summary=add($('detail'),'div',undefined,'summary');add(summary,'span',entry?.display_name||dataset,'pill');add(summary,'span',`${original?'Original':'Normalizada'} · fila ${rowNumber}`,'pill');add($('detail'),'p',original?'El registro se muestra tal como aparece en el archivo. No se le han asignado IDs normalizados ni embeddings.':'Registro en el esquema común. Los IDs están disponibles; los embeddings y las similitudes se calculan en una etapa posterior.','callout');if(entry?.provenance?.original_path)add($('detail'),'p','Archivo fuente: '+entry.provenance.original_path,'small');if(!original){const ids=add($('detail'),'div',undefined,'refs');for(const field of ['context_id','question_id','answer_id','score_id','paragraph_id'])if(record[field])add(ids,'span',`${field}: ${record[field]}`,'pill')}add($('detail'),'pre',JSON.stringify(record,null,2),'pre')}
async function refreshDatasets(){const data=await(await fetch('/api/datasets')).json();datasetCatalog=data.datasets||[];renderDashboard(datasetCatalog);const select=$('dataset'),previous=select.value;select.replaceChildren();for(const d of datasetCatalog){const count=Number(d.responses||0),state=d.provenance?.label||'Estado desconocido',original=d.provenance?.original_available?'fuente disponible':'fuente no localizada',o=el('option',`${d.display_name||d.name} · ${state} · ${original} · ${count.toLocaleString()} registros`);o.value=d.name;select.appendChild(o)}if(data.datasets.some(d=>d.name===previous))select.value=previous;$('deleteDataset').disabled=!data.datasets.length;return data.datasets}
async function init(){const datasets=await refreshDatasets();configureExplorerMode();for(const [key,label] of Object.entries(simLabels)){const o=el('option',label);o.value=key;$('sim').appendChild(o)}if(!datasets.length){$('status').textContent='Aún no hay vistas. Ejecuta preparar_vistas.py para generar una vista.';return}await search()}
async function loadWorkflowPage(offset=workflowOffset){workflowOffset=Math.max(0,offset);workflowPageSize=Number($('wfPageSize').value);const query=new URLSearchParams({dataset:workflowDataset,limit:String(workflowPageSize),offset:String(workflowOffset)});const data=await(await fetch('/api/search?'+query)).json();if(data.error)throw Error(data.error);workflowTotal=data.total_matches;workflowOffset=data.offset||0;const pages=Math.max(1,Math.ceil(workflowTotal/workflowPageSize));const page=Math.floor(workflowOffset/workflowPageSize)+1;const first=workflowTotal?workflowOffset+1:0;const last=Math.min(workflowOffset+data.matches.length,workflowTotal);$('wfPageInfo').textContent=`Página ${page.toLocaleString()} de ${pages.toLocaleString()} · ${first.toLocaleString()}–${last.toLocaleString()} de ${workflowTotal.toLocaleString()}`;$('wfPageNumber').max=String(pages);$('wfPageNumber').value=String(page);$('wfPrevPage').disabled=workflowOffset===0;$('wfNextPage').disabled=!data.has_next;$('wfDownloadAll').href='/api/dataset/download?dataset='+encodeURIComponent(workflowDataset);const body=$('wfResults');body.replaceChildren();for(const row of data.matches){const tr=add(body,'tr');add(tr,'td',row.ids.answer_id||'','id');add(tr,'td',(row.texts.question||'').slice(0,220));add(tr,'td',(row.texts.candidate_answer||'').slice(0,220));const value=row.cosine_similarities.question_answer;add(tr,'td',value==null?'—':Number(value).toFixed(3));add(tr,'td',row.targets.score==null?'—':`${row.targets.score} ${row.targets.score_scale||''}`)}}
function workflowPageBy(page){const pages=Math.max(1,Math.ceil(workflowTotal/workflowPageSize));loadWorkflowPage((Math.max(1,Math.min(pages,Number(page)||1))-1)*workflowPageSize).catch(e=>{$('wfExportStatus').textContent='Error cargando registros: '+e.message})}
$('wfPageSize').addEventListener('change',()=>workflowPageBy(1));for(const [id,delta]of [['wfPrevPage',-1],['wfNextPage',1]])$(id).addEventListener('click',()=>workflowPageBy(Math.floor(workflowOffset/workflowPageSize)+1+delta));$('wfGoPage').addEventListener('click',()=>workflowPageBy($('wfPageNumber').value));$('wfPageNumber').addEventListener('keydown',e=>{if(e.key==='Enter')workflowPageBy($('wfPageNumber').value)});
function sourceCellValue(row,path){function read(value,parts,index){if(Array.isArray(value)){return value.map(item=>read(item,parts,index)).filter(item=>item!==''&&item!=null).join(' · ')}if(index>=parts.length)return value;if(parts[index]==='*')return read(value,parts,index+1);return read(value&&typeof value==='object'?value[parts[index]]:undefined,parts,index+1)}const value=read(row,path.split('.'),0);if(value==null)return '';return typeof value==='object'?JSON.stringify(value):String(value)}
async function loadSourcePreviewPage(offset=sourceOffset){sourceOffset=Math.max(0,offset);sourcePageSize=Number($('wfSourcePageSize').value);const params=new URLSearchParams({upload_id:workflowDataset,offset:String(sourceOffset),limit:String(sourcePageSize)});const response=await fetch('/api/workflow/preview-page?'+params);const data=await response.json();if(!response.ok||data.error)throw Error(data.error||'No se pudo cargar la página del archivo.');sourceTotal=data.total;sourceOffset=data.offset||0;sourceFields=data.fields||[];const page=Math.floor(sourceOffset/sourcePageSize)+1;const pages=Math.max(1,Math.ceil(sourceTotal/sourcePageSize));const first=sourceTotal?sourceOffset+1:0;const last=Math.min(sourceOffset+data.rows.length,sourceTotal);$('wfSourcePageInfo').textContent=`Archivo original · página ${page.toLocaleString()} de ${pages.toLocaleString()} · filas ${first.toLocaleString()}–${last.toLocaleString()} de ${sourceTotal.toLocaleString()}`;$('wfSourcePageNumber').max=String(pages);$('wfSourcePageNumber').value=String(page);$('wfSourcePrev').disabled=sourceOffset===0;$('wfSourceNext').disabled=!data.has_next;const head=$('wfSourceHead');head.replaceChildren();const header=add(head,'tr');for(const field of sourceFields)add(header,'th',field);const body=$('wfSourceRows');body.replaceChildren();for(const record of data.rows){const tr=add(body,'tr');for(const field of sourceFields){const value=sourceCellValue(record,field);const cell=add(tr,'td',value);cell.title=value}}$('wfSourcePager').hidden=false;$('wfSourceTableWrap').hidden=false}
function sourcePageBy(page){const pages=Math.max(1,Math.ceil(sourceTotal/sourcePageSize));loadSourcePreviewPage((Math.max(1,Math.min(pages,Number(page)||1))-1)*sourcePageSize).catch(e=>{$('wfStatus').textContent='Error leyendo página de origen: '+e.message})}
$('wfSourcePageSize').addEventListener('change',()=>sourcePageBy(1));$('wfSourcePrev').addEventListener('click',()=>sourcePageBy(Math.floor(sourceOffset/sourcePageSize)));$('wfSourceNext').addEventListener('click',()=>sourcePageBy(Math.floor(sourceOffset/sourcePageSize)+2));$('wfSourceGo').addEventListener('click',()=>sourcePageBy($('wfSourcePageNumber').value));$('wfSourcePageNumber').addEventListener('keydown',e=>{if(e.key==='Enter')sourcePageBy($('wfSourcePageNumber').value)});
let currentOffset=0;
async function search(reset=true){configureExplorerMode();if(reset)currentOffset=0;const body=$('results');body.replaceChildren();$('status').textContent='Buscando…';$('summary').replaceChildren();$('detailPanel').hidden=true;const pageSize=Number($('limit').value);const p=new URLSearchParams({dataset:$('dataset').value||'',q:$('query').value,field:$('field').value,similarity:$('sim').value,minimum:$('minimum').value,limit:String(pageSize),offset:String(currentOffset),scored:$('scored').checked?'1':'0'});try{const kind=selectedDatasetInfo()?.provenance?.kind,sourceMode=kind==='original'||kind==='normalized_only';const endpoint=kind==='original'?'/api/original/search?':kind==='normalized_only'?'/api/normalized/search?':'/api/search?';const data=await(await fetch(endpoint+p)).json();if(data.error)throw Error(data.error);currentOffset=data.offset||0;const page=Math.floor(currentOffset/pageSize)+1;const pages=Math.max(1,Math.ceil(data.total_matches/pageSize));const first=data.total_matches?currentOffset+1:0;const last=Math.min(currentOffset+data.matches.length,data.total_matches);$('pager').hidden=false;$('pageInfo').textContent=`Página ${page.toLocaleString()} de ${pages.toLocaleString()} · ${first.toLocaleString()}–${last.toLocaleString()} de ${data.total_matches.toLocaleString()}`;$('pageNumber').max=String(pages);$('pageNumber').value=String(page);$('prevPage').disabled=currentOffset===0;$('nextPage').disabled=!data.has_next;$('downloadDataset').href=(kind==='original'?'/api/original/download?dataset=':kind==='normalized_only'?'/api/normalized/download?dataset=':'/api/dataset/download?dataset=')+encodeURIComponent(data.dataset);$('status').textContent=data.scanned<data.total_matches?`Leídos ${data.scanned.toLocaleString()} registros para esta página.`:`Revisados ${data.scanned.toLocaleString()} registros para aplicar los filtros.`;const info=selectedDatasetInfo(),provenance=info?.provenance;add($('summary'),'span',info?.display_name||data.dataset,'pill');if(provenance){const badge=add($('summary'),'span',`Estado: ${provenance.label}`,'pill');badge.classList.add(provenance.kind==='normalized'?'state-normalized':provenance.kind==='original'?'state-original':'state-direct');if(provenance.kind!=='original'){const originalBadge=add($('summary'),'span',provenance.original_available?'Fuente original disponible':'Fuente original no localizada','pill');originalBadge.classList.add(provenance.original_available?'state-normalized':'state-original')}}add($('summary'),'span',`${data.total_matches.toLocaleString()} coincidencias en total`,'pill');if(sourceMode){renderOriginalHeaders(data.fields||[],kind==='original'?'Fila original':'Fila normalizada');for(const match of data.matches){const tr=add(body,'tr');add(tr,'td',`${kind==='original'?'ORIG':'NORM'}-${String(match.row_number).padStart(6,'0')}`,'id');for(const field of data.fields||[]){const full=rawDisplayValue(match.values?.[field]??sourceCellValue(match.record,field));const cell=add(tr,'td',full.length>240?full.slice(0,240)+'…':full);cell.title=full}const action=add(tr,'td');const button=add(action,'button','Ver','secondary');button.onclick=()=>showSourceRecord(data.dataset,match.row_number,match.record)}}else{renderNormalizedHeaders();for(const row of data.matches){const tr=add(body,'tr');add(tr,'td',row.ids.answer_id||'','id');add(tr,'td',(row.texts.question||'').slice(0,180));add(tr,'td',(row.texts.candidate_answer||'').slice(0,180));add(tr,'td',row.cosine_similarities.question_answer==null?'—':Number(row.cosine_similarities.question_answer).toFixed(3));add(tr,'td',row.targets.score==null?'—':`${row.targets.score} ${row.targets.score_scale||''}`);const action=add(tr,'td');const b=add(action,'button','Ver','secondary');b.onclick=()=>showRecord(data.dataset,row.ids.answer_id)}}}catch(e){$('status').textContent='Error: '+e.message;$('status').className='small err'}}
function section(parent,title){const s=add(parent,'section');add(s,'h3',title);return s}
function code(parent,label,value){const p=add(parent,'p');add(p,'b',label+': ');add(p,'code',value||'—','id')}
async function showRecord(dataset,answerId){$('detailPanel').hidden=false;$('detail').replaceChildren(el('p','Cargando registro...','spinner'));$('detailPanel').scrollIntoView({behavior:'smooth'});try{const r=await(await fetch('/api/record?'+new URLSearchParams({dataset,answer_id:answerId}))).json();if(r.error)throw Error(r.error);const root=$('detail');root.replaceChildren();const grid=add(root,'div',undefined,'record-grid');const ids=section(grid,'IDs y procedencia');for(const [k,v]of Object.entries(r.ids||{}))code(ids,k,v);const provenance=datasetCatalog.find(item=>item.name===r.dataset)?.provenance;if(provenance){const badge=add(ids,'p',`Estado de la base: ${provenance.label}`,'pill');badge.classList.add(provenance.kind==='normalized'?'state-normalized':'state-direct');const originalBadge=add(ids,'p',provenance.original_available?'Fuente original disponible':'Fuente original no localizada','pill');originalBadge.classList.add(provenance.original_available?'state-normalized':'state-original');if(provenance.original_available)add(ids,'p',`Ruta del original: ${provenance.original_path}`,'small');else add(ids,'p','No se localizó una copia local del archivo fuente.','small');if(provenance.normalized_path)add(ids,'p',`Archivo normalizado: ${provenance.normalized_path}`,'small')}add(ids,'p',`${r.dataset} · ${r.source?.source_file||''}`,'small');const texts=section(grid,'Pregunta y respuesta');add(texts,'b','Pregunta');add(texts,'div',r.texts.question||'—','pre');add(texts,'b','Respuesta candidata');add(texts,'div',r.texts.candidate_answer||'—','pre');const near=section(grid,'Párrafos seleccionados');for(const [key,title]of [['paragraph_nearest_question','Más cercano a la pregunta'],['paragraph_nearest_answer','Más cercano a la respuesta']]){const x=r.texts[key];add(near,'b',title);add(near,'div',x?.paragraph_text||'Sin contexto o párrafo','pre');if(x)code(near,'paragraph_id',x.paragraph_id)}const simsec=section(root,'Similitudes coseno');for(const [key,label]of Object.entries(simLabels)){const value=r.cosine_similarities[key];const line=add(simsec,'div',undefined,'sim-row');add(line,'span',label);const bar=add(line,'div',undefined,'bar');if(value!=null){const fill=add(bar,'div',undefined,'fill');fill.style.width=((Math.max(-1,Math.min(1,value))+1)*50)+'%';if(value<0)fill.classList.add('bar-negative')}add(line,'b',value==null?'—':Number(value).toFixed(4))}const inp=section(root,'Entradas y objetivos del modelo');add(inp,'b','Entrada para calificar');add(inp,'pre',JSON.stringify(r.model_inputs?.grading,null,2),'pre');add(inp,'b','Entrada para generar respuesta (sin respuesta candidata)');add(inp,'pre',JSON.stringify(r.model_inputs?.answer_generation,null,2),'pre');add(inp,'b','Objetivos disponibles');add(inp,'pre',JSON.stringify(r.targets,null,2),'pre');const refs=section(root,'Embeddings');const refsBox=add(refs,'div',undefined,'refs');const flat={question:r.embedding_refs?.question_embedding_id,answer:r.embedding_refs?.answer_embedding_id,context_centroid:r.embedding_refs?.context_centroid_embedding_id,paragraph_nearest_question:r.embedding_refs?.paragraph_nearest_question?.embedding_id,paragraph_nearest_answer:r.embedding_refs?.paragraph_nearest_answer?.embedding_id};for(const [kind,id]of Object.entries(flat)){if(!id)continue;const b=add(refsBox,'button',kind+': '+id,'secondary');b.onclick=()=>showVector(id)}const vc=add(refs,'pre','','pre');vc.id='vectorPreview';}catch(e){$('detail').textContent='Error: '+e.message}}
async function showVector(id){const node=$('vectorPreview');node.textContent='Cargando vector…';try{const v=await(await fetch('/api/vector?id='+encodeURIComponent(id))).json();if(v.error)throw Error(v.error);node.textContent=`${v.embedding_id} · ${v.kind} · ${v.model} · ${v.dimensions} dimensiones\nPrimeros 12 valores: ${JSON.stringify(v.vector.slice(0,12))}\nEl vector completo se puede descargar desde el enlace siguiente.`;const blob=new Blob([JSON.stringify(v)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=id+'.json';a.textContent='Descargar vector completo como JSON';node.appendChild(document.createElement('br'));node.appendChild(a)}catch(e){node.textContent='Error: '+e.message}}
let pendingDeleteDataset='';
$('deleteDataset').addEventListener('click',()=>{const dataset=$('dataset').value;if(!dataset)return;pendingDeleteDataset=dataset;const selected=selectedDatasetInfo();$('deleteMessage').textContent=selected?.provenance?.kind==='original'?`Vas a quitar ${selected.display_name||dataset} del explorador. El archivo original guardado en qa_dataset_project se conservará.`:`Vas a borrar la vista guardada de ${dataset}, sus exportaciones y, si se importó desde el flujo de trabajo, el archivo cargado asociado. Los originales y datos normalizados del proyecto se conservan. Los vectores compartidos de ChromaDB se mantienen.`;$('deleteDialog').hidden=false;$('deleteConfirm').focus()});
$('deleteCancel').addEventListener('click',()=>{$('deleteDialog').hidden=true;pendingDeleteDataset=''});
$('deleteConfirm').addEventListener('click',async()=>{const dataset=pendingDeleteDataset;if(!dataset)return;const button=$('deleteConfirm');button.disabled=true;try{const response=await fetch('/api/dataset/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset})});const data=await response.json();if(!response.ok||data.error)throw Error(data.error||`El servidor respondió ${response.status}.`);$('deleteDialog').hidden=true;pendingDeleteDataset='';await refreshDatasets();$('detailPanel').hidden=true;$('results').replaceChildren();if($('dataset').value)await search();else{$('summary').replaceChildren(el('span','No hay bases disponibles.','muted'));$('status').textContent='';$('pager').hidden=true}$('status').textContent=data.source_preserved?`Se quitó ${dataset} del explorador. El archivo original se conservó.`:`Se eliminó la vista de ${dataset} y ${data.removed_exports} exportación(es).`;}catch(e){$('deleteMessage').textContent='No se pudo eliminar: '+e.message;}finally{button.disabled=false;$('deleteDataset').disabled=!$('dataset').value}});
$('search').addEventListener('click',()=>search(true));$('query').addEventListener('keydown',e=>{if(e.key==='Enter')search(true)});$('limit').addEventListener('change',()=>search(true));$('prevPage').addEventListener('click',()=>{currentOffset=Math.max(0,currentOffset-Number($('limit').value));search(false)});$('nextPage').addEventListener('click',()=>{currentOffset+=Number($('limit').value);search(false)});$('goPage').addEventListener('click',()=>{const page=Math.max(1,Math.min(Number($('pageNumber').max)||1,Number($('pageNumber').value)||1));currentOffset=(page-1)*Number($('limit').value);search(false)});$('pageNumber').addEventListener('keydown',e=>{if(e.key==='Enter')$('goPage').click()});$('dataset').addEventListener('change',()=>{currentOffset=0;$('detailPanel').hidden=true;configureExplorerMode()});init().catch(e=>{$('status').textContent='Error: '+e.message;$('status').className='small err'});loadNormalizationStatus().catch(e=>{$('normStatus').textContent='No se pudo consultar el proceso: '+e.message});
let normUploadId='',normUploadMeta=null,normFieldCoverage=[],normMappings={};
const normRoles={normQ:['question_text','Pregunta',true],normA:['answer_text','Respuesta',true],normC:['context_text','Contexto',false],normRef:['reference_answer','Respuesta esperada',false],normScore:['score','Calificación',false],normUnans:['is_unanswerable','¿No respondible?',false],normSource:['source_id','ID de origen',false]};
function normalizationFile(file){if(!file)return;$('normFile')._selected=null;normUploadId='';normUploadMeta=null;normQualityReport=null;$('normQualityPanel').hidden=true;$('normQualityRun').disabled=true;$('normCustomStart').disabled=true;$('normSaveOriginal').disabled=true;$('normExploreOriginal').hidden=true;$('normSaveStatus').textContent='';$('normCustomMapping').hidden=true;$('normCustomPreview').hidden=true;const allowed=['.csv','.tsv','.json','.jsonl','.ndjson'];const ext='.'+file.name.split('.').pop().toLowerCase();if(!allowed.includes(ext)){$('normFileInfo').textContent='Formato no admitido. Usa CSV, TSV, JSON o JSONL.';return}if(file.size>20*1024*1024){$('normFileInfo').textContent='El archivo supera el límite de 20 MB de esta vista.';return}normUploadId='';normUploadMeta=null;normQualityReport=null;$('normQualityPanel').hidden=true;$('normQualityRun').disabled=true;$('normCustomStart').disabled=true;$('normCustomMapping').hidden=true;$('normCustomPreview').hidden=true;$('normCustomName').value=file.name.replace(/\.[^.]+$/,'').replace(/[^a-zA-Z0-9_-]+/g,'_').replace(/^_+|_+$/g,'').toLowerCase();$('normFileInfo').textContent=`Archivo seleccionado: ${file.name} · ${(file.size/1048576).toFixed(2)} MB`;$('normFile')._selected=file}
$('normDropZone').addEventListener('click',()=>$('normFile').click());$('normDropZone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('normFile').click()}});$('normFile').addEventListener('change',e=>normalizationFile(e.target.files[0]));for(const eventName of ['dragenter','dragover'])$('normDropZone').addEventListener(eventName,e=>{e.preventDefault();$('normDropZone').classList.add('dragover')});for(const eventName of ['dragleave','drop'])$('normDropZone').addEventListener(eventName,e=>{e.preventDefault();$('normDropZone').classList.remove('dragover')});$('normDropZone').addEventListener('drop',e=>normalizationFile(e.dataTransfer.files[0]));
function normSelector(field){const records=(normUploadMeta?.records_path||'').replace(/\s/g,'');return records&&field.startsWith(records+'.')?field.slice(records.length+1):field}
function normMapValue(id){return $('map_'+id+'_custom')?.value.trim()||$('map_'+id)?.value||''}
let normQualityReport=null;
function invalidateNormalizationQuality(){normQualityReport=null;$('normQualityPanel').hidden=true;$('normCustomStart').disabled=true;$('normQualityRun').disabled=!normUploadId}
$('normMappings').addEventListener('input',()=>{invalidateNormalizationQuality();renderNormMappingRoles()});$('normMappings').addEventListener('change',()=>{invalidateNormalizationQuality();renderNormMappingRoles()});
function normQualityPayload(){const fields={};for(const [id,[key,label,required]]of Object.entries(normRoles)){const value=normMapValue(id);if(required&&!value)throw Error(`Asigna el campo obligatorio: ${label}.`);if(value&&key!=='source_id')fields[key]=value}return {upload_id:normUploadId,records_path:$('normRecordsPath').value.trim(),fields,source_id:normMapValue('normSource')}}
function csvCell(value){return '"'+String(value??'').replace(/"/g,'""')+'"'}
function renderNormalizationQuality(report){normQualityReport=report;$('normQualityPanel').hidden=false;$('normQualitySummary').textContent=report.summary;$('normQualitySummary').className='callout'+(report.ready_for_normalization?'':' err');const labels={source_records:['Registros detectados','Filas encontradas en el archivo.'],usable_records:['Registros utilizables','Filas con pregunta y respuesta (o etiqueta no respondible).'],output_answers:['Respuestas que se crearían','Cantidad de filas normalizadas estimada.'],missing_question:['Pregunta vacía','Se omitirá esa fila.'],missing_answer:['Respuesta vacía','Se omitirá esa fila salvo que esté marcada como no respondible.'],empty_context:['Contexto vacío','Es válido, pero no habrá características basadas en contexto.'],invalid_score:['Calificación no numérica','Error bloqueante: asigna una nota numérica o quita ese mapeo.'],invalid_answer_items:['Respuestas vacías en una lista','Error bloqueante: corrige o elimina elementos vacíos.'],duplicate_source_ids:['IDs de origen repetidos','Advertencia: revisa que se trate del mismo registro.'],duplicate_question_answer:['Parejas pregunta/respuesta repetidas','Advertencia: revisa si son duplicados intencionales.']};const counts=report.counts||{};$('normQualityMetrics').replaceChildren();for(const [key,label]of [['source_records','Filas analizadas'],['usable_records','Filas utilizables'],['output_answers','Respuestas previstas']])add($('normQualityMetrics'),'span',`${label}: ${Number(counts[key]||0).toLocaleString()}`,'pill');const body=$('normQualityCounts');body.replaceChildren();for(const [key,[label,meaning]]of Object.entries(labels)){const count=Number(counts[key]||0);if(key==='source_records'||key==='usable_records'||key==='output_answers'||count){const tr=add(body,'tr');add(tr,'td',label);add(tr,'td',count.toLocaleString());add(tr,'td',meaning,count&&['invalid_score','invalid_answer_items'].includes(key)?'err':'small')}}const issues=$('normQualityIssues');issues.replaceChildren();for(const item of report.issues||[]){const row=add(issues,'tr');add(row,'td',item.row);add(row,'td',item.category);add(row,'td',item.detail,item.blocking?'err':'')}if(report.issues_truncated)add(issues,'tr').appendChild(el('td','La lista visible se limitó a 1.000 hallazgos; el resumen incluye todas las filas.','small'));$('normQualityReady').textContent=report.ready_for_normalization?'Puedes continuar con la normalización.':'Corrige los errores bloqueantes o revisa el mapeo.';$('normCustomStart').disabled=!report.ready_for_normalization;$('normQualityDownload').disabled=!(report.issues||[]).length}
$('normQualityRun').addEventListener('click',async()=>{const button=$('normQualityRun');try{if(!normUploadId)throw Error('Vuelve a previsualizar el archivo.');const payload=normQualityPayload();button.disabled=true;button.textContent='Analizando…';$('normQualityPanel').hidden=true;const response=await fetch('/api/normalize/quality',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const report=await response.json();if(!response.ok||report.error)throw Error(report.error||`El servidor respondió ${response.status}.`);renderNormalizationQuality(report)}catch(e){$('normQualityPanel').hidden=false;$('normQualitySummary').textContent='No se pudo analizar la calidad: '+e.message;$('normQualitySummary').className='callout err';$('normCustomStart').disabled=true}finally{button.disabled=false;button.textContent='Analizar calidad'}});
$('normQualityDownload').addEventListener('click',()=>{if(!normQualityReport)return;const lines=[['fila','hallazgo','detalle'],...(normQualityReport.issues||[]).map(x=>[x.row,x.category,x.detail])].map(row=>row.map(csvCell).join(',')).join('\r\n');const blob=new Blob(['\uFEFF'+lines],{type:'text/csv;charset=utf-8'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=($('normCustomName').value||'dataset')+'_informe_calidad.csv';link.click();URL.revokeObjectURL(link.href)});
function renderNormMappingRoles(){const roles={};for(const [id,[key,label]]of Object.entries(normRoles)){const value=normMapValue(id);if(value)roles[value]=(roles[value]?roles[value]+', ':'')+label}const body=$('normFieldRows');body.replaceChildren();for(const item of normFieldCoverage){const tr=add(body,'tr');add(tr,'td',item.field,'id');add(tr,'td',item.type);add(tr,'td',`${Number(item.present).toLocaleString()} / ${Number(item.total).toLocaleString()} (${(100*Number(item.coverage||0)).toFixed(0)}%)`);add(tr,'td',item.example||'—');const role=roles[normSelector(item.field)]||roles[item.field];add(tr,'td',role||'—',role?'field-role':'small')}$('normFieldCount').textContent=`${normFieldCoverage.length.toLocaleString()} campos detectados en ${Number(normUploadMeta?.count||0).toLocaleString()} registros.`}
function createNormMapping(id,[field,label,required]){const box=add($('normMappings'),'div',undefined,'norm-map');const lab=add(box,'label',label+(required?' *':''));lab.htmlFor='map_'+id;const sel=add(box,'select');sel.id='map_'+id;const empty=el('option','— sin asignar —');empty.value='';sel.appendChild(empty);for(const f of normUploadMeta.fields){const o=el('option',f);o.value=normSelector(f);o.title=f;sel.appendChild(o)}const custom=add(box,'input');custom.id='map_'+id+'_custom';custom.placeholder='Ruta manual opcional (ej. ../../context)';add(box,'p',required?'Obligatorio para generar un registro normalizado.':'Opcional; si falta, el campo queda vacío.','norm-help');sel.addEventListener('change',renderNormMappingRoles);custom.addEventListener('input',renderNormMappingRoles)}
$('normRecordsPath').addEventListener('input',()=>{if(normUploadId){normUploadId='';normQualityReport=null;$('normQualityPanel').hidden=true;$('normQualityRun').disabled=true;$('normCustomStart').disabled=true;$('normSaveOriginal').disabled=true;$('normCustomMapping').hidden=true;$('normFileInfo').textContent='Cambiaste la ruta de registros. Previsualiza otra vez para actualizar los campos.'}});
let lastRawDataset='';
$('normSaveOriginal').addEventListener('click',async()=>{if(!normUploadId||!normUploadMeta){$('normSaveStatus').textContent='Primero previsualiza el archivo.';return}const name=$('normCustomName').value.trim();if(!name){$('normSaveStatus').textContent='Escribe un nombre para la base.';return}if(!window.confirm(`Guardar “${name}” como original sin normalizar. No se crearán IDs, embeddings ni similitudes. ¿Continuar?`))return;const button=$('normSaveOriginal');button.disabled=true;$('normSaveStatus').textContent='Guardando una copia original…';try{const response=await fetch('/api/original/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({upload_id:normUploadId,name,records_path:$('normRecordsPath').value.trim(),fields:normUploadMeta.fields.map(normSelector)})});const data=await response.json();if(!response.ok||data.error)throw Error(data.error||`El servidor respondió ${response.status}.`);lastRawDataset=data.dataset;await refreshDatasets();$('normSaveStatus').textContent=`Guardada sin normalizar: ${Number(data.records).toLocaleString()} registros. Archivo original: ${data.source_file}`;$('normExploreOriginal').hidden=false}catch(e){$('normSaveStatus').textContent='No se pudo guardar: '+e.message}finally{button.disabled=false}});
$('normExploreOriginal').addEventListener('click',async()=>{if(!lastRawDataset)return;await refreshDatasets();$('dataset').value=lastRawDataset;configureExplorerMode();tab('explore');await search();$('exploreView').scrollIntoView({behavior:'smooth'})});
$('normPreview').addEventListener('click',async()=>{const file=$('normFile')._selected;if(!file){$('normFileInfo').textContent='Selecciona o arrastra un archivo primero.';return}const button=$('normPreview');button.disabled=true;$('normFileInfo').textContent='Leyendo el archivo y detectando campos…';try{const response=await fetch('/api/workflow/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:file.name,content:await file.text(),records_path:$('normRecordsPath').value.trim()})});const data=await response.json();if(!response.ok||data.error)throw Error(data.error||`El servidor respondió ${response.status}.`);normUploadId=data.upload_id;normUploadMeta=data;normFieldCoverage=data.field_coverage||[];normQualityReport=null;$('normQualityPanel').hidden=true;$('normCustomStart').disabled=true;$('normQualityRun').disabled=false;$('normSaveOriginal').disabled=false;$('normRecordsPath').value=data.records_path||'';$('normFileInfo').textContent=`Vista previa lista: ${data.format.toUpperCase()} · ${Number(data.count).toLocaleString()} registros · ${data.fields.length} campos detectados.`;$('normCustomPreview').replaceChildren();$('normCustomPreview').hidden=false;add($('normCustomPreview'),'h4','Muestra de datos');add($('normCustomPreview'),'pre',JSON.stringify(data.preview,null,2),'pre');if(data.record_paths?.length)add($('normCustomPreview'),'p','Rutas JSON detectadas: '+data.record_paths.join(' · '),'small');$('normMappings').replaceChildren();for(const [id,spec]of Object.entries(normRoles))createNormMapping(id,spec);autoMapNormFields();renderNormMappingRoles();$('normCustomMapping').hidden=false}catch(e){$('normFileInfo').textContent='No se pudo leer el archivo: '+e.message}finally{button.disabled=false}});
function autoMapNormFields(){const aliases={normQ:['question','questiontext','question_text','pregunta','prompt'],normA:['answer','answertext','answer_text','respuesta','candidateanswer','response'],normC:['context','contexttext','context_text','contexto','paragraph','passage'],normRef:['referenceanswer','reference_answer','goldanswer','expectedanswer','respuestaesperada'],normScore:['score','grade','rating','calificacion','nota'],normUnans:['unanswerable','is_unanswerable','norespondible'],normSource:['id','sourceid','source_id','idorigen']};const clean=x=>String(x).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,'');for(const [id,names]of Object.entries(aliases)){const select=$('map_'+id);const match=[...select.options].find(o=>names.includes(clean((o.title||o.value).split('.').pop())));if(match)select.value=match.value}}
$('normCustomStart').addEventListener('click',async()=>{const dataset=$('normCustomName').value.trim();if(!normUploadId){$('normFileInfo').textContent='Primero previsualiza el archivo.';return}if(!dataset){$('normFileInfo').textContent='Escribe un nombre para la base.';return}let payload;try{payload=normQualityPayload()}catch(e){$('normFileInfo').textContent=e.message;return}if(!normQualityReport?.ready_for_normalization){$('normFileInfo').textContent='Analiza primero la calidad con el mapeo actual.';return}const fields=payload.fields;if(!window.confirm(`Se normalizará “${dataset}”. Si ya existe una base normalizada con ese nombre, sus filas se reemplazarán. ¿Continuar?`))return;const button=$('normCustomStart');button.disabled=true;$('normFileInfo').textContent='Guardando el original y normalizando…';try{const response=await fetch('/api/normalize/custom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...payload,name:dataset})});const state=await response.json();if(!response.ok||state.error)throw Error(state.error||state.message||`El servidor respondió ${response.status}.`);renderNormalizationState(state)}catch(e){$('normFileInfo').textContent='No se pudo iniciar la normalización: '+e.message;button.disabled=false}});
let normalizationPollTimer=null,lastNormalizedDataset='';
function renderNormalizationState(state){const button=$('normStart');button.disabled=state.status==='running';if($('normCustomStart'))$('normCustomStart').disabled=state.status==='running'||!normQualityReport?.ready_for_normalization;$('normStatus').className='small'+(state.status==='error'?' err':'');$('normStatus').textContent=state.message||'Elige una base y ejecuta el proceso.';if(state.dataset&&state.dataset===$('normCustomName').value){if(state.status==='running')$('normFileInfo').textContent=state.message||'Normalizando archivo…';else if(state.status==='completed')$('normFileInfo').textContent=`Normalización completa: ${Number((state.result?.counts_by_dataset||{})[state.dataset]||0).toLocaleString()} respuestas para ${state.dataset}.` ;else if(state.status==='error')$('normFileInfo').textContent='La normalización falló: '+(state.error||state.message)}if(state.status==='running'){$('normSummary').hidden=true;if(normalizationPollTimer)clearTimeout(normalizationPollTimer);normalizationPollTimer=setTimeout(loadNormalizationStatus,1200);return}if(state.status==='completed'&&state.result){const result=state.result;updateNormalizationCommand(state.dataset);lastNormalizedDataset=state.dataset==='all'?'':state.dataset;$('normExploreNormalized').hidden=!lastNormalizedDataset;refreshDatasets().catch(e=>{$('normStatus').textContent='La base se normalizó, pero no se pudo actualizar el selector: '+e.message});$('normSummary').hidden=false;$('normMetrics').replaceChildren();for(const [label,value] of [['Respuestas',result.record_count],['Contextos',result.context_count],['Párrafos',result.paragraph_count]])add($('normMetrics'),'span',`${label}: ${Number(value||0).toLocaleString()}`,'pill');const rows=$('normRows');rows.replaceChildren();for(const name of result.normalized||[]){const row=add(rows,'tr');add(row,'td',name);add(row,'td',Number((result.counts_by_dataset||{})[name]||0).toLocaleString())}$('normOutput').textContent=result.source_file?`Original guardado en ${result.source_file} · Configuración: ${result.config_file} · Salidas: ${result.output_dir}`:`Archivos normalizados guardados en ${result.output_dir}. Las demás bases se conservaron.`;if(result.source_file)$('normSummary').scrollIntoView({behavior:'smooth',block:'center'})}else if(state.status==='error'){ $('normSummary').hidden=false;$('normMetrics').replaceChildren();$('normRows').replaceChildren();$('normOutput').textContent=state.error||'Revisa los archivos y la salida del normalizador.'}else if(state.status==='idle'){$('normSummary').hidden=true}}
async function loadNormalizationStatus(){const response=await fetch('/api/normalize/status');const state=await response.json();renderNormalizationState(state);return state}
function updateNormalizationCommand(dataset=$('normDataset').value){$('normNextCommand').textContent=`.\\.venv\\Scripts\\python.exe preparar_vistas.py --dataset ${dataset}`}
$('normDataset').addEventListener('change',updateNormalizationCommand);updateNormalizationCommand();$('normExploreNormalized').addEventListener('click',async()=>{if(!lastNormalizedDataset)return;await refreshDatasets();const normalView=datasetCatalog.find(item=>item.name===lastNormalizedDataset);$('dataset').value=normalView?normalView.name:`normalized::${lastNormalizedDataset}`;configureExplorerMode();tab('explore');await search();$('exploreView').scrollIntoView({behavior:'smooth'})});
$('normStart').addEventListener('click',async()=>{const button=$('normStart');button.disabled=true;$('normStatus').className='small';$('normStatus').textContent='Enviando solicitud…';$('normSummary').hidden=true;try{const response=await fetch('/api/normalize/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset:$('normDataset').value,download:$('normDownload').checked})});const state=await response.json();if(!response.ok)throw Error(state.error||state.message||`El servidor respondió ${response.status}.`);renderNormalizationState(state)}catch(e){button.disabled=false;$('normStatus').className='small err';$('normStatus').textContent='No se pudo iniciar: '+e.message}});
function tab(which){$('exploreView').hidden=which!=='explore';$('workflowView').hidden=which!=='workflow';$('normalizeView').hidden=which!=='normalize';$('dashboard').hidden=which!=='explore';$('tabExplore').className=which==='explore'?'active':'secondary';$('tabWorkflow').className=which==='workflow'?'active':'secondary';$('tabNormalize').className=which==='normalize'?'active':'secondary'}$('tabExplore').onclick=()=>tab('explore');$('tabWorkflow').onclick=()=>tab('workflow');$('tabNormalize').onclick=()=>tab('normalize');$('dataset').addEventListener('change',()=>{ $('detailPanel').hidden=true; });
function setWorkflowStep(active){document.querySelectorAll('.stepper .step').forEach((node,index)=>{node.classList.toggle('current',index===active);node.classList.toggle('done',index<active)})}
function renderFieldCatalog(){const body=$('wfFieldRows');if(!body)return;body.replaceChildren();const query=$('wfFieldSearch').value.trim().toLocaleLowerCase();const mapped={mapQ:'Pregunta',mapA:'Respuesta',mapC:'Contexto',mapRef:'Respuesta esperada',mapScore:'Calificación',mapUnans:'No respondible',mapSource:'ID de origen'};const roles={};for(const [id,label]of Object.entries(mapped)){const value=mapValue(id);if(value)roles[value]=(roles[value]?roles[value]+', ':'')+label}const visible=workflowCatalog.filter(item=>`${item.field} ${item.type} ${item.example}`.toLocaleLowerCase().includes(query));for(const item of visible){const row=add(body,'tr');add(row,'td',item.field,'id');add(row,'td',item.type);const coverage=add(row,'td',undefined,'field-presence');const track=add(coverage,'div',undefined,'chart-track');const fill=add(track,'div',undefined,'chart-fill');fill.style.width=(100*item.coverage)+'%';add(coverage,'span',`${Number(item.present).toLocaleString()} / ${Number(item.total).toLocaleString()} (${(100*item.coverage).toFixed(0)}%)`,'small');add(row,'td',item.example||'—','field-example');add(row,'td',roles[item.field]||'—',roles[item.field]?'field-role':'small')}$('wfFieldCount').textContent=`Mostrando ${visible.length.toLocaleString()} de ${workflowCatalog.length.toLocaleString()} campos detectados. La cobertura se calcula sobre todos los registros.`}
function selectMap(id,label,required=false,help=''){const box=add($('wfMappings'),'div',undefined,'map-card');const lab=add(box,'label',label+(required?' *':''));lab.htmlFor=id;add(box,'p',help,'map-help');const sel=add(box,'select');sel.id=id;const empty=el('option','— sin asignar —');empty.value='';sel.appendChild(empty);workflowFields.forEach(f=>{const o=el('option',f);o.value=f;sel.appendChild(o)});const custom=add(box,'input');custom.type='text';custom.id=id+'Custom';custom.placeholder='Ruta manual solo si hace falta (ej. ../../context)';add(box,'div','Al elegir una columna, aquí verás qué campo se usará y un ejemplo.','map-preview').id=id+'Preview';return sel}
function mapValue(id){return $(id+'Custom').value.trim()||$(id).value}
$('wfFieldSearch').addEventListener('input',renderFieldCatalog);$('wfMappings').addEventListener('change',()=>{renderFieldCatalog();renderMappingSummary();renderMapPreviews()});$('wfMappings').addEventListener('input',()=>{renderFieldCatalog();renderMappingSummary();renderMapPreviews()});
function autoMapFields(){const normalize=s=>String(s).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,'');const aliases={mapQ:['pregunta','question','questiontext','question_text','query','prompt'],mapA:['respuesta','answer','answertext','answer_text','candidateanswer','candidate_answer','response'],mapC:['contexto','context','paragraph','passage','document'],mapRef:['respuestareferencia','respuesta_referencia','referenceanswer','reference_answer','goldanswer','gold_answer'],mapScore:['calificacion','score','grade','rating','label'],mapUnans:['norespondible','is_unanswerable','unanswerable'],mapSource:['idorigen','id_origen','sourceid','source_id','id']};for(const [id,names] of Object.entries(aliases)){const select=$(id);if(!select)continue;const match=[...select.options].find(o=>names.includes(normalize(o.value)));if(match)select.value=match.value}}
function renderMapPreviews(){for(const [id,label]of Object.entries({mapQ:'Pregunta',mapA:'Respuesta del estudiante',mapC:'Contexto',mapRef:'Respuesta esperada',mapScore:'Calificación',mapUnans:'Etiqueta no respondible',mapSource:'ID original'})){const box=$(id+'Preview');if(!box)continue;box.replaceChildren();const custom=$(id+'Custom').value.trim(),value=mapValue(id);if(!value){box.textContent='Sin asignar: esta información no se tomará del archivo.';continue}const title=add(box,'strong',custom?'Ruta seleccionada: ':'Columna seleccionada: ');add(box,'span',value);if(custom){add(box,'div','Es una ruta manual; verifica que apunte al valor correcto en la vista previa.');continue}const sample=workflowCatalog.find(item=>item.field===value)?.example;if(sample){add(box,'div','Ejemplo del archivo: ');add(box,'em',sample)}else add(box,'div','No hay ejemplo disponible para esta columna.')}}
function renderMappingSummary(){const box=$('wfMappingSummary');if(!box)return;const names={mapQ:'Pregunta',mapA:'Respuesta del estudiante',mapC:'Contexto',mapRef:'Respuesta esperada (objetivo)',mapScore:'Calificación (objetivo)',mapUnans:'Indicador no respondible',mapSource:'ID original'};const selected=Object.entries(names).map(([id,name])=>[name,mapValue(id)]);const inputs=selected.slice(0,3),targets=selected.slice(3,6);box.replaceChildren();add(box,'h4','Resumen de tu configuración');const p=add(box,'p');p.append('Se calcularán características a partir de: ');p.append([...inputs.filter(x=>x[1]).map(x=>x[0]),...(inputs[2][1]?[]:['contexto (sin asignar)'])].join(', ')+'.');const q=add(box,'p');q.append('Objetivos disponibles: '+(targets.filter(x=>x[1]).map(x=>x[0]).join(', ')||'ninguno asignado')+'.');const source=selected[6][1]?'Se conservará el ID original indicado para trazabilidad.':'No hay ID original asignado; se usará la posición del registro como referencia.';add(box,'p',source);if(!inputs[0][1]||!inputs[1][1])add(box,'p','Falta asignar pregunta o respuesta: no se podrá iniciar el procesamiento.','err');else if(!inputs[2][1])add(box,'p','Sin contexto no se calcularán embeddings ni similitudes de párrafos del contexto.','small')}
$('wfPreview').onclick=async()=>{try{const f=$('wfFile').files[0];if(!f)throw Error('Selecciona un archivo.');$('wfStatus').textContent='Leyendo y analizando el archivo…';const content=await f.text();const d=await(await fetch('/api/workflow/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:f.name,content,records_path:$('wfPath').value})})).json();if(d.error)throw Error(d.error);workflowDataset=d.upload_id;workflowFields=d.fields;workflowCatalog=d.field_coverage||[];sourceOffset=0;await loadSourcePreviewPage(0);$('wfPath').value=d.records_path||'';$('wfPreviewBox').replaceChildren();add($('wfPreviewBox'),'p',`${d.format.toUpperCase()} · ${Number(d.count).toLocaleString()} registros detectados · ${d.fields.length} campos · ${(f.size/1048576).toFixed(1)} MB`,'pill');if(d.record_paths?.length){add($('wfPreviewBox'),'p','Rutas candidatas: '+d.record_paths.join(' · '),'small')}add($('wfPreviewBox'),'p','Revisa los ejemplos y confirma que pregunta, respuesta y contexto estén mapeados correctamente antes de procesar.','callout');if(d.field_coverage?.length){const quality=add($('wfPreviewBox'),'div');add(quality,'h3','Calidad de campos · registros con valor');for(const item of d.field_coverage.slice(0,12)){const line=add(quality,'div',undefined,'coverage');add(line,'span',item.field);const track=add(line,'div',undefined,'chart-track');const fill=add(track,'div',undefined,'chart-fill');fill.style.width=(100*item.coverage)+'%';add(line,'b',`${(100*item.coverage).toFixed(0)}%`)}}add($('wfPreviewBox'),'pre',JSON.stringify(d.preview,null,2),'pre');$('wfMappings').replaceChildren();selectMap('mapQ','Pregunta · obligatoria',true,'Enunciado que se le presenta al estudiante. Se calculará su embedding y se comparará con la respuesta y el contexto.');selectMap('mapA','Respuesta del estudiante · obligatoria',true,'Texto que se quiere analizar o calificar. Se calculará su embedding y sus similitudes.');selectMap('mapC','Contexto o pasaje',false,'Texto de apoyo donde se encuentra la información para responder. Se separa en párrafos; de estos se calculan embeddings y similitudes.');selectMap('mapRef','Respuesta esperada · objetivo opcional',false,'Respuesta correcta o de referencia. Guárdala como objetivo de entrenamiento/evaluación; no como entrada del modelo.');selectMap('mapScore','Calificación humana · objetivo opcional',false,'Nota asignada por una persona. Debe ser numérica y se usa como objetivo de predicción.');selectMap('mapUnans','¿No se puede responder? · opcional',false,'Etiqueta sí/no que indica que el contexto no contiene la respuesta.');selectMap('mapSource','ID original del registro · opcional',false,'Identificador estable que ya venga en tu archivo. Ayuda a rastrear el resultado hasta la fila fuente.');autoMapFields();renderFieldCatalog();renderMappingSummary();renderMapPreviews();$('wfMapPanel').hidden=false;$('wfExportPanel').hidden=true;setWorkflowStep(1);$('wfStatus').textContent='Vista previa lista. Revisa los campos detectados y procesa.'}catch(e){$('wfStatus').textContent='Error: '+e.message;$('wfStatus').className='small err'}};
const processButton=$('wfProcess');
processButton.onclick=async()=>{const status=$('wfStatus');status.className='small';$('wfExportPanel').hidden=true;try{if(!workflowDataset)throw Error('Primero carga y previsualiza el archivo.');if(!mapValue('mapQ')||!mapValue('mapA'))throw Error('Asigna pregunta y respuesta.');processButton.disabled=true;status.textContent='Procesando: normalización y cálculo de embeddings en curso…';const payload={upload_id:workflowDataset,name:$('wfName').value,records_path:$('wfPath').value,fields:{question_text:mapValue('mapQ'),answer_text:mapValue('mapA'),context_text:mapValue('mapC')||null,reference_answer:mapValue('mapRef')||null,score:mapValue('mapScore')||null,is_unanswerable:mapValue('mapUnans')||null},keys:{source_id:mapValue('mapSource')||null},model:$('wfModel').value,dimensions:Number($('wfDims').value),api_key_file:$('wfKeyFile').value};const response=await fetch('/api/workflow/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await response.json();if(!response.ok||d.error)throw Error(d.error||`El servidor respondió ${response.status}.`);workflowDataset=d.dataset;await refreshDatasets();if(![...$('dataset').options].some(o=>o.value===d.dataset)){const option=el('option',d.dataset);option.value=d.dataset;$('dataset').appendChild(option)}$('dataset').value=d.dataset;status.textContent=`Listo: ${Number(d.records).toLocaleString()} respuestas procesadas para ${d.dataset}.`;const meta=d.columns;$('wfResult').replaceChildren();add($('wfResult'),'span',`Base: ${d.dataset}`,'pill');add($('wfResult'),'span',`${Number(d.records).toLocaleString()} respuestas · ${Number(d.scored).toLocaleString()} con calificación`,'pill');$('wfColumns').replaceChildren();meta.forEach(c=>{const lab=add($('wfColumns'),'label');const cb=add(lab,'input');cb.type='checkbox';cb.value=c.key;cb.checked=c.default;lab.appendChild(document.createTextNode(' '+c.label))});$('wfExportPanel').hidden=false;setWorkflowStep(2);workflowOffset=0;await loadWorkflowPage(0);$('wfExportPanel').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){status.textContent='No se pudo procesar: '+e.message;status.className='small err';status.scrollIntoView({behavior:'smooth',block:'center'})}finally{processButton.disabled=false}};
$('wfExport').onclick=async()=>{try{const columns=[...$('wfColumns').querySelectorAll('input:checked')].map(x=>x.value);const d=await(await fetch('/api/workflow/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset:workflowDataset,columns,format:$('wfFormat').value,include_vectors:$('wfVectors').checked})})).json();if(d.error)throw Error(d.error);$('wfExportStatus').textContent=`Exportado: ${d.path} · ${Number(d.records).toLocaleString()} filas`;setWorkflowStep(3)}catch(e){$('wfExportStatus').textContent='Error: '+e.message}};
</script></body></html>'''


def fold(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFD", value.casefold())
                   if unicodedata.category(char) != "Mn")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def paragraph_context_matches(normalized: Path, needle: str, search_contexts: bool) -> set[str]:
    matching: set[str] = set()
    paragraph_path = normalized / "parrafos.jsonl"
    if paragraph_path.exists():
        for item in iter_jsonl(paragraph_path):
            if needle in fold(item.get("paragraph_text") or ""):
                matching.add(item.get("context_id"))
    if search_contexts:
        context_path = normalized / "contextos.jsonl"
        if context_path.exists():
            for item in iter_jsonl(context_path):
                if needle in fold(item.get("context_text") or ""):
                    matching.add(item.get("context_id"))
    return matching


def normalization_status():
    with NORMALIZATION_LOCK:
        return {**NORMALIZATION_STATE}


def run_normalization(job_id: str, dataset: str, download: bool, custom: dict | None = None) -> None:
    command = [sys.executable, str(NORMALIZER_SCRIPT), "--dataset", dataset]
    if custom:
        command.extend(["--input", str(custom["source"]), "--config", str(custom["config"])])
    elif dataset in {"boolq_train", "boolq_dev"}:
        source = NORMALIZER_DIR / "originales" / "adicionales" / f"{dataset}.jsonl"
        config = NORMALIZER_DIR / f"config_{dataset}.json"
        if not source.is_file() or not config.is_file():
            with NORMALIZATION_LOCK:
                if NORMALIZATION_STATE["job_id"] == job_id:
                    NORMALIZATION_STATE.update(status="error", message="No se encontraron los archivos locales de BoolQ.",
                                               error=f"Se requiere {source.name} y {config.name} en el proyecto del normalizador.")
            return
        command.extend(["--input", str(source), "--config", str(config)])
    elif download:
        command.append("--download")

    with NORMALIZATION_LOCK:
        if NORMALIZATION_STATE["job_id"] == job_id:
            NORMALIZATION_STATE["message"] = "Ejecutando el normalizador; el tiempo depende del tamaño de la base."
    try:
        completed = subprocess.run(command, cwd=NORMALIZER_DIR, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", check=False)
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "El normalizador terminó con error.").strip()
            with NORMALIZATION_LOCK:
                if NORMALIZATION_STATE["job_id"] == job_id:
                    NORMALIZATION_STATE.update(status="error", message="La normalización no terminó.",
                                               error=diagnostic[-5000:])
            return
        manifest_path = NORMALIZER_DIR / "resultados" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = {"record_count": manifest.get("record_count", 0),
                  "context_count": manifest.get("context_count", 0),
                  "paragraph_count": manifest.get("paragraph_count", 0),
                  "counts_by_dataset": manifest.get("counts_by_dataset", {}),
                  "normalized": [dataset] if dataset != "all" else [
                      "squad2_train", "squad2_dev", "cmu_qa_v1.2", "mohler_v1.0",
                      "mohler_v2.0", "roars_2class", "roars_3class"],
                  "output_dir": str(manifest_path.parent)}
        if custom:
            result["source_file"] = str(custom["source"])
            result["config_file"] = str(custom["config"])
        with NORMALIZATION_LOCK:
            if NORMALIZATION_STATE["job_id"] == job_id:
                NORMALIZATION_STATE.update(status="completed", message="Normalización terminada.",
                                           result=result, error=None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        with NORMALIZATION_LOCK:
            if NORMALIZATION_STATE["job_id"] == job_id:
                NORMALIZATION_STATE.update(status="error", message="No se pudo ejecutar el normalizador.",
                                           error=str(exc))


def start_normalization(dataset: str, download: bool) -> tuple[bool, dict]:
    if dataset not in NORMALIZER_DATASETS:
        raise ValueError("Selecciona una base disponible en el normalizador.")
    if not NORMALIZER_SCRIPT.is_file():
        raise FileNotFoundError(f"No se encontró el normalizador en {NORMALIZER_SCRIPT}.")
    if dataset in {"boolq_train", "boolq_dev"}:
        source = NORMALIZER_DIR / "originales" / "adicionales" / f"{dataset}.jsonl"
        config = NORMALIZER_DIR / f"config_{dataset}.json"
        if not source.is_file() or not config.is_file():
            raise FileNotFoundError(f"Faltan {source.name} o {config.name} en qa_dataset_project.")
    with NORMALIZATION_LOCK:
        if NORMALIZATION_STATE["status"] == "running":
            return False, {**NORMALIZATION_STATE}
        job_id = uuid.uuid4().hex
        NORMALIZATION_STATE.update(job_id=job_id, status="running", dataset=dataset,
                                   message="Preparando la normalización…", result=None, error=None)
    worker = threading.Thread(target=run_normalization, args=(job_id, dataset, download), daemon=True)
    worker.start()
    return True, normalization_status()


def analyze_custom_normalization(body: dict, results_dir: Path) -> dict:
    upload_id = str(body.get("upload_id", ""))
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        raise ValueError("La carga expiró o no es válida. Vuelve a seleccionar el archivo.")
    upload = results_dir / "importaciones" / upload_id
    metadata_path = upload / "preview.json"
    if not metadata_path.is_file():
        raise ValueError("No se encontró la vista previa. Vuelve a cargar el archivo.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_candidates = list((upload / "originales").glob("*"))
    if len(source_candidates) != 1 or not source_candidates[0].is_file():
        raise ValueError("No se encontró el archivo original de esta carga.")
    source = source_candidates[0]
    fields = body.get("fields") or {}
    if not fields.get("question_text") or not fields.get("answer_text"):
        raise ValueError("Asigna pregunta y respuesta antes de analizar la calidad.")
    config = {"format": "jsonl" if source.suffix.lower() in {".jsonl", ".ndjson"} else source.suffix.lower().lstrip("."),
              "records_path": str(body.get("records_path") or metadata.get("records_path") or "*"),
              "fields": fields}
    if config["format"] == "ndjson":
        config["format"] = "jsonl"
    from normalizar_datasets import custom_field, read_custom_rows
    counts = {"source_records": 0, "usable_records": 0, "output_answers": 0,
              "missing_question": 0, "missing_answer": 0, "empty_context": 0,
              "invalid_score": 0, "invalid_answer_items": 0,
              "duplicate_source_ids": 0, "duplicate_question_answer": 0}
    issues, seen_source, seen_pairs = [], {}, set()
    max_issues = 1000

    def issue(line, category, detail, blocking=False):
        if len(issues) < max_issues:
            issues.append({"row": line, "category": category, "detail": detail,
                           "blocking": bool(blocking)})

    source_selector = (body.get("source_id") or "").strip()
    for row, root_path, line_no in read_custom_rows(source, config):
        counts["source_records"] += 1
        question = custom_field(row, root_path, fields.get("question_text"))
        answer = custom_field(row, root_path, fields.get("answer_text"))
        context = custom_field(row, root_path, fields.get("context_text"))
        unanswerable = str(custom_field(row, root_path, fields.get("is_unanswerable")) or "").lower() in {"true", "1", "yes"}
        has_question = question is not None and str(question).strip() != ""
        if not has_question:
            counts["missing_question"] += 1
            issue(line_no, "Pregunta vacía", "La fila se omitirá porque no tiene pregunta.")
        if context is None or str(context).strip() == "":
            counts["empty_context"] += 1
            issue(line_no, "Contexto vacío", "No se calcularán características basadas en contexto para esta fila.")

        answer_values = answer if isinstance(answer, list) else [answer]
        if not answer_values:
            answer_values = [None]
        has_any_answer = any(value is not None and str(value).strip() != "" for value in answer_values)
        if not has_any_answer and not unanswerable:
            counts["missing_answer"] += 1
            issue(line_no, "Respuesta vacía", "La fila se omitirá porque no tiene respuesta y no está marcada como no respondible.")
        row_usable = has_question and (has_any_answer or unanswerable)
        if not row_usable:
            continue
        counts["usable_records"] += 1
        if isinstance(answer, list):
            empty_items = sum(value is None or str(value).strip() == "" for value in answer_values)
            if empty_items and not unanswerable:
                counts["invalid_answer_items"] += empty_items
                issue(line_no, "Elemento de respuesta vacío", f"La lista contiene {empty_items} respuesta(s) vacía(s); el normalizador no puede crear una fila válida para cada elemento.", True)
        output_values = answer_values if isinstance(answer, list) else [answer]
        if unanswerable:
            counts["output_answers"] += len(output_values)
        else:
            counts["output_answers"] += sum(1 for value in output_values if value is not None and str(value).strip() != "")

        score = custom_field(row, root_path, fields.get("score"))
        if score is not None and str(score).strip() != "":
            try:
                float(score)
            except (TypeError, ValueError):
                counts["invalid_score"] += 1
                issue(line_no, "Calificación no numérica", f"No se puede convertir {str(score)[:100]!r} a número.", True)

        if source_selector:
            source_id = custom_field(row, root_path, source_selector)
            if source_id is not None and str(source_id).strip():
                key = str(source_id).strip()
                if key in seen_source:
                    counts["duplicate_source_ids"] += 1
                    issue(line_no, "ID de origen repetido", f"El ID también aparece en la fila {seen_source[key]}.")
                else:
                    seen_source[key] = line_no
        if has_any_answer:
            for value in output_values:
                if value is None or str(value).strip() == "":
                    continue
                pair = (fold(str(question)), fold(str(value)))
                if pair in seen_pairs:
                    counts["duplicate_question_answer"] += 1
                    issue(line_no, "Pregunta y respuesta duplicadas", "La misma pareja de textos aparece más de una vez.")
                else:
                    seen_pairs.add(pair)

    blocking = counts["invalid_score"] + counts["invalid_answer_items"] > 0
    ready = counts["usable_records"] > 0 and not blocking
    return {"ready_for_normalization": ready, "blocking_errors": blocking,
            "counts": counts, "issues": issues, "issues_truncated": len(issues) >= max_issues,
            "summary": ("Datos listos para normalizar." if ready else
                        "Corrige los errores indicados antes de normalizar." if blocking else
                        "No se encontraron filas utilizables con el mapeo actual.")}


def start_custom_normalization(body: dict, results_dir: Path) -> tuple[bool, dict]:
    from workflow_import import safe_name
    upload_id = str(body.get("upload_id", ""))
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        raise ValueError("La carga expiró o no es válida. Vuelve a seleccionar el archivo.")
    dataset = safe_name(str(body.get("name") or ""))
    reserved = NORMALIZER_DATASETS | {"boolq_train", "boolq_dev"}
    if dataset in reserved:
        raise ValueError("Ese nombre está reservado para una base integrada; usa otro nombre.")
    upload = results_dir / "importaciones" / upload_id
    metadata_path = upload / "preview.json"
    if not metadata_path.is_file():
        raise ValueError("No se encontró la vista previa. Vuelve a cargar el archivo.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_candidates = list((upload / "originales").glob("*"))
    if len(source_candidates) != 1 or not source_candidates[0].is_file():
        raise ValueError("No se encontró el archivo original de esta carga.")
    source = source_candidates[0]
    fields = body.get("fields") or {}
    if not fields.get("question_text") or not fields.get("answer_text"):
        raise ValueError("Asigna pregunta y respuesta para poder normalizar.")
    if dataset in {"all", "squad2_train", "squad2_dev", "cmu_qa_v1.2", "mohler_v1.0", "mohler_v2.0", "roars_2class", "roars_3class"}:
        raise ValueError("Ese nombre corresponde a una fuente integrada; elige otro.")
    if not NORMALIZER_SCRIPT.is_file():
        raise FileNotFoundError(f"No se encontró el normalizador en {NORMALIZER_SCRIPT}.")
    quality = analyze_custom_normalization(body, results_dir)
    if not quality["ready_for_normalization"]:
        raise ValueError(quality["summary"])

    normalized_fields = {key: str(value).strip() for key, value in fields.items() if value}
    config = {"dataset": dataset, "format": "jsonl" if source.suffix.lower() in {".jsonl", ".ndjson"} else source.suffix.lower().lstrip("."),
              "records_path": str(body.get("records_path") or metadata.get("records_path") or "*"),
              "fields": normalized_fields,
              "keys": {"source_id": str(body.get("source_id") or "").strip() or None},
              "context_kind": "custom_context"}
    config = {key: value for key, value in config.items() if value is not None}
    with NORMALIZATION_LOCK:
        if NORMALIZATION_STATE["status"] == "running":
            return False, {**NORMALIZATION_STATE}
        job_id = uuid.uuid4().hex
        NORMALIZATION_STATE.update(job_id=job_id, status="running", dataset=dataset,
                                   message="Guardando el original y preparando la normalización…", result=None, error=None)
    additional = NORMALIZER_DIR / "originales" / "adicionales"
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stored_source = additional / f"{dataset}_{digest.hexdigest()[:12]}{source.suffix.lower()}"
    config_path = NORMALIZER_DIR / f"config_{dataset}.json"
    try:
        additional.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stored_source)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        from normalizar_datasets import load_custom
        candidate_rows, _ = load_custom(stored_source, config_path)
        if not candidate_rows:
            raise ValueError("Ninguna fila contiene pregunta y respuesta con el mapeo elegido.")
        shutil.rmtree(upload, ignore_errors=True)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        with NORMALIZATION_LOCK:
            if NORMALIZATION_STATE["job_id"] == job_id:
                NORMALIZATION_STATE.update(status="error", message="No se pudo guardar el original o su configuración.", error=str(exc))
        raise
    custom = {"source": stored_source, "config": config_path}
    worker = threading.Thread(target=run_normalization, args=(job_id, dataset, False, custom), daemon=True)
    worker.start()
    return True, normalization_status()


class AppHandler(BaseHTTPRequestHandler):
    results: Path
    normalized: Path
    api_key_file: Path | None = None
    line_index_cache: dict = {}

    def log_message(self, fmt, *args):
        # Local server logs contain request paths only; never log headers or secrets.
        super().log_message(fmt, *args)

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def normalized_catalog_entry(self, dataset_key: str) -> tuple[str, Path]:
        match = re.fullmatch(r"normalized::([A-Za-z0-9][A-Za-z0-9_.-]{0,99})", dataset_key)
        if not match:
            raise ValueError("Nombre de base normalizada no válido.")
        name = match.group(1)
        path = (self.normalized / "por_base" / f"{name}.jsonl").resolve()
        root = (self.normalized / "por_base").resolve()
        if path.parent != root or not path.is_file():
            raise FileNotFoundError("No se encontró la exportación normalizada.")
        return name, path

    def normalized_search(self, params) -> None:
        name, source = self.normalized_catalog_entry(params.get("dataset", [""])[0])
        query = fold(params.get("q", [""])[0].strip())
        selected_field = params.get("field", ["all"])[0]
        fields = ["dataset", "context_id", "question_id", "answer_id", "score_id",
                  "paragraph_id", "context_text", "question_text", "answer_text",
                  "reference_answer", "score", "score_scale", "is_unanswerable",
                  "source_file", "source_record_id", "source_answer_id", "source_url"]
        if selected_field != "all" and selected_field not in fields:
            raise ValueError("El campo de búsqueda no pertenece a esta base normalizada.")
        offset = max(0, int(params.get("offset", ["0"])[0]))
        limit = max(1, min(200, int(params.get("limit", ["100"])[0])))
        matches, total, scanned = [], 0, 0
        for row_number, record in enumerate(iter_jsonl(source), start=1):
            scanned += 1
            value = record if selected_field == "all" else record.get(selected_field)
            searchable = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
            if query and query not in fold(searchable):
                continue
            if total >= offset and len(matches) < limit:
                matches.append({"row_number": row_number, "record": record,
                                "values": {field: record.get(field) for field in fields}})
            total += 1
        self.send_json({"dataset": f"normalized::{name}", "display_name": name,
                        "fields": fields, "matches": matches, "total_matches": total,
                        "scanned": scanned, "offset": offset,
                        "has_next": offset + len(matches) < total})

    def normalized_download(self, params) -> None:
        _, source = self.normalized_catalog_entry(params.get("dataset", [""])[0])
        data = source.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{source.name}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def raw_catalog_entry(self, dataset_key: str) -> tuple[dict, Path]:
        if not re.fullmatch(r"raw::[a-z0-9][a-z0-9_.-]{0,59}", dataset_key):
            raise ValueError("Nombre de base original no válido.")
        with RAW_CATALOG_LOCK:
            item = load_raw_catalog()["datasets"].get(dataset_key)
        if not item:
            raise ValueError("No se encontró la base original guardada.")
        originals_root = (NORMALIZER_DIR / "originales").resolve()
        source = (originals_root / str(item.get("source_file", ""))).resolve()
        if originals_root not in source.parents or not source.is_file():
            raise FileNotFoundError("No se encontró el archivo original guardado.")
        return item, source

    def save_original_dataset(self, body: dict) -> None:
        from workflow_import import safe_name
        upload_id = str(body.get("upload_id", ""))
        if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
            raise ValueError("La carga expiró o no es válida. Vuelve a previsualizar el archivo.")
        name = safe_name(str(body.get("name") or ""))
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,59}", name):
            raise ValueError("Usa un nombre que empiece por letra o número y contenga letras, números, guiones o puntos.")
        upload = self.workflow_upload_dir(upload_id)
        metadata_path = upload / "preview.json"
        if not metadata_path.is_file():
            raise ValueError("No se encontró la vista previa. Vuelve a cargar el archivo.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = list((upload / "originales").glob("*"))
        if len(files) != 1 or not files[0].is_file():
            raise ValueError("No se encontró el archivo original de esta carga.")
        source = files[0]
        if not int(metadata.get("count", 0)):
            raise ValueError("El archivo no contiene registros para explorar.")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        suffix = source.suffix.lower()
        saved_name = f"{name}_{checksum[:12]}{suffix}"
        originals_root = NORMALIZER_DIR / "originales"
        destination = originals_root / "adicionales" / saved_name
        additional = (originals_root / "adicionales").resolve()
        if additional not in destination.resolve().parents:
            raise ValueError("Ruta de archivo no válida.")
        fields = [str(value).strip() for value in body.get("fields", []) if str(value).strip()]
        entry = {"display_name": name, "source_file": destination.relative_to(originals_root).as_posix(),
                 "format": metadata.get("format"), "records_path": str(body.get("records_path") or metadata.get("records_path") or "*"),
                 "fields": fields, "record_count": int(metadata.get("count", 0)),
                 "sha256": checksum, "file_size": source.stat().st_size}
        key = f"raw::{name}"
        with RAW_CATALOG_LOCK:
            catalog = load_raw_catalog()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            catalog["datasets"][key] = entry
            save_raw_catalog(catalog)
        self.send_json({"dataset": key, "display_name": name, "records": entry["record_count"],
                        "source_file": str(destination.resolve()), "saved": True})

    def original_search(self, params) -> None:
        from normalizar_datasets import custom_field, read_custom_rows
        key = params.get("dataset", [""])[0]
        item, source = self.raw_catalog_entry(key)
        query = fold(params.get("q", [""])[0].strip())
        selected_field = params.get("field", ["all"])[0]
        fields = item.get("fields", [])
        if selected_field != "all" and selected_field not in fields:
            raise ValueError("El campo de búsqueda no pertenece a esta base.")
        offset = max(0, int(params.get("offset", ["0"])[0]))
        limit = max(1, min(200, int(params.get("limit", ["100"])[0])))
        config = {"format": item.get("format", "auto"), "records_path": item.get("records_path", "*")}
        matches, total = [], 0
        for row, root_path, line_no in read_custom_rows(source, config):
            value = row if selected_field == "all" else custom_field(row, root_path, selected_field)
            values = {field: custom_field(row, root_path, field) for field in fields}
            searchable_value = {"record": row, "fields": values} if selected_field == "all" else value
            searchable = json.dumps(searchable_value, ensure_ascii=False, default=str) if not isinstance(searchable_value, str) else searchable_value
            if query and query not in fold(searchable):
                continue
            if total >= offset and len(matches) < limit:
                matches.append({"row_number": line_no, "record": row, "values": values})
            total += 1
        self.send_json({"dataset": key, "display_name": item.get("display_name", key),
                        "fields": fields, "matches": matches, "total_matches": total,
                        "scanned": item.get("record_count", total), "offset": offset,
                        "has_next": offset + len(matches) < total})

    def original_download(self, params) -> None:
        key = params.get("dataset", [""])[0]
        item, source = self.raw_catalog_entry(key)
        data = source.read_bytes()
        filename = source.name.replace('"', "")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def dataset_provenance(self, name: str, metadata: dict) -> dict:
        normalized_file = self.normalized / "por_base" / f"{name}.jsonl"
        import_id = metadata.get("import_id")
        source_name = metadata.get("source")
        source_path = None
        if isinstance(import_id, str) and re.fullmatch(r"[a-f0-9]{32}", import_id):
            import_dir = self.results / "importaciones" / import_id
            source_path = import_dir / "originales" / str(source_name or "")
            normalized_kind = "direct_import"
        else:
            normalized_kind = "normalized" if normalized_file.is_file() else "unknown"
            try:
                first_record = next(iter_jsonl(normalized_file))
            except (OSError, StopIteration, json.JSONDecodeError):
                first_record = {}
            recorded_source = ((first_record.get("source") or {}).get("source_file")
                               or first_record.get("source_file"))
            if recorded_source:
                source_name = Path(str(recorded_source)).name
                candidate = Path(str(recorded_source)).expanduser()
                work_roots = {"cmu_qa_v1.2": "cmu_qa_v1.2",
                              "mohler_v1.0": "mohler_short_answer_v1.0",
                              "mohler_v2.0": "mohler_short_answer_v2.0"}
                archives = {"cmu_qa_v1.2": "cmu_qa_v1.2.tar.gz",
                            "mohler_v1.0": "mohler_short_answer_v1.0.tar.gz",
                            "mohler_v2.0": "mohler_short_answer_v2.0.zip"}
                candidates = [candidate] if candidate.is_absolute() else [
                    self.normalized.parent / "originales" / candidate,
                    self.normalized.parent / "originales" / "adicionales" / candidate,
                    NORMALIZER_DIR / "originales" / candidate,
                    NORMALIZER_DIR / "originales" / "adicionales" / candidate,
                ]
                if name in work_roots:
                    candidates.insert(0, self.normalized.parent / "trabajo" / work_roots[name] / candidate)
                if name in archives:
                    candidates.append(self.normalized.parent / "originales" / archives[name])
                source_path = next((item for item in candidates if item.is_file()), None)
        return {"kind": normalized_kind,
                "label": "Normalizada" if normalized_kind == "normalized" else
                         "Importación directa" if normalized_kind == "direct_import" else "Estado desconocido",
                "original_available": bool(source_path and source_path.is_file()),
                "original_path": str(source_path) if source_path and source_path.is_file() else None,
                "normalized_path": str(normalized_file) if normalized_file.is_file() else None,
                "source_name": source_path.name if source_path and source_path.is_file() else source_name}

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                content = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            elif parsed.path == "/api/datasets":
                manifest_path = self.results / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                datasets = []
                for file in sorted(self.results.glob("*/vista_respuestas.jsonl")):
                    name = file.parent.name
                    metadata = manifest.get("datasets", {}).get(name, {})
                    if name.startswith("demo_") or name == "prueba1":
                        continue
                    datasets.append({"name": name, "responses": metadata.get("responses"),
                                     "with_score_target": metadata.get("with_score_target", 0),
                                     "model": metadata.get("model"), "dimensions": metadata.get("dimensions"),
                                     "source": metadata.get("source"),
                                     "provenance": self.dataset_provenance(name, metadata)})
                with RAW_CATALOG_LOCK:
                    raw_catalog = load_raw_catalog()
                originals_root = (NORMALIZER_DIR / "originales").resolve()
                for key, item in raw_catalog["datasets"].items():
                    source = (originals_root / str(item.get("source_file", ""))).resolve()
                    if originals_root not in source.parents or not source.is_file():
                        available = False
                    else:
                        available = True
                    datasets.append({"name": key, "display_name": item.get("display_name", key.removeprefix("raw::")),
                                     "responses": item.get("record_count", 0), "with_score_target": 0,
                                     "model": None, "dimensions": None, "source": source.name,
                                     "provenance": {"kind": "original", "label": "Original (sin normalizar)",
                                                    "original_available": available,
                                                    "original_path": str(source) if available else None,
                                                    "normalized_path": None,
                                                    "fields": item.get("fields", []),
                                                    "records_path": item.get("records_path", "")}})
                visible_names = {item["name"] for item in datasets}
                normalized_manifest_path = self.normalized / "manifest.json"
                try:
                    normalized_manifest = json.loads(normalized_manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    normalized_manifest = {}
                for file in sorted((self.normalized / "por_base").glob("*.jsonl")):
                    name = file.stem
                    key = f"normalized::{name}"
                    if name in visible_names or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
                        continue
                    count = normalized_manifest.get("counts_by_dataset", {}).get(name)
                    provenance = self.dataset_provenance(name, {})
                    provenance.update(kind="normalized_only", label="Normalizada (sin embeddings)",
                                      fields=["dataset", "context_id", "question_id", "answer_id", "score_id",
                                              "paragraph_id", "context_text", "question_text", "answer_text",
                                              "reference_answer", "score", "score_scale", "is_unanswerable",
                                              "source_file", "source_record_id", "source_answer_id", "source_url"])
                    datasets.append({"name": key, "display_name": name, "responses": count or 0,
                                     "with_score_target": 0, "model": None, "dimensions": None,
                                     "source": provenance.get("source_name"), "provenance": provenance})
                self.send_json({"datasets": datasets})
            elif parsed.path == "/api/normalized/search":
                self.normalized_search(params)
            elif parsed.path == "/api/normalized/download":
                self.normalized_download(params)
            elif parsed.path == "/api/original/search":
                self.original_search(params)
            elif parsed.path == "/api/original/download":
                self.original_download(params)
            elif parsed.path == "/api/normalize/status":
                self.send_json(normalization_status())
            elif parsed.path == "/api/search":
                self.search(params)
            elif parsed.path == "/api/workflow/preview-page":
                self.workflow_preview_page(params)
            elif parsed.path == "/api/dataset/download":
                self.download_dataset(params)
            elif parsed.path == "/api/record":
                self.record(params)
            elif parsed.path == "/api/vector":
                self.vector(params)
            else:
                self.send_json({"error": "Ruta no encontrada"}, 404)
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            self.send_json({"error": str(exc)}, 500)

    def read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 30 * 1024 * 1024:
            raise ValueError("El archivo debe tener entre 1 byte y 30 MB.")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self):
        try:
            host = self.headers.get("Host", "").split(":", 1)[0]
            if host not in {"127.0.0.1", "localhost"}:
                raise ValueError("El explorador solo acepta solicitudes locales.")
            origin = self.headers.get("Origin")
            if origin and not origin.startswith(("http://127.0.0.1:", "http://localhost:")):
                raise ValueError("Origen no permitido para esta aplicación local.")
            body = self.read_body()
            if self.path == "/api/workflow/preview":
                self.workflow_preview(body)
            elif self.path == "/api/workflow/process":
                self.workflow_process(body)
            elif self.path == "/api/workflow/export":
                self.workflow_export(body)
            elif self.path == "/api/dataset/delete":
                self.dataset_delete(body)
            elif self.path == "/api/normalize/start":
                started, state = start_normalization(str(body.get("dataset", "")),
                                                     body.get("download") is True)
                self.send_json(state, 202 if started else 409)
            elif self.path == "/api/original/save":
                self.save_original_dataset(body)
            elif self.path == "/api/normalize/quality":
                self.send_json(analyze_custom_normalization(body, self.results))
            elif self.path == "/api/normalize/custom":
                started, state = start_custom_normalization(body, self.results)
                self.send_json(state, 202 if started else 409)
            else:
                self.send_json({"error": "Ruta no encontrada"}, 404)
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError, KeyError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def workflow_upload_dir(self, upload_id):
        if not re.fullmatch(r"[a-f0-9]{32}", str(upload_id or "")):
            raise ValueError("Carga no válida; vuelve a previsualizar el archivo.")
        return self.results / "importaciones" / upload_id

    def workflow_preview(self, body):
        from workflow_import import inspect_file
        name = Path(str(body.get("filename", "dataset.csv"))).name
        if Path(name).suffix.lower() not in {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}:
            raise ValueError("Formato no admitido. Usa CSV, TSV, JSON o JSONL.")
        content = body.get("content")
        if not isinstance(content, str): raise ValueError("No se recibió el contenido del archivo.")
        upload_id = uuid.uuid4().hex
        folder = self.workflow_upload_dir(upload_id) / "originales"
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / name
        source.write_text(content, encoding="utf-8-sig")
        preview = inspect_file(source, str(body.get("records_path", "")).strip())
        (self.workflow_upload_dir(upload_id) / "preview.json").write_text(
            json.dumps({"filename": name, "records_path": preview.get("records_path", ""),
                        "format": preview["format"], "count": preview["count"],
                        "fields": preview["fields"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.send_json({"upload_id": upload_id, "filename": name, **preview})

    def workflow_preview_page(self, params):
        from workflow_import import read_file_page
        upload = self.workflow_upload_dir(params.get("upload_id", [""])[0])
        metadata_path = upload / "preview.json"
        if not metadata_path.is_file():
            raise ValueError("No se encontró la vista previa; vuelve a cargar el archivo.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        originals = upload / "originales"
        source = originals / metadata["filename"]
        if not source.is_file():
            raise ValueError("No se encontró el archivo original de esta carga.")
        offset = max(0, int(params.get("offset", ["0"])[0]))
        limit = max(1, min(200, int(params.get("limit", ["50"])[0])))
        rows = read_file_page(source, offset, limit, metadata.get("records_path", ""))
        total = int(metadata.get("count", 0))
        self.send_json({"filename": metadata["filename"], "total": total, "offset": offset,
                        "limit": limit, "fields": metadata["fields"], "rows": rows,
                        "has_next": offset + len(rows) < total})

    def workflow_process(self, body):
        from workflow_import import safe_name, write_normalized
        from preparar_vistas import load_api_key, process_dataset
        from vector_store import ChromaVectorStore
        upload = self.workflow_upload_dir(body.get("upload_id"))
        originals = upload / "originales"
        files = [p for p in originals.iterdir() if p.is_file()]
        if len(files) != 1: raise ValueError("No se encontró el archivo cargado.")
        source = files[0]
        dataset = safe_name(body.get("name") or source.stem)
        if (self.results / dataset / "vista_respuestas.jsonl").exists():
            raise ValueError(f"Ya existe la base '{dataset}'. Usa otro nombre para conservar ambos resultados.")
        fields = body.get("fields") or {}
        if not fields.get("question_text") or not fields.get("answer_text"):
            raise ValueError("La pregunta y la respuesta son campos obligatorios.")
        config = {"dataset": dataset, "format": source.suffix.lower().lstrip("."),
                  "records_path": body.get("records_path") or "*", "fields": fields,
                  "keys": body.get("keys") or {}, "context_kind": "custom_context"}
        if config["format"] in {"jsonl", "ndjson"}: config["format"] = "jsonl"
        normalized = upload / "normalizados"
        count = write_normalized(source, config, normalized)
        model = str(body.get("model") or "text-embedding-3-small")
        dimensions = int(body.get("dimensions") or 1536)
        if dimensions < 1 or dimensions > 3072: raise ValueError("Las dimensiones deben estar entre 1 y 3072.")
        key_path = Path(body["api_key_file"]).expanduser() if body.get("api_key_file") else self.api_key_file
        api_key = load_api_key(key_path)
        store = ChromaVectorStore(self.results / "chroma_db", model, dimensions)
        try:
            processed, scored = process_dataset(dataset, normalized, self.results, store, api_key, None)
        finally:
            store.close()
        manifest_path = self.results / "manifest.json"
        try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): manifest = {"datasets": {}}
        manifest.setdefault("datasets", {})[dataset] = {"responses": processed, "with_score_target": scored,
                                                           "model": model, "dimensions": dimensions,
                                                           "source": source.name, "import_id": body["upload_id"]}
        manifest["vector_store"] = "ChromaDB PersistentClient"
        manifest["vector_store_path"] = "chroma_db"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        columns = [
            {"key":"ids", "label":"IDs de contexto, pregunta y respuesta", "default":True},
            {"key":"texts", "label":"Textos y párrafos seleccionados", "default":True},
            {"key":"embedding_refs", "label":"IDs de embeddings", "default":True},
            {"key":"cosine_similarities", "label":"Siete similitudes coseno", "default":True},
            {"key":"targets", "label":"Objetivos (respuesta y calificación)", "default":True},
            {"key":"source", "label":"Procedencia / trazabilidad", "default":True},
            {"key":"model_inputs.grading", "label":"Entrada de modelo para calificación", "default":False},
            {"key":"model_inputs.answer_generation", "label":"Entrada para generación de respuesta", "default":False},
        ]
        granular = [
            ("ids.context_id", "ID de contexto"), ("ids.question_id", "ID de pregunta"),
            ("ids.answer_id", "ID de respuesta"), ("ids.score_id", "ID de calificación"),
            ("texts.question", "Texto de pregunta"), ("texts.candidate_answer", "Texto de respuesta"),
            ("texts.paragraph_nearest_question.paragraph_text", "Párrafo más cercano a pregunta"),
            ("texts.paragraph_nearest_answer.paragraph_text", "Párrafo más cercano a respuesta"),
            ("embedding_refs.question_embedding_id", "Embedding ID de pregunta"),
            ("embedding_refs.answer_embedding_id", "Embedding ID de respuesta"),
            ("embedding_refs.context_centroid_embedding_id", "Embedding ID del centroide"),
            ("embedding_refs.paragraph_nearest_question.embedding_id", "Embedding ID del párrafo cercano a pregunta"),
            ("embedding_refs.paragraph_nearest_answer.embedding_id", "Embedding ID del párrafo cercano a respuesta"),
            ("targets.answer_text", "Respuesta objetivo"), ("targets.score", "Calificación objetivo"),
            ("targets.score_scale", "Escala de calificación"), ("targets.is_unanswerable", "Es no respondible"),
        ] + [(f"cosine_similarities.{key}", f"Similitud: {label}") for key, label in COSINE_FIELDS.items()]
        columns.extend({"key": key, "label": label, "default": False} for key, label in granular)
        self.send_json({"dataset": dataset, "records": processed or count, "scored": scored, "columns": columns})

    def workflow_export(self, body):
        dataset = str(body.get("dataset", ""))
        safe = Path(dataset).name
        source = self.results / safe / "vista_respuestas.jsonl"
        if safe != dataset or not source.is_file(): raise ValueError("No se encontró la vista de la base procesada.")
        fmt = body.get("format", "jsonl")
        if fmt not in {"csv", "json", "jsonl"}: raise ValueError("Formato de exportación inválido.")
        allowed = {"ids", "texts", "embedding_refs", "cosine_similarities", "targets", "source",
                   "model_inputs.grading", "model_inputs.answer_generation",
                   "ids.context_id", "ids.question_id", "ids.answer_id", "ids.score_id",
                   "texts.question", "texts.candidate_answer",
                   "texts.paragraph_nearest_question.paragraph_text",
                   "texts.paragraph_nearest_answer.paragraph_text",
                   "embedding_refs.question_embedding_id", "embedding_refs.answer_embedding_id",
                   "embedding_refs.context_centroid_embedding_id",
                   "embedding_refs.paragraph_nearest_question.embedding_id",
                   "embedding_refs.paragraph_nearest_answer.embedding_id",
                   "cosine_similarities.question_answer", "cosine_similarities.question_context_centroid",
                   "cosine_similarities.answer_context_centroid",
                   "cosine_similarities.question_nearest_question_paragraph",
                   "cosine_similarities.question_nearest_answer_paragraph",
                   "cosine_similarities.answer_nearest_question_paragraph",
                   "cosine_similarities.answer_nearest_answer_paragraph",
                   "targets.answer_text", "targets.score", "targets.score_scale", "targets.is_unanswerable",
                   "source.source_file", "source.source_record_id", "source.source_answer_id"}
        columns = [key for key in body.get("columns", []) if key in allowed]
        if not columns: raise ValueError("Selecciona al menos un bloque para exportar.")
        def get_path(row, path):
            value = row
            for part in path.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            return value
        vector_store = None
        if body.get("include_vectors"):
            from vector_store import ChromaVectorStore
            vector_store = ChromaVectorStore(self.results / "chroma_db")
        outdir = self.results / "exportaciones"
        outdir.mkdir(parents=True, exist_ok=True)
        target = outdir / f"{safe}_entrenamiento.{fmt}"
        temporary = target.with_name(target.name + ".tmp")
        def build_output(row):
            output = {key: get_path(row, key) for key in columns}
            if vector_store:
                refs = row.get("embedding_refs") or {}
                ids = [refs.get("question_embedding_id"), refs.get("answer_embedding_id"), refs.get("context_centroid_embedding_id")]
                ids += [(refs.get(key) or {}).get("embedding_id") for key in ("paragraph_nearest_question", "paragraph_nearest_answer")]
                output["vectors"] = {}
                for eid in ids:
                    if eid:
                        vector = vector_store.get(eid)
                        if vector is not None: output["vectors"][eid] = vector
            return output
        records = 0
        try:
            with temporary.open("w", encoding="utf-8-sig" if fmt == "csv" else "utf-8", newline="" if fmt == "csv" else None) as f:
                if fmt == "csv":
                    fields = columns + (["vectors"] if vector_store else [])
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for row in iter_jsonl(source):
                        output = build_output(row)
                        writer.writerow({k: (" ".join(part.strip() for part in v.splitlines() if part.strip()) if isinstance(v, str) else json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k,v in output.items()})
                        records += 1
                elif fmt == "jsonl":
                    for row in iter_jsonl(source):
                        f.write(json.dumps(build_output(row), ensure_ascii=False, separators=(",", ":")) + "\n")
                        records += 1
                else:
                    f.write("[\n")
                    first = True
                    for row in iter_jsonl(source):
                        if not first: f.write(",\n")
                        f.write(json.dumps(build_output(row), ensure_ascii=False, indent=2))
                        first = False
                        records += 1
                    f.write("\n]\n")
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if vector_store: vector_store.close()
        self.send_json({"path": str(target.resolve()), "records": records})

    def dataset_delete(self, body):
        """Remove one generated view and its local workflow artifacts."""
        dataset = str(body.get("dataset", ""))
        if dataset.startswith("raw::"):
            with RAW_CATALOG_LOCK:
                catalog = load_raw_catalog()
                if dataset not in catalog["datasets"]:
                    raise ValueError("No se encontró la base original.")
                item = catalog["datasets"].pop(dataset)
                save_raw_catalog(catalog)
            self.send_json({"dataset": dataset, "deleted": True, "removed_exports": 0,
                            "removed_import": False, "vectors_retained": True,
                            "source_preserved": True})
            return
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", dataset):
            raise ValueError("Nombre de base no válido.")
        target = (self.results / dataset).resolve()
        results_root = self.results.resolve()
        if target.parent != results_root or not target.is_dir():
            raise ValueError("No se encontró la vista de esa base.")
        if not (target / "vista_respuestas.jsonl").is_file():
            raise ValueError("La carpeta seleccionada no contiene una vista de base válida.")

        manifest_path = self.results / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {"datasets": {}}
        datasets = manifest.setdefault("datasets", {})
        metadata = datasets.get(dataset, {})
        import_id = metadata.get("import_id")

        # Remove the catalogue entry atomically so a partial cleanup cannot leave
        # a deleted view listed as available.
        datasets.pop(dataset, None)
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
        temporary_manifest.replace(manifest_path)

        shutil.rmtree(target)
        exports = self.results / "exportaciones"
        removed_exports = 0
        if exports.is_dir():
            for file in exports.glob(f"{dataset}_entrenamiento.*"):
                if file.is_file():
                    file.unlink()
                    removed_exports += 1

        removed_import = False
        if isinstance(import_id, str) and re.fullmatch(r"[a-f0-9]{32}", import_id):
            still_referenced = any(item.get("import_id") == import_id
                                   for item in datasets.values() if isinstance(item, dict))
            import_dir = (self.results / "importaciones" / import_id).resolve()
            import_root = (self.results / "importaciones").resolve()
            if not still_referenced and import_dir.parent == import_root and import_dir.is_dir():
                shutil.rmtree(import_dir)
                removed_import = True

        self.line_index_cache = {key: value for key, value in self.line_index_cache.items()
                                 if key[0] != str((target / "vista_respuestas.jsonl").resolve())}
        self.send_json({"dataset": dataset, "deleted": True,
                        "removed_exports": removed_exports, "removed_import": removed_import,
                        "vectors_retained": True})

    def selected_file(self, params):
        dataset = params.get("dataset", [""])[0]
        safe = Path(dataset).name
        file = self.results / safe / "vista_respuestas.jsonl"
        if not dataset or safe != dataset or not file.is_file():
            raise ValueError("Selecciona una base que ya tenga vista generada.")
        return safe, file

    @classmethod
    def line_offsets(cls, file):
        stat = file.stat()
        key = (str(file.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = cls.line_index_cache.get(key)
        if cached is not None:
            return cached
        # Compact byte offsets allow direct page access without parsing all JSONL rows.
        offsets = array("Q")
        with file.open("rb") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(position)
        cls.line_index_cache = {k: v for k, v in cls.line_index_cache.items() if k[0] != str(file.resolve())}
        cls.line_index_cache[key] = offsets
        return offsets

    def download_dataset(self, params):
        dataset, file = self.selected_file(params)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Length", str(file.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{dataset}.jsonl"')
        self.end_headers()
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                self.wfile.write(block)

    def search(self, params):
        dataset, file = self.selected_file(params)
        query = params.get("q", [""])[0]
        needle = fold(query.strip())
        field = params.get("field", ["all"])[0]
        similarity = params.get("similarity", [""])[0]
        min_raw = params.get("minimum", [""])[0]
        minimum = float(min_raw) if min_raw else None
        if minimum is not None and not -1 <= minimum <= 1:
            raise ValueError("La similitud mínima debe estar entre -1 y 1.")
        if minimum is not None and not similarity:
            raise ValueError("Elige la similitud que se usará con el mínimo.")
        limit = max(1, min(200, int(params.get("limit", ["100"])[0])))
        offset = max(0, min(10_000_000, int(params.get("offset", ["0"])[0])))
        scored_only = params.get("scored", ["0"])[0] == "1"

        def compact(row):
            return {"ids": row.get("ids") or {}, "texts": row.get("texts") or {},
                    "cosine_similarities": row.get("cosine_similarities") or {},
                    "targets": row.get("targets") or {}}

        # Browsing every row is the common case. An in-memory byte-offset index
        # makes later pages seek directly to only the requested records.
        if not needle and not similarity and minimum is None and not scored_only:
            offsets = self.line_offsets(file)
            selected_offsets = offsets[offset:offset + limit]
            matches = []
            with file.open("rb") as stream:
                for position in selected_offsets:
                    stream.seek(position)
                    matches.append(compact(json.loads(stream.readline().decode("utf-8"))))
            total = len(offsets)
            self.send_json({"dataset": dataset, "scanned": len(matches), "total_matches": total,
                            "offset": offset, "matches": matches,
                            "has_next": offset + len(matches) < total,
                            "truncated": offset + len(matches) < total})
            return

        context_ids = set()
        if needle and field in {"context", "paragraph", "all"}:
            context_ids = paragraph_context_matches(self.normalized, needle, field in {"context", "all"})

        selected_offsets = []
        top_heap = []
        total = 0
        scanned = 0
        with file.open("rb") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                scanned += 1
                row = json.loads(line.decode("utf-8"))
                ids = row.get("ids") or {}
                texts = row.get("texts") or {}
                if scored_only and row.get("targets", {}).get("score") is None:
                    continue
                if needle:
                    id_text = " ".join(str(value or "") for value in ids.values())
                    question = texts.get("question") or ""
                    answer = texts.get("candidate_answer") or ""
                    para_q = (texts.get("paragraph_nearest_question") or {}).get("paragraph_text") or ""
                    para_a = (texts.get("paragraph_nearest_answer") or {}).get("paragraph_text") or ""
                    if field == "ids":
                        matched = needle in fold(id_text)
                    elif field == "question":
                        matched = needle in fold(question)
                    elif field == "answer":
                        matched = needle in fold(answer)
                    elif field in {"context", "paragraph"}:
                        matched = ids.get("context_id") in context_ids
                    else:
                        matched = (needle in fold(id_text + " " + question + " " + answer + " " + para_q + " " + para_a)
                                   or ids.get("context_id") in context_ids)
                    if not matched:
                        continue
                if similarity:
                    value = (row.get("cosine_similarities") or {}).get(similarity)
                    if value is None or (minimum is not None and value < minimum):
                        continue
                rank = (row.get("cosine_similarities") or {}).get(similarity) if similarity else None
                if similarity:
                    entry = (rank, scanned, position)
                    heap_size = offset + limit
                    if len(top_heap) < heap_size:
                        heapq.heappush(top_heap, entry)
                    elif entry > top_heap[0]:
                        heapq.heapreplace(top_heap, entry)
                elif offset <= total < offset + limit:
                    selected_offsets.append(position)
                total += 1

        if similarity:
            selected_offsets = [entry[2] for entry in sorted(top_heap, reverse=True)[offset:offset + limit]]
        matches = []
        with file.open("rb") as stream:
            for position in selected_offsets:
                stream.seek(position)
                matches.append(compact(json.loads(stream.readline().decode("utf-8"))))
        self.send_json({"dataset": dataset, "scanned": scanned, "total_matches": total,
                        "offset": offset, "matches": matches,
                        "has_next": offset + len(matches) < total,
                        "truncated": offset + len(matches) < total})

    def record(self, params):
        dataset, file = self.selected_file(params)
        answer_id = params.get("answer_id", [""])[0]
        for row in iter_jsonl(file):
            if row.get("ids", {}).get("answer_id") == answer_id:
                self.send_json(row)
                return
        self.send_json({"error": "No se encontró ese answer_id en la base elegida."}, 404)

    def vector(self, params):
        embedding_id = params.get("id", [""])[0]
        database = self.results / "chroma_db"
        if not database.exists():
            self.send_json({"error": "No existe el almacén de vectores; primero genera una vista."}, 404)
            return
        store = ChromaVectorStore(database)
        result = store.get_record(embedding_id)
        store.close()
        if result is None:
            self.send_json({"error": "No se encontró ese embedding_id."}, 404)
            return
        self.send_json(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--api-key-file", type=Path, help="Archivo local de clave API para el flujo de importación")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.results_dir.is_dir():
        parser.error(f"No existe la carpeta de resultados: {args.results_dir}")
    if args.port < 1024 or args.port > 65535:
        parser.error("El puerto debe estar entre 1024 y 65535")
    AppHandler.results = args.results_dir.resolve()
    AppHandler.normalized = args.normalized_dir.resolve()
    AppHandler.api_key_file = args.api_key_file.resolve() if args.api_key_file else None
    server = ThreadingHTTPServer(("127.0.0.1", args.port), AppHandler)
    print(f"Explorador QA disponible en http://127.0.0.1:{args.port} (solo en este equipo). Ctrl+C para cerrar.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
