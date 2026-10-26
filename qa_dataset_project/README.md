# Normalización de datasets QA

Este componente transforma archivos de preguntas y respuestas de distintas fuentes a un esquema común. Su salida es la entrada del proyecto qa_embedding_features.

El normalizador no depende de OpenAI ni de una conexión a Internet para trabajar con los originales que ya están en la carpeta del proyecto. Usa Python 3.10 o superior y la biblioteca estándar.

## Qué hace

Para cada registro de origen, el programa:

1. Lee la estructura propia de la fuente o aplica una configuración de campos.
2. Limpia el texto y separa los párrafos del contexto.
3. Asigna IDs de contexto, pregunta y respuesta y conserva las claves de origen.
4. Asocia la calificación cuando la fuente proporciona una.
5. Valida que los IDs requeridos sean únicos y que las respuestas apunten a contextos existentes.
6. Escribe una exportación CSV y otra JSONL por dataset, además de tablas consolidadas y un manifiesto.

El resultado representa una fila por respuesta. Si una pregunta tiene varias respuestas, todas comparten el question_id y cada respuesta recibe un answer_id distinto.

## Estructura

- **normalizar_datasets.py**: extractor, adaptador y generador de las exportaciones.
- **buscar.py**: búsqueda de texto o IDs en las filas normalizadas.
- **preparar_roars.py**: prepara el archivo de trabajo de ROARS excluyendo filas sin respuesta y campos ajenos al análisis.
- **config_*.json**: mapeos de campos para los datasets que no tienen lector integrado.
- **originales/**: archivos fuente. No se modifican al normalizarlos.
- **originales/adicionales/**: fuentes añadidas y archivos derivados de preparación.
- **resultados/por_base/**: exportación independiente por cada dataset.
- **resultados/registros.jsonl y registros.csv**: tabla consolidada de respuestas.
- **resultados/contextos.jsonl y parrafos.jsonl**: contexto completo y párrafos asociados.
- **resultados/manifest.json**: URLs, hashes SHA-256, recuentos y estado de validación.
- **trabajo/**: extracción temporal de archivos comprimidos. Se puede regenerar.
- **demo/**: archivo pequeño de referencia para aprender a crear una configuración.

El archivo JSONL guarda un registro por línea y conserva los saltos de línea normalizados dentro del texto. En CSV, los saltos de línea de una celda se convierten en espacios para que una respuesta ocupe una sola línea física. Ambos formatos representan las mismas filas.

## Datasets incluidos

Los nombres se usan en el parámetro --dataset. Los recuentos son los que figuran en las exportaciones locales actuales.

| Dataset | Respuestas | Notas |
| --- | ---: | --- |
| squad2_train | 130.319 | SQuAD 2.0; incluye preguntas sin respuesta respaldada |
| squad2_dev | 26.247 | Partición de desarrollo de SQuAD 2.0 |
| cmu_qa_v1.2 | 3.975 | La fuente incluye niveles de dificultad |
| mohler_v1.0 | 630 | Respuestas de estudiantes y respuestas de referencia |
| mohler_v2.0 | 2.273 | Conserva la calificación normalizada de la fuente |
| boolq_train | 9.427 | Preguntas sí/no con pasaje |
| boolq_dev | 3.270 | Partición de desarrollo de BoolQ |
| roars_2class | 981 | Etiqueta humana en escala de 0 a 1 |
| roars_3class | 981 | Etiqueta humana en escala de 0 a 2 |

Los datasets SQuAD, CMU, Mohler y ROARS tienen lectores incluidos en el normalizador. BoolQ se conserva como fuente adicional con su configuración JSON; se vuelve a procesar indicando el archivo y el mapeo de campos, como se muestra más adelante.

## Ejecutar desde VS Code o PowerShell

Abre esta carpeta en VS Code o abre Proyecto_QA.code-workspace para trabajar con los dos componentes. En una terminal cuya carpeta actual sea qa_dataset_project, procesa todos los datasets integrados:

~~~powershell
python normalizar_datasets.py --dataset all
~~~

Una ejecución completa vuelve a procesar los siete datasets incorporados como lectores: SQuAD train/dev, CMU, Mohler v1/v2 y las dos escalas ROARS. Conserva los datasets adicionales que ya estén en los resultados, como BoolQ.

Para volver a generar una sola base:

~~~powershell
python normalizar_datasets.py --dataset roars_2class
~~~

Otros valores admitidos: squad2_train, squad2_dev, cmu_qa_v1.2, mohler_v1.0, mohler_v2.0 y roars_3class. Una ejecución individual reemplaza las filas de ese dataset en los archivos consolidados y conserva las otras bases.

La opción --download descarga las fuentes oficiales que falten. Sin ella, se usan los archivos locales y el programa falla con un mensaje claro si falta un original. --no-download se acepta por compatibilidad, pero no hace falta escribirlo.

~~~powershell
python normalizar_datasets.py --dataset squad2_dev --download
~~~

La carpeta de salida puede cambiarse con --output-dir:

~~~powershell
python normalizar_datasets.py --dataset roars_2class --output-dir "D:\datos\qa"
~~~

## Cómo se relacionan los IDs

Los prefijos identifican el tipo de entidad:

| Prefijo | Entidad |
| --- | --- |
| C- | Contexto |
| P- | Pregunta |
| R- | Respuesta |
| CAL- | Calificación |
| PAR- | Párrafo |

Después aparece una etiqueta corta del dataset, por ejemplo SQTR para SQuAD train o R2C para ROARS 2-class, y un sufijo calculado a partir de claves de origen mediante SHA-256. El sufijo hace que el ID sea compacto y reproducible.

La relación se consulta en las columnas, no se deduce leyendo el sufijo:

- question_id apunta a la pregunta asociada con ese contexto.
- answer_id identifica una respuesta concreta de esa pregunta.
- score_id identifica la nota de esa respuesta; queda vacío si no hay calificación.
- paragraph_id identifica un párrafo y context_id indica a qué contexto pertenece.

Las claves de origen permiten reconstruir estos IDs al volver a procesar el mismo archivo. Para mantenerlos estables, asigna claves persistentes en keys; si el normalizador debe recurrir al número de fila, reordenar el archivo puede cambiar los IDs. Si cambian las claves o el nombre del dataset, también pueden cambiar.

## Añadir un archivo propio

Se admiten CSV, TSV, JSONL y JSON. Para una fuente propia, crea una copia de una plantilla y configura:

- **dataset**: nombre corto que se usará en resultados/por_base.
- **format**: csv, tsv, jsonl o json.
- **records_path**: ruta de los registros en un JSON anidado.
- **fields**: correspondencia entre campos normalizados y columnas/rutas de origen.
- **keys**: claves de origen para conservar relaciones entre filas.
- **score_scale** y **source_url**: información opcional de la fuente.

Ejemplo mínimo para un CSV:

~~~json
{
  "dataset": "mi_dataset",
  "format": "csv",
  "encoding": "utf-8-sig",
  "context_kind": "paragraph",
  "score_scale": "0..5",
  "source_url": "https://ejemplo.org/dataset",
  "fields": {
    "context_text": "passage",
    "question_text": "question",
    "answer_text": "student_answer",
    "reference_answer": "expected_answer",
    "score": "grade"
  },
  "keys": {
    "source_id": "record_id",
    "context_id": "passage_id",
    "question_id": "question_id",
    "answer_id": "response_id"
  }
}
~~~

question_text y answer_text son obligatorios. context_text, reference_answer, score e is_unanswerable son opcionales. En CSV/TSV, los valores son nombres de columnas. En JSON, son rutas separadas por puntos.

Para procesar una sola fuente local:

~~~powershell
python normalizar_datasets.py --dataset mi_dataset --input "D:\datos\mi_base.csv" --config "D:\datos\config_mi_dataset.json"
~~~

Para agregarla mientras se regeneran los datasets incluidos:

~~~powershell
python normalizar_datasets.py --dataset all --input "D:\datos\mi_base.csv" --config "D:\datos\config_mi_dataset.json"
~~~

Si la URL apunta directamente al archivo de datos, se puede descargar con --url:

~~~powershell
python normalizar_datasets.py --dataset mi_dataset --url "https://ejemplo.org/mi_base.csv" --config "D:\datos\config_mi_dataset.json"
~~~

La descarga se guarda en originales/adicionales/. Si el enlace no termina en .csv, .tsv, .json o .jsonl, indica --filename con el nombre y extensión correctos. Una página de descripción de GitHub no es un archivo de datos directo; usa el enlace raw del archivo.

### JSON anidado y respuestas múltiples

records_path selecciona qué objetos se interpretan como registros. El comodín * expande listas u objetos. Las rutas de fields se evalúan desde cada registro seleccionado. Para llegar a un campo del objeto padre se puede subir niveles: ../../context.

Si answer_text selecciona una lista como answers.*.text, el normalizador genera una fila por cada respuesta. Los registros que no tienen pregunta o respuesta válida se excluyen; las etiquetas opcionales pueden quedar vacías.

Las plantillas completas están en config_csv.example.json y config_json_anidado.example.json.

## BoolQ y ROARS

BoolQ está integrado mediante config_boolq_train.json y config_boolq_dev.json. Para repetir la normalización local de train:

~~~powershell
python normalizar_datasets.py --dataset boolq_train --input "originales\adicionales\boolq_train.jsonl" --config config_boolq_train.json
~~~

BoolQ tiene respuesta booleana, no una calificación numérica. Por ello score y score_id quedan vacíos.

ROARS tiene dos etiquetas distintas. Se mantienen como datasets separados para no mezclar sus escalas:

~~~powershell
python normalizar_datasets.py --dataset roars_2class
python normalizar_datasets.py --dataset roars_3class
~~~

El proyecto contiene 981 respuestas por escala; se excluyen 87 filas sin respuesta de estudiante. Los originales y el diccionario están en originales/roars_dataset/. El repositorio declara licencia CC BY-NC 4.0: uso no comercial con atribución. Antes de redistribuir los datos o sus derivados, revisa originales/roars_dataset/README_fuente.md y la [fuente ROARS](https://github.com/owenhenkel/ROARS-dataset).

## Buscar registros

La búsqueda trabaja sobre las exportaciones normalizadas; no consulta OpenAI ni calcula similitudes. Desde esta carpeta:

~~~powershell
python buscar.py "Abraham Lincoln" --dataset cmu_qa_v1.2 --field context
python buscar.py "prototype program" --dataset mohler_v1.0 --field question
python buscar.py --id "R-SQTR-IDENTIFICADOR" --limit 5
~~~

Los campos disponibles son question, answer, reference, context e all. La búsqueda ignora mayúsculas y tildes. --limit limita las coincidencias mostradas; --records permite elegir un CSV o JSONL distinto. Cada consulta se guarda en resultados/busquedas/ como CSV y JSONL.

## Esquema de la salida

Los registros normalizados incluyen dataset, IDs, textos, calificación, escala, estado de no respondible y procedencia. source_file, source_record_id, source_answer_id y source_url permiten rastrear la fila hasta la fuente.

Cuando una fuente no aporta nota numérica, score y score_id quedan vacíos. SQuAD 2.0 sí distingue preguntas sin respuesta respaldada mediante is_unanswerable. Mohler no tiene pasaje de lectura por pregunta; el normalizador deja context_text vacío en vez de inventar contexto.

El manifiesto contiene recuentos por dataset, hashes SHA-256 de los archivos de entrada y validaciones de IDs y referencias. Sirve para comprobar qué archivos originaron una exportación y si el esquema quedó consistente.

## Fuentes y atribución

- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- [CMU Question-Answer Dataset](https://www.cs.cmu.edu/~ark/QA-data/)
- [Mohler Short Answer Grading](https://web.eecs.umich.edu/~mihalcea/downloads.html)
- [BoolQ](https://github.com/google-research-datasets/boolean-questions)
- [ROARS](https://github.com/owenhenkel/ROARS-dataset)

Cada fuente tiene condiciones de uso propias. Revisa la licencia indicada por sus autores antes de redistribuir archivos o derivados.
