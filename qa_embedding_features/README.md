# Embeddings, características y explorador QA

Este componente lee las exportaciones JSONL del normalizador y crea una vista detallada por respuesta. Para cada fila guarda los IDs originales, referencias a embeddings, siete similitudes coseno, los párrafos seleccionados y los objetivos disponibles.

El programa prepara características y ofrece entrenamiento supervisado local con scikit-learn. Los modelos predicen un objetivo etiquetado, como una calificación o categoría; no generan texto de respuesta.

## Relación con el normalizador

La carpeta predeterminada de entrada es qa_dataset_project/resultados/. Cada dataset debe tener una exportación en por_base/<nombre>.jsonl, además de contextos.jsonl y, cuando corresponda, parrafos.jsonl.

El lector de embeddings recibe estas tablas normalizadas y escribe resultados propios. No modifica los originales ni las exportaciones del normalizador. Si normalizaste los datos en otra ubicación, indica esa ruta con --normalized-dir.

En **Explorar bases** aparecen vistas con embeddings, bases normalizadas sin embeddings y fuentes originales sin normalizar. El selector identifica cada tipo. En las fuentes originales puedes buscar por campos y consultar o descargar filas sin que se generen IDs ni vectores; en las bases normalizadas puedes revisar los IDs y campos comunes. Los filtros de similitud solo están disponibles para las vistas con embeddings.

## Normalizar desde la aplicación

La pestaña **Normalizar bases** ejecuta `qa_dataset_project/normalizar_datasets.py` desde la interfaz. Puedes subir o arrastrar una base propia, elegir una fuente incorporada, una partición concreta o todas las bases integradas. La ejecución usa los originales que ya están en `qa_dataset_project/originales/`; la casilla de descarga es opcional y solo descarga archivos oficiales que falten.

BoolQ se procesa como fuente adicional: necesita el JSONL y su configuración en el proyecto del normalizador. La opción **Todas las bases integradas** vuelve a procesar SQuAD train/dev, CMU, Mohler v1/v2 y ROARS en sus dos escalas; conserva fuentes adicionales como BoolQ que ya estén normalizadas.

En **Normalizar bases**, después de previsualizar un archivo puedes pulsar **Guardar como original (sin normalizar)**. La app conserva el archivo tal cual y lo añade al selector de **Explorar bases** con la etiqueta *Original (sin normalizar)*. En esa vista puedes revisar filas, buscar texto por campo y descargar el archivo fuente; no se crean IDs normalizados ni embeddings.

Al normalizar un archivo, la exportación común también queda disponible en **Explorar bases** como *Normalizada (sin embeddings)*. Allí se conservan los IDs estandarizados y se puede buscar por sus campos, aunque los filtros de similitud siguen desactivados. Si después generas la vista de embeddings, esa vista calculada se identifica como *Normalizada* y ambas vistas siguen siendo distintas de la original.

### Añadir una base propia desde la vista

En la misma pestaña puedes seleccionar o arrastrar un archivo CSV, TSV, JSON o JSONL de hasta 20 MB. La vista previa presenta una muestra, los campos detectados y su cobertura. Asigna pregunta y respuesta y ejecuta el informe de calidad antes de continuar; el informe estima cuántas respuestas se crearán, señala filas sin pregunta o respuesta, contexto vacío, calificaciones inválidas y duplicados, y permite descargar los hallazgos en CSV. Contexto, respuesta esperada, calificación e ID de origen son campos opcionales. Para JSON anidado puedes corregir la ruta de registros y usar una ruta relativa del campo, como `../../context`.

Al normalizar, el archivo original se copia a `qa_dataset_project/originales/adicionales/` con un nombre derivado de un hash SHA-256 abreviado del archivo. El mapeo se guarda en `config_<nombre>.json`. Si vuelves a cargar el mismo archivo, se reconoce la misma versión y se mantienen los IDs; un archivo con contenido distinto se conserva como otra versión. El normalizador crea un registro estándar por cada respuesta, deriva IDs relacionados desde los identificadores y claves de origen, separa el texto del contexto en párrafos y actualiza `resultados/por_base/<nombre>.jsonl` y `.csv`, junto con las tablas compartidas de registros, contextos y párrafos y el manifiesto. Si se vuelve a usar un nombre existente, se reemplazan las filas normalizadas de esa base; la vista solicita confirmación antes de hacerlo.

Las filas sin pregunta o sin respuesta se omiten. Una calificación no numérica produce un error para que se corrija el mapeo o el archivo. La vista informa cuántas filas quedaron normalizadas y dónde se guardó el original.

La vista muestra el resultado de normalización y la carpeta de salida. Esta etapa crea IDs y organiza contextos, párrafos y respuestas. No genera embeddings ni envía textos a OpenAI. Después, genera la vista de características para una base con `preparar_vistas.py --dataset <nombre>` (consulta la sección siguiente).

## Preparación del entorno

Abre esta carpeta en VS Code. El entorno virtual local contiene ChromaDB y las demás dependencias:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Si .venv ya existe, no hace falta volver a crearlo. El generador usa por defecto el modelo text-embedding-3-small con 1536 dimensiones.

La clave puede llegar mediante la variable de entorno OPENAI_API_KEY o con --api-key-file. También se puede configurar en la vista de importación. El programa lee la clave para autenticar la solicitud y no la escribe en las vistas, manifiestos ni mensajes de registro.

Las preguntas, textos de respuesta y párrafos de contexto que aún no tengan vector se envían a la API de embeddings de OpenAI. La respuesta esperada y la calificación se conservan como objetivos y no se envían para generar embeddings. Los textos enviados pueden estar sujetos a uso facturable según la cuenta de API.

## Generar las vistas

Desde una terminal abierta en qa_embedding_features, genera una vista para una sola base:

~~~powershell
.\.venv\Scripts\python.exe preparar_vistas.py --dataset roars_2class --api-key-file "C:\ruta\KEY.txt"
~~~

ROARS tiene dos escalas y, por eso, se procesa en dos datasets separados:

~~~powershell
.\.venv\Scripts\python.exe preparar_vistas.py --dataset roars_2class --api-key-file "C:\ruta\KEY.txt"
.\.venv\Scripts\python.exe preparar_vistas.py --dataset roars_3class --api-key-file "C:\ruta\KEY.txt"
~~~

Para procesar todas las exportaciones normalizadas, usa --dataset all. Esta opción puede tardar bastante en SQuAD 2.0. Para una ejecución pequeña, --limit 3 procesa las primeras tres respuestas por base seleccionada. No uses ese parámetro en la generación final si necesitas las bases completas.

Si los resultados normalizados no están en la ubicación predeterminada, puedes indicar la carpeta del proyecto normalizador o directamente resultados/:

~~~powershell
.\.venv\Scripts\python.exe preparar_vistas.py --normalized-dir "C:\datos\qa_dataset_project\resultados" --dataset squad2_train --api-key-file "C:\ruta\KEY.txt"
~~~

Los embeddings se solicitan por lotes. Antes de enviar texto, el generador comprueba si su embedding_id ya existe en ChromaDB. Al repetir una ejecución con el mismo modelo y dimensiones, reutiliza los vectores disponibles y solicita solamente los que falten.

## Cómo se calculan las características

Para cada respuesta el programa obtiene:

1. El embedding de la pregunta.
2. El embedding del texto de respuesta que trae el dataset.
3. Un embedding por párrafo del contexto. Los párrafos largos se dividen en fragmentos para ajustarse a los límites de entrada.
4. El centroide del contexto, calculado como el promedio de los vectores de sus párrafos.
5. El párrafo del contexto con mayor similitud coseno respecto a la pregunta.
6. El párrafo del contexto con mayor similitud coseno respecto a la respuesta.
7. Siete similitudes coseno:
   - pregunta con respuesta;
   - pregunta con centroide del contexto;
   - respuesta con centroide del contexto;
   - pregunta con el párrafo más cercano a la pregunta;
   - pregunta con el párrafo más cercano a la respuesta;
   - respuesta con el párrafo más cercano a la pregunta;
   - respuesta con el párrafo más cercano a la respuesta.

La similitud coseno es una medida geométrica entre vectores. Un valor cercano a 1 indica direcciones parecidas, alrededor de 0 indica poca alineación y un valor negativo indica direcciones opuestas. No es una calificación humana ni demuestra por sí sola que una respuesta sea correcta.

Si una fuente no tiene contexto o no contiene párrafos, las referencias de contexto y las similitudes correspondientes quedan nulas. El programa no inventa párrafos ni calificaciones.

## Archivos generados

Cada dataset procesado crea una carpeta resultados/<nombre>/:

- **vista_respuestas.jsonl**: una línea por respuesta con IDs, textos, referencias a embeddings, similitudes, objetivos y procedencia.
- **vista_respuestas.csv**: resumen tabular para Excel y análisis. Los saltos de línea dentro de celdas se convierten en espacios; JSONL conserva el texto completo.
- **entrenamiento_respuestas.jsonl**: entrada de señales de pregunta/contexto y respuesta objetivo.
- **entrenamiento_calificaciones.jsonl**: características de pregunta, respuesta candidata y contexto con la nota como objetivo. Solo tiene filas cuando existe una calificación numérica.

El manifiesto resultados/manifest.json registra cuántas respuestas se procesaron, cuántas tienen nota, qué modelo y dimensión se usaron y cuál fue la fuente normalizada.

### Campos de una fila de vista

- **ids**: context_id, question_id, answer_id, score_id y paragraph_id procedentes del normalizador.
- **embedding_refs**: IDs de los embeddings de pregunta y respuesta, centroide y párrafos elegidos.
- **cosine_similarities**: los siete valores descritos arriba.
- **texts**: pregunta, respuesta y texto de los párrafos seleccionados.
- **targets**: respuesta objetivo, score, escala de la nota y estado de no respondible.
- **source**: archivo y claves de origen para rastrear el registro.
- **model_inputs.grading**: referencias de pregunta/respuesta/contexto y similitudes. Incluye señales derivadas de la respuesta candidata.
- **model_inputs.answer_generation**: referencias derivadas de pregunta y contexto; excluye la respuesta candidata.

Los campos de modelo usan referencias a embeddings. La vista no copia el vector completo cientos de veces. Para leer un vector se usa embedding_id y se consulta ChromaDB.

## Dónde se guardan los vectores

ChromaDB mantiene el almacenamiento persistente en resultados/chroma_db/. Cada colección corresponde a una combinación de modelo y dimensión. Los IDs de vector incorporan el modelo, la dimensión y el texto, por lo que el mismo texto se puede reutilizar entre datasets procesados con la misma configuración.

Para recuperar un vector por su ID:

~~~powershell
.\.venv\Scripts\python.exe consultar_vector.py E-ID-COPIADO-DE-LA-VISTA
~~~

Para guardarlo como JSON:

~~~powershell
.\.venv\Scripts\python.exe consultar_vector.py E-ID-COPIADO-DE-LA-VISTA --output vector.json
~~~

El archivo resultados/embeddings.sqlite3 es un respaldo del almacén anterior. Las ejecuciones actuales consultan ChromaDB. migrar_sqlite_a_chroma.py sirve únicamente para importar datos de ese SQLite si se necesita reconstruir ChromaDB.

## Usar el explorador local

Inicia la aplicación en el puerto usado por el proyecto:

~~~powershell
.\.venv\Scripts\python.exe app_interactiva.py --port 8768
~~~

Abre http://127.0.0.1:8768/. El servidor solo acepta conexiones locales. El perfil Ejecutar explorador QA de VS Code también inicia el puerto 8768.

### Cómo se comunica el navegador con Python

La página está definida dentro de app_interactiva.py. Python la sirve y atiende las solicitudes locales; no hay un servicio web externo ni una compilación separada del frontend. La interfaz usa fetch para pedir datos al servidor:

| Solicitud | Función |
| --- | --- |
| GET /api/datasets | Lee el manifiesto y las carpetas de vistas disponibles |
| GET /api/search | Busca y devuelve una página de resultados del JSONL seleccionado |
| GET /api/record | Recupera el detalle completo de una respuesta |
| GET /api/vector | Consulta un embedding en ChromaDB |
| POST /api/workflow/preview | Guarda temporalmente el archivo y devuelve campos y ejemplos |
| POST /api/workflow/process | Normaliza la importación, calcula embeddings y escribe la vista |
| POST /api/workflow/export | Exporta los bloques seleccionados a CSV, JSON o JSONL |
| POST /api/dataset/delete | Borra una vista después de la confirmación de la interfaz |

La vista previa y la búsqueda se procesan localmente. Durante el cálculo, el backend Python lee la clave y envía a OpenAI solo los textos que necesitan embeddings. Los IDs y vectores quedan en los resultados locales.

En **Explorar bases**:

- selecciona una vista generada;
- busca por texto o ID en preguntas, respuestas, párrafos y contextos;
- recorre resultados por páginas, con hasta 200 filas por página;
- filtra por una similitud coseno y un mínimo;
- activa **Solo registros con calificación** cuando busques ejemplos etiquetados;
- pulsa **Ver** para consultar IDs, procedencia, objetivos, párrafos y similitudes;
- pulsa un ID de embedding para ver el comienzo del vector o descargarlo como JSON.

El filtro de similitud solo incluye filas donde ese valor se calculó. Por ejemplo, Mohler no tiene contexto de lectura, así que los filtros de similitud contextual no aplican a sus filas.

## Preparar otra fuente desde la aplicación

Abre **Preparar datos de entrenamiento** para importar una fuente directamente:

1. Selecciona un archivo CSV, TSV, JSON, JSONL o NDJSON. La carga tiene un límite de 30 MB.
2. Escribe un nombre nuevo y pulsa **Cargar y previsualizar**. La vista previa muestra filas de ejemplo, los campos detectados y su cobertura.
3. En JSON anidado, revisa la ruta de registros. El asterisco expande arrays; las rutas manuales permiten subir al objeto padre, por ejemplo ../../context.
4. Asigna el campo de pregunta y el de respuesta de estudiante. Son obligatorios. Contexto, respuesta esperada, calificación, etiqueta de no respondible e ID de origen son opcionales.
5. Revisa la cobertura de los campos y los ejemplos para confirmar que las asignaciones son correctas.
6. Elige modelo, dimensión y archivo de clave si no está configurado. Pulsa **Normalizar y calcular**.
7. Revisa los resultados y selecciona los bloques o campos que quieres exportar a JSON, JSONL o CSV. La opción de incluir vectores agrega todos los números del vector y puede aumentar mucho el archivo.

Esta importación genera su configuración y archivos normalizados bajo resultados/importaciones/<id>/, la vista bajo resultados/<nombre>/ y sus vectores en ChromaDB. No agrega automáticamente el archivo a qa_dataset_project/resultados/.

El nombre debe ser nuevo para conservar las vistas existentes. Para quitar una vista, selecciónala y pulsa **Eliminar base**. La aplicación pide confirmación; al confirmar elimina la vista y sus exportaciones, y borra el archivo de importación local si ninguna otra vista lo comparte. Los archivos fuente y resultados normalizados del proyecto normalizador permanecen. ChromaDB conserva los vectores compartidos, que podrán reutilizarse si se procesa otra vez la base.

## Entrenamiento supervisado desde la interfaz

La pestaña **Entrenar modelos** permite escoger cualquier vista normalizada o importada que tenga características listas. Para crear una vista nueva, usa **Preparar datos** y luego selecciónala en el campo **Base para entrenar**.

1. Escoge **Regresión** para predecir `targets.score`, o **Clasificación** para predecir `targets.is_unanswerable`, `targets.answer_text` o valores discretos de `targets.score` como categorías.
2. Elige el algoritmo, las características de entrada y el porcentaje reservado para prueba. Las características pueden ser las siete similitudes coseno, texto convertido mediante TF-IDF, vectores completos recuperados desde ChromaDB o combinaciones de estas opciones. Al usar vectores se selecciona una muestra reproducible de hasta 1.000 registros para controlar la memoria.
3. Activa la optimización solo si deseas comparar combinaciones de hiperparámetros. La búsqueda usa validación cruzada en la partición de entrenamiento, con semilla fija para repetir la separación.
4. Pulsa **Entrenar y evaluar**. Los valores objetivo ausentes se excluyen. Para clasificación, la partición se estratifica para mantener representación de las clases; se informa claramente si faltan ejemplos suficientes.
5. Revisa las métricas y descarga el ZIP del modelo, el JSON de configuración y métricas o el CSV de predicciones para los registros de prueba.

La regresión informa MAE y RMSE (errores, menor es mejor) y R². La clasificación informa accuracy, balanced accuracy, F1 macro, precisión macro y recall macro. Son medidas del conjunto de prueba de esa ejecución y deben interpretarse considerando el tamaño, equilibrio y calidad de las etiquetas.

Los archivos se guardan en `resultados/modelos_entrenados/`. El ZIP contiene un pipeline `joblib` con el preprocesamiento y el estimador, más métricas y predicciones. Instala las dependencias del mismo `requirements.txt` al cargar el modelo y usa joblib solo con artefactos de confianza.

La clasificación de `answer_text` interpreta cada respuesta esperada distinta como una clase. Conviene cuando existen respuestas repetidas y pocas clases. Una tarea con muchas respuestas únicas no es una clasificación útil. Para esos datasets, prepara una etiqueta de categoría apropiada o usa regresión si existe una calificación numérica.

## Objetivos y límites de entrenamiento

La entrada para calificar incluye la respuesta candidata y sus similitudes. La entrada para generación de respuesta usa pregunta y señales contextuales, sin incluir la respuesta candidata. Mantener los objetivos separados de las entradas ayuda a evitar fuga de información durante el entrenamiento.

Cuando la fuente trae una respuesta esperada, se usa como objetivo de generación. Si no trae una, el programa usa el texto de respuesta disponible como objetivo. Ese respaldo no convierte automáticamente una respuesta de estudiante en una respuesta correcta.

Solo las fuentes con notas numéricas producen ejemplos en entrenamiento_calificaciones.jsonl. SQuAD, BoolQ y CMU no tienen una calificación equivalente a una nota de respuesta; sus scores quedan nulos. ROARS conserva sus etiquetas en escalas separadas y Mohler conserva las escalas indicadas por la fuente.

## Demos y archivos históricos

resultados/_archivo_pruebas/ conserva vistas e importaciones anteriores. No se incluye al cargar la lista de bases. ejemplos/vectores/ contiene dos vectores JSON pequeños para inspección; no se usan durante el procesamiento.
