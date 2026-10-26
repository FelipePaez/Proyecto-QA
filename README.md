# Proyecto QA

Aplicación local para importar, normalizar y explorar conjuntos de datos de preguntas y respuestas. El proyecto permite conservar una fuente original, convertirla a un esquema común y, opcionalmente, calcular embeddings y similitudes para análisis o preparación de datos de entrenamiento.

> **Importante:** el repositorio contiene código y ejemplos de uso; no distribuye los datasets originales ni resultados generados. Cada dataset conserva sus propios términos de acceso, licencia y redistribución. El programa no entrena por sí mismo un modelo ni infiere respuestas: prepara y explora datos y características.

## Qué incluye

- **Normalización**: lectores para SQuAD 2.0, CMU QA, Mohler y ROARS, además de configuración para incorporar CSV, TSV, JSON y JSONL propios.
- **IDs trazables**: IDs de contexto, pregunta, respuesta, calificación y párrafo relacionados entre sí y con su procedencia.
- **Explorador web local**: busca en fuentes originales, exportaciones normalizadas y vistas con embeddings.
- **Características semánticas**: embeddings de pregunta, respuesta y párrafos; centroide del contexto; similitudes coseno.
- **Preparación de datos de entrenamiento**: importación, asignación de campos y exportación en CSV, JSON o JSONL.
- **Almacenamiento vectorial local** con ChromaDB para reutilizar los embeddings calculados.

## Requisitos

- Windows 10/11, macOS o Linux.
- Python 3.10 o posterior; se recomienda Python 3.11.
- Git para clonar el repositorio.
- Conexión a Internet para descargar una fuente que falte y para solicitar embeddings a OpenAI.
- Una clave de API de OpenAI **solo** si se van a calcular embeddings. La normalización y exploración básica no requieren una clave.

## Descargar y abrir el proyecto

Clona el repositorio desde una terminal:

```bash
git clone https://github.com/FelipePaez/Proyecto-QA.git
cd Proyecto-QA
```

El nombre técnico usa un guion porque GitHub no admite espacios en el nombre del repositorio. En VS Code abre el workspace incluido:

```bash
code Proyecto_QA.code-workspace
```

También puedes abrir la carpeta `Proyecto-QA` desde **Archivo → Abrir carpeta**.

## Instalar dependencias

Desde la raíz del repositorio, crea el entorno virtual e instala las dependencias del explorador:

### Windows PowerShell

```powershell
py -3.11 -m venv qa_embedding_features\.venv
.\qa_embedding_features\.venv\Scripts\python.exe -m pip install --upgrade pip
.\qa_embedding_features\.venv\Scripts\python.exe -m pip install -r qa_embedding_features\requirements.txt
```

### macOS o Linux

```bash
python3 -m venv qa_embedding_features/.venv
qa_embedding_features/.venv/bin/python -m pip install --upgrade pip
qa_embedding_features/.venv/bin/python -m pip install -r qa_embedding_features/requirements.txt
```

El normalizador usa la biblioteca estándar de Python. ChromaDB es la dependencia externa de la aplicación de embeddings y del explorador.

## Primera ejecución: descargar y normalizar SQuAD 2.0 dev

El repositorio no trae los archivos de datos. Descarga y procesa la partición `dev` desde la fuente oficial:

### Windows PowerShell

```powershell
.\qa_embedding_features\.venv\Scripts\python.exe qa_dataset_project\normalizar_datasets.py --dataset squad2_dev --download
```

### macOS o Linux

```bash
qa_embedding_features/.venv/bin/python qa_dataset_project/normalizar_datasets.py --dataset squad2_dev --download
```

El original se guarda localmente en `qa_dataset_project/originales/`; las salidas estandarizadas se crean en `qa_dataset_project/resultados/`. Estas carpetas están excluidas de Git. Para normalizar otra base integrada, sustituye `squad2_dev` por el nombre admitido, por ejemplo `squad2_train`, `cmu_qa_v1.2`, `mohler_v1.0`, `mohler_v2.0`, `roars_2class` o `roars_3class`. Para descargar varias fuentes, revisa sus condiciones de uso y ejecuta el normalizador con la opción `--download`.

También puedes cargar un archivo propio desde la pestaña **Añadir / normalizar bases**. Allí puedes guardarlo como original sin modificar o mapear sus campos y crear una exportación normalizada.

## Iniciar la aplicación

### Windows PowerShell

```powershell
.\qa_embedding_features\.venv\Scripts\python.exe qa_embedding_features\app_interactiva.py --port 8768
```

### macOS o Linux

```bash
qa_embedding_features/.venv/bin/python qa_embedding_features/app_interactiva.py --port 8768
```

Abre [http://127.0.0.1:8768/](http://127.0.0.1:8768/) en tu navegador. Mantén abierta la terminal mientras uses la aplicación. Para detenerla, pulsa `Ctrl+C`.

La interfaz ofrece tres secciones:

1. **Explorar bases**: búsqueda por texto, campos, IDs y similitudes calculadas; paginación, detalle y descarga.
2. **Preparar datos de entrenamiento**: carga un archivo, asigna campos, calcula características y exporta las columnas elegidas.
3. **Añadir / normalizar bases**: guarda el original para explorarlo sin cambios, o normaliza el archivo y crea IDs relacionados.

## Calcular embeddings (opcional)

Los embeddings usan la API de OpenAI y pueden generar cargos. Se recomienda empezar con pocas filas y confirmar los resultados antes de procesar un corpus completo. Guarda la clave fuera del repositorio, por ejemplo en un archivo local `KEY.txt` que nunca subas a GitHub.

### Windows PowerShell

```powershell
.\qa_embedding_features\.venv\Scripts\python.exe qa_embedding_features\preparar_vistas.py --dataset squad2_dev --limit 3 --api-key-file "C:\ruta\KEY.txt"
```

Cuando hayas revisado la prueba, quita `--limit 3` para generar la vista completa:

```powershell
.\qa_embedding_features\.venv\Scripts\python.exe qa_embedding_features\preparar_vistas.py --dataset squad2_dev --api-key-file "C:\ruta\KEY.txt"
```

La clave también se puede pasar mediante la variable de entorno `OPENAI_API_KEY`. No pegues claves en el código, README, configuraciones que vayan al repositorio, capturas ni commits. Si una clave se publica por accidente, revócala desde el panel del proveedor y crea otra.

## Dónde se guardan los datos

| Carpeta | Contenido | ¿Se publica en Git? |
|---|---|---|
| `qa_dataset_project/originales/` | Fuentes originales descargadas o agregadas | No |
| `qa_dataset_project/resultados/` | CSV, JSONL y manifiestos normalizados | No |
| `qa_embedding_features/resultados/` | Vistas, vectores, ChromaDB y exportaciones | No |
| `qa_dataset_project/` | Normalizador, configuraciones de ejemplo y documentación | Sí |
| `qa_embedding_features/` | Aplicación, scripts, dependencias y documentación | Sí |

Para reconstruir resultados, conserva los archivos originales en tu copia local y vuelve a ejecutar los comandos. Los datos normalizados se pueden regenerar.

## Estructura del repositorio

```text
Proyecto-QA/
├── Proyecto_QA.code-workspace
├── README.md
├── qa_dataset_project/
│   ├── normalizar_datasets.py
│   ├── buscar.py
│   ├── config_*.json
│   ├── originales/              # local, no versionado
│   └── resultados/              # local, no versionado
└── qa_embedding_features/
    ├── app_interactiva.py
    ├── preparar_vistas.py
    ├── vector_store.py
    ├── requirements.txt
    └── resultados/               # local, no versionado
```

## Ayuda y documentación

- [Normalizador: formatos, IDs, configuración y búsqueda](qa_dataset_project/README.md)
- [Embeddings, similitudes y explorador](qa_embedding_features/README.md)
- [Ejemplos de vectores](qa_embedding_features/ejemplos/README.md)

## Uso de datasets y atribución

El código no concede derechos sobre los datos de terceros. Descarga cada fuente desde su sitio oficial y revisa sus condiciones antes de usarla, redistribuirla o publicar derivados:

- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- [CMU Question-Answer Dataset](https://www.cs.cmu.edu/~ark/QA-data/)
- [Mohler Short Answer Grading](https://web.eecs.umich.edu/~mihalcea/downloads.html)
- [ROARS](https://github.com/owenhenkel/ROARS-dataset) — su fuente indica condiciones CC BY-NC 4.0; confirma los términos actuales antes de uso o redistribución.

No se incluye una licencia de software en este repositorio todavía. Si quieres permitir que otras personas reutilicen o modifiquen el código, el propietario debe elegir y añadir una licencia explícita.

