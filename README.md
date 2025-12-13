# BioRAG — Hybrid RAG sobre Europe PMC (bioRxiv)

Este proyecto implementa un pipeline RAG “híbrido” (léxico + semántico) para:
1) buscar papers en Europe PMC (filtrando bioRxiv),
2) rankear candidatos combinando BM25 + embeddings (FAISS),
3) extraer snippets cortos como evidencia,
4) generar una respuesta en inglés con citas [n] usando un modelo local vía Ollama.

---

## Pipeline (end-to-end)

### A. Entrada del usuario
- El usuario escribe una pregunta (en español o inglés) en la UI (Gradio).

### B. Traducción/normalización de la consulta (para Europe PMC)
- Si la pregunta está en español (o es “libre”), se convierte a una query en inglés estilo Lucene (AND/OR/NOT, frases entre comillas, etc.).
- Esto se hace con un modelo local vía Ollama (por defecto: `codellama:7b-instruct`) y un prompt few-shot.

**Objetivo:** producir una query robusta para el buscador de Europe PMC.

### C. Búsqueda en Europe PMC
- Se consulta el endpoint de Europe PMC con la query generada.
- En tu implementación actual se filtra a preprints de bioRxiv.
- Se descargan metadatos (título, abstract, año, DOI, links) y, si existe PMCID, se intenta traer full-text en XML y parsearlo.

**Salida:** lista de documentos “Doc” normalizados.

### D. Index local (BM25 + embeddings) + persistencia
Se construye un índice local para esos documentos:

1) **BM25 (similitud léxica)**
- Se tokenizan los textos (title + abstract + fulltext) y se entrena BM25.

2) **Embeddings + FAISS (similitud contextual)**
- Se embebe cada documento con SentenceTransformers (default: `all-MiniLM-L6-v2`, embeddings normalizados).
- Se guardan/reusan embeddings en `./biorag_store/`:
  - `embeddings.npy` (vectores)
  - `meta.parquet` (metadatos + hash por texto normalizado)
- Con esos embeddings se construye un índice FAISS (`IndexFlatIP`) para búsqueda por producto interno (equivalente a coseno si están normalizados).

**Objetivo:** no recalcular embeddings si el mismo texto ya existía (dedupe por hash).

### E. Ranking híbrido (score final)
Para rankear:
- Score léxico: BM25 sobre “palabras clave” extraídas desde la query Lucene.
- Score contextual: FAISS sobre embedding de la pregunta original.
- Score final: combinación lineal
  - `alpha * score_vector + (1 - alpha) * score_bm25_normalizado`

### F. Extracción de evidencia (snippets)
- Para los top docs, se arma un snippet por documento (máx. N snippets).
- Se cortan oraciones del título+abstract y se priorizan oraciones con patrones típicos (genes, p-values, números, etc.).
- Cada snippet queda corto (ej. <=200 chars) para entrar fácil en el contexto del generador.

### G. Generación (answer + referencias) con citas [n]
- Se construye un prompt con:
  - lista de snippets como contexto, cada línea termina con su cita [n]
  - instrucción: responder en inglés y **solo** con evidencia del contexto, citando cada afirmación con [n]
- Se llama a Ollama con un modelo local (por defecto: `gemma3:1b`).
- Se devuelve:
  - `answer` (con citas inline [n])
  - `refs` (lista formateada de referencias [n] con título, fuente, año, link/DOI)

---

## Estructura del código

- `orquestador.py`:
  - Une todo el pipeline y expone una UI en Gradio (entrada: “Consulta”, salida: query final, respuesta con citas, referencias y top recuperados).
- `rag_index.py`:
  - Traducción de consulta a Lucene, búsqueda en Europe PMC, normalización de resultados.
  - `Index_coincidencias`: BM25 + embeddings + FAISS + score híbrido.
- `vector_store.py`:
  - Persistencia de embeddings y metadatos en `./biorag_store/` (reuso + dedupe).
- `retriever.py`:
  - Split simple en oraciones y selección de snippets “representativos”.
- `generador.py`:
  - Arma el prompt con citas y llama a Ollama de forma robusta (generate/chat).
  - Devuelve respuesta + bloque de referencias.

---

## Cómo correr
1) Instalar dependencias:
   - `pip install -r requirements.txt`
2) Tener Ollama corriendo y con modelos descargados:
   - Query: `codellama:7b-instruct`
   - Answer: `gemma3:1b`
3) Ejecutar:
   - `python orquestador.py`
4) Entrar a la interfaz local en:
👉 http://127.0.0.1:7862


Notas:
- Parámetros útiles: `page_size`, `alpha`, `top_k`, `max_snippets`.
- Persistencia: se crea `./biorag_store/` automáticamente.

## Arquitectura del RAG

![Arquitectura del RAG](img/rag_pipeline.png)

