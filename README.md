# biorag

## Motivation
Large Language Models (LLMs) are powerful tools for biomedical research, but they often **hallucinate** when facing very specific domain questions. In biomedical sciences, this is particularly critical: wrong gene-disease associations or invented citations can mislead research and waste valuable time.  
The challenge is how to **ground LLM answers in reliable biomedical evidence** and avoid hallucinations while keeping the flexibility of natural language queries.

## What is biorag?
**biorag** is a lightweight Retrieval-Augmented Generation (RAG) pipeline designed for biomedical question answering.  
It combines **BM25 lexical search** and **vector-based embeddings (FAISS + SentenceTransformers)** with local LLMs through **Ollama**.  
The goal is to ensure that model answers are supported by retrieved documents, reducing hallucination risks.

## File overview
- **`orquestador.py`** → main entrypoint, runs the full pipeline.  
- **`rag_index.py`** → builds and manages BM25 + vector indices.  
- **`retriever.py`** → retrieves top relevant snippets from the indices.  
- **`vector_store.py`** → manages embeddings and FAISS storage.  
- **`generador.py`** → generates the final answer, formatting with citations.  
- **`README.md`** → project documentation.  

## How to use biorag

1. Open a terminal in the project directory.  
2. Run:  
   python orquestador.py
3. Wait for the pipeline to initialize.
4. Open the local interface at:
👉 http://127.0.0.1:7862

# biorag

## Motivación
Los LLMs son herramientas poderosas para la investigación biomédica, pero suelen alucinar cuando se enfrentan a preguntas muy específicas del dominio. En biomedicina esto es crítico: asociaciones falsas gen-enfermedad o citas inventadas pueden desviar la investigación y hacer perder tiempo.
El desafío es cómo anclar las respuestas de los LLM en evidencia biomédica confiable, evitando alucinaciones y manteniendo la flexibilidad de consultas en lenguaje natural.

## ¿Qué es biorag?
biorag es un pipeline sencillo de RAG (Retrieval-Augmented Generation) para preguntas biomédicas.
Combina búsqueda léxica BM25 y vectores de embeddings (FAISS + SentenceTransformers) con LLMs locales a través de Ollama.
El objetivo es que las respuestas estén siempre respaldadas por documentos recuperados, reduciendo el riesgo de alucinaciones.

## Archivos
- orquestador.py → punto de entrada principal, ejecuta todo el flujo.

- rag_index.py → construcción y gestión de índices BM25 + vectores.

- retriever.py → recuperación de snippets más relevantes.

- vector_store.py → gestión de embeddings y almacenamiento en FAISS.

- generador.py → genera la respuesta final con citas.

- README.md → documentación del proyecto.

## Cómo usar

1. Abrir una terminal en el directorio del proyecto.
2. Ejecutar:
   python orquestador.py
3. Esperar que se inicialice el pipeline.
4. Entrar a la interfaz local en:
👉 http://127.0.0.1:7862


