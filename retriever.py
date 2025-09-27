# ========= retriever.py

from __future__ import annotations
import re
from typing import List

# Mini separador de oraciones sin NLTK (para evitar descargas)
_CORTA_ORACIONES = re.compile(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9(])')

def split_oraciones(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    return [s.strip() for s in _CORTA_ORACIONES.split(text) if s.strip()]


def _acortar(s: str, max_chars=200):
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars] + "…")

def top_snippets(question: str, docs_scored, max_snippets: int = 4):
    #Máx 4 snippets, cada uno <= 200 chars, 1 por doc para diversidad
    snippets = []
    for d, _ in docs_scored[:5]:
        text = f"{(d.title or '').strip()}. {(d.abstract or '').strip()}"
        sents = split_oraciones(text)
        # prioriza oraciones con números/genes/p-valores (sujeto a modificacion, puse algunos genes que fui viendo pero fue arbitrario)
        sents.sort(key=lambda s: int(bool(re.search(
            r'\b(p<|p=|[0-9]+%|[0-9]+\.[0-9]+|IL-|TNF-|BRCA|UBA|EGFR|MAPK|AKT|NF-?kB)\b', s, re.I
        ))), reverse=True)
        for s in sents[:1]:  # 1 por doc
            snippets.append((d, _acortar(s)))
            if len(snippets) >= max_snippets:
                return snippets
    return snippets