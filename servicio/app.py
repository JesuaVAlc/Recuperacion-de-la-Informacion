import gzip
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from nltk.stem import PorterStemmer
from pydantic import BaseModel, Field

K1, B = 1.5, 0.75
RUTA_INDICE = Path(__file__).resolve().parent / "datos" / "indice.json.gz"

# ---------------------------------------------------------------- carga
if not RUTA_INDICE.exists():
    raise RuntimeError("Falta datos/indice.json.gz: ejecuta la celda de exportación del notebook "
                       "y copia el archivo a la carpeta datos/ antes de desplegar.")
with gzip.open(RUTA_INDICE, "rt", encoding="utf-8") as f:
    _datos = json.load(f)

STOPWORDS = set(_datos["stopwords"])
POSTINGS = _datos["postings"]          
LONGITUDES = _datos["longitudes"]      
DOCS = _datos["docs"]                  
N_DOCS = len(LONGITUDES)
AVGDL = sum(LONGITUDES.values()) / N_DOCS

# ---------------------------------------------------------------- preprocesamiento
_stemmer = PorterStemmer()
_PATRON = re.compile(r"[a-z0-9]+")
_cache = {}


def _stem(t):
    if t not in _cache:
        _cache[t] = _stemmer.stem(t)
    return _cache[t]


def preprocesar(texto):
    return [_stem(t) for t in _PATRON.findall(texto.lower())
            if t not in STOPWORDS and len(t) > 1]


def top_k_scores(scores, k):
    return heapq.nlargest(k, scores.items(), key=lambda x: (x[1], x[0]))


# ---------------------------------------------------------------- TF-IDF + coseno
IDF_TFIDF = {t: math.log(N_DOCS / len(docs)) for t, docs in POSTINGS.items()}
PESOS = {t: {d: (1.0 + math.log(tf)) * IDF_TFIDF[t] for d, tf in docs.items()}
         for t, docs in POSTINGS.items()}
_cuadrados = defaultdict(float)
for _docs in PESOS.values():
    for _d, _w in _docs.items():
        _cuadrados[_d] += _w * _w
NORMAS = {d: math.sqrt(s) for d, s in _cuadrados.items()}
del _cuadrados


def rankear_tfidf(tokens, top_k):
    vq = {}
    for t, tf in Counter(tokens).items():
        idf = IDF_TFIDF.get(t, 0.0)
        if idf > 0:
            vq[t] = (1.0 + math.log(tf)) * idf
    norma_q = math.sqrt(sum(w * w for w in vq.values()))
    if norma_q == 0:
        return []
    acum = defaultdict(float)
    for t, wq in vq.items():
        for d, wd in PESOS[t].items():
            acum[d] += wq * wd
    scores = {d: s / (NORMAS[d] * norma_q) for d, s in acum.items() if NORMAS.get(d, 0) > 0}
    return top_k_scores(scores, top_k)


# ---------------------------------------------------------------- BM25
IDF_BM25 = {t: math.log(1.0 + (N_DOCS - len(docs) + 0.5) / (len(docs) + 0.5))
            for t, docs in POSTINGS.items()}


def rankear_bm25(tokens, top_k):
    acum = defaultdict(float)
    for t in set(tokens):
        if t not in POSTINGS:
            continue
        idf = IDF_BM25[t]
        for d, tf in POSTINGS[t].items():
            norm_long = 1.0 - B + B * LONGITUDES[d] / AVGDL
            acum[d] += idf * tf * (K1 + 1.0) / (tf + K1 * norm_long)
    return top_k_scores(acum, top_k)


MODELOS = {"tfidf": rankear_tfidf, "bm25": rankear_bm25}


def buscar(query, model, top_k):
    ranking = MODELOS[model](preprocesar(query), top_k)
    return {
        "model": model,
        "results": [
            {
                "rank": i,
                "document_id": doc_id,
                "score": round(float(score), 4),
                "title": DOCS[doc_id][0],
                "snippet": DOCS[doc_id][1],
            }
            for i, (doc_id, score) in enumerate(ranking, start=1)
        ],
    }


# ---------------------------------------------------------------- API
app = FastAPI(title="Búsqueda NFCorpus: TF-IDF y BM25", version="1.0")


class Consulta(BaseModel):
    query: str = Field(..., min_length=1)
    model: Literal["bm25", "tfidf"] = "bm25"
    top_k: int = Field(10, ge=1, le=100)


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, exc: RequestValidationError):
    errores = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": errores})


@app.get("/")
def raiz():
    return {"servicio": "NFCorpus IR", "endpoints": ["GET /health", "POST /search", "GET /docs"]}


@app.get("/health")
def health():
    return {"status": "ok", "documents": N_DOCS, "vocabulary": len(POSTINGS),
            "models": sorted(MODELOS)}


@app.post("/search")
def search(c: Consulta):
    if not c.query.strip():
        return JSONResponse(status_code=400, content={"detail": ["query: no puede estar vacía"]})
    return buscar(c.query, c.model, c.top_k)
