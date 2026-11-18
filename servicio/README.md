# Servicio de búsqueda NFCorpus (TF-IDF y BM25)

    servicio/
      app.py                 # FastAPI (el entrypoint que Vercel detecta)
      requirements.txt
      datos/indice.json.gz   # lo genera el notebook (celda de exportación)

Local:   pip install -r requirements.txt uvicorn && uvicorn app:app --reload
Docs:    http://127.0.0.1:8000/docs
Vercel:  vercel --prod   (o importar el repo de GitHub en vercel.com)
Render:  Build `pip install -r requirements.txt` | Start `uvicorn app:app --host 0.0.0.0 --port $PORT`
