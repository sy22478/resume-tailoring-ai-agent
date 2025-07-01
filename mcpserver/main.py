from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from .schemas import StoreRequest, RetrieveRequest, TailorRequest
from .memory_store import store_context, retrieve_context, store_tailored_resume, get_all_tailored_resumes
import sys
import tempfile
import os
import numpy as np
from sentence_transformers import SentenceTransformer
sys.path.append('..')
from main import process_resume

app = FastAPI()

@app.post("/mcp/store")
def store(req: StoreRequest):
    store_context(req.context_items)
    return {"status": "success"}

@app.post("/mcp/retrieve")
def retrieve(req: RetrieveRequest):
    items = retrieve_context(req.keys)
    return {"results": [item.dict() for item in items]}

@app.post("/mcp/tailor")
def tailor(
    pdf: UploadFile = File(...),
    job_description: str = Form(...),
    key: str = Form(...)
):
    try:
        # Save uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.file.read())
            tmp_path = tmp_file.name
        tailored_resume = process_resume(tmp_path, job_description)
        os.remove(tmp_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"PDF file not found: {pdf.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Store tailored resume under the provided key
    store_tailored_resume(key, tailored_resume, job_description)
    return {"key": key, "tailored_resume": tailored_resume}

@app.post("/mcp/retrieve_similar")
def retrieve_similar(job_description: str = Form(...), top_k: int = 1):
    # Compute embedding for the new job description
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = embedding_model.encode(job_description)
    all_resumes = get_all_tailored_resumes()
    if not all_resumes:
        raise HTTPException(status_code=404, detail="No tailored resumes stored.")
    # Compute cosine similarity
    results = []
    for key, data in all_resumes.items():
        stored_emb = data['embedding']
        sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
        results.append((sim, key, data['tailored_resume']))
    results.sort(reverse=True)
    top_results = results[:top_k]
    return {"results": [{"key": key, "similarity": float(sim), "tailored_resume": resume} for sim, key, resume in top_results]}
