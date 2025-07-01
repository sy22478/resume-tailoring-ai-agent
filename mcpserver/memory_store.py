from .schemas import ContextItem
from sentence_transformers import SentenceTransformer
import numpy as np

memory = {}

# Load embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def store_context(items: list[ContextItem]):
    for item in items:
        memory[item.key] = item

def retrieve_context(keys: list[str]):
    return [memory[k] for k in keys if k in memory]

def store_tailored_resume(key, tailored_resume, job_description):
    embedding = embedding_model.encode(job_description)
    memory[key] = {
        'tailored_resume': tailored_resume,
        'job_description': job_description,
        'embedding': embedding
    }

def get_all_tailored_resumes():
    return memory
