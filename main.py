from utils import extract_text_from_pdf, format_text
import openai
import os
from dotenv import load_dotenv

# Placeholder imports for LangChain, ChromaDB, HuggingFace
# from langchain.embeddings import HuggingFaceEmbeddings
# from chromadb import Client

load_dotenv()

def process_resume(pdf_path, job_description):
    # 1. Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_path)
    formatted_resume = format_text(resume_text)

    # 2. Vectorize and store in ChromaDB (placeholder)
    # embeddings = HuggingFaceEmbeddings()
    # chroma_client = Client()
    # chroma_client.add_document(formatted_resume)

    # 3. Retrieve relevant chunks (placeholder)
    # relevant_chunks = chroma_client.query(job_description)
    relevant_chunks = formatted_resume  # Placeholder

    # 4. Call OpenAI Assistant API (new syntax)
    prompt = (
        f"You are an expert resume writer. Given the following resume and job description, "
        f"tailor the resume to best fit the job description.\n\n"
        f"Resume:\n{relevant_chunks}\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Return only the tailored resume, formatted professionally."
    )
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that tailors resumes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    tailored_resume = response.choices[0].message.content.strip()
    return tailored_resume 