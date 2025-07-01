# Resume Tailoring AI Agent

An AI-powered agent that tailors user resumes based on job descriptions using Retrieval-Augmented Generation (RAG), Streamlit UI, and OpenAI's Assistant API.

## Features
- **Streamlit UI**: Simple web interface for uploading resumes and pasting job descriptions.
- **RAG System**: Uses LangChain, ChromaDB, and HuggingFace embeddings to vectorize and retrieve resume content.
- **OpenAI Assistant**: Leverages the OpenAI Assistant API to generate a tailored resume based on retrieved content and job description.
- **FastAPI Backend**: Provides endpoints for storing, retrieving, and tailoring resumes, as well as similarity search.

## Project Structure
```
Resume_builder/
├── app.py                # Streamlit UI
├── main.py               # Core logic: resume parsing, vectorization, retrieval, OpenAI API
├── utils.py              # PDF extraction and text formatting
├── requirements.txt      # Project dependencies
├── mcpserver/            # FastAPI backend
│   ├── main.py           # API endpoints
│   ├── memory_store.py   # In-memory storage for resumes
│   ├── schemas.py        # Pydantic schemas
│   ├── requirements.txt  # Backend dependencies
│   └── ...
└── ...
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Resume_builder
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   For backend-specific dependencies:
   ```bash
   pip install -r mcpserver/requirements.txt
   ```
3. **Set your OpenAI API key:**
   - Create a `.env` file in the root directory:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```

## Running the Application
### Streamlit UI
```bash
streamlit run app.py
```
- Upload your resume (PDF) and paste the job description to get a tailored resume.

### FastAPI Backend
```bash
uvicorn mcpserver.main:app --reload
```
- Access API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints
- `POST /mcp/store` — Store context items
- `POST /mcp/retrieve` — Retrieve context items by keys
- `POST /mcp/tailor` — Tailor a resume to a job description (PDF upload)
- `POST /mcp/retrieve_similar` — Retrieve most similar tailored resumes for a job description

## How It Works
1. **PDF Extraction**: Extracts text from uploaded PDF resumes.
2. **Vectorization & Retrieval**: (Planned) Uses embeddings and vector DB to retrieve relevant resume sections.
3. **AI Tailoring**: Sends relevant content and job description to OpenAI's GPT-4o for resume rewriting.

## Requirements
- Python 3.8+
- See `requirements.txt` and `mcpserver/requirements.txt` for dependencies

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace](https://huggingface.co/) 