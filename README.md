# 📄 Resume Tailoring AI Agent

An AI-powered agent that tailors user resumes based on job descriptions using Retrieval-Augmented Generation (RAG), Streamlit UI, and OpenAI's Assistant API (GPT-4o). This system combines PDF processing, semantic similarity search, and AI-driven content generation to create customized resumes optimized for specific job applications.

---

## 📌 Project Overview

An AI-powered resume optimization system using OpenAI GPT-4o for resume tailoring with PDF processing capabilities. Features a Streamlit UI frontend and FastAPI MCP server backend with in-memory storage and similarity search using sentence transformers.

### Key Capabilities
- **Streamlit UI**: Simple web interface for uploading resumes (PDF) and pasting job descriptions
- **RAG System**: Uses LangChain, ChromaDB, and HuggingFace embeddings to vectorize and retrieve resume content
- **OpenAI Assistant**: Leverages the OpenAI Assistant API (GPT-4o) to generate tailored resumes based on retrieved content and job descriptions
- **FastAPI Backend**: Provides RESTful endpoints for storing, retrieving, and tailoring resumes, as well as similarity search
- **MCP Protocol**: Model Context Protocol for context management and resume storage

## Complete Architecture

### System Architecture (Actual Implementation)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  FastAPI MCP     │    │   In-Memory     │
│   (app.py)      │◄──►│  Server          │◄──►│   Storage       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  OpenAI GPT-4o  │             │
         │              │  Chat API       │             │
         │              └─────────────────┘             │
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  PyPDF2         │             │
         │              │  Text Extract   │             │
         │              └─────────────────┘             │
         │                                               │
         └──────────────────────┬──────────────────────┘
                                │
                    ┌──────────▼──────────┐
                    │  SentenceTransformer│
                    │  all-MiniLM-L6-v2   │
                    └─────────────────────┘
```

**Note:** ChromaDB and LangChain are listed as dependencies but not actively implemented. The system uses placeholder comments for future RAG implementation.

## Complete Tech Stack

### Core Technologies (Verified Implementation)
- **UI Framework:** Streamlit with basic file upload and text area components
- **Backend API:** FastAPI with Model Context Protocol (FastMCP)
- **Document Processing:** PyPDF2 for PDF text extraction
- **Storage:** In-memory Python dictionary (memory_store.py)

### AI/ML Stack (Actual Implementation)
- **Language Model:** OpenAI GPT-4o via chat completions API
- **Embeddings:** SentenceTransformer 'all-MiniLM-L6-v2' for similarity search
- **PDF Processing:** PyPDF2.PdfReader for text extraction
- **Similarity Computation:** Cosine similarity using numpy operations

### Data Management (Current Implementation)
- **Storage:** Simple in-memory dictionary with key-value pairs
- **Embedding Generation:** SentenceTransformer for job description embeddings
- **Context Management:** Basic MCP server with store/retrieve endpoints
- **File Handling:** Temporary file processing for PDF uploads

### Dependencies Status
- **Active:** streamlit, openai, PyPDF2, sentence-transformers, fastapi, fastmcp
- **Installed but Unused:** langchain, chromadb, weaviate-client (placeholders for future implementation)

## Skills Developed

### AI-Powered Document Processing (Actual Implementation)
- **PDF Processing:** PyPDF2 integration for text extraction from uploaded resumes
- **AI Integration:** OpenAI GPT-4o chat completions for resume tailoring
- **Similarity Search:** SentenceTransformer embeddings with cosine similarity matching
- **Content Generation:** Prompt-based resume optimization with professional formatting

### Code Implementation Examples

**PDF Text Extraction:**
```python
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
```

**OpenAI GPT-4o Integration:**
```python
def process_resume(pdf_path, job_description):
    resume_text = extract_text_from_pdf(pdf_path)
    formatted_resume = format_text(resume_text)

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
    return response.choices[0].message.content.strip()
```

**Similarity Search Implementation:**
```python
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_similar(job_description: str, top_k: int = 1):
    query_emb = embedding_model.encode(job_description)
    all_resumes = get_all_tailored_resumes()
    results = []
    for key, data in all_resumes.items():
        stored_emb = data['embedding']
        sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
        results.append((sim, key, data['tailored_resume']))
    results.sort(reverse=True)
    return results[:top_k]
```

### Full-Stack Development
- **Frontend Development:** Streamlit component design, file upload handling
- **API Development:** RESTful endpoints, file handling, async processing
- **Data Flow Design:** Request/response handling, state management
- **Error Handling:** Graceful failure handling, user feedback systems

### NLP & AI Integration
- **OpenAI API Integration:** Assistant API usage, prompt optimization
- **Embedding Techniques:** Semantic similarity, vector operations
- **Text Processing:** Content analysis, format standardization
- **Context Understanding:** Job description parsing, skill extraction

## Technical Achievements (Verified Implementation)

**Actual Features Implemented:**
- **Dual Interface:** Streamlit UI (app.py) + FastAPI MCP server (mcpserver/main.py)
- **PDF Processing:** Complete PyPDF2 text extraction pipeline
- **AI Integration:** OpenAI GPT-4o chat completions with custom prompting
- **Similarity Search:** SentenceTransformer embeddings with cosine similarity
- **MCP Protocol:** Basic context storage and retrieval endpoints
- **File Handling:** Temporary PDF file processing with cleanup

**API Endpoints Implemented:**
- `POST /mcp/store` - Store context items
- `POST /mcp/retrieve` - Retrieve context by keys
- `POST /mcp/tailor` - Upload PDF and tailor resume
- `POST /mcp/retrieve_similar` - Find similar tailored resumes

**Architecture Limitations:**
- **Storage:** In-memory only (data lost on restart)
- **RAG:** Not fully implemented (placeholder comments exist)
- **Vector Database:** ChromaDB/Weaviate listed as dependencies but unused
- **Chunking:** No document chunking - processes entire resume as single text

---

## 🗂️ Project Structure

```
resume-tailoring-ai-agent/
├── app.py                    # Streamlit UI application
├── main.py                   # Core logic: resume parsing, vectorization, retrieval, OpenAI API
├── utils.py                  # PDF extraction (PyPDF2) and text formatting utilities
├── requirements.txt          # Project dependencies (Streamlit, OpenAI, PyPDF2)
├── .env                      # Environment variables (OPENAI_API_KEY)
├── mcpserver/                # FastAPI backend for Model Context Protocol
│   ├── main.py               # API endpoints (store, retrieve, tailor, retrieve_similar)
│   ├── memory_store.py       # In-memory storage for resumes and embeddings
│   ├── schemas.py            # Pydantic schemas for request/response validation
│   ├── requirements.txt      # Backend dependencies (FastAPI, FastMCP)
│   └── ...
└── README.md                 # Project documentation
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sy22478/resume-tailoring-ai-agent.git
cd resume-tailoring-ai-agent
```

### 2. Install Dependencies

**Frontend (Streamlit UI):**
```bash
pip install -r requirements.txt
```

**Backend (FastAPI MCP Server):**
```bash
pip install -r mcpserver/requirements.txt
```

### 3. Set Your OpenAI API Key

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
```

Or export as environment variable:
```bash
export OPENAI_API_KEY=your_openai_api_key
```

---

## 🎯 Running the Application

### Streamlit UI (Frontend)
```bash
streamlit run app.py
```

**Usage:**
1. Upload your resume (PDF format)
2. Paste the job description in the text area
3. Click "Tailor Resume" to generate a customized resume
4. View and download the tailored resume

### FastAPI Backend (MCP Server)
```bash
uvicorn mcpserver.main:app --reload
```

The API server will be available at `http://localhost:8000`

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🔌 API Endpoints

### Context Management
- **`POST /mcp/store`** — Store context items in memory
  - Request: `{"key": "string", "value": "string"}`
  - Response: `{"status": "stored"}`

- **`POST /mcp/retrieve`** — Retrieve context items by keys
  - Request: `{"keys": ["key1", "key2"]}`
  - Response: `{"key1": "value1", "key2": "value2"}`

### Resume Tailoring
- **`POST /mcp/tailor`** — Tailor a resume to a job description (PDF upload)
  - Request: `multipart/form-data` with `file` (PDF) and `job_description` (text)
  - Response: `{"tailored_resume": "string"}`

- **`POST /mcp/retrieve_similar`** — Retrieve most similar tailored resumes for a job description
  - Request: `{"job_description": "string", "top_k": 1}`
  - Response: `[{"similarity": 0.95, "key": "uuid", "tailored_resume": "string"}]`

---

## ⚙️ How It Works

### 1. PDF Extraction
Extracts text from uploaded PDF resumes using PyPDF2:

```python
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
```

### 2. Vectorization & Retrieval
Uses SentenceTransformer embeddings for semantic similarity search:

```python
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_similar(job_description: str, top_k: int = 1):
    query_emb = embedding_model.encode(job_description)
    all_resumes = get_all_tailored_resumes()
    results = []
    for key, data in all_resumes.items():
        stored_emb = data['embedding']
        sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
        results.append((sim, key, data['tailored_resume']))
    results.sort(reverse=True)
    return results[:top_k]
```

### 3. AI Tailoring
Sends relevant content and job description to OpenAI's GPT-4o for resume rewriting:

```python
def process_resume(pdf_path, job_description):
    resume_text = extract_text_from_pdf(pdf_path)
    formatted_resume = format_text(resume_text)

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
    return response.choices[0].message.content.strip()
```

---

## 📦 Requirements

### Frontend (requirements.txt)
```
streamlit
openai
PyPDF2
sentence-transformers
langchain
chromadb
weaviate-client
python-dotenv
numpy
```

### Backend (mcpserver/requirements.txt)
```
fastapi
fastmcp
uvicorn
sentence-transformers
pydantic
numpy
```

**Python Version:** 3.8+

---

## 🔧 Features

### ✅ Implemented Features
- **PDF Processing**: Complete PyPDF2 text extraction pipeline
- **AI Integration**: OpenAI GPT-4o chat completions with custom prompting
- **Similarity Search**: SentenceTransformer embeddings with cosine similarity
- **Dual Interface**: Streamlit UI + FastAPI MCP server
- **File Handling**: Temporary PDF file processing with cleanup
- **MCP Protocol**: Basic context storage and retrieval endpoints

### ⚠️ Architecture Limitations
- **Storage**: In-memory only (data lost on restart) - no persistent database
- **RAG**: Not fully implemented (placeholder comments exist for ChromaDB/Weaviate)
- **Vector Database**: ChromaDB/Weaviate listed as dependencies but unused
- **Chunking**: No document chunking - processes entire resume as single text
- **Scalability**: Single-server deployment with no load balancing

---

## 💡 Future Enhancements

- **Persistent Storage**: Implement PostgreSQL or MongoDB for resume storage
- **Full RAG Implementation**: Integrate ChromaDB or Weaviate for vector storage
- **Document Chunking**: Split resumes into semantic chunks for better retrieval
- **User Authentication**: Add JWT-based authentication for multi-user support
- **Resume Templates**: Provide multiple ATS-friendly resume templates
- **Batch Processing**: Support bulk resume tailoring for multiple job descriptions
- **Analytics Dashboard**: Track tailoring history, success rates, and performance metrics
- **Export Formats**: Support PDF, DOCX, and LaTeX export formats
- **SHAP Explanations**: Add explainability for AI-generated content

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

## 📧 Contact

For questions, improvements, or collaboration:

- **Email**: sonu.yadav19997@gmail.com
- **LinkedIn**: [Sonu Yadav](https://www.linkedin.com/in/sonu-yadav-a61046245/)
- **GitHub**: [@sy22478](https://github.com/sy22478)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) - Web UI framework
- [OpenAI](https://openai.com/) - GPT-4o API for resume tailoring
- [LangChain](https://python.langchain.com/) - RAG framework (planned integration)
- [ChromaDB](https://www.trychroma.com/) - Vector database (planned integration)
- [HuggingFace](https://huggingface.co/) - SentenceTransformers embeddings

---

*This project demonstrates AI-powered document processing, semantic similarity search, and automated content generation for resume optimization using OpenAI GPT-4o and modern NLP techniques.*
