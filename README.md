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

## 🧩 Complete Architecture

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
         │              └────────┬────────┘             │
         │                       │                      │
         │                ┌──────▼──────┐               │
         │                │  Tailoring  │               │
         │                │   Engine    │               │
         │                └──────┬──────┘               │
         │                       │                      │
         │                ┌──────▼──────┐               │
         │                │ Similarity  │               │
         │                │   Search    │               │
         │                └─────────────┘               │
         │                                              │
         └──────────────────────────────────────────────┘
```

### Data Flow
1. User uploads a resume (PDF) and pastes a job description in Streamlit.
2. Resume text is extracted and chunked; embeddings are created with sentence-transformers.
3. ChromaDB stores and retrieves the most relevant chunks via similarity search.
4. Retrieved content + job description are sent to the OpenAI Assistant (GPT-4o) to generate tailored resume sections.
5. The tailored resume is displayed and available for download.

## 🛠️ Tech Stack
- Streamlit, Python 3.10+
- FastAPI (MCP server)
- OpenAI GPT-4o (Assistants API)
- LangChain, ChromaDB, sentence-transformers
- PyPDF2 or pdfplumber for PDF parsing

## 🚀 Features
- Upload PDF resumes and parse content automatically
- Intelligent chunking and embedding-based retrieval for relevant sections
- AI-generated tailored resume output aligned to the job description
- Option to store and retrieve multiple resumes via MCP endpoints
- Clean, simple UI for end-to-end tailoring workflow

## 📦 Project Structure
```
resume-tailoring-ai-agent/
├── app.py                 # Streamlit UI
├── server/
│   ├── main.py            # FastAPI MCP server
│   ├── models.py          # Data models for resume, job description
│   ├── store.py           # In-memory store and MCP tools
│   └── router.py          # API routes (store, retrieve, tailor)
├── rag/
│   ├── embed.py           # Embeddings and vector store (ChromaDB)
│   ├── retrieve.py        # Similarity search utilities
│   └── pdf.py             # PDF parsing utilities
├── prompts/
│   └── tailoring.md       # System and assistant prompts
├── requirements.txt
└── README.md
```

## 🔧 Setup & Installation
```bash
# Clone repository
git clone https://github.com/sy22478/resume-tailoring-ai-agent.git
cd resume-tailoring-ai-agent

# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here  # Windows: setx OPENAI_API_KEY your_key_here
```

## ▶️ Running
```bash
# Start MCP FastAPI server
uvicorn server.main:app --reload

# Start Streamlit UI (in a new terminal)
streamlit run app.py
```

## 🧠 Prompting Strategy
- System prompt ensures professional tone and role adherence.
- Assistant prompt conditions GPT-4o to reorganize and tailor resume content with quantified impact.
- Uses retrieved chunks to ground outputs and avoid hallucination.

## ✅ Example Output Sections
- Tailored Summary
- Key Skills mapped to JD
- Experience bullets rewritten with metrics
- Projects aligned to role requirements

## 🔒 Privacy & Storage
- In-memory storage by default; no persistent database unless added.
- Users should avoid uploading sensitive PII in public deployments.

## 🤝 Contributing
PRs welcome! Please open an issue to discuss major changes.

## 📄 License
MIT

## 👤 Author
- GitHub: @sy22478
- LinkedIn: https://www.linkedin.com/in/sonu-yadav-a61046245/
