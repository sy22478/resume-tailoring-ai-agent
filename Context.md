{
  "name": "Resume Tailoring AI Agent",
  "description": "An AI-powered agent that tailors user resumes based on job descriptions using RAG (Retrieval-Augmented Generation), Streamlit UI, and OpenAI's Assistant API.",
  "rootPath": ".",
  "includeGlob": [
    "**/*.py",
    "**/*.txt",
    "**/*.md"
  ],
  "excludeGlob": [
    "__pycache__/**",
    "*.pyc",
    "*.db",
    "*.sqlite",
    ".git/**"
  ],
  "language": "python",
  "customInstructions": {
    "projectPurpose": "The goal of this project is to build an AI agent that retrieves relevant parts of a user's resume (using RAG) and generates a tailored version optimized for a given job description.",
    "mainComponents": {
      "Streamlit_UI": "Used to create a simple web interface for uploading resumes and pasting job descriptions.",
      "RAG_System": "Uses LangChain + ChromaDB + HuggingFace embeddings to vectorize and retrieve resume content.",
      "OpenAI_Assistant": "Leverages the OpenAI Assistant API to generate a tailored resume based on retrieved content and job description."
    },
    "keyFiles": {
      "app.py": "Main UI built with Streamlit where users upload their resume and input job descriptions.",
      "main.py": "Contains core logic: resume parsing, vectorization, retrieval, and calling the OpenAI Assistant API.",
      "utils.py": "Helper functions like PDF extraction and text formatting.",
      "requirements.txt": "List of dependencies needed for the project."
    },
    "howToRun": "1. Install dependencies via `pip install -r requirements.txt`\n2. Set your OpenAI API key in environment variables\n3. Run the app with `streamlit run app.py`",
    "aiModelUsed": "OpenAI GPT-4o via Assistant API",
    "toolsUsed": ["Streamlit", "LangChain", "ChromaDB", "HuggingFace Embeddings", "OpenAI SDK"]
  }
}