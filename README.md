# Resume Tailoring AI Agent

An AI-powered agent that tailors user resumes based on job descriptions using Retrieval-Augmented Generation (RAG), Streamlit UI, and OpenAI's Assistant API.

## Features
- Upload your resume (PDF)
- Paste a job description
- AI tailors your resume to fit the job
- Uses RAG (LangChain + ChromaDB + HuggingFace Embeddings) for relevant content retrieval
- Powered by OpenAI GPT-4o

## Tech Stack
- Streamlit (UI)
- LangChain
- ChromaDB
- HuggingFace Embeddings
- OpenAI API (GPT-4o)
- PyPDF2 (PDF extraction)

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/sy22478/resume-tailoring-ai-agent.git
   cd resume-tailoring-ai-agent
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up your OpenAI API key:**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your-openai-api-key-here
     ```
4. **Run the app:**
   ```sh
   streamlit run app.py
   ```

## Usage
- Upload your resume (PDF format)
- Paste the job description
- Click "Tailor My Resume"
- View your AI-tailored resume in the app

## Security
- **Never commit your `.env` file or API keys to git.**
- `.env` is included in `.gitignore` by default.

## License
MIT 