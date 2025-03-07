# HireSift 🔍 

<div align="center">


[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.19-00873E)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-Powered Recruitment Assistant for Technical Hiring**

</div>

## 🌟 Features

- **🔎 Smart Resume Search**: Find candidates matching specific skills or job descriptions
- **📊 Intelligent Matching**: AI-powered ranking of candidate resumes against job requirements
- **📅 Automated Scheduling**: Schedule interviews with candidates matching your criteria
- **📧 Email Drafting**: Generate professional interview invitation emails
- **🧠 Multi-Model Support**: Switch between different LLM models for various tasks
- **🔄 RAG & RAG Fusion**: Advanced retrieval techniques for more accurate candidate matching
- **🧩 Category-Based Search**: Find candidates by roles (Frontend, Backend, Full Stack, GenAI, etc.)

## 🛠️ Technologies

- **Frontend**: Streamlit
- **AI/ML**: LangChain, Google Generative AI, Groq, VoyageAI (embeddings)
- **Vector Database**: FAISS
- **Document Processing**: PyPDF, Pandas
- **Natural Language Processing**: Transformers, Sentence-Transformers

## 📋 Prerequisites

- Python 3.9+
- API keys:
  - Groq API Key (for LLM access)
  - VoyageAI API Key (for embeddings)
  - Google Generative AI API Key (for LLM access)
 
## Setting Up Environment Variables

Create a `.env` file in the root directory of the project with the following configuration:

### Configuration Details:

- **API Keys**:
  - `GROQ_API_KEY`: Your Groq API key for LLM access
  - `VOYAGE_API_KEY`: Your VoyageAI API key for embeddings

- **Storage Paths**:
  - `FAISS_PATH`: Directory where vector indices will be stored
  - `GENERATED_DATA_PATH`: Directory for processed CSV data
  - `DEFAULT_DATA_PATH`: Path to default data (can be empty)

- **Model Settings**:
  - `EMBEDDING_MODEL`: Embedding model to use for document vectorization

### Important Notes:

- Ensure your API keys are kept confidential and not committed to public repositories
- Create all necessary directories before running the application
- Relative paths can be used, but absolute paths are recommended

## 🚀 How to run?

```bash
git clone https://github.com/abhilekhborah/hiresift.git
cd hiresift
streamlit run main.py
