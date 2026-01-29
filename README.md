# 🤖 AI PDF Chatbot with Ollama
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-FF6C37?logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A professional-grade AI chatbot that lets you interact with PDF documents using local LLMs via Ollama. All processing happens on your machine - no data leaves your computer.

## ✨ Features

- **🔒 Privacy First**: All processing is local - no API calls, no data sent to the cloud
- **📄 Multi-Page PDF Support**: Process and chat with multi-page documents
- **💬 Intelligent Q&A**: Ask questions, get answers based only on your document
- **🔍 Semantic Search**: FAISS vector database for accurate information retrieval
- **📊 Processing Analytics**: Real-time metrics and performance tracking
- **💾 Export Conversations**: Save your chat history as JSON
- **🎨 Professional UI**: Modern, responsive interface built with Streamlit

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain)
![Ollama](https://img.shields.io/badge/Ollama-FF6C37?logo=ollama)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-purple)

- **Frontend**: Streamlit
- **AI Framework**: LangChain
- **LLM**: Ollama (Llama3, Mistral, etc.)
- **Vector Database**: FAISS
- **Embeddings**: Sentence Transformers
- **PDF Processing**: PyPDF

## 🚀 Quick Start

### Prerequisites
1. **Python 3.8+**
2. **Ollama** installed ([download here](https://ollama.ai))
3. **Llama3 model** pulled: `ollama pull llama3`

### Installation


 1. Clone the repository
git clone https://github.com/Hani-Reza/ai-document-intelligence-rag.git
cd ai-document-intelligence-rag

 2. Install Python dependencies
pip install -r requirements.txt

 3. Make sure Ollama is running
ollama serve

 4. Configure environment variables
bash 
cp .env.example .env
 Edit .env with your preferred settings
-  Key configurations:
- EMBEDDING_MODEL=all-MiniLM-L6-v2
-  LLM_MODEL=llama3:7b
-  CHUNK_SIZE=1000
-  CHUNK_OVERLAP=200

 5. Run the application
streamlit run app.py

### System Architecture
## Core Components
PDF Document → Text Extraction → Chunking → Embedding Generation → Vector Store → Query Processing → LLM Response

1. Document Processing Layer (PyPDF2/pdfplumber)
- PDF text extraction with metadata preservation
- Intelligent chunking with configurable overlap

2. Embedding Generation (sentence-transformers)
- Local embedding models (all-MiniLM-L6-v2 or similar)
- Batch processing for efficiency
- Dimensionality: 384-768 (configurable)

3. Vector Search Engine (FAISS/ChromaDB)
- Approximate nearest neighbor search
- Cosine similarity for semantic matching
- Configurable top-k retrieval

4. LLM Integration (Ollama/llama.cpp)
- Local LLM inference (Llama 3, Mistral, or Gemma variants)
- Prompt engineering for document-aware responses
- Context window optimization

5. Application Interface (Streamlit)
- Interactive document upload and processing
- Real-time Q&A interface
- Processing metrics and visualization

### Design Decisions and Rationale

Component	      |          Choice	             |                       Rationale
Embedding Model	  |      Sentence Transformers	 |       Lightweight, local inference strong semantic understanding
Vector Store	  |          FAISS	             |       High-performance similarity search, minimal dependencies
LLM Framework	  |          Ollama	             |       Simplified local LLM management, cross-platform support
UI Framework	  |          Streamlit	         |       Rapid prototyping with production capabilities
Chunking Strategy |  Recursive text splitter	 |    Preserves document structure while creating searchable segments


## Skill Demonstrated
- AI/ML Engineering: RAG pipeline design, embedding generation, vector search implementation

- Software Architecture: Modular design, separation of concerns, configurable components

- Local LLM Operations: Model quantization, prompt engineering, inference optimization

- Data Processing: PDF parsing, text normalization, chunking strategies

- Production Considerations: Error handling, logging, performance monitoring

## 📁 Project Structure
```
document-intelligence-system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── setup.py              # Package configuration
├── LICENSE.txt           # MIT License
├── .env                  # Environment configuration
├── .gitignore           # Git exclusion patterns
└── assets/              # Static resources
    └── landing page.png         # Application screenshot
```

### ⚠️ Limitations & Considerations

## Current Constraints
- Document Size: Best performance with documents under 100 pages

- Processing Speed: Initial embedding generation may take 30-60 seconds per document

- Memory Usage: Larger models (13B+ parameters) require significant RAM

- Model Knowledge: Limited to document content only, no general knowledge

- Format Complexity: Complex PDF layouts (multi-column, forms) may reduce accuracy

## Performance Notes
- First-time embedding model download: ~80MB

- Vector store creation: Scales linearly with document size

- Inference speed: 2-10 tokens/second on CPU, faster with GPU acceleration


### 🔮 Future Improvements
## Short-term Enhancements
1. Enhanced PDF Parsing: Support for scanned documents (OCR integration)
2. Multi-document Search: Cross-document querying and synthesis
3. Caching Layer: Persistent vector stores for repeated document use
4. Batch Processing: Queue system for multiple document ingestion

## Architectural Evolution
1. Microservices Design: Separate embedding, vector, and inference services
2. Advanced Retrieval: Hybrid search (keyword + semantic), re-ranking models
3. Evaluation Framework: Automated RAG pipeline quality assessment
4. Docker Deployment: Containerized setup for consistent environments

## Production Features
1. Authentication Layer: User management and document access control
2. Audit Logging: Comprehensive activity tracking
3. Health Monitoring: System metrics and performance dashboards
4. API Endpoints: RESTful interface for integration with other 


### 🤝 Contributing
While this is primarily a demonstration project, suggestions and improvements are welcome. Please ensure any proposed changes:

- Maintain the local-only, no-API architecture
- Include appropriate testing
- Document new functionality
- Consider cross-platform compatibility

### 📄 License
This project is licensed under the MIT License - see the LICENSE.txt file for details.

## 👨‍💻 Author
Hani Reza
AI Engineer & Full-Stack Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:your.email@example.com)

Looking For: AI/ML Engineering roles in UAE/GCC region with focus on government digital transformation.


## 🙏 Acknowledgments
UAE Government for digital transformation inspiration

Streamlit team for the excellent web framework

Open-source community for continuous learning resources

<div align="center">
Built with ❤️ for the AI Engineering Community

Professional • Production-Ready • Portfolio Project



</div> ```
