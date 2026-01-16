![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-FF6C37?logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

# 🤖 AI PDF Chatbot with Ollama
![Demo](assets/demo.png)

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

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-pdf-chatbot.git
cd ai-pdf-chatbot

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Make sure Ollama is running
ollama serve

# 4. Run the application
streamlit run app.py