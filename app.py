"""
🚀 AI PDF Chatbot - GitHub Portfolio Edition
Professional, feature-rich, and impressive for recruiters
"""

import streamlit as st
import tempfile
import os
import json
import time
from datetime import datetime

# Core LangChain imports (keep it simple, working)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# ============================================
# 1. ENHANCED UI WITH CUSTOM CSS
# ============================================
st.set_page_config(
    page_title="PDF AI Assistant | GitHub Portfolio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Makes it look like a SaaS product
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Elegant header */
    .portfolio-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .portfolio-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .portfolio-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.2);
    }
    
    .ai-message {
        background: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #e9ecef;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Tech badge */
    .tech-badge {
        display: inline-block;
        background: #e9ecef;
        color: #495057;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-family: 'Monaco', 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. SESSION STATE MANAGEMENT
# ============================================
def initialize_session_state():
    """Professional state management - shows you know what you're doing"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'processing_metrics' not in st.session_state:
        st.session_state.processing_metrics = {}
    
    if 'file_info' not in st.session_state:
        st.session_state.file_info = {}
    
    # NEW: Track if we should show processing
    if 'show_processing' not in st.session_state:
        st.session_state.show_processing = False

# ============================================
# 3. IMPRESSIVE SIDEBAR WITH TECH STACK
# ============================================
def render_sidebar():
    """Sidebar that showcases your tech skills"""
    with st.sidebar:
        # Portfolio badge
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem; border-radius: 10px;'>
                <h3 style='margin: 0;'>🚀 Portfolio Project</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>AI PDF Chatbot</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tech stack showcase
        st.markdown("### 🛠️ Tech Stack")
        cols = st.columns(3)
        tech_stack = ["Streamlit", "LangChain", "Ollama", "FAISS", "PyPDF", "HuggingFace"]
        
        for i, tech in enumerate(tech_stack):
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"<div class='tech-badge'>{tech}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model configuration
        st.markdown("### ⚙️ Configuration")
        
        model_options = {
            "llama3": "Meta's Latest (Recommended)",
            "mistral": "Fast & Efficient",
            "llama2": "Proven & Stable",
            "neural-chat": "Conversation Optimized"
        }
        
        selected_model = st.selectbox(
            "AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: f"{x} - {model_options[x]}"
        )
        
        # Advanced settings in expander
        with st.expander("🔧 Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 
                                  help="Larger chunks = more context, but slower processing")
            chunk_overlap = st.slider("Chunk Overlap", 50, 300, 100,
                                     help="Overlap between chunks for better context continuity")
            search_k = st.slider("Search Results", 1, 5, 3,
                                help="Number of document chunks to consider for each answer")
        
        st.markdown("---")
        
        # Conversation stats
        st.markdown("### 📊 Statistics")
        if st.session_state.messages:
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            ai_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Messages", user_msgs)
            with col2:
                st.metric("AI Responses", ai_msgs)
        
        # Export conversation
        if st.session_state.messages:
            if st.button("💾 Export Conversation", use_container_width=True):
                export_conversation()
        
        st.markdown("---")
        
        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                initialize_session_state()
                st.rerun()
        
        # Footer with GitHub link
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0; color: #666;'>
            <p style='margin: 0;'>⭐ Star on GitHub</p>
            <p style='margin: 0; font-size: 0.9rem;'>github.com/yourusername/ai-pdf-chatbot</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# 4. PROFESSIONAL PDF PROCESSING WITH METRICS
# ============================================
def process_pdf_with_metrics(uploaded_file, chunk_size=1000, chunk_overlap=100):
    """Process PDF and collect metrics - shows engineering skills"""
    
    metrics = {
        'start_time': time.time(),
        'file_size': uploaded_file.size,
        'file_name': uploaded_file.name
    }
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a container for live metrics
    metrics_container = st.container()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Step 1: Load PDF
        status_text.text("📖 Loading PDF document...")
        progress_bar.progress(10)
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        metrics['page_count'] = len(documents)
        
        # Update metrics display
        with metrics_container:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Pages", metrics['page_count'])
        
        if not documents:
            st.error("No readable content found in PDF")
            return False
        
        # Step 2: Split text
        status_text.text("✂️ Splitting document into chunks...")
        progress_bar.progress(30)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        metrics['chunk_count'] = len(chunks)
        
        with metrics_container:
            with cols[1]:
                st.metric("Chunks", metrics['chunk_count'])
        
        # Step 3: Create embeddings
        status_text.text("🔤 Creating embeddings (this may take a moment)...")
        progress_bar.progress(60)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Step 4: Create vector store
        status_text.text("🗄️ Building search index...")
        progress_bar.progress(80)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Step 5: Initialize LLM
        llm = OllamaLLM(model="llama3", temperature=0.1)
        
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.pdf_processed = True
        
        # Calculate final metrics
        metrics['processing_time'] = time.time() - metrics['start_time']
        metrics['embedding_model'] = "all-MiniLM-L6-v2"
        st.session_state.processing_metrics = metrics
        
        # Final update
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Show success with metrics
        with st.container():
            st.success(f"**{uploaded_file.name}** processed successfully!")
            
            with st.expander("📈 View Processing Metrics"):
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Time", f"{metrics['processing_time']:.1f}s")
                with cols[1]:
                    st.metric("Pages", metrics['page_count'])
                with cols[2]:
                    st.metric("Chunks", metrics['chunk_count'])
                with cols[3]:
                    st.metric("File Size", f"{metrics['file_size']/1024:.1f} KB")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ============================================
# 5. ENHANCED CHAT WITH SOURCES AND CITATIONS
# ============================================
def get_ai_response_with_sources(question):
    """Get AI response with source citations - shows attention to detail"""
    
    if not st.session_state.retriever or not st.session_state.llm:
        return "System not ready. Please upload a PDF first.", []
    
    try:
        # Retrieve relevant documents
        docs = st.session_state.retriever.invoke(question)
        
        # Prepare context with citations
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Extract metadata
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            # Format with citation marker
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            context_parts.append(f"[Source {i}, Page {page}]: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for better answers
        prompt = f"""You are an expert AI assistant analyzing a document.

DOCUMENT CONTEXT (with citations):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer STRICTLY based on the provided context above
2. Cite your sources using [Source X] notation when referencing specific information
3. If the answer cannot be found in the context, say: "Based on the document, I cannot find specific information about this."
4. Keep answers concise but informative
5. Format your response professionally with clear paragraphs

ANSWER:"""
        
        # Get response
        response = st.session_state.llm.invoke(prompt)
        
        # Track conversation
        st.session_state.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'sources_used': len(docs)
        })
        
        return response, docs
        
    except Exception as e:
        return f"Error: {str(e)}", []

# ============================================
# 6. FEATURES THAT IMPRESS ON GITHUB
# ============================================
def export_conversation():
    """Export conversation as JSON - shows you think about data"""
    conversation_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'project': 'AI PDF Chatbot',
            'model': 'Ollama Llama3'
        },
        'conversation': st.session_state.messages,
        'processing_metrics': st.session_state.get('processing_metrics', {})
    }
    
    # Create downloadable JSON
    json_str = json.dumps(conversation_data, indent=2)
    
    st.download_button(
        label="⬇️ Download Conversation JSON",
        data=json_str,
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

def show_demo_mode():
    """Showcase features when no PDF is uploaded"""
    with st.container():
        st.markdown("""
        <div class='portfolio-header'>
            <h1>🤖 AI PDF Assistant</h1>
            <p>Professional-grade document analysis with local AI</p>
            <p style='font-size: 1rem; margin-top: 1rem; opacity: 0.8;'>
                GitHub Portfolio Project • Built with Streamlit, LangChain & Ollama
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("## ✨ Professional Features")
    
    cols = st.columns(3)
    features = [
        ("🔒 Local & Private", "All processing happens on your machine. No data leaves your system."),
        ("📊 Smart Analytics", "Processing metrics, conversation tracking, and export capabilities."),
        ("🔍 Semantic Search", "FAISS-powered vector search for accurate information retrieval."),
        ("💬 Context-Aware Chat", "Follow-up questions, source citations, and conversation memory."),
        ("🎯 Multiple Models", "Switch between Llama3, Mistral, and other Ollama models."),
        ("📈 Performance Metrics", "Real-time processing stats and optimization insights.")
    ]
    
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"<div class='feature-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
    
    # Upload area with better styling
    st.markdown("## 🚀 Get Started")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; border: 2px dashed #667eea; border-radius: 15px;'>
                <h3 style='color: #667eea;'>📤 Upload Your First PDF</h3>
                <p>Drag and drop or click to browse</p>
                <p style='font-size: 0.9rem; color: #666;'>Supported: Research papers, reports, books, articles</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# 7. MAIN APPLICATION FLOW - FIXED VERSION
# ============================================
def main():
    """Main application - clean, professional flow"""
    
    # Initialize session
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if not st.session_state.pdf_processed:
        # Show demo/upload mode
        show_demo_mode()
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            label_visibility="collapsed",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            # Set flag to show processing
            st.session_state.show_processing = True
            
            # Process the PDF
            with st.spinner("Processing your document..."):
                success = process_pdf_with_metrics(uploaded_file)
                
                if success:
                    st.balloons()  # Celebration effect
                    st.session_state.file_info = {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'upload_time': datetime.now().isoformat()
                    }
                    
                    # FIX: Clear the processing flag and force app to show chat interface
                    st.session_state.show_processing = False
                    
                    # Add a button to continue to chat
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🚀 Start Chatting", type="primary", use_container_width=True):
                            # Force rerun to show chat interface
                            st.rerun()
                    
                    # Alternative: Auto-refresh after 2 seconds
                    time.sleep(2)
                    st.rerun()
    
    else:
        # PDF is processed, show chat interface
        st.markdown("## 💬 Chat with Your Document")
        
        # Document info bar
        with st.container():
            cols = st.columns(4)
            with cols[0]:
                st.metric("Document", st.session_state.file_info.get('name', 'PDF'))
            with cols[1]:
                st.metric("Status", "Ready")
            with cols[2]:
                if st.session_state.processing_metrics:
                    st.metric("Pages", st.session_state.processing_metrics.get('page_count', 0))
            with cols[3]:
                if st.session_state.messages:
                    st.metric("Messages", len(st.session_state.messages))
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.messages:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='ai-message'><strong>🤖 Assistant:</strong> {message['content']}</div>", 
                                   unsafe_allow_html=True)
            else:
                st.info("💡 No messages yet. Start by asking a question below!")
        
        # Chat input - MOVE THIS OUTSIDE THE CONTAINER
        if question := st.chat_input("Ask a question about your PDF..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get AI response
            with st.spinner("Analyzing document..."):
                response, sources = get_ai_response_with_sources(question)
                
                # Add AI response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Show sources if available
                if sources and len(sources) > 0:
                    with st.expander(f"📄 View {len(sources)} Source(s)"):
                        for i, source in enumerate(sources, 1):
                            page = source.metadata.get('page', 'N/A')
                            content_preview = source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content
                            st.markdown(f"**Source {i} (Page {page}):**")
                            st.text(content_preview)
            
            # Rerun to update display
            st.rerun()
        
        # Quick questions suggestions
        if not st.session_state.messages:
            st.markdown("### 💡 Try asking:")
            cols = st.columns(4)
            questions = [
                "What is this document about?",
                "Summarize the main points",
                "What are the key findings?",
                "List the recommendations"
            ]
            
            for i, q in enumerate(questions):
                with cols[i]:
                    if st.button(q, use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.rerun()

# ============================================
# 8. RUN THE APPLICATION
# ============================================
if __name__ == "__main__":
    main()