import os
import json
import streamlit as st
from vector_store import VectorStore
from rag_engine import RAGEngine
import google.generativeai as genai

# Setup page config
st.set_page_config(
    page_title="Rechtwijs AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states if they don't exist
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None

# Title and description
st.title("🇮🇩 Asisten AI berbahasa indonesia")
st.markdown("AI dengan basis data peraturan dan literatur berbahasa indonesia, By: Psatir-420(Gungde)")

# Sidebar for settings and actions
with st.sidebar:
    st.header("Settings")
    
    # API Key input
    api_key = st.text_input(
        "Pass Key", 
        value=st.session_state.gemini_api_key,
        type="password",
        help="not a member of tuweb ? Get your own pass key from https://aistudio.google.com/"
    )
    
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        
        # Reset RAG engine if API key changes
        st.session_state.rag_engine = None
        st.success("Password berhasil di submit")
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Menunggu koneksi database"):
            try:
                # Initialize vector store
                data_dir = "processed_data"
                st.session_state.vector_store = VectorStore(data_dir)
                st.session_state.vector_store.load_documents()
                
                # Show success message with document count
                doc_count = len(st.session_state.vector_store.documents)
                chunk_count = sum(len(doc["chunks"]) for doc in st.session_state.vector_store.documents)
                st.success(f"Successfully loaded {doc_count} documents with {chunk_count} chunks!")
                
                # Initialize RAG engine if we have a key
                if st.session_state.gemini_api_key:
                    st.session_state.rag_engine = RAGEngine(
                        st.session_state.vector_store,
                        st.session_state.gemini_api_key
                    )
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Show data stats if loaded
    if st.session_state.vector_store and st.session_state.vector_store.documents:
        st.metric("Documents Loaded", len(st.session_state.vector_store.documents))
        st.metric("Total Chunks", sum(len(doc["chunks"]) for doc in st.session_state.vector_store.documents))
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    Program ini hanyalah asisten semata, gunakan secukupnya dan dengan bertanggung jawab. Semua jawaban AI didasarkan pada data yang digunakan untuk melatih. Punya literatur hukum atau peraturan yang ingin anda tambahkan ? Chat langsung Gungde/ kirim PDF ke Grup Tuweb    
- Masukkan kunci akses Anda untuk menggunakan layanan ini
- Muat data hukum dari direktori data
- Ajukan pertanyaan terkait hukum Indonesia
- Dapatkan jawaban akurat dengan sumber kutipan
    """)

# Main content - RAG Query Interface
st.header("Hallo, Selamat Belajar")

# Check if we have required components
if not st.session_state.vector_store or len(st.session_state.vector_store.documents) == 0:
    st.warning("Masukkan kunci terlebih dahulu lalu load data untuk menggunakan AI")
elif not st.session_state.gemini_api_key:
    st.error("Masukkan kunci terlebih dahulu lalu load data untuk menggunakan AI")
elif not st.session_state.rag_engine:
    # Try to initialize RAG engine if vector store and API key are available
    try:
        st.session_state.rag_engine = RAGEngine(
            st.session_state.vector_store,
            st.session_state.gemini_api_key
        )
        st.success("RAG engine initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing RAG engine: {str(e)}")
else:
    # Query interface
    query = st.text_area("Enter your question about Indonesian law:", 
                        height=100, 
                        max_chars=1000,
                        help="Ask a question about the Indonesian laws in the data")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_results = st.number_input("Mau pakai berapa sumber ? Maximum 10 sumber sob(semakin banyak semakin lambat)", 
                                    min_value=1, 
                                    max_value=10, 
                                    value=3)
    with col2:
        show_sources = st.checkbox("dokumen sumber", value=True)
    
    if st.button("Buat Jawaban"):
        if query:
            with st.spinner("Lagi Mikir........"):
                # Get response from RAG engine
                response = st.session_state.rag_engine.generate_response(
                    query, 
                    num_results=num_results
                )
                
                # Display response
                st.subheader("Answer")
                st.markdown(response["answer"])
                
                # Display sources if requested
                if show_sources and response["sources"]:
                    st.subheader("Sources")
                    for i, source in enumerate(response["sources"]):
                        with st.expander(f"Source {i+1}: {os.path.basename(source['source'])}"):
                            st.write(f"**Source:** {source['source']}")
                            st.write(f"**Pages:** {source['metadata']['page_start']}-{source['metadata']['page_end']}")
                            st.write("**Content:**")
                            st.text_area(f"Source content {i+1}", 
                                        value=source['text'], 
                                        height=150,
                                        key=f"source_{i}")
        else:
            st.error("Please enter a query.")

# Footer
st.divider()
st.caption("Asisten AI Hukum Indonesia | By: Psatir-420 (Gungde)")
