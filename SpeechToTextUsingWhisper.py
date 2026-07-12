import os
import shutil
from dotenv import load_dotenv
import gradio as gr
import gc
import time
import uuid
import torch
import requests
import json
import traceback
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_manager import RAGManager
from langchain_community.chat_models.ollama import ChatOllama
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import asyncio
import soundfile as sf 
# Voice processing imports
import whisper
import tempfile
import numpy as np
from gtts import gTTS
import pygame
from io import BytesIO
import threading

rag_manager = None 

# Load environment variables
load_dotenv()

# Voice processing setup
whisper_model = None
pygame_initialized = False

def initialize_voice_components():
    """Initialize Whisper and pygame for voice processing"""
    global whisper_model, pygame_initialized
    
    try:
        print("🎤 Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        whisper_model = None
    
    try:
        pygame.mixer.init()
        pygame_initialized = True
        print("✅ Pygame audio initialized")
    except Exception as e:
        print(f"❌ Error initializing pygame: {e}")
        pygame_initialized = False

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper"""
    global whisper_model

    if not whisper_model:
        return "❌ Whisper model not loaded"

    if not os.path.exists(audio_file):
        return f"❌ Audio file not found: {audio_file}"

    try:
        print(f"🎤 Transcribing audio file: {audio_file}")
        result = whisper_model.transcribe(audio_file)
        transcribed_text = result["text"].strip()
        print(f"📝 Transcribed: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        error_msg = f"❌ Error transcribing audio: {str(e)}"
        print(error_msg)
        return error_msg

def process_voice_input(audio):
    """Process voice input and return transcribed text"""
    if audio is None:
        return "❌ No audio file provided"
    
    try:
        # Check if audio is a tuple (numpy array, sample rate) or file path
        if isinstance(audio, tuple) and len(audio) == 2:
            # It's a numpy array with sample rate
            data, samplerate = audio
            
            # Check if data is valid
            if data is None or len(data) == 0:
                return "❌ Empty audio data"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, data, samplerate)
                transcribed_text = transcribe_audio(tmp.name)
                os.unlink(tmp.name)
                return transcribed_text
        
        elif isinstance(audio, str) and os.path.exists(audio):
            # It's a file path
            return transcribe_audio(audio)
        
        else:
            return "❌ Invalid audio format"
            
    except Exception as e:
        error_msg = f"❌ Error processing voice input: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg

def text_to_speech(text, lang='en'):
    """Convert text to speech using gTTS and play it"""
    global pygame_initialized
    
    if not pygame_initialized:
        return "❌ Audio system not initialized"
    
    try:
        print(f"🔊 Converting text to speech: {text[:50]}...")
        
        # Create TTS object
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            tmp_filename = tmp_file.name
        
        # Play the audio
        pygame.mixer.music.load(tmp_filename)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Clean up temporary file
        os.unlink(tmp_filename)
        
        print("✅ Text-to-speech playback completed")
        return "✅ Audio played successfully"
        
    except Exception as e:
        error_msg = f"❌ Error in text-to-speech: {str(e)}"
        print(error_msg)
        return error_msg

# Shared configuration and helpers (deduplicated across entry points)
from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    BASE_PERSIST_DIR,
    BASE_VECTOR_DIR,
    DEVICE as device,
)
from ollama_utils import check_ollama_connection, stream_ollama_response
from neo4j_utils import create_neo4j_driver

# Neo4j connection (optional; only used for GraphRAG)
neo4j_driver = create_neo4j_driver()

print(f"🚀 Using device: {device}")

# Import your custom modules with better error handling
MODULES_AVAILABLE = True
missing_modules = []

try:
    from url_text_extractor import extract_text_from_url
except ImportError as e:
    missing_modules.append("url_text_extractor")
    print(f"⚠️ Warning: url_text_extractor not found: {e}")

try:
    from document_loader import load_documents_from_files
except ImportError as e:
    missing_modules.append("document_loader")
    print(f"⚠️ Warning: document_loader not found: {e}")

try:
    from graph_rag_manager import GraphRAGManager
except ImportError as e:
    missing_modules.append("GraphRAGManager")
    print(f"⚠️ Warning: GraphRAGManager not found: {e}")
    MODULES_AVAILABLE = False

try:
    from rag_chain import generate_graph_answer_ollama
except ImportError as e:
    missing_modules.append("rag_chain")
    print(f"⚠️ Warning: rag_chain not found: {e}")

try:
    from rag import generate_answer_ollama
except ImportError as e:
    missing_modules.append("rag")
    print(f"⚠️ Warning: rag not found: {e}")

if missing_modules:
    print(f"⚠️ Missing modules: {', '.join(missing_modules)}")

# Global variables
graph_rag_manager = None
embeddings = None
current_persist_dir = None
current_vector_dir = None
current_rag_type = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_graph_rag_context(question):
    """Get relevant context from GraphRAG system"""
    global graph_rag_manager
    
    try:
        if not graph_rag_manager:
            return "No GraphRAG context available"
        
        # Method 1: Try to get relevant context from the graph
        context = ""
        
        # Try to get knowledge graph store
        if hasattr(graph_rag_manager, 'knowledge_graph_store'):
            kg_store = graph_rag_manager.knowledge_graph_store
            
            # Try to get triplets
            if hasattr(kg_store, 'get_triplets'):
                triplets = kg_store.get_triplets()
                # Filter relevant triplets based on question keywords
                question_words = question.lower().split()
                relevant_triplets = []
                
                for triplet in triplets[:100]:  # Limit to avoid too much context
                    triplet_text = ' '.join([str(x) for x in triplet]).lower()
                    if any(word in triplet_text for word in question_words):
                        relevant_triplets.append(triplet)
                
                if relevant_triplets:
                    context = "Relevant knowledge from graph:\n"
                    for triplet in relevant_triplets[:15]:  # Top 15 most relevant
                        context += f"- {triplet[0]} -> {triplet[1]} -> {triplet[2]}\n"
        
        # Method 2: Try to get documents or chunks
        if not context and hasattr(graph_rag_manager, 'vector_store'):
            vector_store = graph_rag_manager.vector_store
            if hasattr(vector_store, 'similarity_search'):
                docs = vector_store.similarity_search(question, k=3)
                context = "Relevant documents:\n"
                for doc in docs:
                    context += f"- {doc.page_content[:500]}...\n"
        
        return context if context else "No relevant GraphRAG context found"
        
    except Exception as e:
        print(f"Error getting GraphRAG context: {e}")
        return f"Error retrieving GraphRAG context: {str(e)}"

def get_rag_context(question, k=3):
    """Get relevant context from the RAG vector store."""
    global rag_manager
    try:
        if rag_manager and rag_manager.is_ready():
            docs = rag_manager.vector_store.similarity_search(question, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        else:
            return "No RAG context available"
    except Exception as e:
        print(f"Error getting RAG context: {e}")
        return "Error retrieving context"

def stream_rag_answer(question):
    """Stream RAG answer with improved error handling"""
    global current_rag_type
    
    try:
        yield "🔍 Retrieving relevant context...\n\n"
        
        # Get context based on RAG type
        if current_rag_type == "GraphRAG":
            context = get_graph_rag_context(question)
            yield f"🔍 Found GraphRAG context, generating answer...\n\n"
        else:
            context = get_rag_context(question)
            yield f"🔍 Found RAG context, generating answer...\n\n"
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question comprehensively and accurately.

Context:
{context}

Question: {question}

Answer:"""

        # Stream the response
        answer_started = False
        for response in stream_ollama_response(prompt):
            if not answer_started:
                yield f"🔍 Context retrieved, generating answer...\n\n**Answer:**\n{response}"
                answer_started = True
            else:
                yield f"🔍 Context retrieved, generating answer...\n\n**Answer:**\n{response}"
                
    except Exception as e:
        error_msg = f"❌ Error in stream_rag_answer: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        yield error_msg

def on_ask_with_voice(question, enable_tts=False):
    """Handle question answering with optional text-to-speech output"""
    global graph_rag_manager, current_persist_dir, current_vector_dir, current_rag_type, rag_manager
    
    if not question or not question.strip():
        error_msg = "❌ Please enter a question"
        if enable_tts:
            threading.Thread(target=text_to_speech, args=(error_msg,)).start()
        yield error_msg
        return
    
    if not MODULES_AVAILABLE:
        error_msg = "❌ Required modules not found. Please ensure all custom modules are available."
        if enable_tts:
            threading.Thread(target=text_to_speech, args=(error_msg,)).start()
        yield error_msg
        return
    
    if not current_rag_type:
        error_msg = "⚠️ Please upload documents first to build the knowledge base."
        if enable_tts:
            threading.Thread(target=text_to_speech, args=(error_msg,)).start()
        yield error_msg
        return
    
    ollama_status, ollama_msg = check_ollama_connection()
    if not ollama_status:
        if enable_tts:
            threading.Thread(target=text_to_speech, args=(f"Ollama connection failed: {ollama_msg}",)).start()
        yield f"🔌 {ollama_msg}"
        return
    
    try:
        if current_rag_type == "GraphRAG":
            # Use GraphRAG
            if not graph_rag_manager:
                if current_persist_dir and os.path.exists(current_persist_dir):
                    try:
                        loading_msg = "Loading GraphRAG knowledge base..."
                        yield f"🔄 {loading_msg}\n\n"
                        if enable_tts:
                            threading.Thread(target=text_to_speech, args=(loading_msg,)).start()
                        
                        print(f"🔄 Loading GraphRAG from {current_persist_dir}")
                        graph_rag_manager = GraphRAGManager(
                            persist_dir=current_persist_dir,
                            api_key=None,
                            device=device
                        )
                        
                        # Try to load the knowledge graph
                        if hasattr(graph_rag_manager, 'load_knowledge_graph'):
                            graph_rag_manager.load_knowledge_graph()
                            print("✅ Knowledge graph loaded successfully")
                        else:
                            print("⚠️ GraphRAGManager doesn't have load_knowledge_graph method")
                            
                    except Exception as e:
                        error_msg = f"⚠️ Error loading knowledge graph: {str(e)}"
                        print(error_msg)
                        traceback.print_exc()
                        if enable_tts:
                            threading.Thread(target=text_to_speech, args=(error_msg,)).start()
                        yield error_msg
                        return
                else:
                    error_msg = "⚠️ Please upload content first to build the GraphRAG knowledge graph."
                    if enable_tts:
                        threading.Thread(target=text_to_speech, args=(error_msg,)).start()
                    yield error_msg
                    return
            
            # Stream GraphRAG answer with TTS
            try:
                full_answer = ""
                for response in stream_rag_answer(question):
                    yield response
                    # Extract just the answer part for TTS
                    if "**Answer:**\n" in response:
                        full_answer = response.split("**Answer:**\n", 1)[1].strip()
                
                # Convert final answer to speech
                if enable_tts and full_answer:
                    threading.Thread(target=text_to_speech, args=(full_answer,)).start()
                    
            except Exception as e:
                error_msg = f"❌ Error generating GraphRAG answer: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                if enable_tts:
                    threading.Thread(target=text_to_speech, args=(error_msg,)).start()
                yield error_msg
        
        else:  # Regular RAG
            # Check if vector store is loaded
            try:
                if not rag_manager or not rag_manager.is_ready():
                    if current_vector_dir and os.path.exists(current_vector_dir):
                        loading_msg = "Loading RAG vector store..."
                        yield f"🔄 {loading_msg}\n\n"
                        if enable_tts:
                            threading.Thread(target=text_to_speech, args=(loading_msg,)).start()
                        
                        rag_manager = RAGManager(vector_store_path=current_vector_dir)
                        if not rag_manager.load_vector_store():
                            error_msg = "❌ Failed to load vector store"
                            if enable_tts:
                                threading.Thread(target=text_to_speech, args=(error_msg,)).start()
                            yield error_msg
                            return
                    else:
                        error_msg = "⚠️ Please upload content first to build the RAG vector store."
                        if enable_tts:
                            threading.Thread(target=text_to_speech, args=(error_msg,)).start()
                        yield error_msg
                        return
                        
                # Stream RAG answer with TTS
                full_answer = ""
                for response in stream_rag_answer(question):
                    yield response
                    # Extract just the answer part for TTS
                    if "**Answer:**\n" in response:
                        full_answer = response.split("**Answer:**\n", 1)[1].strip()
                
                # Convert final answer to speech
                if enable_tts and full_answer:
                    threading.Thread(target=text_to_speech, args=(full_answer,)).start()
                    
            except Exception as e:
                error_msg = f"❌ Error generating RAG answer: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                if enable_tts:
                    threading.Thread(target=text_to_speech, args=(error_msg,)).start()
                yield error_msg
    
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        if enable_tts:
            threading.Thread(target=text_to_speech, args=(error_msg,)).start()
        yield error_msg

def handle_inputs(urls, files, rag_type, progress=gr.Progress()):
    """
    Handles building the knowledge base from URLs and files for RAG or GraphRAG.
    """
    global rag_manager, graph_rag_manager, current_persist_dir, current_vector_dir, current_rag_type

    try:
        progress(0, desc="Starting knowledge base build...")

        # Parse URLs
        url_list = []
        if urls:
            url_list = [u.strip() for u in urls.split(",") if u.strip()]
        docs = []

        # Extract text from URLs
        if url_list:
            progress(0.1, desc="Extracting text from URLs...")
            for url in url_list:
                try:
                    text = extract_text_from_url(url)
                    docs.append({"content": text, "source": url})
                except Exception as e:
                    yield f"❌ Error extracting from {url}: {e}"

        # Load files
        file_paths = []
        if files:
            if isinstance(files, list):
                file_paths = [f.name if hasattr(f, "name") else f for f in files]
            else:
                file_paths = [files.name if hasattr(files, "name") else files]
        if file_paths:
            progress(0.2, desc="Loading files...")
            loaded_docs = load_documents_from_files(file_paths)
            for doc in loaded_docs:
                docs.append({"content": doc.page_content, "source": doc.metadata.get("source", "uploaded file")})

        if not docs:
            yield "⚠️ No documents found. Please provide URLs or upload files."
            return

        # Build knowledge base
        if rag_type == "GraphRAG":
            progress(0.4, desc="Building GraphRAG knowledge base...")
            persist_dir = os.path.join(BASE_PERSIST_DIR, f"graphrag_{uuid.uuid4().hex[:8]}")
            os.makedirs(persist_dir, exist_ok=True)
            graph_rag_manager = GraphRAGManager(persist_dir=persist_dir, device=device)
            status = graph_rag_manager.build_knowledge_graph([Document(page_content=d["content"], metadata={"source": d["source"]}) for d in docs])
            current_persist_dir = persist_dir
            current_rag_type = "GraphRAG"
            yield f"✅ GraphRAG knowledge base built!\n{status}"
        else:
            progress(0.4, desc="Building RAG vector store...")
            vector_dir = os.path.join(BASE_VECTOR_DIR, f"rag_{uuid.uuid4().hex[:8]}")
            os.makedirs(vector_dir, exist_ok=True)
            rag_manager = RAGManager(vector_store_path=vector_dir)
            success = rag_manager.create_vector_store(docs)
            current_vector_dir = vector_dir
            current_rag_type = "RAG"
            if success:
                yield f"✅ RAG vector store built with {len(docs)} documents."
            else:
                yield "❌ Failed to build RAG vector store."

    except Exception as e:
        error_msg = f"❌ Error building knowledge base: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg

def get_current_status():
    """Get current system status for display in Gradio."""
    global current_rag_type, current_persist_dir, current_vector_dir, graph_rag_manager, rag_manager, neo4j_driver

    status_lines = []

    # Ollama status
    ollama_status, ollama_msg = check_ollama_connection()
    status_lines.append(f"🤖 Ollama: {'✅ Connected' if ollama_status else '❌ Not Connected'}")
    if not ollama_status:
        status_lines.append(f"   {ollama_msg}")

    # Neo4j status
    neo4j_status = "✅ Connected" if neo4j_driver else "❌ Not Connected"
    status_lines.append(f"🗄️ Neo4j: {neo4j_status}")

    # Voice system status
    status_lines.append(f"🎤 Whisper: {'✅ Loaded' if whisper_model else '❌ Not Loaded'}")
    status_lines.append(f"🔊 Audio System: {'✅ Ready' if pygame_initialized else '❌ Not Ready'}")

    # RAG/GraphRAG status
    if current_rag_type:
        status_lines.append(f"🧠 RAG Type: {current_rag_type}")
        if current_rag_type == "GraphRAG":
            status_lines.append(f"📁 GraphRAG Dir: {current_persist_dir}")
            if graph_rag_manager:
                status_lines.append("📊 GraphRAG: ✅ Loaded")
                if hasattr(graph_rag_manager, "get_graph_stats"):
                    stats = graph_rag_manager.get_graph_stats()
                    status_lines.append(f"   Nodes: {stats.get('nodes', 0)}, Edges: {stats.get('edges', 0)}, Chunks: {stats.get('chunks', 0)}")
            else:
                status_lines.append("📊 GraphRAG: ⚠️ Not Loaded")
        else:
            status_lines.append(f"📁 Vector Dir: {current_vector_dir}")
            if rag_manager and hasattr(rag_manager, "get_stats"):
                stats = rag_manager.get_stats()
                status_lines.append(f"   Chunks: {stats.get('num_chunks', 0)}, Vector Dim: {stats.get('vector_dimension', 0)}")
            else:
                status_lines.append("📊 RAG: ⚠️ Not Loaded")
    else:
        status_lines.append("🧠 RAG Type: Not selected or built yet.")

    return "\n".join(status_lines)

def create_interface():
    """Create and configure the Gradio interface with voice features"""
    
    with gr.Blocks(title="RAG & Graph RAG System with Voice", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 RAG & Graph RAG System with Voice Features")
        gr.Markdown("Upload documents, ask questions via text or voice, and get audio responses!")
        
        with gr.Tab("📤 Upload & Build"):
            with gr.Row():
                with gr.Column(scale=1):
                    urls_input = gr.Textbox(
                        label="🌐 URLs (comma-separated)",
                        placeholder="https://example.com, https://another-site.com",
                        lines=2
                    )
                    files_input = gr.File(
                        label="📁 Upload Files",
                        file_count="multiple",
                        file_types=["text", "pdf", "docx", "markdown"]
                    )
                    rag_type = gr.Radio(
                        choices=["RAG", "GraphRAG"],
                        value="GraphRAG",
                        label="🧠 RAG Type",
                        info="GraphRAG builds knowledge graphs, regular RAG uses vector similarity"
                    )
                    build_btn = gr.Button("🚀 Build Knowledge Base", variant="primary")
                
                with gr.Column(scale=1):
                    build_output = gr.Textbox(
                        label="📊 Build Status",
                        lines=15,
                        max_lines=20
                    )
        
        with gr.Tab("❓ Ask Questions"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Text input
                    question_input = gr.Textbox(
                        label="❓ Your Question (Text)",
                        placeholder="Ask anything about your uploaded documents...",
                        lines=3
                    )
                    
                    # Voice input
                    gr.Markdown("### 🎤 Voice Input")
                    voice_input = gr.Audio(
                        label="🎙️ Record your question",
                        type="numpy",
                        show_label=True
                    )
                    
                    transcribe_btn = gr.Button("📝 Transcribe Voice", variant="secondary")
                    transcribed_text = gr.Textbox(
                        label="📝 Transcribed Text",
                        placeholder="Transcribed text will appear here...",
                        lines=2
                    )
                    
                    # Control buttons
                    with gr.Row():
                        ask_btn = gr.Button("🔍 Ask Question (Text)", variant="primary")
                        ask_voice_btn = gr.Button("🎤 Ask Question (Voice)", variant="primary")
                    
                    # TTS option
                    enable_tts = gr.Checkbox(
                        label="🔊 Enable Text-to-Speech Output",
                        value=True,
                        info="Convert answers to speech"
                    )
                    
                with gr.Column(scale=1):
                    status_btn = gr.Button("📊 Check Status")
                    status_output = gr.Textbox(
                        label="🔄 System Status",
                        lines=10,
                        max_lines=15
                    )
            
            answer_output = gr.Textbox(
                label="💡 Answer",
                lines=15,
                max_lines=25
            )
        
        with gr.Tab("🔧 System Info"):
            gr.Markdown("## System Configuration")
            
            info_text = f"""
            **🤖 Ollama Configuration:**
            - Base URL: `{OLLAMA_BASE_URL}`
            - Model: `{OLLAMA_MODEL}`
            
            **🗄️ Neo4j Configuration:**
            - URI: `{NEO4J_URI}`
            - User: `{NEO4J_USER}`
            - Password: `{'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else 'Not Set'}`
            
            **📁 Storage Directories:**
            - Graph DB: `{BASE_PERSIST_DIR}`
            - Vector DB: `{BASE_VECTOR_DIR}`
            
            **🔧 Device:**
            - Current Device: `{device}`
            - CUDA Available: `{torch.cuda.is_available()}`
            
            **📦 Module Status:**
              - Missing Modules: `{', '.join(missing_modules) if missing_modules else 'All modules available'}`
            """
            
            gr.Markdown(info_text)
            
            # Add system check button
            check_system_btn = gr.Button("🔍 Check All Systems", variant="secondary")
            system_check_output = gr.Textbox(
                label="🔍 System Check Results",
                lines=10,
                placeholder="Click 'Check All Systems' to see detailed status..."
            )
        
        # Event handlers
        def handle_build_knowledge_base(urls, files, rag_type, progress=gr.Progress()):
            """Wrapper for handle_inputs with progress tracking"""
            for result in handle_inputs(urls, files, rag_type, progress):
                yield result
        
        def handle_transcribe(audio):
            """Handle voice transcription"""
            if audio is None:
                return "❌ No audio provided"
            
            result = process_voice_input(audio)
            return result
        
        def handle_ask_text(question, enable_tts):
            """Handle text-based questions"""
            if not question.strip():
                return "❌ Please enter a question"
            
            for response in on_ask_with_voice(question, enable_tts):
                yield response
        
        def handle_ask_voice(transcribed_text, enable_tts):
            """Handle voice-based questions"""
            if not transcribed_text or not transcribed_text.strip():
                return "❌ Please transcribe your voice input first"
            
            for response in on_ask_with_voice(transcribed_text, enable_tts):
                yield response
        
        def handle_system_check():
            """Perform comprehensive system check"""
            results = []
            
            # Check Ollama
            ollama_ok, ollama_msg = check_ollama_connection()
            results.append(f"🤖 Ollama: {ollama_msg}")
            
            # Check Neo4j
            if neo4j_driver:
                try:
                    with neo4j_driver.session() as session:
                        result = session.run("RETURN 1 as test")
                        record = result.single()
                        if record and record["test"] == 1:
                            results.append("🗄️ Neo4j: ✅ Connected and responding")
                        else:
                            results.append("🗄️ Neo4j: ⚠️ Connected but unexpected response")
                except Exception as e:
                    results.append(f"🗄️ Neo4j: ❌ Connection error: {str(e)}")
            else:
                results.append("🗄️ Neo4j: ❌ Not connected")
            
            # Check voice components
            results.append(f"🎤 Whisper: {'✅ Loaded' if whisper_model else '❌ Not loaded'}")
            results.append(f"🔊 Audio System: {'✅ Ready' if pygame_initialized else '❌ Not ready'}")
            
            # Check directories
            results.append(f"📁 Graph DB Dir: {'✅ Exists' if os.path.exists(BASE_PERSIST_DIR) else '❌ Missing'}")
            results.append(f"📁 Vector DB Dir: {'✅ Exists' if os.path.exists(BASE_VECTOR_DIR) else '❌ Missing'}")
            
            # Check current RAG status
            if current_rag_type:
                results.append(f"🧠 Current RAG Type: {current_rag_type}")
                if current_rag_type == "GraphRAG" and current_persist_dir:
                    results.append(f"📊 GraphRAG Dir: {'✅ Exists' if os.path.exists(current_persist_dir) else '❌ Missing'}")
                elif current_rag_type == "RAG" and current_vector_dir:
                    results.append(f"📊 Vector Store Dir: {'✅ Exists' if os.path.exists(current_vector_dir) else '❌ Missing'}")
            else:
                results.append("🧠 RAG System: ⚠️ No knowledge base built yet")
            
            # Check GPU/CUDA
            if torch.cuda.is_available():
                results.append(f"🚀 CUDA: ✅ Available (Device: {torch.cuda.get_device_name()})")
                results.append(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            else:
                results.append("🚀 CUDA: ❌ Not available (using CPU)")
            
            return "\n".join(results)
        
        # Wire up the event handlers
        build_btn.click(
            fn=handle_build_knowledge_base,
            inputs=[urls_input, files_input, rag_type],
            outputs=[build_output]
        )
        
        transcribe_btn.click(
            fn=handle_transcribe,
            inputs=[voice_input],
            outputs=[transcribed_text]
        )
        
        ask_btn.click(
            fn=handle_ask_text,
            inputs=[question_input, enable_tts],
            outputs=[answer_output]
        )
        
        ask_voice_btn.click(
            fn=handle_ask_voice,
            inputs=[transcribed_text, enable_tts],
            outputs=[answer_output]
        )
        
        status_btn.click(
            fn=get_current_status,
            inputs=[],
            outputs=[status_output]
        )
        
        check_system_btn.click(
            fn=handle_system_check,
            inputs=[],
            outputs=[system_check_output]
        )
        
        # Auto-populate transcribed text to question input
        def copy_transcribed_to_question(transcribed):
            return transcribed
        
        transcribed_text.change(
            fn=copy_transcribed_to_question,
            inputs=[transcribed_text],
            outputs=[question_input]
        )
        
    return interface

def cleanup_resources():
    """Clean up system resources"""
    global neo4j_driver, pygame_initialized, whisper_model
    
    try:
        if neo4j_driver:
            neo4j_driver.close()
            print("✅ Neo4j connection closed")
    except Exception as e:
        print(f"⚠️ Error closing Neo4j: {e}")
    
    try:
        if pygame_initialized:
            pygame.mixer.quit()
            print("✅ Pygame mixer closed")
    except Exception as e:
        print(f"⚠️ Error closing pygame: {e}")
    
    # Clear CUDA cache if available
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ CUDA cache cleared")
    except Exception as e:
        print(f"⚠️ Error clearing CUDA cache: {e}")
    
    # Force garbage collection
    gc.collect()
    print("✅ Garbage collection completed")

def main():
    """Main function to run the application"""
    print("🚀 Starting RAG & Graph RAG System with Voice...")
    
    try:
        # Initialize voice components
        print("🎤 Initializing voice components...")
        initialize_voice_components()
        
        # Create directories if they don't exist
        os.makedirs(BASE_PERSIST_DIR, exist_ok=True)
        os.makedirs(BASE_VECTOR_DIR, exist_ok=True)
        print(f"📁 Created directories: {BASE_PERSIST_DIR}, {BASE_VECTOR_DIR}")
        
        # Check initial system status
        print("🔍 Checking system status...")
        ollama_ok, ollama_msg = check_ollama_connection()
        print(f"🤖 {ollama_msg}")
        
        if neo4j_driver:
            print("🗄️ Neo4j: ✅ Connected")
        else:
            print("🗄️ Neo4j: ❌ Not connected (optional for RAG)")
        
        # Create and launch interface
        print("🌐 Creating Gradio interface...")
        interface = create_interface()
        
        print("✅ System initialized successfully!")
        print("🚀 Launching web interface...")
        print(f"📱 Access the interface at: http://localhost:7860")
        print("🔄 Use Ctrl+C to stop the server")
        
        # Launch with specific configuration
        interface.launch(
            server_name="127.0.0.1",  # Allow external access
            server_port=7860,
            share=False,  # Set to True if you want a public link
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n🔄 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up resources...")
        cleanup_resources()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()