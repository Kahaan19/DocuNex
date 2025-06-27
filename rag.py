import os
import json
import shutil
import requests
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pickle
import gc
import torch
from rag_manager import RAGManager


# Initialize global RAG manager
rag_manager = None

def initialize_rag_system(vector_store_path: str = "./vector_store") -> RAGManager:
    """Initialize the RAG system"""
    global rag_manager
    rag_manager = RAGManager(vector_store_path)
    return rag_manager

def generate_answer_ollama(question: str, rag_manager: RAGManager, ollama_base_url: str, model: str) -> str:
    """Generate answer using RAG approach with Ollama"""
    try:
        # Check if rag_manager is valid and ready
        if not rag_manager:
            return "⚠️ RAG manager is None. Please initialize the RAG system first."
        
        if not isinstance(rag_manager, RAGManager):
            return "⚠️ Invalid RAG manager object. Expected RAGManager instance."
        
        if not hasattr(rag_manager, 'is_ready'):
            return "⚠️ RAG manager missing is_ready method. Please update your RAGManager class."
        
        if not rag_manager.is_ready():
            return "⚠️ RAG system not ready. Please build or load the vector store first."
        
        # Retrieve relevant chunks
        relevant_chunks = rag_manager.retrieve_relevant_chunks(question, k=5)
        
        if not relevant_chunks:
            return "⚠️ No relevant documents found for your question."
        
        # Build context from chunks
        context_parts = []
        context_parts.append("RELEVANT DOCUMENT EXCERPTS:")
        
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk.get('source', 'Unknown source')
            content = chunk['content'][:600]  # Limit chunk size
            if len(chunk['content']) > 600:
                content += "..."
            
            context_parts.append(f"\n[Document {i}] Source: {source}")
            context_parts.append(f"Relevance Score: {chunk.get('relevance_score', 0):.3f}")
            context_parts.append(f"Content: {content}")
        
        context_text = "\n".join(context_parts)
        
        # Compose the enhanced prompt for Ollama
        prompt = f"""You are an expert document analyst. Answer the question based on the provided document excerpts. Be thorough, accurate, and cite specific information from the documents.

{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Answer based only on the information provided in the document excerpts above
- Include specific details and examples from the documents
- If the documents don't contain enough information, state what is missing
- Structure your response clearly with main points and supporting details
- Be thorough but concise

ANSWER:"""

        # Generate response using Ollama
        response = call_ollama(prompt, ollama_base_url, model)
        
        if "error" in response:
            return f"⚠️ Ollama Error: {response['error']}"
        
        answer_text = response.get('response', 'No response generated')
        
        # Format the final response
        formatted_response = format_rag_response_ollama(answer_text, relevant_chunks)
        
        return formatted_response

    except Exception as e:
        return f"⚠️ Error generating RAG answer: {str(e)}"

def call_ollama(prompt: str, base_url: str, model: str) -> Dict:
    """Call Ollama API with the given prompt"""
    try:
        url = f"{base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 1024
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Try a shorter question or check if Ollama is running."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Ollama. Make sure it's running: `ollama serve`"}
    except Exception as e:
        return {"error": str(e)}

def format_rag_response_ollama(answer: str, chunks: List[Dict]) -> str:
    """Format the final RAG response with additional information"""
    
    # Build the main response
    response_parts = [f"### 🦙 Ollama RAG Answer:\n{answer}"]
    
    # Add source information
    if chunks:
        response_parts.append(f"\n---\n### 📚 Document Sources:")
        unique_sources = list(set([chunk.get('source', 'Unknown') for chunk in chunks]))
        for i, source in enumerate(unique_sources, 1):
            # Truncate long source names
            display_source = source if len(source) < 60 else source[:57] + "..."
            response_parts.append(f"{i}. {display_source}")
        
        # Add relevance scores
        response_parts.append(f"\n### 🎯 Retrieved Chunks:")
        for i, chunk in enumerate(chunks[:3], 1):  # Show top 3
            score = chunk.get('relevance_score', 0)
            source = chunk.get('source', 'Unknown')[:30] + "..." if len(chunk.get('source', '')) > 30 else chunk.get('source', 'Unknown')
            response_parts.append(f"- **Chunk {i}**: {source} (Score: {score:.3f})")
    
    # Add processing info
    response_parts.append(f"\n### ⚡ Processing Info:")
    response_parts.append(f"- 🔍 Search method: **Vector similarity (FAISS)**")
    response_parts.append(f"- 🦙 Powered by Ollama (no API limits)")
    response_parts.append(f"- 📄 Document chunks analyzed: {len(chunks)}")
    response_parts.append(f"- 🧠 Embedding model: **sentence-transformers/all-MiniLM-L6-v2**")
    
    return "\n".join(response_parts)

def check_rag_status(rag_manager) -> Dict[str, Any]:
    """Check the status of the RAG system"""
    if not rag_manager:
        return {
            "status": "not_initialized",
            "message": "RAG manager not created",
            "ready": False
        }
    
    # Check if it's a RAGManager instance
    if not isinstance(rag_manager, RAGManager):
        return {
            "status": "invalid_type",
            "message": f"Expected RAGManager, got {type(rag_manager)}",
            "ready": False
        }
    
    # Check if it has the required methods
    if not hasattr(rag_manager, 'is_ready') or not hasattr(rag_manager, 'get_stats'):
        return {
            "status": "invalid_manager",
            "message": "RAG manager missing required methods",
            "ready": False
        }
    
    try:
        stats = rag_manager.get_stats()
        return {
            "status": "ready" if stats['is_ready'] else "not_ready",
            "ready": stats['is_ready'],
            "stats": stats,
            "message": f"RAG system {'ready' if stats['is_ready'] else 'not ready'} - {stats['num_chunks']} chunks loaded"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking RAG status: {str(e)}",
            "ready": False
        }

def load_or_create_rag(rag_manager, documents: List[Dict] = None) -> bool:
    """Load existing RAG or create new one with documents"""
    try:
        # Validate RAG manager
        if not rag_manager or not isinstance(rag_manager, RAGManager):
            print("❌ Invalid RAG manager")
            return False
        
        # Try to load existing vector store first
        if rag_manager.load_vector_store():
            print("✅ Loaded existing vector store")
            return True
        
        # If no existing store and documents provided, create new one
        if documents:
            print("📦 Creating new vector store...")
            return rag_manager.create_vector_store(documents)
        else:
            print("❌ No existing vector store found and no documents provided")
            return False
            
    except Exception as e:
        print(f"❌ Error in load_or_create_rag: {str(e)}")
        return False

# Legacy function for backward compatibility (if needed)
def generate_answer(question, vectorstore):
    """Legacy function - kept for backward compatibility"""
    try:
        # This is a simplified version using the old interface
        # You might need to adapt this based on your vectorstore format
        docs = vectorstore.similarity_search(question, k=3) if hasattr(vectorstore, 'similarity_search') else []
        
        context_text = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs])
        
        # Since we're moving to Ollama, this function should ideally be replaced
        # with generate_answer_ollama, but keeping it for compatibility
        
        return {
            "answer": f"Please use the new generate_answer_ollama function for better results with Ollama integration.\n\nFound {len(docs)} relevant documents.",
            "sources": docs,
            "context": context_text[:500] + "..." if len(context_text) > 500 else context_text
        }
        
    except Exception as e:
        return {
            "answer": f"⚠️ Error generating answer: {str(e)}",
            "sources": [],
            "context": ""
        }

# Helper functions for better integration
def get_rag_manager() -> Optional[RAGManager]:
    """Get the global RAG manager instance"""
    return rag_manager

def reset_rag_system():
    """Reset the global RAG system"""
    global rag_manager
    if rag_manager:
        rag_manager.cleanup_vector_store()
    rag_manager = None

def add_documents_to_rag(rag_manager, documents: List[Dict]) -> bool:
    """Add new documents to existing RAG system"""
    if not rag_manager:
        print("❌ RAG manager not initialized")
        return False
    
    if not isinstance(rag_manager, RAGManager):
        print(f"❌ Expected RAGManager, got {type(rag_manager)}")
        return False
    
    return rag_manager.add_documents(documents)