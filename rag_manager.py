import os
import json
import shutil
import pickle
import gc
import torch
import numpy as np
from typing import List, Dict, Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
class RAGManager:
    def __init__(self, vector_store_path: str = "./vector_store"):
        """Initialize RAG Manager with vector store configuration"""
        self.vector_store_path = vector_store_path
        self.embedding_model = None
        self.vector_store = None
        self.document_store = []
        self.vector_dimension = 384  # Default for all-MiniLM-L6-v2

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def initialize_embeddings(self) -> bool:
        """Initialize the embedding model"""
        try:
            print("🔄 Loading embedding model...")
            self.embedding_model = HuggingFaceBgeEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            print("✅ Embedding model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading embedding model: {str(e)}")
            return False

    def is_ready(self) -> bool:
        """Check if RAG system is ready for use"""
        return (
            self.embedding_model is not None and 
            self.vector_store is not None and 
            len(self.document_store) > 0
        )

    def create_vector_store(self, documents: List[Dict]) -> bool:
        """Create vector store from documents"""
        try:
            if not self.embedding_model:
                if not self.initialize_embeddings():
                    return False

            # Clean up old vector store
            self.cleanup_vector_store()

            print(f"📄 Processing {len(documents)} documents...")

            # Process documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = self.process_document(doc)
                all_chunks.extend(chunks)

            if not all_chunks:
                print("❌ No chunks created from documents")
                return False

            print(f"🔧 Created {len(all_chunks)} text chunks")
            docs = [Document(page_content=chunk['content'], metadata={'source': chunk['source']}) for chunk in all_chunks]
            self.vector_store = FAISS.from_documents(docs, self.embedding_model)
            self.document_store = all_chunks
            self.vector_store.save_local(self.vector_store_path)
            print(f"✅ Vector store created with {len(all_chunks)} chunks")
            return True
            print(f"✅ Vector store created with {len(all_chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            return False

    def process_document(self, document: Dict) -> List[Dict]:
        """Process a single document into chunks"""
        try:
            content = document.get('content', '')
            source = document.get('source', 'Unknown')

            if not content.strip():
                return []

            # Split document into chunks
            doc_obj = Document(page_content=content, metadata={'source': source})
            chunks = self.text_splitter.split_documents([doc_obj])

            # Convert to our format
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'content': chunk.page_content,
                    'source': source,
                    'chunk_id': f"{source}_{i}",
                    'metadata': chunk.metadata
                })

            return processed_chunks

        except Exception as e:
            print(f"❌ Error processing document {document.get('source', 'Unknown')}: {str(e)}")
            return []

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        try:
            if not self.is_ready():
                print("❌ Vector store not ready. Please create or load a vector store first.")
                return []

            results = self.vector_store.similarity_search_with_score(query, k)
            relevant_chunks = []
            for i, (doc, score) in enumerate(results):
                chunk={
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'relevance_score': score,
                    'rank': i+1,
                    'metadata': doc.metadata
                }
                relevant_chunks.append(chunk)
            return relevant_chunks 

        except Exception as e:
            print(f"❌ Error retrieving chunks: {str(e)}")
            return []

    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Alternative method name for document search"""
        return self.retrieve_relevant_chunks(query, k)

    def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """Return just the content of relevant chunks (for compatibility)"""
        chunks = self.retrieve_relevant_chunks(query, k)
        return [chunk['content'] for chunk in chunks]

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Return content and scores as tuples"""
        chunks = self.retrieve_relevant_chunks(query, k)
        return [(chunk['content'], chunk['relevance_score']) for chunk in chunks]

    def save_vector_store(self):
        try:
            if self.vector_store is not None:
                self.vector_store.save_local(self.vector_store_path)
                print("💾 Vector store saved successfully")
        except Exception as e:
            print(f"❌ Error saving vector store: {str(e)}")

    def load_vector_store(self) -> bool:
        """Load existing vector store from disk"""
        try:
            if not os.path.exists(self.vector_store_path):
                print("📂 No existing vector store found")
                return False

            # Load metadata first
            metadata_path = os.path.join(self.vector_store_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.vector_dimension = metadata.get('vector_dimension', 384)

            self.vector_store = FAISS.load_local(self.vector_store_path, self.embedding_model)
            if hasattr(self.vector_store, "docstore") and hasattr(self.vector_store.docstore,"search"):
                self.document_store = [doc for doc in self.vector_store.docstore._dict.values()]
            else:
                print("❌ Could not load document store from FAISS vector store")
                return False

            # Initialize embedding model
            if not self.embedding_model:
                if not self.initialize_embeddings():
                    return False

            print(f"✅ Vector store loaded with {len(self.document_store)} chunks")
            return True

        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            return False

    def cleanup_vector_store(self):
        """Clean up existing vector store"""
        try:
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
                print("🧹 Cleaned up old vector store")

            # Clear memory
            self.vector_store = None
            self.document_store = []

            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"⚠️ Error during cleanup: {str(e)}")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'is_ready': self.is_ready(),
            'num_chunks': len(self.document_store),
            'vector_dimension': self.vector_dimension,
            'has_embeddings': self.embedding_model is not None,
            'has_vector_store': self.vector_store is not None
        }

    def add_documents(self, documents: List[Dict]) -> bool:
        """Add new documents to existing vector store"""
        try:
            if not self.is_ready():
                print("❌ Vector store not ready. Create a vector store first.")
                return False

            print(f"📄 Adding {len(documents)} new documents...")

            # Process new documents
            new_chunks = []
            for doc in documents:
                chunks = self.process_document(doc)
                new_chunks.extend(chunks)

            if not new_chunks:
                print("❌ No chunks created from new documents")
                return False

            # Combine old and new chunks
            all_chunks = self.document_store + new_chunks
            docs = [Document(page_content=chunk['content'], metadata={'source': chunk['source']}) for chunk in all_chunks]

            # Re-create the FAISS vector store with all documents
            self.vector_store = FAISS.from_documents(docs, self.embedding_model)
            self.document_store = all_chunks

            # Save updated vector store
            self.save_vector_store()

            print(f"✅ Added {len(new_chunks)} new chunks to vector store")
            return True

        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            return False