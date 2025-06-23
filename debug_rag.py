# debug_rag.py
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import torch

def debug_vector_store(vector_dir=None):
    """Debug vector store issues"""
    print("🔍 RAG Debug Utility")
    print("=" * 50)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📍 Device: {device}")
    
    # Check embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print("✅ Embeddings initialized successfully")
        print(f"📊 Embedding dimension: {embeddings.client.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return
    
    # Find vector directories
    if not vector_dir:
        base_dirs = [d for d in os.listdir('.') if d.startswith('vector_db')]
        if not base_dirs:
            print("❌ No vector_db directories found")
            return
        vector_dir = base_dirs[0]
        print(f"📁 Using directory: {vector_dir}")
    
    # Check vector store files
    print(f"\n📂 Checking directory: {vector_dir}")
    if not os.path.exists(vector_dir):
        print(f"❌ Directory doesn't exist: {vector_dir}")
        return
    
    files = os.listdir(vector_dir)
    print(f"📋 Files in directory: {files}")
    
    required_files = ['index.faiss', 'index.pkl']
    for file in required_files:
        if file in files:
            file_path = os.path.join(vector_dir, file)
            size = os.path.getsize(file_path)
            print(f"✅ {file}: {size} bytes")
        else:
            print(f"❌ Missing: {file}")
            return
    
    # Try loading vector store
    try:
        print("\n🔄 Loading vector store...")
        vector_store = FAISS.load_local(
            vector_dir, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded successfully!")
        
        # Check vector store properties
        print(f"📊 Vector store type: {type(vector_store)}")
        if hasattr(vector_store, 'index'):
            print(f"📊 Index type: {type(vector_store.index)}")
            if hasattr(vector_store.index, 'ntotal'):
                print(f"📊 Number of vectors: {vector_store.index.ntotal}")
        
        # Test similarity search
        print("\n🔍 Testing similarity search...")
        results = vector_store.similarity_search("test query", k=1)
        print(f"✅ Search returned {len(results)} results")
        if results:
            print(f"📄 First result preview: {results[0].page_content[:100]}...")
            
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_vector_store()