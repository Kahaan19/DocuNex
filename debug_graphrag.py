# debug_graphrag.py
import os
from graph_rag_manager import GraphRAGManager
import torch

def debug_graphrag_manager():
    """Debug GraphRAG manager to understand its structure"""
    print("🧠 GraphRAG Debug Utility")
    print("=" * 50)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📍 Device: {device}")
    
    # Find GraphRAG directories
    base_dirs = [d for d in os.listdir('.') if d.startswith('graph_db')]
    if not base_dirs:
        print("❌ No graph_db directories found")
        return
    
    graph_dir = base_dirs[0]
    print(f"📁 Using directory: {graph_dir}")
    
    try:
        # Initialize GraphRAG manager
        print(f"\n🔄 Loading GraphRAG manager from {graph_dir}...")
        graph_rag_manager = GraphRAGManager(
            persist_dir=graph_dir,
            api_key=None,
            device=device
        )
        
        # Try to load knowledge graph
        if hasattr(graph_rag_manager, 'load_knowledge_graph'):
            graph_rag_manager.load_knowledge_graph()
            print("✅ Knowledge graph loaded")
        
        print(f"📊 GraphRAG Manager type: {type(graph_rag_manager)}")
        print(f"📊 GraphRAG Manager attributes:")
        
        # List all methods and attributes
        all_attrs = dir(graph_rag_manager)
        methods = [attr for attr in all_attrs if not attr.startswith('_') and callable(getattr(graph_rag_manager, attr, None))]
        properties = [attr for attr in all_attrs if not attr.startswith('_') and not callable(getattr(graph_rag_manager, attr, None))]
        
        print(f"🔧 Methods: {methods}")
        print(f"📦 Properties: {properties}")
        
        # Test specific methods
        test_question = "What is this document about?"
        
        # Test Method 1: query
        if hasattr(graph_rag_manager, 'query'):
            try:
                print(f"\n🧪 Testing query method...")
                result = graph_rag_manager.query(test_question)
                print(f"✅ Query result type: {type(result)}")
                print(f"✅ Query result preview: {str(result)[:200]}...")
            except Exception as e:
                print(f"❌ Query method error: {e}")
        
        # Test Method 2: retrieve_context
        if hasattr(graph_rag_manager, 'retrieve_context'):
            try:
                print(f"\n🧪 Testing retrieve_context method...")
                result = graph_rag_manager.retrieve_context(test_question)
                print(f"✅ retrieve_context result type: {type(result)}")
                print(f"✅ retrieve_context result preview: {str(result)[:200]}...")
            except Exception as e:
                print(f"❌ retrieve_context method error: {e}")
        
        # Test Method 3: similarity_search
        if hasattr(graph_rag_manager, 'similarity_search'):
            try:
                print(f"\n🧪 Testing similarity_search method...")
                result = graph_rag_manager.similarity_search(test_question, k=3)
                print(f"✅ similarity_search result type: {type(result)}")
                print(f"✅ similarity_search result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            except Exception as e:
                print(f"❌ similarity_search method error: {e}")
        
        # Check for vector_store
        if hasattr(graph_rag_manager, 'vector_store'):
            print(f"\n📊 Vector store found: {type(graph_rag_manager.vector_store)}")
            if hasattr(graph_rag_manager.vector_store, 'similarity_search'):
                try:
                    result = graph_rag_manager.vector_store.similarity_search(test_question, k=3)
                    print(f"✅ Vector store similarity_search works: {len(result)} results")
                except Exception as e:
                    print(f"❌ Vector store similarity_search error: {e}")
        
        # Check for knowledge_graph_store
        if hasattr(graph_rag_manager, 'knowledge_graph_store'):
            print(f"\n🧠 Knowledge graph store found: {type(graph_rag_manager.knowledge_graph_store)}")
        
        print("\n✅ GraphRAG manager analysis complete!")
        
    except Exception as e:
        print(f"❌ Error analyzing GraphRAG manager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_graphrag_manager()