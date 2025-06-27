# rag_fixed.py
import requests
import json

def generate_answer_ollama(question, vector_store, ollama_base_url, model_name):
    """Generate answer using Ollama with better error handling"""
    
    # Debug information
    print(f"🔍 Starting RAG query...")
    print(f"📝 Question: {question}")
    print(f"🗄️ Vector store type: {type(vector_store)}")
    print(f"🌐 Ollama URL: {ollama_base_url}")
    print(f"🤖 Model: {model_name}")
    
    try:
        # Validate vector store
        if not vector_store:
            raise ValueError("Vector store is None")
        
        if not hasattr(vector_store, 'similarity_search'):
            raise ValueError(f"Vector store of type {type(vector_store)} doesn't have similarity_search method")
        
        # Perform similarity search
        print("🔍 Performing similarity search...")
        relevant_docs = vector_store.similarity_search(question, k=5)
        print(f"📄 Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            print("⚠️ No relevant documents found")
            return "I couldn't find any relevant information in the documents to answer your question."
        
        # Prepare context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"📝 Context length: {len(context)} characters")
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer: """
        
        print("🤖 Sending request to Ollama...")
        
        # Send request to Ollama
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024
                }
            },
            timeout=120
        )
        
        print(f"📡 Ollama response status: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        response_data = response.json()
        answer = response_data.get("response", "No response generated")
        
        print(f"✅ Generated answer length: {len(answer)} characters")
        return answer
        
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Ollama. Make sure Ollama is running."
        print(f"❌ {error_msg}")
        return f"Connection Error: {error_msg}"
    
    except requests.exceptions.Timeout:
        error_msg = "Ollama request timed out."
        print(f"❌ {error_msg}")
        return f"Timeout Error: {error_msg}"
    
    except ValueError as ve:
        error_msg = f"Vector store error: {str(ve)}"
        print(f"❌ {error_msg}")
        return f"Vector Store Error: {error_msg}"
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return f"Error: {error_msg}"