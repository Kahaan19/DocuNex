import os
from neo4j import GraphDatabase
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
rag_manager = None 

# Load environment variables
load_dotenv()

# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

neo4j_driver = None
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Test the connection
    with neo4j_driver.session() as session:
        session.run("RETURN 1")
    print("✅ Connected to Neo4j successfully")
except Exception as e:
    print(f"⚠️ Could not connect to Neo4j: {e}")
    neo4j_driver = None

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma:2b"

BASE_PERSIST_DIR = "./graph_db"
BASE_VECTOR_DIR = "./vector_db"

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def initialize_embeddings(self) -> bool:
    """Initialize the embedding model"""
    try:
        print("🔄 Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        print("✅ Embedding model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading embedding model: {str(e)}")
        return False

def clear_neo4j_database():
    """Clear all nodes and relationships in Neo4j."""
    if not neo4j_driver:
        return False
    try:
        with neo4j_driver.session() as session:
            # Clear in batches to avoid memory issues
            session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n")
            # Check if there are more nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            if count > 0:
                print(f"⚠️ Still {count} nodes remaining - may need multiple clears")
        print("✅ Cleared Neo4j database")
        return True
    except Exception as e:
        print(f"❌ Error clearing Neo4j: {e}")
        return False

def sanitize_rel_type(rel_type):
    """Sanitize relationship type for Neo4j"""
    if not rel_type:
        return "RELATED_TO"
    
    # Replace non-alphanumeric characters with underscores and uppercase
    rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', str(rel_type)).upper()
    
    # Remove multiple consecutive underscores
    rel_type = re.sub(r'_+', '_', rel_type)
    
    # Remove leading/trailing underscores
    rel_type = rel_type.strip('_')
    
    # Neo4j relationship types cannot start with a number or be empty
    if not rel_type or rel_type[0].isdigit():
        rel_type = "RELATED_TO"
    
    # Ensure it's not too long
    if len(rel_type) > 50:
        rel_type = rel_type[:50]
    
    return rel_type

def sanitize_node_label(label):
    """Sanitize node label for Neo4j"""
    if not label:
        return "Entity"
    
    # Replace non-alphanumeric characters with underscores
    label = re.sub(r'[^a-zA-Z0-9_]', '_', str(label))
    
    # Remove multiple consecutive underscores
    label = re.sub(r'_+', '_', label)
    
    # Remove leading/trailing underscores
    label = label.strip('_')
    
    # Ensure it starts with a letter
    if not label or label[0].isdigit():
        label = "Entity_" + label if label else "Entity"
    
    # Ensure it's not too long
    if len(label) > 50:
        label = label[:50]
    
    return label

def insert_graph_to_neo4j(nodes, relationships):
    """Insert knowledge graph data into Neo4j with better error handling."""
    if not neo4j_driver:
        print("❌ Neo4j not connected.")
        return False

    try:
        with neo4j_driver.session() as session:
            # Create nodes with better error handling
            nodes_created = 0
            for node in nodes:
                try:
                    node_id = str(node.get("id", f"node_{uuid.uuid4().hex[:8]}"))
                    node_label = sanitize_node_label(node.get("label", "Entity"))
                    node_props = node.get("properties", {})
                    
                    # Ensure node_props are serializable
                    clean_props = {}
                    for key, value in node_props.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_props[key] = value
                        else:
                            clean_props[key] = str(value)

                    session.run(
                        f"MERGE (n:{node_label} {{id: $id}}) SET n += $props",
                        id=node_id, props=clean_props
                    )
                    nodes_created += 1
                except Exception as e:
                    print(f"⚠️ Error creating node {node.get('id', 'unknown')}: {e}")
                    continue

            # Create relationships with better error handling
            relationships_created = 0
            for rel in relationships:
                try:
                    source_id = str(rel.get("source", ""))
                    target_id = str(rel.get("target", ""))
                    rel_type = sanitize_rel_type(rel.get("type", "RELATED_TO"))
                    rel_props = rel.get("properties", {})
                    
                    if not source_id or not target_id:
                        continue
                    
                    # Ensure rel_props are serializable
                    clean_props = {}
                    for key, value in rel_props.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_props[key] = value
                        else:
                            clean_props[key] = str(value)

                    session.run(
                        f"""
                        MATCH (a {{id: $source}}), (b {{id: $target}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += $props
                        """,
                        source=source_id, target=target_id, props=clean_props
                    )
                    relationships_created += 1
                except Exception as e:
                    print(f"⚠️ Error creating relationship {rel.get('source', 'unknown')} -> {rel.get('target', 'unknown')}: {e}")
                    continue

        print(f"✅ Successfully inserted {nodes_created} nodes and {relationships_created} relationships into Neo4j")
        return True
    except Exception as e:
        print(f"❌ Error inserting graph into Neo4j: {e}")
        traceback.print_exc()
        return False

def extract_graph_data_from_manager():
    """Extract nodes and relationships from GraphRAGManager for Neo4j insertion."""
    global graph_rag_manager
    
    if not graph_rag_manager:
        print("❌ GraphRAGManager not initialized")
        return [], []
    
    try:
        print("🔍 Attempting to extract graph data from GraphRAGManager...")
        
        # Method 1: Direct graph data extraction
        if hasattr(graph_rag_manager, 'get_graph_data'):
            print("📊 Using get_graph_data method")
            graph_data = graph_rag_manager.get_graph_data()
            nodes = graph_data.get('nodes', [])
            relationships = graph_data.get('relationships', [])
            print(f"📊 Extracted {len(nodes)} nodes and {len(relationships)} relationships")
            return nodes, relationships
        
        # Method 2: Separate node and relationship methods
        elif hasattr(graph_rag_manager, 'get_nodes') and hasattr(graph_rag_manager, 'get_relationships'):
            print("📊 Using separate get_nodes and get_relationships methods")
            nodes = graph_rag_manager.get_nodes()
            relationships = graph_rag_manager.get_relationships()
            print(f"📊 Extracted {len(nodes)} nodes and {len(relationships)} relationships")
            return nodes, relationships
        
        # Method 3: Access knowledge graph store directly
        elif hasattr(graph_rag_manager, 'knowledge_graph_store'):
            print("📊 Accessing knowledge_graph_store directly")
            kg_store = graph_rag_manager.knowledge_graph_store
            
            # Try to get data from the store
            if hasattr(kg_store, 'get_triplets'):
                triplets = kg_store.get_triplets()
                print(f"📊 Found {len(triplets)} triplets")
                
                # Convert triplets to nodes and relationships
                nodes = []
                relationships = []
                node_ids = set()
                
                for triplet in triplets:
                    if len(triplet) >= 3:
                        source, relation, target = triplet[0], triplet[1], triplet[2]
                        
                        # Add source node
                        if source not in node_ids:
                            nodes.append({
                                "id": source,
                                "label": "Entity",
                                "properties": {"name": source}
                            })
                            node_ids.add(source)
                        
                        # Add target node
                        if target not in node_ids:
                            nodes.append({
                                "id": target,
                                "label": "Entity", 
                                "properties": {"name": target}
                            })
                            node_ids.add(target)
                        
                        # Add relationship
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": relation,
                            "properties": {}
                        })
                
                print(f"📊 Converted to {len(nodes)} nodes and {len(relationships)} relationships")
                return nodes, relationships
            
            else:
                print("⚠️ knowledge_graph_store doesn't have get_triplets method")
                print(f"Available methods: {[method for method in dir(kg_store) if not method.startswith('_')]}")
        
        # Method 4: Try to access internal data structures
        else:
            print("⚠️ No standard graph extraction methods found")
            print(f"Available methods: {[method for method in dir(graph_rag_manager) if not method.startswith('_')]}")
            
            # Try to find any data structures that might contain graph data
            for attr_name in dir(graph_rag_manager):
                if not attr_name.startswith('_'):
                    attr = getattr(graph_rag_manager, attr_name)
                    if hasattr(attr, '__len__') and not callable(attr):
                        try:
                            print(f"📊 Found attribute '{attr_name}' with length {len(attr)}")
                        except:
                            pass
        
        print("⚠️ Could not extract graph data using any known method")
        return [], []
        
    except Exception as e:
        print(f"❌ Error extracting graph data: {e}")
        traceback.print_exc()
        return [], []

def check_ollama_connection():
    """Check if Ollama is running and model is available"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            if OLLAMA_MODEL in model_names or any(OLLAMA_MODEL in name for name in model_names):
                return True, f"✅ Ollama connected. Available models: {', '.join(model_names[:3])}"
            else:
                return False, f"❌ Model '{OLLAMA_MODEL}' not found. Available: {', '.join(model_names[:3])}\nRun: ollama pull {OLLAMA_MODEL}"
        return False, "❌ Ollama not responding"
    except requests.exceptions.ConnectionError:
        return False, f"❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}\nMake sure Ollama is running: ollama serve"
    except requests.exceptions.Timeout:
        return False, "❌ Ollama connection timeout"
    except Exception as e:
        return False, f"❌ Ollama connection failed: {str(e)}"

def cleanup_old_databases():
    """Properly cleanup old databases and their directories"""
    global graph_rag_manager, current_persist_dir, current_vector_dir
    
    # Cleanup GraphRAG
    if graph_rag_manager is not None:
        try:
            if hasattr(graph_rag_manager, 'close'):
                graph_rag_manager.close()
            del graph_rag_manager
        except Exception as e:
            print(f"Warning: Error closing graph manager: {e}")
        graph_rag_manager = None
    
        
    # Cleanup GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)
    
    # Remove old directories
    for directory in [current_persist_dir, current_vector_dir]:
        if directory and os.path.exists(directory):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(directory)
                    print(f"✅ Cleaned up old database: {directory}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt + 1}/{max_retries}: Waiting for file handles to release...")
                        time.sleep(1)
                        gc.collect()
                    else:
                        print(f"⚠️ Warning: Could not remove old directory {directory}: {e}")

def handle_inputs(urls, files, rag_type, progress=gr.Progress()):
    global graph_rag_manager, current_persist_dir, current_vector_dir, current_rag_type
    
    if not MODULES_AVAILABLE:
        return "❌ Required modules not found. Please ensure GraphRAGManager is available."
    
    all_docs = []

    # Check Ollama connection first
    ollama_status, ollama_msg = check_ollama_connection()
    if not ollama_status:
        return f"🔌 {ollama_msg}"

    progress(0.1, desc="Processing inputs...")

    try:
        # Process URLs
        if urls:
            if 'url_text_extractor' in missing_modules:
                return "❌ URL text extractor module not found. Cannot process URLs."
            
            url_list = [u.strip() for u in urls.split(",") if u.strip()]
            for i, url in enumerate(url_list):
                try:
                    progress(0.1 + (0.2 * i / len(url_list)), desc=f"Extracting from URL {i+1}/{len(url_list)}")
                    text = extract_text_from_url(url)
                    all_docs.append(Document(page_content=text, metadata={"source": url}))
                except Exception as e:
                    print(f"Error extracting from {url}: {e}")
                    return f"❌ Error extracting from URL {url}: {str(e)}"

        # Process files
        if files:
            if 'document_loader' in missing_modules:
                return "❌ Document loader module not found. Cannot process files."
            
            file_paths = [file.name for file in files]
            progress(0.3, desc="Loading document files...")
            try:
                docs_from_files = load_documents_from_files(file_paths)
                all_docs.extend(docs_from_files)
            except Exception as e:
                print(f"Error loading files: {e}")
                return f"❌ Error loading files: {str(e)}"

        if not all_docs:
            return "❗ No valid input provided. Please upload files or enter URLs."

        progress(0.4, desc="Cleaning up old databases...")
        cleanup_old_databases()

        # Store current RAG type
        current_rag_type = rag_type

        if rag_type == "GraphRAG":
            # Build Knowledge Graph
            progress(0.5, desc="Initializing Graph RAG Manager...")
            
            # Clear Neo4j database for fresh start
            if neo4j_driver:
                clear_neo4j_database()

            # Create a new unique persist directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            new_persist_dir = f"{BASE_PERSIST_DIR}_{timestamp}_{uuid.uuid4().hex[:6]}"

            try:
                graph_rag_manager = GraphRAGManager(
                    persist_dir=new_persist_dir,
                    api_key=None,
                    device=device
                )
                
                progress(0.7, desc="Building knowledge graph... (This may take a while)")
                status_msg = graph_rag_manager.build_knowledge_graph(all_docs)
                current_persist_dir = new_persist_dir
                
                print(f"GraphRAG build status: {status_msg}")
                
            except Exception as e:
                error_msg = f"❌ Error building GraphRAG: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                return error_msg
            
            # Insert graph into Neo4j
            progress(0.9, desc="Inserting graph into Neo4j...")
            neo4j_status = ""
            if neo4j_driver:
                try:
                    nodes, relationships = extract_graph_data_from_manager()
                    if nodes and relationships:
                        success = insert_graph_to_neo4j(nodes, relationships)
                        if success:
                            neo4j_status = f"\n✅ Knowledge graph inserted into Neo4j ({len(nodes)} nodes, {len(relationships)} relationships)"
                        else:
                            neo4j_status = "\n⚠️ Failed to insert graph into Neo4j"
                    else:
                        neo4j_status = "\n⚠️ No graph data extracted for Neo4j insertion"
                except Exception as e:
                    neo4j_status = f"\n❌ Error inserting into Neo4j: {str(e)}"
                    print(f"Neo4j insertion error: {e}")
                    traceback.print_exc()
            else:
                neo4j_status = "\n⚠️ Neo4j not connected - graph not stored in Neo4j"
            
            progress(1.0, desc="Complete!")
            return f"✅ GraphRAG Knowledge Graph built successfully!\n{ollama_msg}\n{status_msg}{neo4j_status}\nUsing folder: {current_persist_dir}"
        
        else:  # Regular RAG
            progress(0.5, desc="Creating vector store...")

            global rag_manager, current_vector_dir
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            current_vector_dir = f"{BASE_VECTOR_DIR}_{timestamp}_{uuid.uuid4().hex[:6]}"
            rag_manager = RAGManager(vector_store_path=current_vector_dir)
            success = rag_manager.create_vector_store([{"content": doc.page_content, "source": doc.metadata.get("source", "uploaded")} for doc in all_docs])
            if not success:
                return "❌ Failed to create vector store"
            
            progress(1.0, desc="Complete!")
            return f"✅ RAG Vector Store created successfully!\n{ollama_msg}\nProcessed {len(all_docs)} documents\nUsing folder: {current_vector_dir}"
    
    except Exception as e:
        error_msg = f"❌ Error creating {rag_type}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg
    
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
# NEW STREAMING FUNCTIONS
def stream_rag_answer(question):
    global current_rag_type
    yield "🔍 Retrieving relevant context...\n\n"
    
    if current_rag_type == "GraphRAG":
        context = get_graph_rag_context(question)
    else:
        context = get_rag_context(question)
    
    # Create the prompt here
    prompt = f"""Based on the following context, please answer the question comprehensively and accurately.
Context:
{context}
Question: {question}
Answer:"""
    
    # Send initial message once
    yield "🔍 Context retrieved, generating answer...\n\n**Answer:**\n"
    
    # Then stream only the answer part
    async def async_stream():
        callback = AsyncIteratorCallbackHandler()
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            streaming=True,
            callbacks=[callback]
        )
        task = asyncio.create_task(llm.ainvoke(prompt))
        answer_so_far = ""
        async for chunk in callback.aiter():
            answer_so_far += chunk 
            yield f"🔍 Context retrieved, generating answer...\n\n**Answer:**\n{answer_so_far}"
        await task

    # Rest of your existing run_async_gen code...
    # Helper to run async generator in sync context
    def run_async_gen(async_gen):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agen = async_gen()
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(agen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
                loop.close()

    for chunk in run_async_gen(async_stream):
        yield chunk

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


def on_ask_streaming(question):
    """Handle streaming question answering"""
    global graph_rag_manager, current_persist_dir, current_vector_dir, current_rag_type, rag_manager
    
    if not MODULES_AVAILABLE:
        yield "❌ Required modules not found. Please ensure all custom modules are available."
        return
    
    if not current_rag_type:
        yield "⚠️ Please upload documents first to build the knowledge base."
        return
    
    ollama_status, ollama_msg = check_ollama_connection()
    if not ollama_status:
        yield f"🔌 {ollama_msg}"
        return
    
    try:
        if current_rag_type == "GraphRAG":
            # Use GraphRAG
            if not graph_rag_manager:
                if current_persist_dir and os.path.exists(current_persist_dir):
                    try:
                        yield "🔄 Loading GraphRAG knowledge base...\n\n"
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
                        yield error_msg
                        return
                else:
                    yield "⚠️ Please upload content first to build the GraphRAG knowledge graph."
                    return
            
            # Stream GraphRAG answer
            try:
                for response in stream_rag_answer(question):
                    yield response
                    
            except Exception as e:
                error_msg = f"❌ Error generating GraphRAG answer: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                yield error_msg
        
        else:  # Regular RAG
            # Check if vector store is loaded
            try:
                if not rag_manager or not rag_manager.is_ready():
                    if current_vector_dir and os.path.exists(current_vector_dir):
                        yield "🔄 Loading RAG vector store...\n\n"
                        rag_manager = RAGManager(vector_store_path=current_vector_dir)
                        if not rag_manager.load_vector_store():
                            yield "❌ Failed to load vector store"
                            return
                    else:
                        yield "⚠️ Please upload content first to build the RAG vector store."
                        return
                        
                # Stream RAG answer
                for response in stream_rag_answer(question):
                    yield response
                    
            except Exception as e:
                error_msg = f"❌ Error generating RAG answer: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                yield error_msg
    
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        yield error_msg

def on_ask(question):
    """Handle question answering with permanent streaming"""
    global graph_rag_manager, current_persist_dir, current_vector_dir, current_rag_type, rag_manager
    
    if not MODULES_AVAILABLE:
        yield "❌ Required modules not found. Please ensure all custom modules are available."
        return
    
    if not current_rag_type:
        yield "⚠️ Please upload documents first to build the knowledge base."
        return
    
    ollama_status, ollama_msg = check_ollama_connection()
    if not ollama_status:
        yield f"🔌 {ollama_msg}"
        return
    
    try:
        if current_rag_type == "GraphRAG":
            # Use GraphRAG
            if not graph_rag_manager:
                if current_persist_dir and os.path.exists(current_persist_dir):
                    try:
                        yield "🔄 Loading GraphRAG knowledge base...\n\n"
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
                        yield error_msg
                        return
                else:
                    yield "⚠️ Please upload content first to build the GraphRAG knowledge graph."
                    return
            
            # Stream GraphRAG answer
            try:
                for response in stream_rag_answer(question):
                    yield response
                    
            except Exception as e:
                error_msg = f"❌ Error generating GraphRAG answer: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                yield error_msg
        
        else:  # Regular RAG
            # Check if vector store is loaded
            try:
                if not rag_manager or not rag_manager.is_ready():
                    if current_vector_dir and os.path.exists(current_vector_dir):
                        yield "🔄 Loading RAG vector store...\n\n"
                        rag_manager = RAGManager(vector_store_path=current_vector_dir)
                        if not rag_manager.load_vector_store():
                            yield "❌ Failed to load vector store"
                            return
                    else:
                        yield "⚠️ Please upload content first to build the RAG vector store."
                        return
                        
                # Stream RAG answer
                for response in stream_rag_answer(question):
                    yield response
                    
            except Exception as e:
                error_msg = f"❌ Error generating RAG answer: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                yield error_msg
    
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        yield error_msg

def get_graph_stats():
    """Get statistics about the current graph"""
    global graph_rag_manager
    
    if not graph_rag_manager:
        return "No GraphRAG knowledge base loaded."
    
    try:
        stats = []
        
        # Try to get node count
        if hasattr(graph_rag_manager, 'get_graph_data'):
            graph_data = graph_rag_manager.get_graph_data()
            nodes = graph_data.get('nodes', [])
            relationships = graph_data.get('relationships', [])
            stats.append(f"📊 Nodes: {len(nodes)}")
            stats.append(f"📊 Relationships: {len(relationships)}")
        
        # Try to get triplet count
        elif hasattr(graph_rag_manager, 'knowledge_graph_store'):
            kg_store = graph_rag_manager.knowledge_graph_store
            if hasattr(kg_store, 'get_triplets'):
                triplets = kg_store.get_triplets()
                stats.append(f"📊 Triplets: {len(triplets)}")
        
        # Get Neo4j stats
        if neo4j_driver:
            try:
                with neo4j_driver.session() as session:
                    node_result = session.run("MATCH (n) RETURN count(n) as count")
                    node_count = node_result.single()["count"]
                    
                    rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                    rel_count = rel_result.single()["count"]
                    
                    stats.append(f"🗄️ Neo4j Nodes: {node_count}")
                    stats.append(f"🗄️ Neo4j Relationships: {rel_count}")
            except Exception as e:
                stats.append(f"⚠️ Neo4j stats error: {str(e)}")
        
        return "\n".join(stats) if stats else "No graph statistics available."
        
    except Exception as e:
        return f"❌ Error getting graph stats: {str(e)}"

def get_vector_stats():
    """Get statistics about the current vector store"""
    global rag_manager
    
    if not rag_manager or not rag_manager.is_ready():
        return "No RAG vector store loaded."
    
    try:
        # Get vector store statistics
        vector_store = rag_manager.vector_store
        if hasattr(vector_store, 'index'):
            total_docs = vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else "Unknown"
        else:
            total_docs = "Unknown"
        
        return f"📊 Vector Store Documents: {total_docs}"
        
    except Exception as e:
        return f"❌ Error getting vector stats: {str(e)}"

def get_current_status():
    """Get current system status"""
    global current_rag_type, current_persist_dir, current_vector_dir
    
    status_lines = []
    
    # Ollama status
    ollama_status, ollama_msg = check_ollama_connection()
    status_lines.append(f"🤖 Ollama: {'✅ Connected' if ollama_status else '❌ Not Connected'}")
    
    # Neo4j status
    neo4j_status = "✅ Connected" if neo4j_driver else "❌ Not Connected"
    status_lines.append(f"🗄️ Neo4j: {neo4j_status}")
    
    # RAG Type
    if current_rag_type:
        status_lines.append(f"🧠 RAG Type: {current_rag_type}")
        
        if current_rag_type == "GraphRAG":
            status_lines.append(f"📁 GraphRAG Dir: {current_persist_dir}")
            if graph_rag_manager:
                status_lines.append("📊 GraphRAG: ✅ Loaded")
                stats = get_graph_stats()
                if stats and "No GraphRAG" not in stats:
                    status_lines.append(stats)
            else:
                status_lines.append("📊 GraphRAG: ⚠️ Not Loaded")
        else:
            status_lines.append(f"📁 Vector Dir: {current_vector_dir}")
            if rag_manager and rag_manager.is_ready():
                status_lines.append("📊 Vector Store: ✅ Loaded") 
                stats = get_vector_stats()
                if stats and "No RAG" not in stats:
                    status_lines.append(stats)
            else:
                status_lines.append("📊 Vector Store: ⚠️ Not Loaded")
    else:
        status_lines.append("🧠 RAG Type: None (Upload documents first)")
    
    # Missing modules
    if missing_modules:
        status_lines.append(f"⚠️ Missing Modules: {', '.join(missing_modules)}")
    
    return "\n".join(status_lines)

# Initialize embeddings at startup

# Create Gradio interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="RAG & Graph RAG System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 RAG & Graph RAG System")
        gr.Markdown("Upload documents or enter URLs to build a knowledge base, then ask questions!")
        
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
                    question_input = gr.Textbox(
                        label="❓ Your Question",
                        placeholder="Ask anything about your uploaded documents...",
                        lines=3
                    )
                    ask_btn = gr.Button("🔍 Ask Question", variant="primary")
                    
                with gr.Column(scale=1):
                    status_btn = gr.Button("📊 Check Status")
                    status_output = gr.Textbox(
                        label="🔄 System Status",
                        lines=8,
                        max_lines=12
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
            - Status: {'✅ Connected' if neo4j_driver else '❌ Not Connected'}
            
            **📁 Storage Directories:**
            - Graph DB Base: `{BASE_PERSIST_DIR}`
            - Vector DB Base: `{BASE_VECTOR_DIR}`
            
            **🧩 Available Modules:**
            - Missing: `{', '.join(missing_modules) if missing_modules else 'None'}`
            - GraphRAG Available: `{'✅ Yes' if MODULES_AVAILABLE else '❌ No'}`
            """
            
            gr.Markdown(info_text)
        
        # Event handlers
        build_btn.click(
            fn=handle_inputs,
            inputs=[urls_input, files_input, rag_type],
            outputs=build_output,
            show_progress=True
        )
        
        ask_btn.click(
            fn=on_ask,
            inputs=question_input,
            outputs=answer_output,
            show_progress=False
        )
        
        status_btn.click(
            fn=get_current_status,
            outputs=status_output
        )
        
        question_input.submit(
            fn=on_ask,
            inputs=question_input,
            outputs=answer_output,
            show_progress=False
        )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    print(f"🚀 Starting DocuNex with GPU: {device}")
    
    # Print module availability status
    if missing_modules:
        print(f"⚠️ Missing modules: {', '.join(missing_modules)}")
        print("Some functionality may be limited.")
    else:
        print("✅ All modules loaded successfully")
    
    demo = create_interface()
    
    try:
        print("🌐 Launching Gradio interface...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            inbrowser=True
        )
    except Exception as e:
        print(f"❌ Error launching Gradio: {e}")
        # Try alternative port
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=7861,
                share=False,
                inbrowser=True
            )
        except Exception as e2:
            print(f"❌ Alternative launch failed: {e2}")