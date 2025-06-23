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
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from rag_manager import RAGManager
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

def initialize_embeddings():
    """Initialize embeddings for vector store"""
    global embeddings
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
            print("✅ Embeddings initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing embeddings: {e}")
            embeddings = None
    return embeddings

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

'''def create_vector_store(documents):
    """Create and save vector store from documents"""
    global vector_store, current_vector_dir, embeddings
    
    try:
        # Initialize embeddings if not already done
        if not initialize_embeddings():
            raise Exception("Failed to initialize embeddings")
        
        # Split documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        print(f"📄 Creating vector store with {len(all_chunks)} chunks...")
        
        # Create timestamp for unique directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        current_vector_dir = f"{BASE_VECTOR_DIR}_{timestamp}_{uuid.uuid4().hex[:6]}"
        
        # Create vector store
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        
        # Save vector store
        os.makedirs(current_vector_dir, exist_ok=True)
        vector_store.save_local(current_vector_dir)
        
        print(f"✅ Vector store created and saved to {current_vector_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        traceback.print_exc()
        return False'''
'''
def load_vector_store():
    """Load existing vector store"""
    global vector_store, embeddings
    
    try:
        if not current_vector_dir or not os.path.exists(current_vector_dir):
            print(f"❌ Vector store directory not found: {current_vector_dir}")
            return False
            
        if not initialize_embeddings():
            print("❌ Failed to initialize embeddings")
            return False
        
        # Check if the vector store files exist
        index_file = os.path.join(current_vector_dir, "index.faiss")
        pkl_file = os.path.join(current_vector_dir, "index.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            print(f"❌ Vector store files missing in {current_vector_dir}")
            return False
            
        vector_store = FAISS.load_local(
            current_vector_dir, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Vector store loaded from {current_vector_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print(f"Current vector dir: {current_vector_dir}")
        traceback.print_exc()
        return False
'''
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
            from rag_manager import RAGManager  # Make sure this import is at the top

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

def on_ask(question):
    global graph_rag_manager, current_persist_dir, current_vector_dir, current_rag_type, rag_manager
    
    if not MODULES_AVAILABLE:
        return "❌ Required modules not found. Please ensure all custom modules are available."
    
    if not current_rag_type:
        return "⚠️ Please upload documents first to build the knowledge base."
    
    ollama_status, ollama_msg = check_ollama_connection()
    if not ollama_status:
        return f"🔌 {ollama_msg}"
    
    try:
        if current_rag_type == "GraphRAG":
            # Use GraphRAG
            if not graph_rag_manager:
                if current_persist_dir and os.path.exists(current_persist_dir):
                    try:
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
                        return error_msg
                else:
                    return "⚠️ Please upload content first to build the GraphRAG knowledge graph."
            
            # Check if rag_chain module is available
            if 'rag_chain' in missing_modules:
                return "❌ rag_chain module not found. Cannot use GraphRAG functionality."
            
            try:
                print(f"🧠 Generating GraphRAG answer for: {question[:50]}...")
                
                # Method 1: Try with GraphRAGManager directly
                try:
                    return generate_graph_answer_ollama(question, graph_rag_manager, OLLAMA_BASE_URL, OLLAMA_MODEL)
                except Exception as graph_error:
                    print(f"❌ GraphRAG method failed: {str(graph_error)}")
                    
                    # Method 2: Try to extract RAG manager from GraphRAGManager
                    if hasattr(graph_rag_manager, 'rag_manager'):
                        print("🔄 Trying with internal RAG manager...")
                        return generate_graph_answer_ollama(question, graph_rag_manager.rag_manager, OLLAMA_BASE_URL, OLLAMA_MODEL)
                    
                    # Method 3: Try to use query method directly
                    elif hasattr(graph_rag_manager, 'query'):
                        print("🔄 Using direct query method...")
                        return graph_rag_manager.query(question)
                    
                    # Method 4: Fallback to manual implementation
                    else:
                        print("🔄 Using fallback implementation...")
                        return fallback_graph_query(question, graph_rag_manager)
                        
            except Exception as e:
                error_msg = f"❌ Error generating GraphRAG answer: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                return error_msg
        
        else:  # Regular RAG
            # Check if vector store is loaded
            try:
                if not rag_manager or not rag_manager.is_ready():
                    if current_vector_dir and os.path.exists(current_vector_dir):
                        rag_manager = RAGManager(vector_store_path=current_vector_dir)
                        if not rag_manager.load_vector_store():
                            return "❌ Failed to load vector store"
                    else:
                        return "⚠️ Please upload content first to build the RAG vector store."
                return generate_answer_ollama(question, rag_manager, OLLAMA_BASE_URL, OLLAMA_MODEL)
            except Exception as rag_error:
                error_msg = f"❌ Error generating RAG answer: {str(rag_error)}"
                print(error_msg)
                traceback.print_exc()
                return error_msg
    
    except Exception as e:
        error_msg = f"❌ Error processing question: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def fallback_graph_query(question, graph_rag_manager):
    """
    Fallback implementation for GraphRAG querying when standard methods fail
    """
    try:
        print("🔧 Using fallback GraphRAG implementation...")
        
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
                
                for triplet in triplets[:50]:  # Limit to avoid too much context
                    triplet_text = ' '.join([str(x) for x in triplet]).lower()
                    if any(word in triplet_text for word in question_words):
                        relevant_triplets.append(triplet)
                
                if relevant_triplets:
                    context = "Relevant knowledge from graph:\n"
                    for triplet in relevant_triplets[:10]:  # Top 10 most relevant
                        context += f"- {triplet[0]} -> {triplet[1]} -> {triplet[2]}\n"
        
        # Method 2: Try to get documents or chunks
        if not context and hasattr(graph_rag_manager, 'vector_store'):
            print("🔄 Trying vector store fallback...")
            vector_store = graph_rag_manager.vector_store
            if hasattr(vector_store, 'similarity_search'):
                docs = vector_store.similarity_search(question, k=3)
                context = "Relevant documents:\n"
                for doc in docs:
                    context += f"- {doc.page_content[:300]}...\n"
        
        # If we have context, generate answer using Ollama
        if context:
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
            
            # Call Ollama API directly
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"❌ Error calling Ollama: {response.status_code}"
        
        # If no context found, return a generic response
        return "⚠️ No relevant information found in the knowledge graph. Please ensure the documents contain information related to your question."
        
    except Exception as e:
        print(f"❌ Fallback query failed: {str(e)}")
        traceback.print_exc()
        return f"❌ Error in fallback query: {str(e)}"

# Add debugging function to inspect GraphRAGManager
def debug_graph_rag_manager():
    """Debug function to inspect GraphRAGManager structure"""
    global graph_rag_manager
    
    if not graph_rag_manager:
        print("❌ GraphRAGManager is None")
        return
    
    print(f"📊 GraphRAGManager type: {type(graph_rag_manager)}")
    print(f"📊 GraphRAGManager attributes: {[attr for attr in dir(graph_rag_manager) if not attr.startswith('_')]}")
    
    # Check for common attributes
    attrs_to_check = ['rag_manager', 'vector_store', 'knowledge_graph_store', 'query', 'get_relevant_context']
    for attr in attrs_to_check:
        if hasattr(graph_rag_manager, attr):
            attr_value = getattr(graph_rag_manager, attr)
            print(f"✅ Has {attr}: {type(attr_value)}")
        else:
            print(f"❌ Missing {attr}")

# Alternative implementation if the above doesn't work
def alternative_graph_query(question):
    """Alternative implementation that works directly with available data"""
    global graph_rag_manager
    
    try:
        # Debug the manager first
        debug_graph_rag_manager()
        
        # Try different approaches based on what's available
        if hasattr(graph_rag_manager, 'retriever'):
            print("🔄 Using retriever approach...")
            docs = graph_rag_manager.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs[:3]])
        
        elif hasattr(graph_rag_manager, 'vector_store'):
            print("🔄 Using vector store approach...")
            docs = graph_rag_manager.vector_store.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
        
        else:
            return "❌ No compatible query method found in GraphRAGManager"
        
        # Generate answer using context
        if context:
            prompt = f"""Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer:"""
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"❌ Ollama API error: {response.status_code}"
        
        return "❌ No context retrieved for the question"
        
    except Exception as e:
        print(f"❌ Alternative query failed: {str(e)}")
        traceback.print_exc()
        return f"❌ Error in alternative query: {str(e)}"

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="DocuNex", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🦙🧠 DocuNex")
        gr.Markdown("Upload large documents and ask questions using **Ollama + RAG/GraphRAG** with **Neo4j Knowledge Graph Integration**!")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔥 **Features:**")
                gr.Markdown("- **🦙 Ollama Integration** (No API limits)\n- **🧠 GraphRAG + Regular RAG** Options\n- **📊 Neo4j Graph Storage** (Use Neo4j Browser to visualize)\n- **🎮 GPU Acceleration**\n- **📚 Large Document** Support")
            with gr.Column():
                # Check status on startup
                ollama_status, ollama_msg = check_ollama_connection()
                ollama_color = "green" if ollama_status else "red"
                
                neo4j_status = "✅ Connected" if neo4j_driver else "❌ Not Connected"
                neo4j_color = "green" if neo4j_driver else "red"
                
                gr.Markdown(f"### 🔌 **System Status:**")
                gr.Markdown(f"**Ollama:** <span style='color: {ollama_color}'>{ollama_msg}</span>")
                gr.Markdown(f"**Neo4j:** <span style='color: {neo4j_color}'>{neo4j_status}</span>")
                if neo4j_driver:
                    gr.Markdown("**📊 View Graph:** Open Neo4j Browser at http://localhost:7474")

        with gr.Group():
            gr.Markdown("### 📁 **Upload Your Documents & Choose RAG Type**")
            
            # Single RAG Type Selection
            rag_type = gr.Radio(
                choices=["RAG", "GraphRAG"],
                value="GraphRAG",
                label="🤖 Choose RAG Type",
                info="RAG: Fast vector-based retrieval | GraphRAG: Advanced knowledge graph with entity relationships (enables Neo4j visualization)"
            )
            
            with gr.Row():
                url_input = gr.Textbox(
                    label="🌐 Enter URLs (comma-separated)", 
                    placeholder="https://example.com, https://research-paper.com",
                    lines=1,
                )
                file_input = gr.File(
                    file_types=[".pdf", ".txt", ".docx"], 
                    label="📁 Upload Documents", 
                    file_count="multiple"
                )
            
            upload_btn = gr.Button("🚀 Build Knowledge Base", variant="primary", size="lg")
            upload_output = gr.Textbox(label="Processing Status", lines=4)

        with gr.Group():
            gr.Markdown("### ❓ **Ask Questions About Your Documents**")
            gr.Markdown("*Questions will be answered using the RAG type selected above*")
            
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask questions about your documents...",
                    lines=3,
                    label=None,
                )
                ask_btn = gr.Button("🧠 Ask", variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            answer_output = gr.Markdown(label="Answer")

        # Event handlers
        upload_btn.click(
            fn=handle_inputs, 
            inputs=[url_input, file_input, rag_type], 
            outputs=[upload_output]
        )
        
        ask_btn.click(fn=on_ask, inputs=[question_input], outputs=[answer_output])
        clear_btn.click(fn=clear_conversation, inputs=[], outputs=[question_input, answer_output])

        # Example questions
        gr.Markdown("### 💡 **Example Questions:**")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**🔍 Good for RAG:**")
                rag_examples = [
                    "What are the main topics discussed?",
                    "Can you summarize the key findings?",
                    "What are the conclusions?"
                ]
                for example in rag_examples:
                    gr.Button(example, variant="secondary", size="sm").click(
                        lambda x=example: x, outputs=[question_input]
                    )
            
            with gr.Column():
                gr.Markdown("**🧠 Good for GraphRAG:**")
                graph_examples = [
                    "What relationships exist between entities?",
                    "How are the main concepts connected?",
                    "What patterns emerge from the knowledge graph?"
                ]
                for example in graph_examples:
                    gr.Button(example, variant="secondary", size="sm").click(
                        lambda x=example: x, outputs=[question_input]
                    )
    
    return demo

def clear_conversation():
    """Clears the question and answer fields in the Gradio UI."""
    return "", ""

# Cleanup on app shutdown
import atexit
def cleanup_on_exit():
    cleanup_old_databases()
    if neo4j_driver:
        neo4j_driver.close()

atexit.register(cleanup_on_exit)

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