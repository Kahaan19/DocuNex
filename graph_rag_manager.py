import os
import json
import pickle
import spacy
import networkx as nx
import torch
from typing import List, Dict, Tuple, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import warnings
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

class KnowledgeGraphStore:
    """Wrapper for NetworkX graph to provide compatibility"""
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def get_nodes(self):
        return list(self.graph.nodes(data=True))
    
    def get_edges(self):
        return list(self.graph.edges(data=True))
    
    def query(self, query: str) -> List:
        # Simple text-based node search
        results = []
        query_lower = query.lower()
        for node, data in self.graph.nodes(data=True):
            if query_lower in node.lower():
                results.append((node, data))
        return results

class VectorStoreRetriever:
    """Simple retriever wrapper for FAISS vector store"""
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def get_relevant_documents(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)

class GraphRAGManager:
    def __init__(self, persist_dir: str, api_key: str = None, device: str = "auto"):
        self.persist_dir = persist_dir
        self.api_key = api_key  # Made optional for Ollama
        
        # Auto-detect best device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"✅ CUDA detected: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                print("ℹ️ Using CPU (CUDA not available)")
        else:
            self.device = device
        
        # Create persist directory
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Initialize graph
        self.knowledge_graph = nx.Graph()
        self.entity_embeddings = {}
        self.document_chunks = []
        self.chunk_embeddings = []
        
        # Initialize vector store for fallback RAG
        self.vector_store = None
        self.retriever = None
        
        # Add knowledge graph store compatibility
        self.knowledge_graph_store = KnowledgeGraphStore(self.knowledge_graph)
        
        print("✅ GraphRAGManager initialized successfully")

    def get_nodes(self):
        """
        Returns a list of node dicts for Neo4j/D3.js.
        Each node has: id, label, name, properties.
        """
        nodes = []
        for node_id, data in self.knowledge_graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get("label", "Entity"),
                "name": node_id,
                "properties": {
                    "chunk_ids": data.get("chunk_ids", []),
                    "frequency": data.get("frequency", 1)
                }
            })
        return nodes 

    def get_relationships(self):
        """
        Returns a list of relationship dicts for Neo4j/D3.js.
        Each relationship has: source, target, type, properties.
        """
        relationships = []
        for source, target, data in self.knowledge_graph.edges(data=True):
            rel_types = data.get("relations", ["related_to"])
            for rel_type in rel_types:
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": rel_type,
                    "properties": {
                        "chunk_ids": data.get("chunk_ids", [])
                    }
                })
        return relationships   
    
    def _init_models(self):
        """Initialize NLP models with GPU support"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ SpaCy model loaded")
            except OSError:
                print("❌ SpaCy model not found. Run: python -m spacy download en_core_web_sm")
                # Fallback to basic tokenization
                self.nlp = None
            
            # Load sentence transformer for embeddings (GPU accelerated)
            print(f"🔄 Loading SentenceTransformer on {self.device}...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            # Also initialize HuggingFace embeddings for LangChain compatibility
            self.hf_embeddings = HuggingFaceEmbeddings(
                model_name='all-MiniLM-L6-v2',
                model_kwargs={'device': self.device}
            )
            
            # Verify GPU usage
            if self.device == "cuda":
                # Force model to GPU and test
                dummy_text = ["test"]
                with torch.no_grad():
                    _ = self.sentence_model.encode(dummy_text, convert_to_tensor=True)
                print(f"✅ SentenceTransformer loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                print("✅ SentenceTransformer loaded on CPU")
            
            # Google embeddings for consistency (optional for Ollama)
            if self.api_key:
                try:
                    self.google_embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001", 
                        api_key=self.api_key
                    )
                    print("✅ Google embeddings initialized")
                except Exception as e:
                    print(f"⚠️ Google embeddings failed: {e}")
                    self.google_embeddings = None
            else:
                self.google_embeddings = None
                print("ℹ️ Google embeddings skipped (using local models)")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            raise
    
    def extract_entities_and_relations(self, text: str) -> Dict:
        """Extract entities and their relationships from text using NLP"""
        if self.nlp is None:
            # Fallback basic entity extraction using regex
            return self._basic_entity_extraction(text)
        
        try:
            doc = self.nlp(text)
            
            entities = []
            relations = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                    entities.append({
                        'text': ent.text.strip(),
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            # Extract relationships using dependency parsing
            for token in doc:
                if (token.dep_ in ['nsubj', 'dobj', 'pobj', 'nsubjpass'] and 
                    token.head.pos_ == 'VERB' and 
                    any(child.ent_type_ in ['PERSON', 'ORG', 'GPE'] for child in token.subtree)):
                    
                    objects = [child.text for child in token.head.children 
                              if child != token and child.pos_ in ['NOUN', 'PROPN']]
                    
                    if objects:
                        relations.append({
                            'subject': token.text.strip(),
                            'predicate': token.head.lemma_.strip(),
                            'object': objects
                        })
            
            return {'entities': entities, 'relations': relations}
            
        except Exception as e:
            print(f"⚠️ NLP processing error: {e}")
            return self._basic_entity_extraction(text)
    
    def _basic_entity_extraction(self, text: str) -> Dict:
        """Fallback basic entity extraction using regex patterns"""
        entities = []
        
        # Basic patterns for common entities
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'ORG': r'\b[A-Z][A-Za-z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Organization)\b',
            'GPE': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+City|\s+State|\s+Country)?\b'
        }
        
        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group().strip(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return {'entities': entities, 'relations': []}
    
    def build_knowledge_graph(self, documents: List[Document]) -> str:
        """Build knowledge graph from documents with GPU acceleration and vector store"""
        print("🔄 Building knowledge graph and vector store...")
        
        total_entities = 0
        total_relations = 0
        
        # Clear existing data
        self.knowledge_graph.clear()
        self.entity_embeddings = {}
        self.document_chunks = []
        self.chunk_embeddings = []
        
        # Process each document
        all_langchain_docs = []  # For vector store
        
        for i, doc in enumerate(documents):
            print(f"📄 Processing document {i+1}/{len(documents)}")
            
            # Split document into chunks
            chunks = self._split_text(doc.page_content)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                # Store chunk for retrieval
                chunk_id = f"doc_{i}_chunk_{chunk_idx}"
                chunk_data = {
                    'id': chunk_id,
                    'content': chunk,
                    'source': doc.metadata.get('source', 'unknown'),
                    'doc_index': i
                }
                self.document_chunks.append(chunk_data)
                
                # Create LangChain document for vector store
                langchain_doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': doc.metadata.get('source', 'unknown'),
                        'chunk_id': chunk_id,
                        'doc_index': i
                    }
                )
                all_langchain_docs.append(langchain_doc)
                
                # Extract entities and relations
                extracted = self.extract_entities_and_relations(chunk)
                
                # Add entities to graph
                for entity in extracted['entities']:
                    entity_text = entity['text'].lower().strip()
                    if len(entity_text) > 2 and len(entity_text) < 100:  # Filter length
                        if self.knowledge_graph.has_node(entity_text):
                            # Update existing node
                            node_data = self.knowledge_graph.nodes[entity_text]
                            node_data['chunk_ids'].append(chunk_id)
                            node_data['frequency'] += 1
                        else:
                            # Add new node
                            self.knowledge_graph.add_node(
                                entity_text,
                                label=entity['label'],
                                chunk_ids=[chunk_id],
                                frequency=1
                            )
                        total_entities += 1
                
                # Add relations to graph
                for relation in extracted['relations']:
                    if relation['object']:
                        subj = relation['subject'].lower().strip()
                        pred = relation['predicate'].lower().strip()
                        for obj_text in relation['object']:
                            obj = obj_text.lower().strip()
                            if (len(subj) > 2 and len(obj) > 2 and 
                                len(subj) < 100 and len(obj) < 100):
                                
                                # Ensure nodes exist
                                if not self.knowledge_graph.has_node(subj):
                                    self.knowledge_graph.add_node(subj, chunk_ids=[chunk_id], frequency=1)
                                if not self.knowledge_graph.has_node(obj):
                                    self.knowledge_graph.add_node(obj, chunk_ids=[chunk_id], frequency=1)
                                
                                # Add edge
                                if self.knowledge_graph.has_edge(subj, obj):
                                    edge_data = self.knowledge_graph.edges[subj, obj]
                                    if 'relations' not in edge_data:
                                        edge_data['relations'] = []
                                    edge_data['relations'].append(pred)
                                else:
                                    self.knowledge_graph.add_edge(
                                        subj, obj,
                                        relations=[pred],
                                        chunk_ids=[chunk_id]
                                    )
                                total_relations += 1
        
        # Build vector store for fallback RAG
        print("🔄 Building vector store...")
        try:
            if all_langchain_docs:
                self.vector_store = FAISS.from_documents(
                    all_langchain_docs, 
                    self.hf_embeddings
                )
                # Create retriever
                self.retriever = VectorStoreRetriever(self.vector_store)
                print(f"✅ Vector store created with {len(all_langchain_docs)} documents")
            else:
                print("⚠️ No documents to add to vector store")
        except Exception as e:
            print(f"⚠️ Vector store creation failed: {e}")
            self.vector_store = None
            self.retriever = None
        
        print(f"🧠 Generating embeddings on {self.device}...")
        # Generate embeddings for entities and chunks
        self._generate_embeddings()
        
        # Update knowledge graph store
        self.knowledge_graph_store = KnowledgeGraphStore(self.knowledge_graph)
        
        # Save the graph
        print("💾 Saving knowledge graph...")
        self._save_graph()
        
        status = f"""
📊 **Knowledge Graph Statistics:**
- 🏷️ Entities: {self.knowledge_graph.number_of_nodes()}
- 🔗 Relations: {self.knowledge_graph.number_of_edges()}
- 📄 Document Chunks: {len(self.document_chunks)}
- 🗂️ Vector Store: {'✅ Available' if self.vector_store else '❌ Not Available'}
- 🧠 Device: {self.device}
- 🚀 GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "N/A"

"""
        
        return status
    
    def query(self, question: str, k: int = 5) -> str:
        """Unified query method that tries Graph RAG first, falls back to vector RAG"""
        try:
            # Try Graph RAG approach first
            graph_result = self.retrieve_relevant_subgraph(question, k=k)
            
            if (graph_result['entities'] or graph_result['chunks']):
                # Build context from Graph RAG results
                context_parts = []
                
                if graph_result['entities']:
                    context_parts.append("Key entities found:")
                    for entity in graph_result['entities'][:5]:
                        score = graph_result['entity_scores'].get(entity, 0)
                        context_parts.append(f"- {entity} (relevance: {score:.3f})")
                
                if graph_result['chunks']:
                    context_parts.append("\nRelevant content:")
                    for chunk in graph_result['chunks'][:3]:
                        content = chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content']
                        context_parts.append(f"- {content}")
                
                return "\n".join(context_parts)
            
            # Fallback to vector RAG if Graph RAG didn't find anything
            elif self.vector_store:
                print("🔄 Falling back to vector store search...")
                docs = self.vector_store.similarity_search(question, k=k)
                if docs:
                    context_parts = ["Found relevant information:"]
                    for doc in docs:
                        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                        context_parts.append(f"- {content}")
                    return "\n".join(context_parts)
            
            return f"No relevant information found for: {question}"
            
        except Exception as e:
            print(f"❌ Error in query method: {e}")
            return f"Error processing query: {str(e)}"
    
    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with intelligent boundary detection"""
        if len(text) <= chunk_size:
            return [text.strip()]
            
        chunks = []
        start = 0
        text = text.strip()

        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Get the chunk to evaluate breaking points
            chunk = text[start:end]
            break_point = None

            # Try to break at sentence boundary
            sentence_breaks = [m.end() for m in re.finditer(r'[.!?]+\s+', chunk)]
            if sentence_breaks:
                break_point = start + max(sentence_breaks)
            else:
                # Try paragraph break
                para_break = chunk.rfind('\n\n')
                if para_break > chunk_size // 2:
                    break_point = start + para_break
                else:
                    # Try line break
                    line_break = chunk.rfind('\n')
                    if line_break > chunk_size // 2:
                        break_point = start + line_break
                    else:
                        # Try word break
                        word_break = chunk.rfind(' ')
                        if word_break > chunk_size // 2:
                            break_point = start + word_break
                        else:
                            break_point = end  # Default fallback

            # Final safeguard
            if break_point <= start or break_point > len(text):
                break_point = min(start + chunk_size, len(text))

            chunks.append(text[start:break_point].strip())

            # Update start for next chunk
            start = break_point - overlap
            if start <= break_point:
                start = break_point + 1  # Avoid infinite loop

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def _generate_embeddings(self):
        """Generate embeddings for entities and document chunks using GPU acceleration"""
        print("🔄 Generating embeddings...")
        
        # Generate entity embeddings in batches
        entities = list(self.knowledge_graph.nodes())
        if entities:
            print(f"📊 Processing {len(entities)} entities...")
            batch_size = 64 if self.device == "cuda" else 32
            
            try:
                with torch.no_grad():
                    entity_embeddings = self.sentence_model.encode(
                        entities, 
                        batch_size=batch_size,
                        show_progress_bar=True,
                        convert_to_tensor=True,
                        device=self.device,
                        normalize_embeddings=True
                    )
                    
                    # Convert to CPU numpy for storage
                    if isinstance(entity_embeddings, torch.Tensor):
                        entity_embeddings = entity_embeddings.cpu().numpy()
                    
                    for i, entity in enumerate(entities):
                        self.entity_embeddings[entity] = entity_embeddings[i]
                        
                print(f"✅ Generated {len(entities)} entity embeddings")
                
            except Exception as e:
                print(f"⚠️ Error generating entity embeddings: {e}")
                # Fallback to smaller batches
                self.entity_embeddings = {}
                for entity in entities:
                    try:
                        emb = self.sentence_model.encode([entity], device=self.device)[0]
                        self.entity_embeddings[entity] = emb
                    except:
                        continue
        
        # Generate chunk embeddings in batches
        chunk_texts = [chunk['content'] for chunk in self.document_chunks]
        if chunk_texts:
            print(f"📊 Processing {len(chunk_texts)} chunks...")
            batch_size = 32 if self.device == "cuda" else 16
            
            try:
                with torch.no_data():
                    chunk_embeddings = self.sentence_model.encode(
                        chunk_texts,
                        batch_size=batch_size,
                        show_progress_bar=True,
                        convert_to_tensor=True,
                        device=self.device,
                        normalize_embeddings=True
                    )
                    
                    # Convert to CPU numpy for storage
                    if isinstance(chunk_embeddings, torch.Tensor):
                        self.chunk_embeddings = chunk_embeddings.cpu().numpy()
                    else:
                        self.chunk_embeddings = chunk_embeddings
                        
                print(f"✅ Generated {len(chunk_texts)} chunk embeddings")
                
            except Exception as e:
                print(f"⚠️ Error generating chunk embeddings: {e}")
                # Fallback processing
                embeddings = []
                for text in chunk_texts:
                    try:
                        emb = self.sentence_model.encode([text], device=self.device)[0]
                        embeddings.append(emb)
                    except:
                        embeddings.append(np.zeros(384))  # Default embedding size
                self.chunk_embeddings = np.array(embeddings)
    
    def retrieve_relevant_subgraph(self, query: str, k: int = 10, threshold: float = 0.3) -> Dict:
        """Retrieve relevant subgraph and chunks based on query with improved similarity"""
        print(f"🔍 Searching for: '{query}'")
        
        # Get query embedding
        try:
            with torch.no_grad():
                query_embedding = self.sentence_model.encode(
                    [query], 
                    device=self.device,
                    normalize_embeddings=True
                )[0]
                
                if isinstance(query_embedding, torch.Tensor):
                    query_embedding = query_embedding.cpu().numpy()
        except Exception as e:
            print(f"⚠️ Error encoding query: {e}")
            return {'entities': [], 'subgraph': nx.Graph(), 'chunks': [], 'entity_scores': {}, 'chunk_scores': {}}
        
        # Find similar entities
        entity_scores = {}
        for entity, embedding in self.entity_embeddings.items():
            try:
                if isinstance(embedding, np.ndarray) and embedding.size > 0:
                    similarity = np.dot(query_embedding, embedding)
                    entity_scores[entity] = float(similarity)
            except Exception as e:
                continue
        
        # Get top-k entities above threshold
        top_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:k*2]
        relevant_entities = [entity for entity, score in top_entities if score > threshold][:k]
        
        # Find similar chunks
        chunk_scores = []
        if len(self.chunk_embeddings) > 0:
            try:
                similarities = np.dot(self.chunk_embeddings, query_embedding)
                for i, similarity in enumerate(similarities):
                    chunk_scores.append((i, float(similarity)))
            except Exception as e:
                print(f"⚠️ Error computing chunk similarities: {e}")
                chunk_scores = [(i, 0.0) for i in range(len(self.document_chunks))]
        
        # Get top-k chunks above threshold
        top_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:k*2]
        relevant_chunks = [
            self.document_chunks[i] for i, score in top_chunks 
            if score > threshold and i < len(self.document_chunks)
        ][:k]
        
        # Extract subgraph with connected components
        if relevant_entities:
            subgraph_nodes = set(relevant_entities)
            
            # Add connected entities (neighbors)
            for entity in relevant_entities:
                if self.knowledge_graph.has_node(entity):
                    neighbors = list(self.knowledge_graph.neighbors(entity))
                    subgraph_nodes.update(neighbors[:3])  # Limit neighbors
            
            subgraph = self.knowledge_graph.subgraph(subgraph_nodes)
        else:
            subgraph = nx.Graph()
        
        result = {
            'entities': relevant_entities,
            'subgraph': subgraph,
            'chunks': relevant_chunks,
            'entity_scores': dict(top_entities[:k]),
            'chunk_scores': dict(top_chunks[:k])
        }
        
        print(f"✅ Found {len(relevant_entities)} entities, {len(relevant_chunks)} chunks")
        return result
    
    def _save_graph(self):
        """Save knowledge graph and embeddings with NetworkX 3.x compatibility"""
        try:
            # Save NetworkX graph using pickle (NetworkX 3.x compatible)
            with open(os.path.join(self.persist_dir, "knowledge_graph.pkl"), "wb") as f:
                pickle.dump(self.knowledge_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save embeddings
            with open(os.path.join(self.persist_dir, "entity_embeddings.pkl"), "wb") as f:
                pickle.dump(self.entity_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save chunks and their embeddings
            if len(self.chunk_embeddings) > 0:
                np.save(os.path.join(self.persist_dir, "chunk_embeddings.npy"), self.chunk_embeddings)
            
            with open(os.path.join(self.persist_dir, "document_chunks.json"), "w", encoding='utf-8') as f:
                json.dump(self.document_chunks, f, indent=2, ensure_ascii=False)
            
            # Save vector store if available
            if self.vector_store:
                try:
                    vector_store_path = os.path.join(self.persist_dir, "vector_store")
                    self.vector_store.save_local(vector_store_path)
                    print("✅ Vector store saved")
                except Exception as e:
                    print(f"⚠️ Could not save vector store: {e}")
            
            # Save metadata
            metadata = {
                'num_entities': self.knowledge_graph.number_of_nodes(),
                'num_relations': self.knowledge_graph.number_of_edges(),
                'num_chunks': len(self.document_chunks),
                'device_used': self.device,
                'embedding_dim': len(list(self.entity_embeddings.values())[0]) if self.entity_embeddings else 0,
                'has_vector_store': self.vector_store is not None
            }
            
            with open(os.path.join(self.persist_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            print("✅ Knowledge graph saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving graph: {e}")
            raise
    
    def load_knowledge_graph(self):
        """Load saved knowledge graph and embeddings with error handling"""
        try:
            # Load NetworkX graph
            graph_path = os.path.join(self.persist_dir, "knowledge_graph.pkl")
            if os.path.exists(graph_path):
                with open(graph_path, "rb") as f:
                    self.knowledge_graph = pickle.load(f)
                print(f"✅ Loaded graph with {self.knowledge_graph.number_of_nodes()} nodes")
            
            # Load embeddings
            embedding_path = os.path.join(self.persist_dir, "entity_embeddings.pkl")
            if os.path.exists(embedding_path):
                with open(embedding_path, "rb") as f:
                    self.entity_embeddings = pickle.load(f)
                print(f"✅ Loaded {len(self.entity_embeddings)} entity embeddings")
            
            # Load chunks and their embeddings
            chunk_emb_path = os.path.join(self.persist_dir, "chunk_embeddings.npy")
            if os.path.exists(chunk_emb_path):
                self.chunk_embeddings = np.load(chunk_emb_path)
                print(f"✅ Loaded chunk embeddings: {self.chunk_embeddings.shape}")
            
            chunks_path = os.path.join(self.persist_dir, "document_chunks.json")
            if os.path.exists(chunks_path):
                with open(chunks_path, "r", encoding='utf-8') as f:
                    self.document_chunks = json.load(f)
                print(f"✅ Loaded {len(self.document_chunks)} document chunks")
            
            # Load vector store
            vector_store_path = os.path.join(self.persist_dir, "vector_store")
            if os.path.exists(vector_store_path):
                try:
                    self.vector_store = FAISS.load_local(
                        vector_store_path, 
                        self.hf_embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.retriever = VectorStoreRetriever(self.vector_store)
                    print("✅ Vector store loaded")
                except Exception as e:
                    print(f"⚠️ Could not load vector store: {e}")
                    self.vector_store = None
                    self.retriever = None
            
            # Update knowledge graph store
            self.knowledge_graph_store = KnowledgeGraphStore(self.knowledge_graph)
            
            # Load metadata
            metadata_path = os.path.join(self.persist_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    print(f"📊 Metadata: {metadata}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading knowledge graph: {e}")
            return False
    
    def get_graph_stats(self) -> Dict:
        """Get comprehensive statistics about the knowledge graph"""
        stats = {
            'nodes': self.knowledge_graph.number_of_nodes(),
            'edges': self.knowledge_graph.number_of_edges(),
            'chunks': len(self.document_chunks),
            'embeddings': len(self.entity_embeddings),
            'has_vector_store': self.vector_store is not None,
            'device': self.device
        }
        
        if self.knowledge_graph.number_of_nodes() > 0:
            # Calculate degree statistics
            degrees = dict(self.knowledge_graph.degree())
            stats.update({
                'avg_degree': np.mean(list(degrees.values())),
                'max_degree': max(degrees.values()),
                'min_degree': min(degrees.values())
            })
            
            # Most connected entities
            top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_entities'] = top_entities
        
        return stats
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for entities matching the query"""
        results = []
        query_lower = query.lower()
        
        for entity, data in self.knowledge_graph.nodes(data=True):
            if query_lower in entity.lower():
                results.append({
                    'entity': entity,
                    'label': data.get('label', 'Unknown'),
                    'frequency': data.get('frequency', 1),
                    'chunk_count': len(data.get('chunk_ids', []))
                })
        
        # Sort by frequency and limit results
        results.sort(key=lambda x: x['frequency'], reverse=True)
        return results[:limit]
    
    def get_entity_context(self, entity: str, context_window: int = 3) -> Dict:
        """Get detailed context for a specific entity"""
        if not self.knowledge_graph.has_node(entity):
            return {'found': False, 'message': f"Entity '{entity}' not found in graph"}
        
        node_data = self.knowledge_graph.nodes[entity]
        
        # Get neighbors and their relationships
        neighbors = []
        for neighbor in self.knowledge_graph.neighbors(entity):
            edge_data = self.knowledge_graph.edges[entity, neighbor]
            neighbors.append({
                'entity': neighbor,
                'relations': edge_data.get('relations', ['connected_to']),
                'chunk_ids': edge_data.get('chunk_ids', [])
            })
        
        # Get relevant chunks
        chunk_ids = node_data.get('chunk_ids', [])
        relevant_chunks = []
        
        for chunk_data in self.document_chunks:
            if chunk_data['id'] in chunk_ids:
                # Extract sentences around the entity mention
                content = chunk_data['content']
                sentences = content.split('.')
                
                for i, sentence in enumerate(sentences):
                    if entity.lower() in sentence.lower():
                        start_idx = max(0, i - context_window)
                        end_idx = min(len(sentences), i + context_window + 1)
                        context = '. '.join(sentences[start_idx:end_idx]).strip()
                        
                        relevant_chunks.append({
                            'chunk_id': chunk_data['id'],
                            'source': chunk_data.get('source', 'unknown'),
                            'context': context
                        })
                        break
        
        return {
            'found': True,
            'entity': entity,
            'label': node_data.get('label', 'Unknown'),
            'frequency': node_data.get('frequency', 1),
            'neighbors': neighbors[:10],  # Limit neighbors
            'chunks': relevant_chunks[:5],  # Limit chunks
            'total_connections': len(neighbors)
        }
    
    def export_graph_data(self, format: str = 'json') -> Union[Dict, str]:
        """Export graph data in various formats"""
        if format.lower() == 'json':
            return {
                'nodes': self.get_nodes(),
                'relationships': self.get_relationships(),
                'stats': self.get_graph_stats()
            }
        
        elif format.lower() == 'cypher':
            # Generate Cypher CREATE statements for Neo4j
            cypher_statements = []
            
            # Create nodes
            for node in self.get_nodes():
                props = ', '.join([f"{k}: {json.dumps(v)}" for k, v in node['properties'].items()])
                cypher = f"CREATE (:{node['label']} {{name: {json.dumps(node['name'])}, {props}}})"
                cypher_statements.append(cypher)
            
            # Create relationships
            for rel in self.get_relationships():
                props = ', '.join([f"{k}: {json.dumps(v)}" for k, v in rel['properties'].items()])
                props_str = f" {{{props}}}" if props else ""
                cypher = f"MATCH (a {{name: {json.dumps(rel['source'])}}}), (b {{name: {json.dumps(rel['target'])}}}) CREATE (a)-[:{rel['type'].upper()}{props_str}]->(b)"
                cypher_statements.append(cypher)
            
            return '\n'.join(cypher_statements)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def visualize_subgraph(self, entities: List[str], depth: int = 2) -> Dict:
        """Create a subgraph for visualization focused on specific entities"""
        if not entities:
            return {'nodes': [], 'edges': []}
        
        # Start with seed entities
        nodes_to_include = set()
        for entity in entities:
            if self.knowledge_graph.has_node(entity):
                nodes_to_include.add(entity)
        
        # Expand by depth
        for _ in range(depth):
            new_nodes = set()
            for node in nodes_to_include:
                neighbors = list(self.knowledge_graph.neighbors(node))
                # Limit neighbors to avoid overcrowding
                new_nodes.update(neighbors[:5])
            nodes_to_include.update(new_nodes)
        
        # Create subgraph
        subgraph = self.knowledge_graph.subgraph(nodes_to_include)
        
        # Convert to visualization format
        vis_nodes = []
        for node in subgraph.nodes():
            node_data = self.knowledge_graph.nodes[node]
            vis_nodes.append({
                'id': node,
                'label': node,
                'size': min(node_data.get('frequency', 1) * 10, 50),
                'color': self._get_node_color(node_data.get('label', 'Unknown'))
            })
        
        vis_edges = []
        for source, target in subgraph.edges():
            edge_data = self.knowledge_graph.edges[source, target]
            relations = edge_data.get('relations', ['connected_to'])
            
            vis_edges.append({
                'source': source,
                'target': target,
                'label': ', '.join(relations[:2]),  # Limit relation labels
                'weight': len(relations)
            })
        
        return {
            'nodes': vis_nodes,
            'edges': vis_edges,
            'stats': {
                'total_nodes': len(vis_nodes),
                'total_edges': len(vis_edges),
                'depth': depth
            }
        }
    
    def _get_node_color(self, label: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            'PERSON': '#FF6B6B',
            'ORG': '#4ECDC4',
            'GPE': '#45B7D1',
            'EVENT': '#96CEB4',
            'PRODUCT': '#FFEAA7',
            'WORK_OF_ART': '#DDA0DD',
            'LAW': '#98D8C8',
            'LANGUAGE': '#F7DC6F'
        }
        return color_map.get(label, '#BDC3C7')
    
    def cleanup_graph(self, min_frequency: int = 1, min_connections: int = 0):
        """Clean up the graph by removing low-frequency or isolated nodes"""
        nodes_to_remove = []
        
        for node, data in self.knowledge_graph.nodes(data=True):
            frequency = data.get('frequency', 1)
            connections = self.knowledge_graph.degree(node)
            
            if frequency < min_frequency or connections < min_connections:
                nodes_to_remove.append(node)
        
        # Remove nodes and update embeddings
        for node in nodes_to_remove:
            self.knowledge_graph.remove_node(node)
            if node in self.entity_embeddings:
                del self.entity_embeddings[node]
        
        # Update knowledge graph store
        self.knowledge_graph_store = KnowledgeGraphStore(self.knowledge_graph)
        
        print(f"🧹 Removed {len(nodes_to_remove)} nodes during cleanup")
        return len(nodes_to_remove)
    
    def get_path_between_entities(self, entity1: str, entity2: str, max_length: int = 5) -> List[List[str]]:
        """Find paths between two entities in the knowledge graph"""
        if not (self.knowledge_graph.has_node(entity1) and self.knowledge_graph.has_node(entity2)):
            return []
        
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.knowledge_graph, 
                entity1, 
                entity2, 
                cutoff=max_length
            ))
            
            # Sort by length (shorter paths first)
            paths.sort(key=len)
            return paths[:10]  # Return top 10 paths
            
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            print(f"⚠️ Error finding paths: {e}")
            return []
    
    def get_central_entities(self, metric: str = 'degree', top_k: int = 10) -> List[Tuple[str, float]]:
        """Get most central entities using various centrality metrics"""
        if self.knowledge_graph.number_of_nodes() == 0:
            return []
        
        try:
            if metric == 'degree':
                centrality = nx.degree_centrality(self.knowledge_graph)
            elif metric == 'betweenness':
                centrality = nx.betweenness_centrality(self.knowledge_graph)
            elif metric == 'closeness':
                centrality = nx.closeness_centrality(self.knowledge_graph)
            elif metric == 'pagerank':
                centrality = nx.pagerank(self.knowledge_graph)
            else:
                centrality = nx.degree_centrality(self.knowledge_graph)
            
            # Sort and return top-k
            sorted_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return sorted_entities[:top_k]
            
        except Exception as e:
            print(f"⚠️ Error computing centrality: {e}")
            return []
    
    def semantic_search_chunks(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Perform semantic search on document chunks"""
        if len(self.chunk_embeddings) == 0:
            return []
        
        try:
            # Get query embedding
            with torch.no_grad():
                query_embedding = self.sentence_model.encode(
                    [query], 
                    device=self.device,
                    normalize_embeddings=True
                )[0]
                
                if isinstance(query_embedding, torch.Tensor):
                    query_embedding = query_embedding.cpu().numpy()
            
            # Calculate similarities
            similarities = np.dot(self.chunk_embeddings, query_embedding)
            
            # Get top-k results above threshold
            results = []
            for i, similarity in enumerate(similarities):
                if similarity > threshold and i < len(self.document_chunks):
                    results.append({
                        'chunk': self.document_chunks[i],
                        'similarity': float(similarity),
                        'rank': len(results) + 1
                    })
            
            # Sort by similarity and limit
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]
            
        except Exception as e:
            print(f"⚠️ Error in semantic search: {e}")
            return []
    
    def __del__(self):
        """Cleanup method to clear CUDA memory if needed"""
        if hasattr(self, 'device') and self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except:
                pass

# Utility functions for external use
def create_graph_rag_manager(persist_dir: str, api_key: str = None, device: str = "auto") -> GraphRAGManager:
    """Factory function to create a GraphRAGManager instance"""
    return GraphRAGManager(persist_dir=persist_dir, api_key=api_key, device=device)

def load_documents_from_directory(directory: str) -> List[Document]:
    """Load documents from a directory (supports .txt, .md files)"""
    documents = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('.txt', '.md')):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={'source': filename, 'path': filepath}
                    ))
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
    
    return documents

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("🚀 GraphRAG Manager - Advanced Knowledge Graph RAG System")
    
    # Initialize manager
    manager = create_graph_rag_manager(
        persist_dir="./knowledge_graph_data",
        device="auto"  # Will auto-detect CUDA/CPU
    )
    
    # Example: Load from directory
    # documents = load_documents_from_directory("./documents")
    
    # Example: Create sample documents
    sample_docs = [
        Document(
            page_content="John Doe works at OpenAI and is interested in artificial intelligence research. He collaborates with Jane Smith on machine learning projects.",
            metadata={'source': 'sample1.txt'}
        ),
        Document(
            page_content="Jane Smith is a researcher at Stanford University. She focuses on natural language processing and works with John Doe on various AI projects.",
            metadata={'source': 'sample2.txt'}
        )
    ]
    
    # Build knowledge graph
    if sample_docs:
        status = manager.build_knowledge_graph(sample_docs)
        print(status)
        
        # Example queries
        queries = [
            "Who works at OpenAI?",
            "What is Jane Smith's research focus?",
            "Tell me about AI research collaborations"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            result = manager.query(query)
            print(f"📝 Answer: {result}")
        
        # Get graph statistics
        stats = manager.get_graph_stats()
        print(f"\n📊 Graph Statistics: {stats}")
        
        # Export graph data
        graph_data = manager.export_graph_data('json')
        print(f"\n📤 Graph exported with {len(graph_data['nodes'])} nodes and {len(graph_data['relationships'])} relationships")
    
    print("\n✅ GraphRAG Manager demonstration completed!")