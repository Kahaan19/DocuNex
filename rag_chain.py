import requests
import json
import traceback
import networkx as nx
from typing import Dict, List, Any, Optional, Union

def generate_graph_answer_ollama(question: str, manager, ollama_base_url: str, model_name: str) -> str:
    """
    Generate answer using GraphRAGManager with Ollama - Enhanced version
    
    Args:
        question (str): The question to answer
        manager: Either RAGManager or GraphRAGManager instance
        ollama_base_url (str): Base URL for Ollama API
        model_name (str): Name of the Ollama model to use
        
    Returns:
        str: Generated answer or error message
    """
    try:
        # Validate inputs
        if not question or not question.strip():
            return "❌ Question cannot be empty"
        
        if not manager:
            return "❌ Manager object is None"
        
        # Check what type of manager we have
        manager_type = type(manager).__name__
        print(f"📊 Manager type: {manager_type}")
        
        context = ""
        metadata = {}
        
        # Method 1: GraphRAGManager with retrieve_relevant_subgraph (preferred for Graph RAG)
        if hasattr(manager, 'retrieve_relevant_subgraph'):
            print("🔄 Using GraphRAGManager with subgraph retrieval...")
            try:
                retrieval_result = manager.retrieve_relevant_subgraph(question, k=15)
                
                # Extract information from retrieval result
                relevant_entities = retrieval_result.get('entities', [])
                subgraph = retrieval_result.get('subgraph', nx.Graph())
                relevant_chunks = retrieval_result.get('chunks', [])
                entity_scores = retrieval_result.get('entity_scores', {})
                
                # Build enhanced context using graph structure
                graph_context = build_graph_context(subgraph, relevant_entities, entity_scores)
                chunk_context = build_chunk_context(relevant_chunks)
                
                context = f"KNOWLEDGE GRAPH INFORMATION:\n{graph_context}\n\nDOCUMENT EXCERPTS:\n{chunk_context}"
                
                metadata = {
                    'method': 'graph_rag_subgraph',
                    'entities_count': len(relevant_entities),
                    'chunks_count': len(relevant_chunks),
                    'graph_nodes': len(subgraph.nodes()),
                    'graph_edges': len(subgraph.edges())
                }
                
            except Exception as e:
                print(f"❌ GraphRAG subgraph retrieval failed: {e}")
                traceback.print_exc()
                # Fall back to other methods
                context = ""
        
        # Method 2: Standard RAGManager with retriever
        if not context and hasattr(manager, 'retriever') and hasattr(manager.retriever, 'get_relevant_documents'):
            print("🔄 Using standard RAGManager approach...")
            try:
                docs = manager.retriever.get_relevant_documents(question)
                context = "\n".join([doc.page_content for doc in docs[:5]])
                metadata = {
                    'method': 'standard_rag',
                    'documents_count': len(docs)
                }
            except Exception as e:
                print(f"❌ Standard RAG retrieval failed: {e}")
        
        # Method 3: GraphRAGManager with vector_store
        if not context and hasattr(manager, 'vector_store') and hasattr(manager.vector_store, 'similarity_search'):
            print("🔄 Using GraphRAGManager vector_store approach...")
            try:
                docs = manager.vector_store.similarity_search(question, k=5)
                context = "\n".join([doc.page_content for doc in docs])
                metadata = {
                    'method': 'vector_store',
                    'documents_count': len(docs)
                }
            except Exception as e:
                print(f"❌ Vector store retrieval failed: {e}")
        
        # Method 4: Manager with query method
        if not context and hasattr(manager, 'query'):
            print("🔄 Using manager query method...")
            try:
                result = manager.query(question)
                # If the query method returns a full response, return it directly
                if isinstance(result, str) and len(result) > 50:
                    return result
                else:
                    context = str(result)
                    metadata = {'method': 'direct_query'}
            except Exception as query_error:
                print(f"❌ Manager query failed: {query_error}")
        
        # Method 5: Knowledge graph store approach
        if not context and hasattr(manager, 'knowledge_graph_store'):
            print("🔄 Using knowledge_graph_store approach...")
            try:
                kg_store = manager.knowledge_graph_store
                
                if hasattr(kg_store, 'get_triplets'):
                    triplets = kg_store.get_triplets()
                    # Filter relevant triplets
                    question_words = [word.lower() for word in question.split() if len(word) > 2]
                    relevant_triplets = []
                    
                    for triplet in triplets[:200]:  # Search more triplets
                        triplet_text = ' '.join([str(x) for x in triplet]).lower()
                        if any(word in triplet_text for word in question_words):
                            relevant_triplets.append(triplet)
                    
                    if relevant_triplets:
                        context = "Knowledge Graph Information:\n"
                        for triplet in relevant_triplets[:15]:
                            context += f"• {triplet[0]} → {triplet[1]} → {triplet[2]}\n"
                        metadata = {
                            'method': 'knowledge_graph',
                            'triplets_count': len(relevant_triplets)
                        }
                    else:
                        context = "No directly relevant information found in knowledge graph."
                else:
                    context = "Knowledge graph store available but no triplets method found."
            except Exception as e:
                print(f"❌ Knowledge graph retrieval failed: {e}")
        
        # Method 6: Try internal rag_manager
        if not context and hasattr(manager, 'rag_manager'):
            print("🔄 Using internal rag_manager...")
            try:
                internal_manager = manager.rag_manager
                if hasattr(internal_manager, 'retriever'):
                    docs = internal_manager.retriever.get_relevant_documents(question)
                    context = "\n".join([doc.page_content for doc in docs[:5]])
                    metadata = {
                        'method': 'internal_rag',
                        'documents_count': len(docs)
                    }
                else:
                    context = "Internal RAG manager found but no retriever available."
            except Exception as e:
                print(f"❌ Internal RAG manager failed: {e}")
        
        # Method 7: Direct document chunks access for GraphRAGManager
        if not context and hasattr(manager, 'document_chunks') and manager.document_chunks:
            print("🔄 Using direct document chunks access...")
            try:
                question_words = set(question.lower().split())
                relevant_chunks = []
                
                for chunk in manager.document_chunks:
                    chunk_words = set(chunk['content'].lower().split())
                    # Simple keyword matching
                    overlap = len(question_words.intersection(chunk_words))
                    if overlap > 0:
                        relevant_chunks.append((chunk, overlap))
                
                # Sort by relevance and take top chunks
                relevant_chunks.sort(key=lambda x: x[1], reverse=True)
                
                if relevant_chunks:
                    context = "Document Information:\n"
                    for chunk, score in relevant_chunks[:5]:
                        context += f"Source: {chunk.get('source', 'Unknown')}\n"
                        context += f"{chunk['content'][:800]}...\n\n"
                    
                    metadata = {
                        'method': 'direct_chunks',
                        'chunks_count': len(relevant_chunks)
                    }
                else:
                    context = "No relevant chunks found in document store."
            except Exception as e:
                print(f"❌ Direct chunks access failed: {e}")
        
        # If no context found, provide debug information
        if not context or not context.strip():
            available_methods = [method for method in dir(manager) if not method.startswith('_')]
            print(f"📋 Available methods in manager: {available_methods}")
            
            debug_info = debug_manager_capabilities(manager)
            
            return f"""⚠️ No relevant context found for the question: '{question}'

{debug_info}

Possible solutions:
1. Ensure your documents contain information related to this topic
2. Check if the GraphRAGManager is properly initialized
3. Verify that the knowledge graph has been built: manager.build_knowledge_graph(documents)
4. Try loading an existing graph: manager.load_knowledge_graph()
5. Check if documents were properly processed

For GraphRAG, make sure to:
- Build the knowledge graph first: manager.build_knowledge_graph(your_documents)
- Verify the graph has nodes: manager.get_graph_stats()"""
        
        # Generate answer using Ollama
        print(f"📝 Context length: {len(context)} characters")
        print(f"🔧 Using method: {metadata.get('method', 'unknown')}")
        
        # Create enhanced prompt
        prompt = create_enhanced_prompt(question, context, metadata.get('method', 'unknown'))
        
        # Call Ollama API
        response = call_ollama_api(ollama_base_url, model_name, prompt)
        
        if "error" in response:
            return f"❌ Ollama Error: {response['error']}"
        
        answer = response.get('response', 'No response generated')
        
        # Format final response based on method used
        if metadata.get('method') == 'graph_rag_subgraph':
            # Enhanced formatting for Graph RAG
            formatted_response = format_graph_rag_response(
                answer, 
                metadata.get('entities_count', 0),
                metadata.get('chunks_count', 0),
                metadata.get('graph_nodes', 0),
                metadata.get('graph_edges', 0)
            )
        else:
            # Standard formatting
            method_info = f"*Answer generated using {manager_type} ({metadata.get('method', 'unknown')}) with {len(context)} characters of context*"
            formatted_response = f"{answer}\n\n---\n{method_info}"
        
        return formatted_response
        
    except requests.exceptions.ConnectionError:
        return f"❌ Cannot connect to Ollama at {ollama_base_url}. Please make sure Ollama is running with: `ollama serve`"
    except requests.exceptions.Timeout:
        return "❌ Request to Ollama timed out. Please try again or use a shorter question."
    except Exception as e:
        error_msg = f"❌ Error generating graph answer: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def create_enhanced_prompt(question: str, context: str, method: str) -> str:
    """Create an enhanced prompt based on the retrieval method used"""
    
    if method == 'graph_rag_subgraph':
        return f"""You are an expert document analyst with access to a knowledge graph and document excerpts. Provide comprehensive, well-structured answers based on the available information.

{context}

QUESTION: {question}

INSTRUCTIONS:
- Analyze both the structured knowledge graph and detailed document text
- Identify key entities, relationships, and patterns
- Provide specific examples and evidence from the documents
- Structure your response clearly with main points and supporting details
- If you find interesting connections in the knowledge graph, highlight them
- Be thorough but concise
- Use the relationship information to provide deeper insights

ANSWER:"""
    
    else:
        return f"""You are a helpful AI assistant. Based on the provided context, please answer the user's question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
- Use only the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be specific and cite relevant details from the context
- Format your response in a clear and readable manner
- Provide examples where available

Answer:"""

def call_ollama_api(base_url: str, model: str, prompt: str) -> Dict:
    """Call Ollama API with enhanced configuration"""
    try:
        url = f"{base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Slightly more deterministic
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 8192,  # Larger context window
                "num_predict": 2048,  # Longer responses
                "repeat_penalty": 1.1
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers, timeout=180)
        
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

def build_graph_context(subgraph: nx.Graph, entities: List[str], entity_scores: Dict[str, float]) -> str:
    """Build context string from graph structure"""
    if not entities:
        return "No relevant entities found in the knowledge graph."
    
    context_parts = []
    
    # Add entity information with scores
    context_parts.append("Key Entities Found:")
    for entity in entities[:12]:
        score = entity_scores.get(entity, 0)
        if subgraph.has_node(entity):
            node_data = subgraph.nodes.get(entity, {})
            label = node_data.get('label', 'CONCEPT')
            frequency = node_data.get('frequency', 1)
            context_parts.append(f"• {entity} ({label}) [Relevance: {score:.3f}, Frequency: {frequency}]")
        else:
            context_parts.append(f"• {entity} [Relevance: {score:.3f}]")
    
    # Add relationship information
    context_parts.append("\nEntity Relationships:")
    edges_added = 0
    for edge in subgraph.edges(data=True):
        if edges_added >= 20:
            break
        
        source, target, data = edge
        relations = data.get('relations', ['related_to'])
        for relation in relations[:2]:  # Limit relations per edge
            context_parts.append(f"• {source} --[{relation}]--> {target}")
            edges_added += 1
            if edges_added >= 20:
                break
    
    if edges_added == 0:
        context_parts.append("• No direct relationships found between entities")
    
    # Add graph statistics
    context_parts.append(f"\nGraph Statistics:")
    context_parts.append(f"• Total entities in subgraph: {len(subgraph.nodes())}")
    context_parts.append(f"• Total relationships: {len(subgraph.edges())}")
    if len(subgraph.nodes()) > 1:
        density = nx.density(subgraph)
        context_parts.append(f"• Graph density: {density:.3f}")
        if nx.is_connected(subgraph):
            context_parts.append(f"• Graph is connected")
        else:
            components = nx.number_connected_components(subgraph)
            context_parts.append(f"• Connected components: {components}")
    
    return "\n".join(context_parts)

def build_chunk_context(chunks: List[Dict]) -> str:
    """Build context string from relevant document chunks"""
    if not chunks:
        return "No relevant document chunks found."
    
    context_parts = []
    context_parts.append("Relevant Document Sections:")
    
    for i, chunk in enumerate(chunks[:8]):  # Limit to 8 chunks
        source = chunk.get('source', 'Unknown source')
        chunk_id = chunk.get('id', f'chunk_{i}')
        content = chunk.get('content', '')
        
        # Truncate very long content
        if len(content) > 1200:
            content = content[:1200] + "..."
        
        context_parts.append(f"\n[Section {i+1}] Source: {source} (ID: {chunk_id})")
        context_parts.append(f"{content}")
        context_parts.append("")  # Add spacing
    
    return "\n".join(context_parts)

def format_graph_rag_response(answer: str, entities_count: int, chunks_count: int, 
                             graph_nodes: int, graph_edges: int) -> str:
    """Format the final response with Graph RAG information"""
    
    response_parts = [f"### 🦙 Ollama Graph RAG Answer:\n\n{answer}"]
    
    # Add processing information
    response_parts.append(f"\n---\n### 📊 Graph RAG Analysis:")
    response_parts.append(f"- 🏷️ **Entities analyzed:** {entities_count}")
    response_parts.append(f"- 📄 **Document chunks processed:** {chunks_count}")
    response_parts.append(f"- 🔗 **Graph nodes in subgraph:** {graph_nodes}")
    response_parts.append(f"- 🌐 **Relationships discovered:** {graph_edges}")
    
    if graph_nodes > 0 and graph_edges > 0:
        if graph_nodes > 1:
            density = graph_edges / (graph_nodes * (graph_nodes - 1) / 2)
            response_parts.append(f"- 📈 **Graph connectivity:** {density:.3f}")
    
    # Add technical info
    response_parts.append(f"\n### ⚡ Processing Info:")
    response_parts.append(f"- 🦙 **Powered by:** Ollama (local processing)")
    response_parts.append(f"- 🔄 **Method:** Graph RAG with knowledge graph")
    response_parts.append(f"- 🎯 **Enhanced with:** Entity relationships and document context")
    
    return "\n".join(response_parts)

def validate_manager(manager) -> tuple[bool, str]:
    """
    Validate and provide information about the manager object
    
    Returns:
        tuple: (is_valid, info_message)
    """
    if manager is None:
        return False, "Manager is None"
    
    manager_type = type(manager).__name__
    available_attrs = [attr for attr in dir(manager) if not attr.startswith('_')]
    
    # Check for expected attributes
    expected_attrs = [
        'retrieve_relevant_subgraph',  # Graph RAG specific
        'retriever', 
        'vector_store', 
        'knowledge_graph_store', 
        'query', 
        'rag_manager',
        'document_chunks',  # GraphRAG specific
        'knowledge_graph'   # GraphRAG specific
    ]
    
    found_attrs = [attr for attr in expected_attrs if hasattr(manager, attr)]
    
    info = f"Manager type: {manager_type}, Found attributes: {found_attrs}"
    
    if found_attrs:
        return True, info
    else:
        return False, f"No expected attributes found. Available: {available_attrs[:15]}"

def debug_manager_capabilities(manager) -> str:
    """Debug function to understand manager capabilities"""
    if not manager:
        return "Manager is None"
    
    debug_info = []
    debug_info.append(f"Manager Type: {type(manager).__name__}")
    
    # Check all attributes
    attrs = [attr for attr in dir(manager) if not attr.startswith('_')]
    debug_info.append(f"Available attributes: {attrs}")
    
    # Test specific capabilities
    capabilities = []
    if hasattr(manager, 'retrieve_relevant_subgraph'):
        capabilities.append("✅ Graph RAG subgraph retrieval")
    if hasattr(manager, 'retriever'):
        capabilities.append("✅ Standard RAG retrieval")
    if hasattr(manager, 'vector_store'):
        capabilities.append("✅ Vector store access")
    if hasattr(manager, 'knowledge_graph_store'):
        capabilities.append("✅ Knowledge graph store")
    if hasattr(manager, 'query'):
        capabilities.append("✅ Direct query method")
    if hasattr(manager, 'document_chunks'):
        chunks_count = len(manager.document_chunks) if manager.document_chunks else 0
        capabilities.append(f"✅ Document chunks ({chunks_count} chunks)")
    if hasattr(manager, 'knowledge_graph'):
        if hasattr(manager.knowledge_graph, 'number_of_nodes'):
            nodes = manager.knowledge_graph.number_of_nodes()
            edges = manager.knowledge_graph.number_of_edges()
            capabilities.append(f"✅ Knowledge graph ({nodes} nodes, {edges} edges)")
        else:
            capabilities.append("✅ Knowledge graph (structure unknown)")
    
    if capabilities:
        debug_info.append("Detected capabilities:")
        debug_info.extend(capabilities)
    else:
        debug_info.append("❌ No recognized capabilities found")
    
    # Additional GraphRAG specific checks
    if hasattr(manager, 'get_graph_stats'):
        try:
            stats = manager.get_graph_stats()
            debug_info.append(f"Graph statistics: {stats}")
        except Exception as e:
            debug_info.append(f"⚠️ Could not get graph stats: {e}")
    
    return "\n".join(debug_info)

def analyze_retrieval_quality(question: str, context: str) -> Dict[str, Any]:
    """Analyze the quality of retrieved context for the given question"""
    question_words = set(question.lower().split())
    context_words = set(context.lower().split())
    
    # Calculate word overlap
    overlap = question_words.intersection(context_words)
    overlap_ratio = len(overlap) / len(question_words) if question_words else 0
    
    return {
        'context_length': len(context),
        'word_overlap_ratio': overlap_ratio,
        'overlapping_words': list(overlap),
        'context_sentences': len(context.split('.')),
        'has_meaningful_content': len(context.strip()) > 50
    }