# retrieval_service.py
import logging
from typing import List, Dict, Any, Tuple
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
import numpy as np

class RetrievalService:
    """
    Dual retrieval service (Sparse + Dense) with reranking.
    """
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def sparse_retrieval(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform sparse retrieval using full-text search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        try:
            self.logger.info(f"Performing sparse retrieval for: {query}")
            
            # Full-text search across Clause and Regulation
            sparse_query = """
            CALL {
                // Search in Clause text
                CALL db.index.fulltext.queryNodes("clause_text_fulltext", $query)
                YIELD node, score
                WHERE node:Clause
                RETURN node, score, 'clause' as type
                
                UNION
                
                // Search in Regulation title
                CALL db.index.fulltext.queryNodes("regulation_title_fulltext", $query)
                YIELD node, score
                WHERE node:Regulation
                RETURN node, score, 'regulation' as type
            }
            RETURN node, score, type
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            results = self.graph.query(sparse_query, params={
                "query": query,
                "top_k": top_k
            })
            
            formatted_results = []
            for result in results:
                node_data = dict(result['node'])
                formatted_results.append({
                    'content': node_data.get('text') or node_data.get('title', ''),
                    'metadata': {
                        'type': result['type'],
                        'score': result['score'],
                        'node_id': node_data.get('id') or node_data.get('number'),
                        'node_properties': node_data
                    }
                })
            
            self.logger.info(f"Sparse retrieval found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def dense_retrieval(self, query: str, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform dense retrieval using vector similarity.
        
        Args:
            query: Search query
            embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        try:
            self.logger.info(f"Performing dense retrieval for: {query}")
            
            # Vector similarity search
            dense_query = """
            CALL {
                // Search in Clause embeddings
                CALL db.index.vector.queryNodes("clause_embedding_vector", $top_k, $embedding)
                YIELD node, score
                WHERE node:Clause
                RETURN node, score, 'clause' as type
                
                UNION
                
                // Search in Regulation embeddings
                CALL db.index.vector.queryNodes("regulation_embedding_vector", $top_k, $embedding)
                YIELD node, score
                WHERE node:Regulation
                RETURN node, score, 'regulation' as type
            }
            RETURN node, score, type
            ORDER BY score DESC
            """
            
            results = self.graph.query(dense_query, params={
                "embedding": embedding,
                "top_k": top_k
            })
            
            formatted_results = []
            for result in results:
                node_data = dict(result['node'])
                formatted_results.append({
                    'content': node_data.get('text') or node_data.get('title', ''),
                    'metadata': {
                        'type': result['type'],
                        'score': float(result['score']),
                        'node_id': node_data.get('id') or node_data.get('number'),
                        'node_properties': node_data
                    }
                })
            
            self.logger.info(f"Dense retrieval found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def rerank_results(self, results: List[Dict[str, Any]], query: str, 
                      title_boost: float = 2.0, number_boost: float = 1.5, 
                      clause_boost: float = 1.2) -> List[Dict[str, Any]]:
        """
        Rerank results with boosts for title/number/clauses.
        
        Args:
            results: List of retrieval results
            query: Original query
            title_boost: Boost factor for title matches
            number_boost: Boost factor for number matches  
            clause_boost: Boost factor for clause matches
            
        Returns:
            Reranked results
        """
        try:
            self.logger.info("Reranking results with boosts")
            
            for result in results:
                metadata = result['metadata']
                score = metadata['score']
                node_type = metadata['type']
                
                # Apply boosts based on type and content
                if node_type == 'regulation':
                    # Boost for regulation titles
                    score *= title_boost
                    
                    # Additional boost if query contains numbers that match regulation number
                    if any(char.isdigit() for char in query):
                        score *= number_boost
                
                elif node_type == 'clause':
                    # Boost for clauses
                    score *= clause_boost
                
                metadata['reranked_score'] = score
            
            # Sort by reranked score
            reranked = sorted(results, key=lambda x: x['metadata']['reranked_score'], reverse=True)
            
            self.logger.info(f"Reranked {len(reranked)} results")
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return results
    
    def expand_graph_context(self, node_id: str, node_type: str, window_size: int = 2) -> Dict[str, Any]:
        """
        Expand graph context with one-hop relationships and clause windows.
        
        Args:
            node_id: ID of the node to expand
            node_type: Type of node ('clause' or 'regulation')
            window_size: Number of adjacent clauses to include
            
        Returns:
            Expanded context information
        """
        try:
            self.logger.info(f"Expanding graph context for {node_type} {node_id}")
            
            if node_type == 'clause':
                expansion_query = """
                MATCH (c:Clause {id: $node_id})
                OPTIONAL MATCH (c)-[:NEXT_CLAUSE*0..$window_size]->(next:Clause)
                OPTIONAL MATCH (prev:Clause)-[:NEXT_CLAUSE*0..$window_size]->(c)
                OPTIONAL MATCH (c)<-[:HAS_CLAUSE]-(r:Regulation)
                OPTIONAL MATCH (r)-[:AFFECTED_BY]->(l:Law)
                RETURN 
                    c as current_clause,
                    collect(DISTINCT next) as next_clauses,
                    collect(DISTINCT prev) as previous_clauses,
                    collect(DISTINCT r) as regulations,
                    collect(DISTINCT l) as laws
                """
            else:  # regulation
                expansion_query = """
                MATCH (r:Regulation {number: $node_id})
                OPTIONAL MATCH (r)-[:HAS_CLAUSE]->(c:Clause)
                OPTIONAL MATCH (r)-[:AFFECTED_BY]->(l:Law)
                WITH r, collect(DISTINCT c) as clauses, collect(DISTINCT l) as laws
                UNWIND clauses as clause
                WITH r, clause, laws
                ORDER BY clause.text
                RETURN 
                    r as regulation,
                    collect(clause) as clauses,
                    laws
                """
            
            results = self.graph.query(expansion_query, params={
                "node_id": node_id,
                "window_size": window_size
            })
            
            if results:
                context = dict(results[0])
                self.logger.info(f"Expanded context retrieved successfully")
                return context
            else:
                self.logger.warning(f"No expansion context found for {node_id}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Graph expansion failed: {e}")
            return {}
    
    def dual_retrieval(self, query: str, query_embedding: List[float], 
                      top_k: int = 10, fusion_alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform dual retrieval (sparse + dense) with fusion.
        
        Args:
            query: Search query
            query_embedding: Query embedding vector
            top_k: Number of results to return
            fusion_alpha: Weight for sparse vs dense (0.5 = equal)
            
        Returns:
            Fused and reranked results
        """
        try:
            self.logger.info(f"Performing dual retrieval for: {query}")
            
            # Get results from both retrieval methods
            sparse_results = self.sparse_retrieval(query, top_k * 2)
            dense_results = self.dense_retrieval(query, query_embedding, top_k * 2)
            
            # Normalize scores and fuse
            all_results = {}
            
            # Add sparse results
            for result in sparse_results:
                node_id = result['metadata']['node_id']
                all_results[node_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'sparse_score': result['metadata']['score'],
                    'dense_score': 0.0
                }
            
            # Add dense results and update existing entries
            for result in dense_results:
                node_id = result['metadata']['node_id']
                dense_score = result['metadata']['score']
                
                if node_id in all_results:
                    all_results[node_id]['dense_score'] = dense_score
                else:
                    all_results[node_id] = {
                        'content': result['content'],
                        'metadata': result['metadata'],
                        'sparse_score': 0.0,
                        'dense_score': dense_score
                    }
            
            # Calculate fused scores
            fused_results = []
            for node_id, data in all_results.items():
                fused_score = (fusion_alpha * data['sparse_score'] + 
                             (1 - fusion_alpha) * data['dense_score'])
                
                fused_results.append({
                    'content': data['content'],
                    'metadata': {
                        **data['metadata'],
                        'fused_score': fused_score,
                        'sparse_score': data['sparse_score'],
                        'dense_score': data['dense_score']
                    }
                })
            
            # Sort by fused score
            fused_results.sort(key=lambda x: x['metadata']['fused_score'], reverse=True)
            top_results = fused_results[:top_k]
            
            # Rerank with boosts
            reranked_results = self.rerank_results(top_results, query)
            
            self.logger.info(f"Dual retrieval completed with {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Dual retrieval failed: {e}")
            return []