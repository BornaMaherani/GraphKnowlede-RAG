# retrieval_service.py (updated for related nodes type)
import logging
from typing import List, Dict, Any
from langchain_community.graphs import Neo4jGraph
import numpy as np

class RetrievalService:
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.labels = [
            'Ordinary_Law', 'Single_Article_Law', 'Executive_Regulation',
            'Circular_Directive', 'Amendment_Addition', 'Related_Laws',
            'Article', 'Note_Clause', 'Sub_section', 'Sub_paragraph'
        ]
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def sparse_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Full-text search using general fulltext index.
        """
        try:
            search_query = """
            CALL db.index.fulltext.queryNodes('content_fulltext', $query) YIELD node, score
            RETURN elementId(node) as node_id, labels(node)[0] as type, node.content as content, 
                   apoc.map.removeKey(properties(node), 'embedding') as node_props, score
            ORDER BY score DESC LIMIT $top_k
            """
            results = self.graph.query(search_query, {'query': query, 'top_k': top_k * 2})  # Get more to fuse
            return [{'content': r['content'], 'metadata': {'node_id': r['node_id'], 'type': r['type'], 
                    'node': dict(r['node_props']), 'score': r['score']}} for r in results]
        except Exception as e:
            self.logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def dense_retrieval(self, query: str, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Vector similarity search across all label-specific indexes.
        """
        try:
            all_results = []
            per_label_k = top_k * 2 // len(self.labels) + 1  # Distribute
            
            for label in self.labels:
                index_name = f"{label.lower()}_embedding_vector"
                search_query = """
                CALL db.index.vector.queryNodes($index_name, $per_k, $embedding)
                YIELD node, score
                RETURN elementId(node) as node_id, labels(node)[0] as type, node.content as content, 
                       apoc.map.removeKey(properties(node), 'embedding') as node_props, score
                """
                params = {'index_name': index_name, 'per_k': per_label_k, 'embedding': query_embedding}
                res = self.graph.query(search_query, params)
                all_results.extend(res)
            
            # Sort all and take top_k
            all_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = all_results[:top_k]
            
            return [{'content': r['content'], 'metadata': {'node_id': r['node_id'], 'type': r['type'], 
                    'node': dict(r['node_props']), 'score': r['score']}} for r in top_results]
        except Exception as e:
            self.logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def get_related_nodes(self, node_id: str, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Traverse relationships and filter related nodes relevant to query.
        """
        try:
            traverse_query = """
            MATCH (n)-[r]-(m)
            WHERE elementId(n) = $node_id AND m.embedding IS NOT NULL
            RETURN elementId(m) as rel_id, labels(m)[0] as type, m.content as content, 
                   apoc.map.removeKey(properties(m), 'embedding') as node_props, 
                   type(r) as rel_type, properties(r) as rel_props,
                   vector.similarity.cosine(m.embedding, $query_embedding) as rel_score
            ORDER BY rel_score DESC LIMIT $top_k
            """
            results = self.graph.query(traverse_query, {'node_id': node_id, 'query_embedding': query_embedding, 'top_k': top_k})
            return [{'content': r['content'], 'metadata': {'node_id': r['rel_id'], 'type': r['type'], 
                    'node': dict(r['node_props']), 'rel_type': r['rel_type'], 'rel_props': r['rel_props'], 
                    'score': r['rel_score']}} for r in results]
        except Exception as e:
            self.logger.error(f"Related nodes retrieval failed: {e}")
            return []
    
    def enrich_with_graph_context(self, docs: List[Dict[str, Any]], query_embedding: List[float]) -> List[Dict[str, Any]]:
        enriched = []
        for doc in docs:
            node_id = doc['metadata']['node_id']
            related = self.get_related_nodes(node_id, query_embedding)
            doc['metadata']['related_nodes'] = [{'type': r['metadata']['type'], 'properties': r['metadata']['node']} for r in related]
            enriched.append(doc)
        return enriched
    
    def dual_retrieval(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        sparse = self.sparse_retrieval(query, top_k)
        dense = self.dense_retrieval(query, query_embedding, top_k)
        
        # Simple fusion: combine and sort by normalized scores
        def normalize_scores(results, key='score'):
            if not results:
                return []
            max_score = max(r['metadata'][key] for r in results)
            for r in results:
                r['metadata']['norm_score'] = r['metadata'][key] / max_score if max_score > 0 else 0
            return results
        
        sparse = normalize_scores(sparse, 'score')
        dense = normalize_scores(dense, 'score')
        
        all_docs = {d['metadata']['node_id']: d for d in sparse + dense}
        for doc_id, doc in all_docs.items():
            sparse_score = next((r['metadata']['norm_score'] for r in sparse if r['metadata']['node_id'] == doc_id), 0)
            dense_score = next((r['metadata']['norm_score'] for r in dense if r['metadata']['node_id'] == doc_id), 0)
            doc['metadata']['fused_score'] = (sparse_score + dense_score) / 2
        
        fused = sorted(all_docs.values(), key=lambda x: x['metadata']['fused_score'], reverse=True)[:top_k]
        
        # Enrich with graph context
        enriched = self.enrich_with_graph_context(fused, query_embedding)
        return enriched