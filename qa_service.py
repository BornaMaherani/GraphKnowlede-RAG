# qa_service.py (rewritten)
import logging
from typing import Dict, Any, List
from embedding_service import EmbeddingService
from retrieval_service import RetrievalService
from generation_service import GenerationService
from langchain_community.graphs import Neo4jGraph

class QAService:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openrouter_api_key: str, database: str = "neo4j"):
        self.graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password, database=database)
        self.embedding_service = EmbeddingService()
        self.embedding_service.create_vector_indexes(self.graph)
        self.embedding_service.update_node_embeddings(self.graph)
        self.retrieval_service = RetrievalService(self.graph)
        self.generation_service = GenerationService(openrouter_api_key)
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            query_embedding = self.embedding_service.generate_embeddings([question])[0]
            retrieved_docs = self.retrieval_service.dual_retrieval(question, query_embedding, top_k)
            logging.info(retrieved_docs)
            response = self.generation_service.generate_response(question, retrieved_docs)
            response['metadata']['retrieval_details'] = {'docs': len(retrieved_docs)}
            return response
        except Exception as e:
            return {"response": "Error", "metadata": {"error": f"{e}"}}