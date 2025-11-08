# qa_service.py
import logging
from typing import Dict, Any
from embedding_service import EmbeddingService
from retrieval_service import RetrievalService
from generation_service import GenerationService
from langchain_community.graphs import Neo4jGraph

class QAService:
    """
    Main QA service that orchestrates retrieval and generation.
    """
    
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_user: str, 
                 neo4j_password: str,
                 openrouter_api_key: str,
                 database: str = "neo4j"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openrouter_api_key = openrouter_api_key
        self.database = database
        
        self.graph = None
        self.embedding_service = None
        self.retrieval_service = None
        self.generation_service = None
        
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_services()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_services(self):
        """Initialize all services"""
        try:
            # Initialize Neo4j graph
            self.logger.info("Initializing Neo4j graph connection...")
            self.graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                database=self.database
            )
            
            # Initialize embedding service
            self.logger.info("Initializing embedding service...")
            self.embedding_service = EmbeddingService()
            
            # Create vector indexes
            self.embedding_service.create_vector_indexes(self.graph)
            
            # Update embeddings for existing nodes
            self.embedding_service.update_node_embeddings(self.graph)
            
            # Initialize retrieval service
            self.logger.info("Initializing retrieval service...")
            self.retrieval_service = RetrievalService(self.graph)
            
            # Initialize generation service
            self.logger.info("Initializing generation service...")
            self.generation_service = GenerationService(
                openrouter_api_key=self.openrouter_api_key
            )
            
            self.logger.info("All services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            raise
    
    def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main method to answer a question using RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            Complete answer with metadata
        """
        try:
            self.logger.info(f"Processing question: {question}")
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embeddings([question])[0]
            
            # Perform dual retrieval
            retrieved_docs = self.retrieval_service.dual_retrieval(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Generate response
            response = self.generation_service.generate_response(question, retrieved_docs)
            
            # Add retrieval metadata to response
            response['metadata']['retrieval_details'] = {
                'total_docs_retrieved': len(retrieved_docs),
                'query_embedding_length': len(query_embedding)
            }
            
            self.logger.info("Question processing completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Question processing failed: {e}")
            return {
                "response": "I encountered an error while processing your question. Please try again.",
                "metadata": {
                    "error": str(e)
                }
            }
    
    def close(self):
        """Clean up resources"""
        if self.graph:
            # Neo4jGraph connection cleanup if needed
            pass