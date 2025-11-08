# embedding_service.py
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.graphs import Neo4jGraph

class EmbeddingService:
    """
    Service for generating embeddings and managing vector indexes in Neo4j.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_model()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts).tolist()
            self.logger.info("Embeddings generated successfully")
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def create_vector_indexes(self, graph: Neo4jGraph):
        """
        Create vector indexes for embeddings in Neo4j.
        
        Args:
            graph: Neo4jGraph instance
        """
        try:
            # Create full-text index for Clause.text
            fulltext_queries = [
                # Full-text index for Clause text
                "CREATE FULLTEXT INDEX clause_text_fulltext IF NOT EXISTS FOR (c:Clause) ON EACH [c.text]",
                
                # Full-text index for Regulation title  
                "CREATE FULLTEXT INDEX regulation_title_fulltext IF NOT EXISTS FOR (r:Regulation) ON EACH [r.title]",
                
                # Vector index for Clause embeddings
                """
                CREATE VECTOR INDEX clause_embedding_vector IF NOT EXISTS
                FOR (c:Clause) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                
                # Vector index for Regulation embeddings
                """
                CREATE VECTOR INDEX regulation_embedding_vector IF NOT EXISTS
                FOR (r:Regulation) ON (r.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
                """
            ]
            
            for query in fulltext_queries:
                graph.query(query)
                self.logger.debug(f"Executed index query: {query}")
            
            self.logger.info("Vector and full-text indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create vector indexes: {e}")
            raise
    
    def update_node_embeddings(self, graph: Neo4jGraph):
        """
        Update nodes with their embeddings.
        
        Args:
            graph: Neo4jGraph instance
        """
        try:
            # Update Clause embeddings
            self.logger.info("Updating Clause embeddings...")
            clauses_query = """
            MATCH (c:Clause) 
            WHERE c.text IS NOT NULL AND c.embedding IS NULL
            RETURN c.id as clause_id, c.text as text
            """
            clauses = graph.query(clauses_query)
            
            if clauses:
                texts = [clause['text'] for clause in clauses if clause['text']]
                embeddings = self.generate_embeddings(texts)
                
                for i, clause in enumerate(clauses):
                    if i < len(embeddings):
                        update_query = """
                        MATCH (c:Clause {id: $clause_id})
                        SET c.embedding = $embedding
                        """
                        graph.query(update_query, params={
                            "clause_id": clause['clause_id'],
                            "embedding": embeddings[i]
                        })
            
            # Update Regulation embeddings
            self.logger.info("Updating Regulation embeddings...")
            regulations_query = """
            MATCH (r:Regulation) 
            WHERE r.title IS NOT NULL AND r.embedding IS NULL
            RETURN r.number as regulation_number, r.title as title
            """
            regulations = graph.query(regulations_query)
            
            if regulations:
                texts = [reg['title'] for reg in regulations if reg['title']]
                embeddings = self.generate_embeddings(texts)
                
                for i, reg in enumerate(regulations):
                    if i < len(embeddings):
                        update_query = """
                        MATCH (r:Regulation {number: $regulation_number})
                        SET r.embedding = $embedding
                        """
                        graph.query(update_query, params={
                            "regulation_number": reg['regulation_number'],
                            "embedding": embeddings[i]
                        })
            
            self.logger.info("Node embeddings updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update node embeddings: {e}")
            raise