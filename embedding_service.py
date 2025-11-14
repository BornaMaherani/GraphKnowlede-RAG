# embedding_service.py (updated)
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.graphs import Neo4jGraph

class EmbeddingService:
    """
    Service for generating embeddings and managing vector indexes in Neo4j for the new graph schema.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
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
        Create vector indexes for embeddings in Neo4j based on new schema.
        
        Args:
            graph: Neo4jGraph instance
        """
        try:
            # Define labels from new schema
            labels = [
                'Ordinary_Law', 'Single_Article_Law', 'Executive_Regulation',
                'Circular_Directive', 'Amendment_Addition', 'Related_Laws',
                'Article', 'Note_Clause', 'Sub_section', 'Sub_paragraph'
            ]
            
            for label in labels:
                # Create text index on content
                text_index_query = f"CREATE TEXT INDEX {label.lower()}_content_text IF NOT EXISTS FOR (n:{label}) ON (n.content)"
                graph.query(text_index_query)
                self.logger.debug(f"Created text index for {label}")
                
                # Create vector index on embedding
                vector_query = f"""
                CREATE VECTOR INDEX {label.lower()}_embedding_vector IF NOT EXISTS
                FOR (n:{label}) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
                graph.query(vector_query)
                self.logger.debug(f"Created vector index for {label}")
            
            # Create general fulltext index across all nodes
            fulltext_query = "CREATE FULLTEXT INDEX content_fulltext IF NOT EXISTS FOR (n:Ordinary_Law|Single_Article_Law|Executive_Regulation|Circular_Directive|Amendment_Addition|Related_Laws|Article|Note_Clause|Sub_section|Sub_paragraph) ON EACH [n.content]"
            graph.query(fulltext_query)
            self.logger.debug("Created general fulltext index")
            
            self.logger.info("Vector and text indexes created successfully for new schema")
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
            raise
    
    def update_node_embeddings(self, graph: Neo4jGraph):
        """
        Update all nodes with their embeddings based on content.
        
        Args:
            graph: Neo4jGraph instance
        """
        try:
            # Get all nodes without embeddings
            query = """
            MATCH (n)
            WHERE n.content IS NOT NULL AND n.embedding IS NULL
            RETURN elementId(n) as node_id, labels(n)[0] as label, n.content as content
            """
            nodes = graph.query(query)
            
            if nodes:
                texts = [node['content'] for node in nodes]
                embeddings = self.generate_embeddings(texts)
                
                for i, node in enumerate(nodes):
                    label = node['label']
                    update_query = f"""
                    MATCH (n:{label}) WHERE elementId(n) = $node_id
                    SET n.embedding = $embedding
                    """
                    graph.query(update_query, params={
                        "node_id": node['node_id'],
                        "embedding": embeddings[i]
                    })
                    self.logger.debug(f"Updated embedding for node {node['node_id']} ({label})")
            
            self.logger.info("All node embeddings updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update node embeddings: {e}")
            raise