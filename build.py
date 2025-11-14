# build.py (rewritten)
import logging
import os
from regulation_graph_builder import RegulationGraphBuilder
from loader import JsonDataLoader  # Assuming JsonDataLoader is implemented as per the provided code
from embedding_service import EmbeddingService
from dotenv import load_dotenv
load_dotenv()

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Neo4j connection details
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
    
    # JSON file paths (adjust these to your actual file paths)
    JSON_FILES = [
        "law_nodes.json",
        "law_edges.json",
        # Add if exists, or other relevant JSONs
        # Add other JSON files as needed, e.g., "excel_edges.json" if separate
    ]
    
    try:
        # Initialize graph builder with LangChain Neo4jGraph
        graph_builder = RegulationGraphBuilder(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        
        # Create constraints and indexes based on new schema
        graph_builder.create_constraints_and_indexes()
        
        # Initialize JSON loader
        json_loader = JsonDataLoader(JSON_FILES)
        
        # Process data and build graph
        graph_builder.process_json_data(json_loader)
        
        # Initialize embedding service and create vector indexes
        embedding_service = EmbeddingService()
        embedding_service.create_vector_indexes(graph_builder.graph)
        embedding_service.update_node_embeddings(graph_builder.graph)
        
        # Get statistics
        stats = graph_builder.get_graph_statistics()
        print(f"Graph Statistics: {stats}")
        
        # Example custom query
        nodes_count = graph_builder.query_graph("MATCH (n) RETURN count(n) as count")
        print(f"Total nodes in graph: {nodes_count[0]['count']}")
        
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
    finally:
        if 'graph_builder' in locals():
            graph_builder.close()

if __name__ == "__main__":
    main()