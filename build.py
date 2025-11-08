# build.py (updated)
# usage_example.py
import logging'
import os
from regulation_graph_builder import RegulationGraphBuilder
from loader import ExcelDataLoader
from embedding_service import EmbeddingService

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Neo4j connection details
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
    
    # Excel file path
    EXCEL_FILE = "Documents/vazir/1-494.xlsx"
    
    try:
        # Initialize graph builder with LangChain Neo4jGraph
        graph_builder = RegulationGraphBuilder(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        
        # Create constraints and indexes
        graph_builder.create_constraints_and_indexes()
        
        # Initialize Excel loader
        excel_loader = ExcelDataLoader(EXCEL_FILE)
        
        # Process data and build graph
        graph_builder.process_excel_data(excel_loader, method='pandas')
        
        # Initialize embedding service and create vector indexes
        embedding_service = EmbeddingService()
        embedding_service.create_vector_indexes(graph_builder.graph)
        embedding_service.update_node_embeddings(graph_builder.graph)
        
        # Get statistics
        stats = graph_builder.get_graph_statistics()
        print(f"Graph Statistics: {stats}")
        
        # Example custom query
        regulations_count = graph_builder.query_graph("MATCH (r:Regulation) RETURN count(r) as count")
        print(f"Total regulations in graph: {regulations_count[0]['count']}")
        
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
    finally:
        if 'graph_builder' in locals():
            graph_builder.close()

if __name__ == "__main__":
    main()