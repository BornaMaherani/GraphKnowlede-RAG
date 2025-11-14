# regulation_graph_builder.py (rewritten)
from langchain_community.graphs import Neo4jGraph
from typing import List, Dict, Any
import logging
import os
from datetime import datetime
import re

class RegulationGraphBuilder:
    """
    Builds a Neo4j graph database from JSON law data using LangChain's Neo4jGraph.
    Creates nodes for laws, articles, clauses, etc., with appropriate relationships based on the new schema.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, database: str = "neo4j", log_level: int = logging.INFO):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.database = database
        self.graph = None
        self._setup_logging(log_level)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Neo4jGraph
        self._init_neo4j_graph()
        
        self.logger.info("RegulationGraphBuilder initialized with LangChain Neo4jGraph")
    
    def _setup_logging(self, log_level: int):
        """Setup comprehensive logging configuration"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'law_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
    
    def _init_neo4j_graph(self):
        """Initialize LangChain Neo4jGraph"""
        try:
            self.graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                database=self.database
            )
            self.logger.info("LangChain Neo4jGraph initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4jGraph: {e}", exc_info=True)
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.graph:
            # Neo4jGraph doesn't have explicit close method in current version
            # The connection is typically managed automatically
            self.logger.info("Neo4jGraph connection active")
    
    def create_constraints_and_indexes(self):
        """
        Create necessary constraints and indexes for optimal performance.
        For the new schema, we create general indexes (e.g., on content for potential searches).
        Unique constraints on unique_id are created dynamically during processing.
        """
        indexes_queries = [
            # Removed invalid general text index
        ]
        
        try:
            for query in indexes_queries:
                self.graph.query(query)
                self.logger.debug(f"Executed index query: {query}")
            
            self.logger.info("Successfully created general Neo4j indexes")
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}", exc_info=True)
            raise
    
    def normalize_label(self, type_str: str) -> str:
        """
        Normalize node type to a valid Neo4j label (replace spaces, slashes, hyphens with underscores).
        """
        return re.sub(r'[\s/\\-]+', '_', type_str).strip('_')
    
    def process_json_data(self, json_loader):
        """
        Process JSON data (nodes and edges) and build the graph database based on the new schema.
        """
        self.logger.info("Starting JSON data processing for graph construction")
        
        try:
            # Load nodes and edges from JSON
            nodes = json_loader.get_all_nodes()
            edges = json_loader.get_all_edges()
            self.logger.info(f"Loaded {len(nodes)} nodes and {len(edges)} edges from JSON")
            
            # Collect unique labels and create constraints dynamically
            unique_labels = set(self.normalize_label(node['type']) for node in nodes)
            for label in unique_labels:
                constraint_query = f"CREATE CONSTRAINT {label}_unique_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.unique_id IS UNIQUE"
                self.graph.query(constraint_query)
                self.logger.debug(f"Created unique constraint for label: {label}")
            
            # Create or update nodes
            for node in nodes:
                label = self.normalize_label(node['type'])
                props = node.get('properties', {})
                props['id'] = node['id']
                props['unique_id'] = node['unique_id']
                
                query = f"""
                MERGE (n:{label} {{unique_id: $unique_id}})
                SET n += $props
                """
                self.graph.query(query, params={"unique_id": props['unique_id'], "props": props})
                self.logger.debug(f"Created/Updated node: {node['unique_id']} with label {label}")
            
            # Create text indexes per label on content after nodes are created
            for label in unique_labels:
                index_query = f"CREATE TEXT INDEX {label}_content_index IF NOT EXISTS FOR (n:{label}) ON (n.content)"
                self.graph.query(index_query)
                self.logger.debug(f"Created text index for label: {label}")
            
            # Create relationships (edges)
            for edge in edges:
                rel_type = edge['type']
                props = edge.get('properties', {})
                source_id = edge['source_id']
                target_id = edge['target_id']
                
                query = f"""
                MATCH (s) WHERE s.unique_id = $source_id
                MATCH (t) WHERE t.unique_id = $target_id
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $props
                """
                self.graph.query(query, params={"source_id": source_id, "target_id": target_id, "props": props})
                self.logger.debug(f"Created/Updated relationship: {source_id} -[{rel_type}]-> {target_id}")
            
            self.logger.info(f"Graph construction completed: {len(nodes)} nodes, {len(edges)} relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to process JSON data: {e}", exc_info=True)
            raise
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph database.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            # Count nodes by label
            node_query = """
            CALL apoc.meta.stats()
            YIELD labels
            RETURN labels
            """
            node_result = self.graph.query(node_query)
            labels = node_result[0]['labels'] if node_result else {}
            
            # Count relationships by type
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            """
            rel_results = self.graph.query(rel_query)
            relationships = {record['rel_type']: record['count'] for record in rel_results}
            
            stats = {
                'node_counts': labels,
                'relationship_counts': relationships,
                'total_nodes': sum(labels.values()),
                'total_relationships': sum(relationships.values())
            }
            
            self.logger.info(f"Graph statistics: {stats}")
            return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get graph statistics: {e}", exc_info=True)
            return {}
    
    def clear_database(self):
        """
        Clear all data from the database (use with caution!).
        """
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            self.logger.warning("Cleared all data from the database")
        except Exception as e:
            self.logger.error(f"Failed to clear database: {e}", exc_info=True)
            raise
    
    def query_graph(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query on the graph.
        
        Args:
            query: Cypher query string
            params: Parameters for the query
            
        Returns:
            Query results
        """
        try:
            result = self.graph.query(query, params=params or {})
            self.logger.debug(f"Executed custom query: {query}")
            return result
        except Exception as e:
            self.logger.error(f"Query failed: {e}", exc_info=True)
            raise