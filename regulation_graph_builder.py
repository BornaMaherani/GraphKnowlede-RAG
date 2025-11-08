import pandas as pd
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.graphs import Neo4jGraph
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime
import uuid

class RegulationGraphBuilder:
    """
    Builds a Neo4j graph database from Excel regulation data using LangChain's Neo4jGraph.
    Creates Regulations, Clauses, and Laws with appropriate relationships.
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
                logging.FileHandler(f'regulation_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
        """
        constraints_queries = [
            # Unique constraint for Regulation number
            "CREATE CONSTRAINT regulation_number_unique IF NOT EXISTS FOR (r:Regulation) REQUIRE r.number IS UNIQUE",
            
            # Index for Regulation type
            "CREATE INDEX regulation_type_index IF NOT EXISTS FOR (r:Regulation) ON (r.type)",
            
            # Index for Regulation status
            "CREATE INDEX regulation_status_index IF NOT EXISTS FOR (r:Regulation) ON (r.status)",
            
            # Index for Clause kind
            "CREATE INDEX clause_kind_index IF NOT EXISTS FOR (c:Clause) ON (c.kind)",
            
            # Index for Law text (for faster lookups)
            "CREATE INDEX law_text_index IF NOT EXISTS FOR (l:Law) ON (l.text)"
        ]
        
        try:
            for query in constraints_queries:
                self.graph.query(query)
                self.logger.debug(f"Executed constraint/index query: {query}")
            
            self.logger.info("Successfully created Neo4j constraints and indexes")
        except Exception as e:
            self.logger.error(f"Failed to create constraints and indexes: {e}", exc_info=True)
            raise
    
    def _clean_data_value(self, value: Any) -> Any:
        """
        Clean and prepare data values for Neo4j insertion.
        """
        if pd.isna(value) or value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return value
    
    def create_regulation_node(self, regulation_data: Dict[str, Any]) -> str:
        """
        Create a Regulation node in Neo4j.
        
        Args:
            regulation_data: Dictionary containing regulation properties
            
        Returns:
            Regulation number
        """
        regulation_number = self._clean_data_value(regulation_data.get('شماره'))
        if not regulation_number:
            self.logger.warning("Skipping regulation with missing number")
            return None
        
        properties = {
            'number': regulation_number,
            'title': self._clean_data_value(regulation_data.get('عنوان/موضوع')),
            'type': self._clean_data_value(regulation_data.get('نوع')),
            'valid_from': self._clean_data_value(regulation_data.get('تاریخ تصویب')),
            'issued_on': self._clean_data_value(regulation_data.get('تاریخ صدور')),
            'effective_from': self._clean_data_value(regulation_data.get('تاریخ لازم الاجرا')),
            'issuer': self._clean_data_value(regulation_data.get('مرجع صادرکننده')),
            'status': self._clean_data_value(regulation_data.get('وضعیت نهایی'))
        }
        
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}
        
        try:
            query = """
            MERGE (r:Regulation {number: $number})
            SET r += $properties
            RETURN r.number
            """
            result = self.graph.query(query, params={"number": regulation_number, "properties": properties})
            if result and len(result) > 0:
                regulation_number = result[0]['r.number']
                self.logger.debug(f"Created/Updated Regulation: {regulation_number}")
                return regulation_number
            return None
                
        except Exception as e:
            self.logger.error(f"Failed to create regulation node {regulation_number}: {e}", exc_info=True)
            return None
    
    def create_clause_node(self, clause_data: Dict[str, Any], regulation_number: str) -> str:
        """
        Create a Clause node and link it to its Regulation.
        
        Args:
            clause_data: Dictionary containing clause properties
            regulation_number: The regulation this clause belongs to
            
        Returns:
            Clause ID
        """
        clause_id = str(uuid.uuid4())  # Generate unique ID for clause
        
        properties = {
            'id': clause_id,
            'kind': self._clean_data_value(clause_data.get('جزء، بند، تبصره یا ماده')),
            'text': self._clean_data_value(clause_data.get('متن'))
        }
        
        # Remove None values but keep id
        properties = {k: v for k, v in properties.items() if v is not None or k == 'id'}
        
        try:
            # Create clause and link to regulation
            query = """
            MATCH (r:Regulation {number: $regulation_number})
            CREATE (c:Clause $properties)
            CREATE (r)-[:HAS_CLAUSE]->(c)
            RETURN c.id
            """
            result = self.graph.query(
                query, 
                params={
                    "regulation_number": regulation_number, 
                    "properties": properties
                }
            )
            if result and len(result) > 0:
                clause_id = result[0]['c.id']
                self.logger.debug(f"Created Clause {clause_id} for Regulation {regulation_number}")
                return clause_id
            return None
                
        except Exception as e:
            self.logger.error(f"Failed to create clause for regulation {regulation_number}: {e}", exc_info=True)
            return None
    
    def create_law_relationship(self, regulation_number: str, law_text: str):
        """
        Create a Law node and AFFECTED_BY relationship.
        
        Args:
            regulation_number: The regulation affected by the law
            law_text: The text of the law
        """
        law_text_clean = self._clean_data_value(law_text)
        if not law_text_clean:
            return
        
        try:
            # Create law and relationship
            query = """
            MATCH (r:Regulation {number: $regulation_number})
            MERGE (l:Law {text: $law_text})
            MERGE (r)-[:AFFECTED_BY]->(l)
            """
            self.graph.query(
                query, 
                params={
                    "regulation_number": regulation_number, 
                    "law_text": law_text_clean
                }
            )
            self.logger.debug(f"Created AFFECTED_BY relationship for Regulation {regulation_number}")
                
        except Exception as e:
            self.logger.error(f"Failed to create law relationship for regulation {regulation_number}: {e}", exc_info=True)
    
    def create_clause_sequence(self, regulation_number: str, clause_ids: List[str]):
        """
        Create NEXT_CLAUSE relationships between clauses of the same regulation.
        
        Args:
            regulation_number: The regulation containing the clauses
            clause_ids: List of clause IDs in order
        """
        if len(clause_ids) < 2:
            return
        
        try:
            # Create sequential relationships between clauses
            for i in range(len(clause_ids) - 1):
                current_clause_id = clause_ids[i]
                next_clause_id = clause_ids[i + 1]
                
                query = """
                MATCH (c1:Clause {id: $current_id})
                MATCH (c2:Clause {id: $next_id})
                WHERE EXISTS((:Regulation {number: $regulation_number})-[:HAS_CLAUSE]->(c1))
                AND EXISTS((:Regulation {number: $regulation_number})-[:HAS_CLAUSE]->(c2))
                MERGE (c1)-[:NEXT_CLAUSE]->(c2)
                """
                self.graph.query(
                    query,
                    params={
                        "current_id": current_clause_id,
                        "next_id": next_clause_id,
                        "regulation_number": regulation_number
                    }
                )
            
            self.logger.debug(f"Created NEXT_CLAUSE sequence for {len(clause_ids)} clauses in Regulation {regulation_number}")
                
        except Exception as e:
            self.logger.error(f"Failed to create clause sequence for regulation {regulation_number}: {e}", exc_info=True)
    
    def process_excel_data(self, excel_loader, method: str = 'pandas'):
        """
        Process Excel data and build the graph database.
        
        Args:
            excel_loader: Instance of ExcelDataLoader
            method: 'pandas' or 'langchain' for data loading
        """
        self.logger.info("Starting Excel data processing for graph construction")
        
        try:
            # Load data from Excel
            rows = excel_loader.get_rows_as_dicts(method)
            self.logger.info(f"Loaded {len(rows)} rows from Excel")
            
            # Group rows by regulation number
            regulations_dict = {}
            for row in rows:
                regulation_number = self._clean_data_value(row.get('شماره'))
                if regulation_number:
                    if regulation_number not in regulations_dict:
                        regulations_dict[regulation_number] = []
                    regulations_dict[regulation_number].append(row)
            
            self.logger.info(f"Found {len(regulations_dict)} unique regulations")
            
            total_clauses = 0
            total_laws = 0
            
            # Process each regulation
            for regulation_number, regulation_rows in regulations_dict.items():
                self.logger.info(f"Processing regulation: {regulation_number} with {len(regulation_rows)} clauses")
                
                # Create regulation node (use first row for regulation properties)
                regulation_data = regulation_rows[0]
                created_regulation = self.create_regulation_node(regulation_data)
                
                if not created_regulation:
                    self.logger.warning(f"Skipping regulation {regulation_number} due to creation failure")
                    continue
                
                clause_ids = []
                
                # Create clauses for this regulation
                for clause_data in regulation_rows:
                    clause_id = self.create_clause_node(clause_data, regulation_number)
                    if clause_id:
                        clause_ids.append(clause_id)
                        total_clauses += 1
                    
                    # Create law relationships
                    law_text = clause_data.get('قانون یا مقرره اثرگذار')
                    if law_text:
                        self.create_law_relationship(regulation_number, law_text)
                        total_laws += 1
                
                # Create clause sequence
                self.create_clause_sequence(regulation_number, clause_ids)
                
                self.logger.debug(f"Completed processing regulation {regulation_number}")
            
            self.logger.info(f"Graph construction completed: {len(regulations_dict)} regulations, {total_clauses} clauses, {total_laws} law relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to process Excel data: {e}", exc_info=True)
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