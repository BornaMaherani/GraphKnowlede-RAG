# generation_service.py (rewritten for new schema)
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class GenerationService:
    """
    Generation service for RAG responses with new graph schema.
    """
    
    def __init__(self, 
                 openrouter_api_key: str,
                 model_name: str = "tngtech/deepseek-r1t2-chimera:free",
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.openrouter_api_key = openrouter_api_key
        self.model_name = model_name
        self.base_url = base_url
        self.llm = None
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_llm()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_llm(self):
        """Initialize the LLM"""
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.openrouter_api_key,
                openai_api_base=self.base_url,
                temperature=0.1,
                max_tokens=1500
            )
            self.logger.info("LLM initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _format_node_context(self, node_type: str, node_props: Dict[str, Any]) -> str:
        lines = [f"{node_type}: {node_props.get('id', 'Unknown')}"]
        for key, value in sorted(node_props.items()):
            if key not in ['id', 'unique_id', 'embedding']:
                if isinstance(value, list):
                    value_str = ', '.join(map(str, value))
                else:
                    value_str = str(value)
                lines.append(f"  {key.capitalize()}: {value_str}")
        return "\n".join(lines)
    
    def _format_complete_context(self, doc: Dict[str, Any]) -> str:
        lines = []
        metadata = doc.get('metadata', {})
        node_type = metadata.get('type', 'Unknown')
        node_props = metadata.get('node', {})
        related = metadata.get('related_nodes', [])
        
        lines.append(f"\n[Document - Type: {node_type.upper()}, Score: {metadata.get('fused_score', 0):.3f}]")
        lines.append("=" * 60)
        lines.append(self._format_node_context(node_type, node_props))
        
        if related:
            lines.append("\nRelated Nodes:")
            lines.append("-" * 40)
            for rel in related:
                rel_type = rel.get('type', 'Unknown')
                rel_props = rel.get('properties', {})
                lines.append(self._format_node_context(rel_type, rel_props))
                lines.append("-" * 40)
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def build_system_prompt(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        parts = [
            """You are a highly specialized legal AI assistant focused on Iranian government tax laws and regulations, particularly the Direct Tax Law (قانون مالیات‌های مستقیم - ق.م.م) and related legal frameworks. Your responses must always be in formal, precise legal language in Farsi (Persian), using professional terminology consistent with Iranian legal standards. Base your answers strictly on the provided retrieved documents from the graph database. Do not add external knowledge, assumptions, or unsubstantiated information. If the retrieved documents do not sufficiently address the query, state this clearly and suggest rephrasing the question for better retrieval.

### Graph Database Schema
The graph database is structured as follows:

#### Nodes:
Each node represents a legal entity with a label (type) and properties. Key node types include:
- **Ordinary_Law**: Represents main laws like "قانون مالیات‌های مستقیم" (ق.م.م). Properties: id (e.g., "ق.م.م"), unique_id (e.g., "qmm"), title (e.g., "قانون مالیات‌های مستقیم"), content (text of the law), keywords (array of relevant terms).
- **Single_Article_Law**: Laws with a single article.
- **Executive_Regulation**: Executive bylaws (آیین‌نامه اجرایی).
- **Circular_Directive**: Circulars or directives (بخشنامه یا دستورالعمل).
- **Amendment_Addition**: Amendments or additions (اصلاحیه یا الحاقیه).
- **Related_Laws**: Related or dependent laws (قوانین مرتبط).
- **Article**: Main articles (ماده اصلی, e.g., "ماده ۱ ق.م.م").
- **Note_Clause**: Notes or clauses (تبصره, e.g., "تبصره ۱ ماده ۱ ق.م.م").
- **Sub_section**: Sub-sections (بند, e.g., "بند ۱ ماده ۱ ق.م.م").
- **Sub_paragraph**: Sub-paragraphs (جزء, e.g., "جزء ۱ بند الف ماده ۱ ق.م.م").

All nodes may have: content (main text), keywords (array), and other properties like title or id for hierarchy.

#### Edges (Relationships):
Edges connect nodes and have types with properties:
- **AMENDS**: Current document amends another (properties: reason).
- **REPEALS**: Current document repeals another (properties: reason).
- **INTERPRETS**: Current document interprets another (properties: reason).
- **REFERS_TO**: Current document refers to another without direct change (properties: reason).
- **IS_TEMPORARY**: Document is temporary (properties: reason, expiration_date in YYYY-MM-DD if available).

Use these relationships to infer connections, e.g., traverse REFERS_TO for referenced laws or AMENDS for modifications.

### Retrieved Documents Format
Retrieved documents are provided as a JSON array of objects. Each object represents a relevant node with:
- **content**: The main text of the node.
- **metadata**:
  - **node_id**: Internal Neo4j ID (e.g., "4:bc14718e-4af8-4d41-a6ba-24b068de627f:603").
  - **type**: Node label (e.g., "Sub_section").
  - **node**: Detailed properties of the node, including:
    - unique_id (e.g., "qmm_m402_b2").
    - id (e.g., "بند ب ماده ۴۰۲ ق.م.م").
    - keywords (array, e.g., ["نام", "ابلاغ", "نسبت", "مؤدی"]).
    - content (same as top-level content).
  - **score**, **norm_score**, **fused_score**: Relevance scores (higher is better; fused_score combines sparse and dense search).
  - **related_nodes**: Array of connected nodes (via edges), each with type and properties (similar to node structure). Use these to expand context, e.g., related articles or amendments.

Documents are ranked by fused_score descending. Cite sources using node.id and node.type (e.g., "بر اساس بند ب ماده ۴۰۲ ق.م.م (Sub_section)...").

### Response Guidelines
- Always respond in Farsi, using formal legal tone (e.g., avoid contractions, use precise terms like "مطابق ماده" instead of casual language).
- Structure answers clearly: Start with a summary, then detailed explanation with citations, and end with implications if relevant.
- Cite every claim: Reference exact node.id and type from retrieved docs (e.g., "مطابق بند ب ماده ۴۰۲ ق.م.م...").
- Consider hierarchy and relationships: E.g., interpret a Sub_section in context of its parent Article.
- If query involves multiple nodes, compare/contrast them.
- If no relevant docs: State "اطلاعات کافی در اسناد بازیابی‌شده موجود نیست" and suggest refinements.
- Keep responses concise, accurate, and objective."""
        ]
        if retrieved_docs:
            parts.append("\n\n### Retrieved Documents\n")
            for doc in retrieved_docs:
                parts.append(self._format_complete_context(doc))
        else:
            parts.append("\nNo relevant documents found.")
        return "\n".join(parts)
    
    def generate_response(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            system_prompt = self.build_system_prompt(retrieved_docs)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            response = self.llm.invoke(messages)
            return {"response": response.content, "metadata": {"docs": len(retrieved_docs)}}
        except Exception as e:
            return {"response": "Error", "metadata": {"error": str(e)}}