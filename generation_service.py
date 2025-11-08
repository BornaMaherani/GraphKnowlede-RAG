# generation_service.py
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

class GenerationService:
    """
    Generation service using OpenRouter for RAG responses.
    """
    
    def __init__(self, 
                 openrouter_api_key: str,
                 model_name: str = "google/gemini-2.0-flash-exp:free",
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
        """Initialize the LLM with OpenRouter"""
        try:
            self.logger.info(f"Initializing LLM with model: {self.model_name}")
            
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.openrouter_api_key,
                openai_api_base=self.base_url,
                temperature=0.1,
                max_tokens=1000
            )
            
            self.logger.info("LLM initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def build_system_prompt(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Build system prompt with retrieved context.
        
        Args:
            retrieved_docs: List of retrieved documents from dual retrieval
            
        Returns:
            System prompt string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content']
            metadata = doc['metadata']
            doc_type = metadata.get('type', 'unknown')
            score = metadata.get('reranked_score', metadata.get('fused_score', 0))
            
            context_parts.append(
                f"[Document {i} - Type: {doc_type}, Relevance: {score:.3f}]\n"
                f"{content}\n"
            )
        
        context = "\n".join(context_parts)
        
        system_prompt = f"""You are a legal expert assistant specialized in Iranian regulations and laws. 
Use the following retrieved legal documents to answer the user's question accurately and comprehensively.

RETRIEVED LEGAL DOCUMENTS:
{context}

INSTRUCTIONS:
1. Base your answer strictly on the provided legal documents
2. If multiple documents are relevant, synthesize information from all of them
3. Cite specific clauses or regulations when possible
4. If the documents don't contain enough information to fully answer, state what information is available and what is missing
5. Provide clear, structured responses with legal precision
6. For complex legal questions, break down the answer into relevant sections

Respond in the same language as the user's question."""
        
        return system_prompt
    
    def generate_response(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using retrieved context.
        
        Args:
            question: User's question
            retrieved_docs: Retrieved documents from dual retrieval
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            self.logger.info(f"Generating response for question: {question}")
            
            if not retrieved_docs:
                return {
                    "response": "I couldn't find relevant legal documents to answer your question. Please try rephrasing or providing more specific details.",
                    "metadata": {
                        "retrieved_docs_count": 0,
                        "warning": "No relevant documents found"
                    }
                }
            
            # Build system prompt with retrieved context
            system_prompt = self.build_system_prompt(retrieved_docs)
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            
            response = self.llm.invoke(messages)
            
            result = {
                "response": response.content,
                "metadata": {
                    "retrieved_docs_count": len(retrieved_docs),
                    "model_used": self.model_name,
                    "retrieved_sources": [
                        {
                            "type": doc['metadata'].get('type'),
                            "score": doc['metadata'].get('reranked_score', doc['metadata'].get('fused_score', 0)),
                            "node_id": doc['metadata'].get('node_id')
                        }
                        for doc in retrieved_docs
                    ]
                }
            }
            
            self.logger.info("Response generated successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return {
                "response": "I encountered an error while generating the response. Please try again.",
                "metadata": {
                    "error": str(e),
                    "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0
                }
            }