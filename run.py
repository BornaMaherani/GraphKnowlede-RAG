# example_usage.py
import os
from qa_service import QAService
from dotenv import load_dotenv
load_dotenv()



def main():
    # Configuration
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    
    # Initialize QA service
    qa_service = QAService(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        openrouter_api_key=OPENROUTER_API_KEY,
        database=NEO4J_DATABASE
    )
    
    # Example questions
    questions = [
        "بند 1 نحوه رسیدگی، تشخیص و تسویه بقایای مالیاتی چخ می گوید؟",
        "موسسات بیمه در مورد تراز نامه سال مالی چه کاری باید انجام دهند؟",
        "چه کسی قانون نحوه رسیدگی و تشخیص بقایای مالیاتی را صادر کرده؟ و در چه تاریخی صادر شده؟",
        # "Find clauses related to environmental protection",
        # "What laws affect regulation number 123?",
        # "Explain the requirements for business licensing"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")
        
        result = qa_service.ask_question(question, top_k=5)
        
        print(f"RESPONSE: {result['response']}")
        print(f"\nMetadata: {result['metadata']}")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    main()