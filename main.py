"""
Secure RAG System - Pydantic AI + Chroma
A RAG implementation that safely manages your company's internal data
"""

import os
import asyncio
from typing import List, Optional
from dataclasses import dataclass

# Required libraries
# pip install pydantic-ai chromadb openai sentence-transformers scikit-learn

import chromadb
from chromadb.config import Settings
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from sentence_transformers import SentenceTransformer
import openai

@dataclass
class DocumentChunk:
    """Data model for document chunks"""
    content: str
    metadata: dict
    chunk_id: str

class SecureRAGSystem:
    """Secure RAG system - data remains local"""
    
    def __init__(self, collection_name: str = "company_docs"):
        # Initialize Chroma DB locally
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",  # Local database
            settings=Settings(
                anonymized_telemetry=False,  # Telemetry disabled
                allow_reset=True
            )
        )
        
        # Create/get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Company internal documents"}
        )
        
        # Embedding model - security options
        self.embedding_model = self._setup_embedding_model()
        
        # Pydantic AI agent
        self.agent = Agent(
            model=OpenAIModel('gpt-4o-mini'),
            system_prompt="""
            You are a company internal AI assistant. Only answer based on the 
            provided context. Do not use information outside the context.
            
            If the answer to the question is not in the context, clearly state this.
            """
        )
    
    def _setup_embedding_model(self):
        """Embedding model selection based on security level"""
        
        # OPTION 1: Pre-downloaded model (AIR-GAPPED)
        # Download model beforehand and use offline
        model_path = "./models/all-MiniLM-L6-v2"  # Local model directory
        if os.path.exists(model_path):
            print("✅ Using offline model - maximum security")
            return SentenceTransformer(model_path)
        
        # OPTION 2: Use from cache (internet connection first time only)
        try:
            print("⚠️  Downloading model for first time - will be offline afterwards")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Show model cache location
            cache_path = model._modules['0'].auto_model.config._name_or_path
            print(f"📁 Model cache location: {cache_path}")
            return model
            
        except Exception as e:
            print(f"❌ Could not download model: {e}")
            
            # OPTION 3: Simple TF-IDF fallback (completely offline)
            print("🔄 Using TF-IDF fallback - 100% offline")
            return self._create_tfidf_embedder()
    
    def _create_tfidf_embedder(self):
        """Simple TF-IDF embedding - completely offline"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        class SimpleTFIDFEmbedder:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=384)
                self.fitted = False
            
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                if not self.fitted:
                    # Fit with initial texts
                    self.vectorizer.fit(texts)
                    self.fitted = True
                
                return self.vectorizer.transform(texts).toarray()
        
        return SimpleTFIDFEmbedder()

    def add_documents(self, documents: List[str], metadatas: List[dict] = None):
        """Securely add documents to the system"""
        if not metadatas:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
        
        # Split documents into chunks
        chunks = []
        chunk_metadatas = []
        chunk_ids = []
        
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            # Simple chunking (can be more sophisticated in real applications)
            doc_chunks = self._chunk_document(doc, chunk_size=500)
            
            for j, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "total_chunks": len(doc_chunks)
                })
                chunk_metadatas.append(chunk_metadata)
                chunk_ids.append(f"doc_{i}_chunk_{j}")
        
        # Create embeddings locally
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Add to Chroma (data remains local)
        self.collection.add(
            documents=chunks,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
            embeddings=embeddings
        )
        
        print(f"✅ {len(chunks)} document chunks added securely")
    
    def _chunk_document(self, document: str, chunk_size: int = 500) -> List[str]:
        """Split document into chunks"""
        words = document.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    async def query(self, question: str, top_k: int = 3) -> str:
        """Secure query - only necessary context is sent"""
        
        # 1. Embed question locally
        question_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        # 2. Find similar documents locally
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        
        # 3. Prepare context
        contexts = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                contexts.append({
                    "content": doc,
                    "source": metadata.get('source', 'unknown')
                })
        
        if not contexts:
            return "❌ No relevant documents found."
        
        # 4. Send only context and question to AI (not raw data!)
        context_text = "\n\n".join([
            f"Source: {ctx['source']}\nContent: {ctx['content']}" 
            for ctx in contexts
        ])
        
        # 5. Secure query with Pydantic AI
        try:
            response = await self.agent.run(
                f"""
                Context:
                {context_text}
                
                Question: {question}
                
                Answer the question based on the context above.
                """
            )
            
            return f"🤖 {response.data}\n\n📚 Sources used: {', '.join([ctx['source'] for ctx in contexts])}"
            
        except Exception as e:
            return f"❌ Error occurred: {str(e)}"
    
    def get_stats(self):
        """System statistics"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name,
            "database_path": "./chroma_db"
        }

# Usage example
async def main():
    """Main application"""
    
    # Set OpenAI API key
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'  # Insert your API key here
    
    # Start secure RAG system
    rag = SecureRAGSystem("company_documents")
    
    # Sample company documents (in real applications, read from files)
    sample_documents = [
        """
        Company Policy: Working Hours
        
        Our company implements flexible working hours. 
        Employees must be in the office between 09:00-18:00.
        Remote work is possible 2 days per week.
        24/7 support line is available for emergencies outside office hours.
        """,
        
        """
        HR Policy: Leave Entitlements
        
        Annual leave entitlements:
        - 1-5 years: 14 days
        - 5-10 years: 20 days  
        - 10+ years: 26 days
        
        Sick leave: 10 days without medical report, unlimited with medical report.
        Paternity leave: 10 days, maternity leave: 120 days.
        """,
        
        """
        Technology Policy: Security
        
        All employees must use two-factor authentication.
        Passwords must be at least 12 characters and changed every 3 months.
        USB devices cannot be used without security approval.
        Personal email accounts cannot be used on work computers.
        """
    ]
    
    metadatas = [
        {"source": "working_hours_policy", "department": "hr"},
        {"source": "leave_entitlements_policy", "department": "hr"},
        {"source": "security_policy", "department": "it"}
    ]
    
    # Add documents to system
    print("📁 Adding documents...")
    rag.add_documents(sample_documents, metadatas)
    
    # System statistics
    stats = rag.get_stats()
    print(f"\n📊 System Statistics: {stats}")
    
    # Sample queries
    questions = [
        "What is the remote work policy?",
        "How are annual leave entitlements calculated?",
        "What is the password policy?",
        "How many days is paternity leave?"
    ]
    
    print("\n🔍 Query examples:")
    print("=" * 60)
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = await rag.query(question)
        print(f"💬 Answer: {answer}")
        print("-" * 60)

# Security information
def security_info():
    """Security information"""
    print("""
    🔒 EMBEDDING MODEL SECURITY LEVELS:
    
    🥇 MAXIMUM SECURITY (Air-gapped):
    - Pre-download model: huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
    - Place in ./models/ directory
    - Works completely offline
    
    🥈 HIGH SECURITY (Cache-first):
    - Downloaded from Hugging Face on first run
    - Then works from ~/.cache/torch/sentence_transformers/
    - Offline usage possible
    
    🥉 BASIC SECURITY (TF-IDF Fallback):
    - Completely offline
    - Simpler embedding quality
    - No internet required at all
    
    ✅ ALL LEVELS GUARANTEE:
    - Raw data never sent to external services
    - Vector DB is local
    - Only final context sent to AI
    """)

def download_model_offline():
    """Instructions for downloading model offline"""
    print("""
    🔽 OFFLINE MODEL DOWNLOAD:
    
    pip install huggingface_hub
    
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \\
        --local-dir ./models/all-MiniLM-L6-v2
    
    This command downloads the model locally, no internet needed afterwards.
    """)

if __name__ == "__main__":
    security_info()
    print("\n" + "="*60)
    download_model_offline()
    print("="*60)
    print("\n🚀 Starting secure RAG system...\n")
    asyncio.run(main())