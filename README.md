# 🔒 Secure RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with Pydantic AI and Chroma that prioritizes data security and privacy for enterprise environments.

## 🎯 Overview

This RAG system ensures your sensitive company data never leaves your infrastructure while still leveraging powerful AI capabilities. Raw documents are processed locally, embeddings are generated on-premises, and only minimal context is sent to external AI services.

## ✨ Key Features

- **🔐 Data Sovereignty**: Raw documents never leave your system
- **🏠 Local Processing**: Embeddings generated locally with SentenceTransformers
- **📊 Local Vector DB**: Chroma database runs entirely on your infrastructure
- **🛡️ Minimal Data Exposure**: Only selected context sent to AI APIs
- **⚡ Multiple Security Levels**: Air-gapped, cache-first, or fallback options
- **🚀 Production Ready**: Built with Pydantic AI for type safety and reliability
- **📈 Scalable**: Easy to extend and modify for enterprise needs

## 🔒 Security Architecture

```mermaid
graph LR
    A[Company Documents] --> B[Local Chunking]
    B --> C[Local Embeddings]
    C --> D[Local Vector DB]
    D --> E[Context Retrieval]
    E --> F[Minimal Context]
    F --> G[AI API]
    G --> H[Secure Response]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
```

### Security Levels

| Level | Description | Internet Required | Security Rating |
|-------|-------------|-------------------|-----------------|
| 🥇 **Air-gapped** | Pre-downloaded model, 100% offline | Never | Maximum |
| 🥈 **Cache-first** | Download once, then offline | First run only | High |
| 🥉 **TF-IDF Fallback** | Simple embeddings, no downloads | Never | Basic |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (or compatible API)
- 2GB+ RAM for embedding models

### Installation

```bash
# Clone the repository
git clone https://github.com/netologist/secure-rag-system.git
cd secure-rag-system

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install pydantic-ai chromadb openai sentence-transformers scikit-learn
```

### Basic Usage

```python
import asyncio
from secure_rag_system import SecureRAGSystem

async def main():
    # Initialize the system
    rag = SecureRAGSystem("my_company_docs")
    
    # Add documents
    documents = [
        "Your company policy document...",
        "Technical documentation...",
        "HR guidelines..."
    ]
    rag.add_documents(documents)
    
    # Query the system
    answer = await rag.query("What is our remote work policy?")
    print(answer)

# Run the system
asyncio.run(main())
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional
export CHROMA_DB_PATH="./chroma_db"
export EMBEDDING_MODEL_PATH="./models/all-MiniLM-L6-v2"
export MAX_CHUNK_SIZE="500"
export TOP_K_RESULTS="3"
```

### Maximum Security Setup (Air-gapped)

For maximum security, pre-download the embedding model:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download model locally
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir ./models/all-MiniLM-L6-v2

# The system will automatically detect and use the local model
```

## 📖 Advanced Usage

### Custom Document Processing

```python
from secure_rag_system import SecureRAGSystem

class CustomRAGSystem(SecureRAGSystem):
    def _chunk_document(self, document: str, chunk_size: int = 500):
        # Implement custom chunking logic
        # E.g., semantic chunking, sentence-based splitting
        return custom_chunks
    
    def add_pdf_documents(self, pdf_paths: List[str]):
        # Add PDF processing capability
        documents = []
        for pdf_path in pdf_paths:
            text = extract_text_from_pdf(pdf_path)
            documents.append(text)
        self.add_documents(documents)
```

### Batch Processing

```python
# Process large document collections
async def process_document_library():
    rag = SecureRAGSystem("document_library")
    
    # Process documents in batches
    batch_size = 10
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        rag.add_documents(batch)
        print(f"Processed batch {i//batch_size + 1}")
    
    return rag
```

### Integration with Different AI Providers

```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

# Use Anthropic Claude
rag.agent = Agent(
    model=AnthropicModel('claude-3-sonnet-20240229'),
    system_prompt="Your custom prompt..."
)

# Use different OpenAI models
rag.agent = Agent(
    model=OpenAIModel('gpt-4-turbo'),
    system_prompt="Your custom prompt..."
)
```

## 🔧 API Reference

### SecureRAGSystem Class

#### Constructor
```python
SecureRAGSystem(collection_name: str = "company_docs")
```

#### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `add_documents()` | Add documents to the system | `documents: List[str]`, `metadatas: List[dict]` |
| `query()` | Query the system | `question: str`, `top_k: int = 3` |
| `get_stats()` | Get system statistics | None |

#### Security Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `_setup_embedding_model()` | Configure embedding model security | `SentenceTransformer` or `TFIDFEmbedder` |
| `_create_tfidf_embedder()` | Create offline TF-IDF embedder | `SimpleTFIDFEmbedder` |

## 🏗️ System Architecture

### Components

1. **Document Processor**: Chunks and preprocesses documents
2. **Embedding Engine**: Generates vector representations locally
3. **Vector Database**: Stores and indexes embeddings (Chroma)
4. **Retrieval Engine**: Finds relevant document chunks
5. **AI Agent**: Generates responses using Pydantic AI
6. **Security Layer**: Ensures data never leaves your control

### Data Flow

1. **Ingestion**: Documents → Chunking → Local Embeddings
2. **Storage**: Embeddings → Local Chroma Database
3. **Retrieval**: Query → Vector Search → Context Selection
4. **Generation**: Context + Query → AI API → Response

## 🔍 Monitoring and Observability

### Built-in Statistics

```python
# Get system statistics
stats = rag.get_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Database path: {stats['database_path']}")
```

### Custom Metrics

```python
# Add custom monitoring
class MonitoredRAGSystem(SecureRAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_count = 0
        self.response_times = []
    
    async def query(self, question: str, top_k: int = 3):
        start_time = time.time()
        result = await super().query(question, top_k)
        end_time = time.time()
        
        self.query_count += 1
        self.response_times.append(end_time - start_time)
        
        return result
```

## 🛡️ Security Best Practices

### Data Handling
- Never log sensitive document content
- Use environment variables for API keys
- Regularly rotate API keys
- Implement access controls on the Chroma database

### Network Security
- Run on isolated networks when possible
- Use VPNs for remote access
- Monitor API calls to external services
- Implement rate limiting

### Compliance
- Maintain audit logs of document access
- Implement data retention policies
- Regular security assessments
- Document data flow for compliance reviews

## 🚨 Troubleshooting

### Common Issues

#### Model Download Fails
```bash
# Check internet connectivity
ping huggingface.co

# Use manual download
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin
```

#### Chroma Database Issues
```python
# Reset database if corrupted
rag.chroma_client.reset()

# Change database path
rag = SecureRAGSystem("new_collection")
```

#### Memory Issues
```python
# Reduce chunk size for large documents
rag.add_documents(docs, chunk_size=200)

# Process documents in smaller batches
for batch in chunks(documents, 5):
    rag.add_documents(batch)
```

### Performance Optimization

#### Embedding Performance
- Use GPU acceleration when available
- Batch process documents
- Optimize chunk sizes for your use case

#### Vector Search Performance
- Adjust `top_k` based on your needs
- Use appropriate distance metrics
- Consider index optimization for large datasets

## 📚 Examples

### Enterprise Document Processing

See `examples/enterprise_setup.py` for a complete enterprise implementation including:
- PDF processing
- Access controls
- Audit logging
- Multi-tenant support

### Integration Examples

- **Slack Bot**: `examples/slack_integration.py`
- **Web API**: `examples/fastapi_server.py`
- **CLI Tool**: `examples/cli_interface.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone for development
git clone https://github.com/netologist/secure-rag-system.git
cd secure-rag-system

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black . && isort . && flake8
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Pydantic AI](https://github.com/pydantic/pydantic-ai) - Type-safe AI agents
- [Chroma](https://github.com/chroma-core/chroma) - Vector database
- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers) - Embedding models


## 📊 Roadmap

- [ ] Multi-modal document support (images, tables)
- [ ] Advanced chunking strategies
- [ ] Integration with more vector databases
- [ ] Kubernetes deployment manifests
- [ ] Federated learning capabilities
- [ ] Real-time document updates

---

**⚠️ Security Notice**: This system is designed for sensitive enterprise data. Always review and test security configurations before production deployment.