# 🔧 Source Code Directory

This directory is reserved for modularized Python source code extracted from the Jupyter notebook.

## 🎯 Purpose

Currently, the RAG pipeline is implemented entirely in `notebooks/rag_wikipedia.ipynb`. As the project matures, core functionality should be refactored into reusable Python modules stored here.

## 📁 Current Status

**Status**: Empty (placeholder)  
**Reason**: Early-stage project with notebook-first development

## 🏗️ Planned Architecture

### Recommended Module Structure

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── scraper.py          # Wikipedia article fetching
│   ├── chunker.py          # Text chunking utilities
│   └── preprocessor.py     # Text cleaning and normalization
├── embeddings/
│   ├── __init__.py
│   ├── encoder.py          # Sentence transformer wrapper
│   └── models.py           # Model configuration and loading
├── retrieval/
│   ├── __init__.py
│   ├── vectorstore.py      # ChromaDB interface
│   └── retriever.py        # Query and retrieval logic
├── generation/
│   ├── __init__.py
│   ├── summarizer.py       # BART summarization wrapper
│   └── prompt.py           # Prompt engineering utilities
├── pipeline/
│   ├── __init__.py
│   └── rag.py              # End-to-end RAG pipeline
├── utils/
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   ├── config.py           # Configuration management
│   └── metrics.py          # Evaluation metrics
└── cli.py                  # Command-line interface
```

## 📦 Module Specifications

### `data/scraper.py`
**Purpose**: Wikipedia article fetching and management

```python
class WikipediaScraper:
    """Fetch and cache Wikipedia articles."""
    
    def __init__(self, language: str = 'en', user_agent: str = None):
        pass
    
    def fetch_article(self, title: str) -> dict:
        """Fetch single article."""
        pass
    
    def fetch_multiple(self, titles: list[str]) -> list[dict]:
        """Fetch multiple articles efficiently."""
        pass
    
    def save_corpus(self, articles: list[dict], path: str):
        """Save to CSV/JSON/Parquet."""
        pass
```

**Key Features**:
- Batch fetching with rate limiting
- Caching to avoid redundant requests
- Error handling for missing articles
- Multiple export formats

---

### `data/chunker.py`
**Purpose**: Intelligent text segmentation

```python
class TextChunker:
    """Split documents into semantic chunks."""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        pass
    
    def chunk_by_words(self, text: str) -> list[str]:
        """Word-based chunking."""
        pass
    
    def chunk_by_sentences(self, text: str) -> list[str]:
        """Sentence-aware chunking."""
        pass
    
    def chunk_by_paragraphs(self, text: str) -> list[str]:
        """Paragraph-based chunking."""
        pass
```

**Improvements**:
- Overlap between chunks for context continuity
- Sentence boundary detection
- Maximum token limit enforcement
- Metadata tracking (chunk IDs, positions)

---

### `embeddings/encoder.py`
**Purpose**: Embedding generation and caching

```python
class EmbeddingEncoder:
    """Generate and manage embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        pass
    
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings with batching."""
        pass
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode single query."""
        pass
```

**Features**:
- Batch processing for efficiency
- GPU acceleration support
- Model caching
- Progress tracking for large batches

---

### `retrieval/vectorstore.py`
**Purpose**: Vector database abstraction

```python
class VectorStore:
    """Unified interface for vector databases."""
    
    def __init__(self, backend: str = "chromadb", persist_dir: str = None):
        pass
    
    def add_documents(self, chunks: list[str], embeddings: np.ndarray, metadata: list[dict]):
        """Add documents to the store."""
        pass
    
    def query(self, query_embedding: np.ndarray, k: int = 4) -> list[dict]:
        """Retrieve top-k similar documents."""
        pass
    
    def delete_collection(self, name: str):
        """Remove collection."""
        pass
```

**Backends**:
- ChromaDB (current)
- Pinecone (cloud option)
- FAISS (performance option)
- Weaviate (production option)

---

### `generation/summarizer.py`
**Purpose**: Text generation wrapper

```python
class Summarizer:
    """BART-based summarization."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        pass
    
    def summarize(self, text: str, max_length: int = 260, min_length: int = 80) -> str:
        """Generate summary."""
        pass
    
    def batch_summarize(self, texts: list[str]) -> list[str]:
        """Summarize multiple texts."""
        pass
```

**Enhancements**:
- Multiple model support (T5, Pegasus)
- Token counting and auto-truncation
- Temperature and sampling controls
- Streaming generation for long outputs

---

### `pipeline/rag.py`
**Purpose**: Orchestrate the complete RAG workflow

```python
class RAGPipeline:
    """End-to-end RAG system."""
    
    def __init__(self, config: dict):
        self.scraper = WikipediaScraper()
        self.chunker = TextChunker()
        self.encoder = EmbeddingEncoder()
        self.vectorstore = VectorStore()
        self.summarizer = Summarizer()
    
    def build_corpus(self, topics: list[str]):
        """Scrape and index topics."""
        pass
    
    def query(self, question: str, k: int = 4) -> dict:
        """Answer a question."""
        return {
            "answer": "...",
            "sources": [...],
            "confidence": 0.95
        }
    
    def evaluate(self, test_set: list[dict]) -> dict:
        """Run evaluation metrics."""
        pass
```

**Capabilities**:
- One-line corpus building
- Configurable retrieval parameters
- Source attribution
- Confidence scoring
- Evaluation harness

---

### `utils/config.py`
**Purpose**: Centralized configuration

```python
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    # Data
    wiki_language: str = "en"
    chunk_size: int = 300
    chunk_overlap: int = 50
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retrieval
    vectorstore_backend: str = "chromadb"
    top_k: int = 4
    
    # Generation
    summarization_model: str = "facebook/bart-large-cnn"
    max_summary_length: int = 260
    min_summary_length: int = 80
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = ".cache"
```

**Benefits**:
- Type-safe configuration
- Easy parameter tuning
- Environment-specific configs (dev/prod)
- YAML/JSON support

---

### `cli.py`
**Purpose**: Command-line interface

```bash
# Build corpus
python -m src.cli build --topics "Federated_learning,Machine_learning"

# Query the system
python -m src.cli query "What is federated learning?"

# Evaluate on test set
python -m src.cli evaluate --test-file tests/qa_pairs.json

# Export results
python -m src.cli export --format json --output results.json
```

## 🚀 Migration Guide

### Converting Notebook to Modules

**Step 1**: Extract data collection (Cells 1-4)
```python
# notebooks/rag_wikipedia.ipynb → src/data/scraper.py
from src.data.scraper import WikipediaScraper

scraper = WikipediaScraper()
articles = scraper.fetch_article("Federated_learning")
```

**Step 2**: Extract chunking (Cell 6)
```python
# notebooks/rag_wikipedia.ipynb → src/data/chunker.py
from src.data.chunker import TextChunker

chunker = TextChunker(chunk_size=300)
chunks = chunker.chunk_by_words(text)
```

**Step 3**: Extract embeddings (Cell 7)
```python
# notebooks/rag_wikipedia.ipynb → src/embeddings/encoder.py
from src.embeddings.encoder import EmbeddingEncoder

encoder = EmbeddingEncoder()
embeddings = encoder.encode(chunks)
```

**Step 4**: Create pipeline wrapper
```python
# New: src/pipeline/rag.py
from src.pipeline.rag import RAGPipeline

pipeline = RAGPipeline()
pipeline.build_corpus(["Federated_learning"])
answer = pipeline.query("What is federated learning?")
```

## 🧪 Testing Structure

```
src/
└── tests/
    ├── __init__.py
    ├── test_scraper.py
    ├── test_chunker.py
    ├── test_encoder.py
    ├── test_retrieval.py
    ├── test_summarizer.py
    ├── test_pipeline.py
    └── fixtures/
        └── sample_articles.json
```

**Example test**:
```python
# src/tests/test_chunker.py
import pytest
from src.data.chunker import TextChunker

def test_chunk_by_words():
    chunker = TextChunker(chunk_size=10)
    text = "word " * 25  # 25 words
    chunks = chunker.chunk_by_words(text)
    
    assert len(chunks) == 3
    assert all(len(c.split()) <= 10 for c in chunks)
```

## 📊 Benefits of Modularization

### Current (Notebook-based)
❌ Hard to test  
❌ No code reuse  
❌ Difficult to version  
❌ No CI/CD integration  
❌ Poor performance (no caching)

### After Modularization
✅ Unit testable  
✅ Reusable components  
✅ Version controlled  
✅ CI/CD ready  
✅ Optimized performance  
✅ API-ready  
✅ Scalable

## 🎯 Implementation Priority

### Phase 1: Core Extraction (Week 1)
- [ ] `data/scraper.py` - Wikipedia fetching
- [ ] `data/chunker.py` - Text chunking
- [ ] `embeddings/encoder.py` - Embedding generation

### Phase 2: Pipeline Integration (Week 2)
- [ ] `retrieval/vectorstore.py` - Vector DB wrapper
- [ ] `generation/summarizer.py` - BART wrapper
- [ ] `pipeline/rag.py` - Orchestration

### Phase 3: Utilities & CLI (Week 3)
- [ ] `utils/config.py` - Configuration
- [ ] `utils/logger.py` - Logging
- [ ] `cli.py` - Command-line interface

### Phase 4: Testing & Documentation (Week 4)
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] API documentation
- [ ] Performance benchmarks

## 📚 Development Guidelines

### Code Style
- **Formatter**: Black (line length 88)
- **Linter**: Flake8 + pylint
- **Type hints**: Required for all functions
- **Docstrings**: Google style

### Example
```python
def encode_chunks(
    chunks: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    """Encode text chunks into embeddings.
    
    Args:
        chunks: List of text strings to encode.
        model_name: Name of the sentence transformer model.
        batch_size: Number of chunks to encode simultaneously.
    
    Returns:
        Array of embeddings with shape (n_chunks, embedding_dim).
    
    Raises:
        ValueError: If chunks list is empty.
    """
    pass
```

## 🔗 Related Documentation

- Main README: `../README.md`
- Notebook documentation: `../notebooks/README.md`
- Contributing guide: `../CONTRIBUTING.md` (to be created)

---

**Status**: Planning phase  
**Target completion**: TBD  
**Maintainer**: @fabriziosulcar

