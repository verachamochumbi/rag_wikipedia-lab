# 📓 Notebooks Directory

This directory contains the main Jupyter notebook implementing the RAG Wikipedia pipeline.

## 📁 Files

### `rag_wikipedia.ipynb`
**Main implementation notebook** - Complete RAG pipeline from data collection to query answering.

## 📋 Notebook Structure

### 🔧 Setup & Installation (Cells 0, 5)
**Purpose**: Install required packages

**Key packages**:
- `wikipedia-api` - Article scraping
- `sentence-transformers` - Embeddings
- `chromadb` - Vector database
- `langchain` - RAG orchestration
- `transformers` - BART model
- `torch` - Deep learning backend

**Note**: Two installation cells exist due to iterative development. In production, consolidate into one.

---

### 📚 Data Collection (Cells 1-4)

#### Cell 1: Wikipedia Scraping
```python
wiki = wikipediaapi.Wikipedia(language='en', user_agent='...')
page = wiki.page("Federated_learning")
```
- Fetches article on **Federated Learning**
- Validates page exists
- Extracts full text content

#### Cell 2: Display Text
- Shows raw text for verification
- Debugging/exploratory cell

#### Cell 3-4: Save & Verify
- Creates `data/wiki_corpus.csv`
- Stores article with ID and title
- Verifies CSV creation

**Output**: `data/wiki_corpus.csv` (1 article, ~4500 words)

---

### 🔪 Text Processing (Cell 6)

**Chunking Strategy**:
- Split by whitespace
- 300 words per chunk
- No overlap (can be improved)

**Output**: 15 chunks from the Federated Learning article

**Why 300 words?**
- Balances context size vs. retrieval precision
- Fits within BART's 1024 token limit
- Small enough for focused retrieval

---

### 🎯 Vector Store Creation (Cells 7-8)

#### Cell 7: ChromaDB Embedding
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
collection.add(ids=..., embeddings=..., documents=...)
```

**Process**:
1. Initialize ChromaDB in-memory client
2. Create `wiki_ai` collection
3. Encode each chunk using `all-MiniLM-L6-v2`
4. Store embeddings with metadata

**Model choice**: `all-MiniLM-L6-v2`
- Fast inference (384-dimensional embeddings)
- Good semantic understanding
- Lightweight (80MB)

#### Cell 8: HuggingFace Token
- Optional authentication setup
- Not required for public models

---

### 🤖 Model Setup (Cells 9-10)

#### Cell 9: BART Summarization
```python
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", ...)
```

**Why BART-large-CNN?**
- State-of-the-art summarization
- Trained on CNN/Daily Mail dataset
- Handles abstractive generation well

**Configuration**:
- `max_length=260`: Upper bound on output
- `min_length=80`: Ensures substantial responses
- `do_sample=False`: Deterministic output

#### Cell 10: LangChain Integration
```python
vectordb = Chroma(collection_name="wiki_ai", ...)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
```

**Why k=4?**
- Provides sufficient context
- Avoids overwhelming the summarizer
- Balances diversity vs. relevance

---

### 🔍 Query Interface (Cell 11)

**Function**: `answer_query(query: str, k: int = 4) -> str`

**Pipeline**:
1. Retrieve top-k relevant documents
2. Combine content into context string
3. **Truncate to 2000 chars** (BART limitation)
4. Construct prompt with context + question
5. Generate answer using BART

**Critical**: Context truncation prevents model errors.

---

### 📊 Demonstration (Cells 12-14)

#### Cell 12: Healthcare Query Example
**Query**: "Challenges of applying federated learning in healthcare"

**Process**:
1. Retrieve 4 relevant docs
2. Combine & truncate text
3. Summarize directly (without prompt)

**Output**: Summary focusing on MedPerf and COVID-19 applications

#### Cell 13: Export Summary
- Saves result to `outputs/rag_summary.md`
- Adds title and formatting

#### Cell 14: Multiple Queries
**Queries**:
1. Conceptual explanation
2. Privacy benefits
3. Healthcare challenges

**Output**: `outputs/retrieval_examples.json` with all Q&A pairs

---

## 🚀 Running the Notebook

### Option 1: Sequential Execution
```bash
jupyter notebook rag_wikipedia.ipynb
# Run: Cell > Run All
```

### Option 2: Step-by-Step
1. Run setup cells (0, 5)
2. Execute data pipeline (1-4, 6-7)
3. Initialize models (9-10)
4. Test queries (11-14)

### Expected Runtime
- **First run**: ~10-15 minutes (model downloads)
- **Subsequent runs**: ~3-5 minutes
- **Per query**: ~2-3 seconds

---

## 🐛 Troubleshooting

### Common Issues

**1. Wikipedia API fails**
```python
# Solution: Check internet connection or use a different article
page = wiki.page("Artificial_intelligence")
```

**2. BART truncation errors**
```python
# Solution: Reduce context size
context = context[:1500]  # Lower threshold
```

**3. ChromaDB collection exists**
```python
# Solution: Delete and recreate
client.delete_collection("wiki_ai")
collection = client.create_collection("wiki_ai")
```

**4. Out of memory**
```python
# Solution: Use CPU instead of GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

---

## 🎯 Customization Guide

### Change Wikipedia Topic
```python
# Cell 1
page = wiki.page("Machine_learning")  # Any Wikipedia article
```

### Adjust Chunk Size
```python
# Cell 6
chunk_size = 500  # Larger chunks = more context per retrieval
```

### Modify Retrieval Count
```python
# Cell 10
retriever = vectordb.as_retriever(search_kwargs={"k": 8})  # More documents
```

### Tune Summarization
```python
# Cell 9
summarizer = pipeline(
    "summarization",
    max_length=300,  # Longer outputs
    min_length=100,
    do_sample=True,  # Add randomness
    temperature=0.7
)
```

### Try Different Embeddings
```python
# Cell 7
model = SentenceTransformer("all-mpnet-base-v2")  # Higher quality
# OR
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Multilingual
```

---

## 📚 Additional Resources

### Notebook Best Practices
- Always run cells in order
- Check outputs after each major section
- Save notebook frequently
- Clear outputs before committing to Git

### Environment Setup
For reproducible runs, consider using:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r ../requirements.txt
```

### Jupyter Extensions (Optional)
```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
# Enable: Table of Contents, Variable Inspector
```

---

## 🔗 Related Files

- Main README: `../README.md`
- Requirements: `../requirements.txt`
- Outputs: `../outputs/README.md`
- Data: `../data/README.md`

---

**Last Updated**: November 2025  
**Notebook Version**: 1.0  
**Python Version**: 3.8+

