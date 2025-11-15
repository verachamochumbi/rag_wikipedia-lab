# 🧠 Wikipedia-based RAG Summarizer

A Retrieval-Augmented Generation (RAG) system that combines Wikipedia articles with vector search and AI-powered summarization to answer domain-specific questions with high factual accuracy.

## 📋 Overview

This project implements a complete RAG pipeline using:
- **Wikipedia API** for knowledge extraction
- **ChromaDB** for vector storage and similarity search
- **Sentence Transformers** for semantic embeddings
- **LangChain** for orchestration
- **BART** (facebook/bart-large-cnn) for natural language generation

The system demonstrates how RAG architectures can provide more reliable, fact-based answers compared to traditional multi-agent workflows by grounding responses in actual document content.

## ✨ Key Features

- 📚 **Automated Wikipedia scraping** with configurable topic selection
- 🔪 **Intelligent text chunking** (300-word segments with overlap)
- 🎯 **Semantic search** using state-of-the-art embeddings (all-MiniLM-L6-v2)
- 💡 **Context-aware question answering** with source attribution
- 📊 **Comparative analysis** of RAG vs Multi-Agent approaches
- 📝 **Automatic summary generation** for retrieved content
- 🔍 **Multiple query examples** with JSON export

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| Vector Database | ChromaDB |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| LLM Framework | LangChain |
| Summarization | Facebook BART-large-CNN |
| Knowledge Source | Wikipedia API |
| Data Processing | Pandas, NumPy |
| Deep Learning | PyTorch, Transformers |

## 📁 Project Structure

```
rag_wikipedia-lab/
├── data/
│   └── wiki_corpus.csv           # Scraped Wikipedia articles
├── notebooks/
│   └── rag_wikipedia.ipynb       # Main implementation notebook
├── outputs/
│   ├── rag_summary.md            # Generated summary on healthcare FL
│   ├── retrieval_examples.json   # Q&A examples with answers
│   └── reflection.md             # RAG vs Multi-Agent comparison
├── src/
│   └── vacio                     # Reserved for future modules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM recommended for model inference
- Internet connection for Wikipedia access

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag_wikipedia-lab
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Optional: Set up Hugging Face token** (for advanced features)
```bash
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

## 💻 Usage

### Running the Notebook

Open and run the Jupyter notebook sequentially:

```bash
jupyter notebook notebooks/rag_wikipedia.ipynb
```

### Pipeline Steps

**1. Data Collection**
```python
import wikipediaapi
wiki = wikipediaapi.Wikipedia(language='en', user_agent='MyProject')
page = wiki.page("Federated_learning")
text = page.text
```

**2. Text Chunking**
```python
words = text.split()
chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
```

**3. Vector Store Creation**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = [model.encode(chunk).tolist() for chunk in chunks]
# Store in ChromaDB
```

**4. Query the System**
```python
query = "What are the key privacy benefits of federated learning?"
answer = answer_query(query)
print(answer)
```

### Example Queries

The system successfully answers questions like:

| Query | Focus Area |
|-------|-----------|
| "Explain federated learning in simple terms." | Conceptual understanding |
| "What are the key privacy benefits of federated learning?" | Technical details |
| "What are the main challenges of federated learning in healthcare?" | Domain-specific insights |

See `outputs/retrieval_examples.json` for full answers.

## 📊 Results

### Generated Outputs

1. **RAG Summary** (`outputs/rag_summary.md`)
   - Focused summary on Federated Learning in Healthcare
   - Based on 4 most relevant document chunks
   - Highlights MedPerf platform and COVID-19 prediction applications

2. **Retrieval Examples** (`outputs/retrieval_examples.json`)
   - 3 diverse queries with complete answers
   - Demonstrates semantic search quality
   - Shows context-aware response generation

3. **Reflection Analysis** (`outputs/reflection.md`)
   - Compares RAG vs Multi-Agent workflows
   - Discusses factuality, ambiguity handling, and coverage
   - Provides guidance on when to use each approach

### Key Findings

**RAG Advantages:**
- ✅ Higher factual accuracy (grounded in documents)
- ✅ No hallucinations (only uses available context)
- ✅ Traceable sources for verification
- ✅ Consistent performance on factual queries

**RAG Limitations:**
- ⚠️ Requires quality source documents
- ⚠️ Limited by corpus coverage
- ⚠️ Less creative/interpretive than multi-agent systems

## 🔧 Configuration

### Customizing the Topic

Change the Wikipedia article in Cell 1:

```python
page = wiki.page("Your_Topic_Here")  # Replace with any Wikipedia article title
```

### Adjusting Chunk Size

Modify in Cell 6:

```python
chunk_size = 300  # Increase for more context per chunk
```

### Tuning Retrieval

Adjust number of documents retrieved:

```python
retriever = vectordb.as_retriever(search_kwargs={"k": 4})  # Change k value
```

### Summary Length

Customize BART output length:

```python
summarizer(text, max_length=260, min_length=80)  # Adjust as needed
```

## 🧪 Testing

To verify the system works correctly:

1. Run all notebook cells sequentially
2. Check that `data/wiki_corpus.csv` is created
3. Verify ChromaDB collection has 15 chunks
4. Confirm outputs directory contains 3 files
5. Review `retrieval_examples.json` for coherent answers

## 📚 Learn More

### RAG Concepts
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Understanding Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Models Used
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Sentence embeddings
- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) - Summarization

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for multiple Wikipedia articles
- [ ] Implement persistent ChromaDB storage
- [ ] Create a web interface with Streamlit/Gradio
- [ ] Add evaluation metrics (BLEU, ROUGE)
- [ ] Support for other languages
- [ ] Integration with other LLMs (GPT, Claude, Llama)

## 📝 License

This project is created for educational purposes as part of a student assignment.

## 🙏 Acknowledgments

- Wikipedia API contributors
- HuggingFace Transformers team
- LangChain community
- ChromaDB developers

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This is a laboratory project demonstrating RAG implementation. For production use, consider additional error handling, security measures, and scalability improvements.
