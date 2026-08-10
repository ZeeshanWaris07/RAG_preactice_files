# Advanced RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) system built with **LangChain, ChromaDB, BM25, Hugging Face embeddings, CrossEncoder reranking, and Google Gemini**.

The project demonstrates how to progressively improve a basic RAG system using **metadata enrichment, dynamic metadata filtering, hybrid search, reranking, and retrieval evaluation**.

---

## Architecture

```text
                         User Question
                              │
                              ▼
                    ┌──────────────────┐
                    │ Query Processing │
                    │ / Query Rewrite  │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Metadata Extraction  │
                  │ Topic / Subtopic /   │
                  │ Difficulty           │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Hybrid Retrieval│
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
          Vector Retrieval            BM25
             ChromaDB             Keyword Search
                 │                       │
                 └───────────┬───────────┘
                             ▼
                       Candidate Chunks
                             │
                             ▼
                    ┌─────────────────┐
                    │   CrossEncoder  │
                    │    Reranker     │
                    └────────┬────────┘
                             │
                             ▼
                       Top K Chunks
                             │
                             ▼
                         Gemini LLM
                             │
                             ▼
                      Generated Answer
```

---

## Features

### 1. PDF Document Loading

PDF documents are loaded using LangChain's `PyPDFLoader`.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/d2l-en.pdf")
documents = loader.load()
```

Multiple PDFs can also be processed from a directory.

---

### 2. Document Chunking

Large documents are divided into smaller chunks using `RecursiveCharacterTextSplitter`.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
```

Chunking improves retrieval by allowing the system to retrieve smaller, more relevant sections instead of entire documents.

---

## Automatic Metadata Generation

Instead of manually assigning metadata to documents, an LLM can analyze each chunk and generate metadata.

Each chunk can contain:

```text
topic
subtopic
difficulty
```

Example:

```json
{
    "topic": "deep learning",
    "subtopic": "convolutional neural networks",
    "difficulty": "intermediate"
}
```

Structured output is used to ensure the metadata follows a predefined schema.

```python
class ChunkMetadata(BaseModel):
    topic: str
    subtopic: str
    difficulty: str
```

The metadata is then stored alongside the chunk:

```python
chunk.metadata["topic"] = metadata.topic
chunk.metadata["subtopic"] = metadata.subtopic
chunk.metadata["difficulty"] = metadata.difficulty
```

This allows metadata to be generated dynamically even when the system does not know the documents beforehand.

---

# Dynamic Metadata Filtering

The system uses an LLM to extract search filters from the user's query.

For example:

```text
User:
"Explain beginner concepts of artificial intelligence."
```

The query-processing LLM can produce:

```json
{
    "question": "Explain beginner concepts of artificial intelligence.",
    "topic": "artificial intelligence",
    "subtopic": null,
    "difficulty": "beginner"
}
```

The filters are then converted into Chroma-compatible filter expressions.

For example:

```python
{
    "$and": [
        {"topic": "artificial intelligence"},
        {"difficulty": "beginner"}
    ]
}
```

This allows the retriever to search only the relevant subset of the vector database.

---

# Vector Search

The project uses:

```text
BAAI/bge-small-en-v1.5
```

to create embeddings.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```

The embeddings are stored in ChromaDB.

Vector search is useful for finding semantically similar content even when the exact words in the query do not appear in the document.

---

# BM25 Search

The project also uses BM25 for keyword-based retrieval.

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    k=10
)
```

BM25 is especially useful when the query contains important exact terms such as:

```text
"ReLU"
"Max Pooling"
"Batch Normalization"
"ResNet"
```

---

# Hybrid Search

Instead of relying exclusively on vector search, the system combines:

* Semantic/vector search
* BM25 keyword search

using `EnsembleRetriever`.

```python
from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    weights=[0.6, 0.4]
)
```

Conceptually:

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Vector Search         BM25
          │                 │
          └────────┬────────┘
                   ▼
            Combined Results
```

This provides better coverage than relying on a single retrieval strategy.

---

# CrossEncoder Reranking

The hybrid retriever produces an initial candidate set.

A CrossEncoder is then used to score the relevance of each:

```text
(query, document)
```

pair.

The current reranker:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)
```

Pairs are created:

```python
pairs = [
    [question, doc.page_content]
    for doc in retrieved_documents
]
```

Scores are generated:

```python
scores = reranker.predict(pairs)
```

The documents are then sorted according to their relevance scores.

```python
ranked_documents = sorted(
    zip(scores, retrieved_documents),
    key=lambda x: x[0],
    reverse=True
)
```

The important point is that the CrossEncoder scores are **ranking scores, not probabilities**.

A higher score means the model considers the query-document pair more relevant.

---

# Why Reranking?

The retrieval pipeline follows a two-stage architecture:

```text
Stage 1: Retrieval

Large document collection
        │
        ▼
Vector + BM25
        │
        ▼
10-20 candidate chunks


Stage 2: Reranking

10-20 candidate chunks
        │
        ▼
CrossEncoder
        │
        ▼
Top 3-5 chunks
```

Retrieval prioritizes **speed and recall**.

Reranking prioritizes **precision and relevance**.

This prevents the computationally expensive CrossEncoder from processing the entire document collection.

---

# Generation

After reranking, the best chunks are passed to the LLM.

The current generation model is Google Gemini.

The final pipeline is:

```text
Question
   ↓
Query Processing
   ↓
Metadata Filtering
   ↓
Hybrid Search
   ↓
CrossEncoder Reranking
   ↓
Top Relevant Chunks
   ↓
Gemini
   ↓
Final Answer
```

---

# RAG Evaluation

A major goal of this project is to evaluate the RAG system rather than simply assuming that retrieval improvements are working.

Evaluation is divided into two major areas:

```text
RAG Evaluation
│
├── Retrieval Evaluation
│
└── Generation Evaluation
```

---

## Retrieval Evaluation

Retrieval quality can be measured using:

### Recall@K

Measures how many relevant chunks were retrieved within the top K results.

```text
Recall@K =
relevant chunks retrieved / total relevant chunks
```

Example:

```text
Ground truth:
[A, B]

Retrieved:
[A, X, Y, B, Z]

Recall@5 = 2 / 2 = 1.0
```

---

### Precision@K

Measures how many of the retrieved chunks are actually relevant.

```text
Precision@K =
relevant retrieved chunks / K
```

Example:

```text
Retrieved:
[A, X, Y, B, Z]

Relevant:
A, B

Precision@5 = 2 / 5 = 0.4
```

---

### MRR

Mean Reciprocal Rank measures how high the first relevant document appears.

If the first relevant document is at position 3:

```text
MRR = 1 / 3
```

A relevant document at position 1 produces:

```text
MRR = 1
```

---

### NDCG

NDCG evaluates ranking quality when documents can have different degrees of relevance.

For example:

```text
3 → Highly relevant
2 → Relevant
1 → Slightly relevant
0 → Irrelevant
```

This is useful for evaluating whether the most useful chunks are placed near the top.

---

# Ground Truth Dataset

Retrieval evaluation requires a ground-truth dataset.

Example:

```python
{
    "question": "What is max pooling?",
    "relevant_chunks": [
        "chunk_182",
        "chunk_183"
    ]
}
```

The ground truth can initially be created with the help of an LLM, but it should be manually verified.

A small evaluation dataset of approximately **30–50 representative questions** is sufficient for the initial project.

Questions should include:

* Simple factual questions
* Conceptual questions
* Multi-chunk questions
* Ambiguous questions
* Comparative questions
* Multi-hop questions

---

# Comparing Retrieval Strategies

One of the goals of evaluation is to determine whether each improvement actually improves the system.

The following configurations can be compared:

```text
1. Vector Search

2. BM25

3. Hybrid Search

4. Hybrid Search + CrossEncoder

5. Hybrid Search + Metadata Filtering

6. Hybrid Search + Metadata Filtering + CrossEncoder
```

Example evaluation table:

| Pipeline          | Recall@5 | MRR@5 | NDCG@5 |
| ----------------- | -------: | ----: | -----: |
| Vector            |        - |     - |      - |
| BM25              |        - |     - |      - |
| Hybrid            |        - |     - |      - |
| Hybrid + Reranker |        - |     - |      - |

The actual values should be generated from the evaluation dataset rather than assumed.

---

# Generation Evaluation

Retrieval quality alone does not guarantee a good RAG system.

The generated answer can also be evaluated using metrics such as:

### Faithfulness

Is the generated answer supported by the retrieved context?

### Answer Relevance

Does the answer actually address the user's question?

### Context Relevance

Is the retrieved context useful for answering the question?

LLM-as-a-judge techniques can be used to automate these evaluations.

Tools such as Ragas can also be introduced later for automated RAG evaluation.

---

# Project Structure

A possible project structure:

```text
multi_pdf/
│
├── data/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
│
├── chroma_db/
│
├── metadata_filtering.py
├── hybrid_retriever.py
├── evaluation.py
│
├── templates.py
├── requirements.txt
└── README.md
```

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install langchain
pip install langchain-community
pip install langchain-chroma
pip install langchain-classic
pip install langchain-huggingface
pip install langchain-google-genai
pip install langchain-ollama
pip install sentence-transformers
pip install rank-bm25
pip install chromadb
pip install pypdf
```

---

# Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
```

---

# Current Technology Stack

| Component           | Technology                     |
| ------------------- | ------------------------------ |
| Language            | Python                         |
| RAG Framework       | LangChain                      |
| PDF Loader          | PyPDF                          |
| Chunking            | RecursiveCharacterTextSplitter |
| Embeddings          | BAAI/bge-small-en-v1.5         |
| Vector Database     | ChromaDB                       |
| Keyword Retrieval   | BM25                           |
| Hybrid Retrieval    | EnsembleRetriever              |
| Reranker            | BAAI/bge-reranker-base         |
| Reranker Framework  | Sentence Transformers          |
| Metadata Generation | Ollama / Llama                 |
| Generation          | Google Gemini                  |
| Structured Output   | Pydantic                       |

---

# Learning Progression

This project is intentionally built progressively:

```text
Basic RAG
   ↓
Metadata
   ↓
Dynamic Metadata Generation
   ↓
Metadata Filtering
   ↓
Hybrid Search
   ↓
CrossEncoder Reranking
   ↓
Retrieval Evaluation
   ↓
Generation Evaluation
```

The objective is not only to build a working RAG application, but to understand **why each component exists and how to measure whether it improves retrieval and answer quality**.

---

# Future Improvements

Potential next steps include:

* [ ] Build an automated evaluation dataset
* [ ] Implement Recall@K
* [ ] Implement Precision@K
* [ ] Implement MRR
* [ ] Implement NDCG
* [ ] Compare vector vs BM25 vs hybrid retrieval
* [ ] Measure the impact of CrossEncoder reranking
* [ ] Evaluate answer faithfulness
* [ ] Evaluate answer relevance
* [ ] Add LLM-as-a-judge evaluation
* [ ] Integrate Ragas
* [ ] Add retrieval latency measurements
* [ ] Optimize batch metadata generation
* [ ] Add caching for embeddings and metadata
* [ ] Add production monitoring

---

## Goal

The ultimate goal is to build a RAG system where improvements are **measured rather than assumed**.

Instead of saying:

> "Hybrid search seems better."

the evaluation pipeline should allow us to say:

> "Hybrid search improved Recall@5 from X to Y, while CrossEncoder reranking improved MRR@5 from X to Y."

This makes the system measurable, reproducible, and much closer to a production-grade RAG architecture.
