# Clinical-RAG: An Intelligent Assistant for Medical Guidelines

**Clinical-RAG** is an advanced **Retrieval-Augmented Generation (RAG)** system designed to provide clinicians and medical researchers with fast, accurate, and evidence-based answers to complex healthcare questions. By leveraging a curated knowledge base of official clinical guidelines, this system combines AI-driven natural language understanding with local inference for secure, offline operation.

---

## The Challenge 🎯

Medical professionals face an ever-growing corpus of dense, unstructured documents. Locating precise information—such as treatment protocols, diagnostic criteria, or specific biomarkers—can be time-consuming and error-prone. Traditional manual searches often slow decision-making, which can impact patient outcomes.

---

## Our Solution: A Conversational Knowledge Base 💡

Clinical-RAG converts a static library of clinical PDFs into a **dynamic, conversational knowledge base**. Using a RAG architecture:

- Clinicians can pose natural language questions.
- The system retrieves relevant context from the knowledge base.
- A local LLM synthesizes concise, context-grounded answers.

All operations are **local**, ensuring complete **data privacy and security**, with no external API calls.

---

## System Architecture ⚙️

```text
User Query
    │
    ▼
[ 1. Embed Query ] --(SentenceTransformer)--> Vector
    │
    ▼
[ 2. Vector Search ] --(Weaviate)--> Retrieved Document Chunks
    │
    ▼
[ 3. Build Prompt ] --(Context + Query)--> Formatted Prompt
    │
    ▼
[ 4. Generate Answer ] --(Local LLM via Llama.cpp)--> Synthesized Text
    │
    ▼
Final Answer & Sources
```

Step Descriptions:

Embed Query – Converts the user question into a vector embedding.

Vector Search – Uses Weaviate to retrieve the most relevant document chunks.

Build Prompt – Combines retrieved context with the original query for the LLM.

Generate Answer – Generates a synthesized response based strictly on the provided context.

Tech Stack 🛠️

Languages:

Python

Frameworks & Libraries:

weaviate-client – vector database operations.

llama-cpp-python – high-performance local inference of quantized LLMs.

Sentence Transformers – high-quality embeddings.

pandas – data analysis and evaluation.

Tools & Technologies:

Weaviate – vector database for storing and searching document embeddings.

Llama.cpp – efficient inference of GGUF-quantized models locally.

Marker – converts medical PDFs into clean Markdown for ingestion.

Conda – environment management.

Key Features ✨

Fully Local & Private – No data leaves your machine.

Comprehensive Pipeline – Includes robust scripts for document conversion, chunking, and embedding.

Rigorous Model Evaluation – Benchmarks multiple LLMs on honesty, reasoning, and faithfulness.

Interactive Q&A – Real-time conversation with the knowledge base via eval.py.

Getting Started 🚀

Prerequisites:

Python 3.11+

Conda

Docker


Installation & Setup:
```bash
# Clone the repository
git clone https://github.com/Ojas1584/Clinical_Decision_Support_RAG
cd <your-repo-name>

# Create and activate Conda environment
conda create -n rag_py311 python=3.11
conda activate rag_py311

# Install dependencies
pip install -r requirements.txt

# Start Weaviate
docker-compose up -d

# Prepare and ingest data
jupyter notebook data_preparation.ipynb
jupyter notebook data_ingestion.ipynb

# Start interacting with the knowledge base
python eval.py
```
