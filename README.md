# 🧠 Clinical-RAG: Intelligent Assistant for Medical Guidelines

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Weaviate](https://img.shields.io/badge/Weaviate-00A98F?style=for-the-badge&logo=weaviate&logoColor=white)](https://weaviate.io/)
[![Llama.cpp](https://img.shields.io/badge/llama.cpp-gray?style=for-the-badge)](https://github.com/ggerganov/llama.cpp)


**Clinical-RAG** is an advanced **Retrieval-Augmented Generation (RAG)** system that provides clinicians and researchers with fast, accurate, and evidence-based answers to complex healthcare questions. It leverages a curated knowledge base of official clinical guidelines and runs entirely **locally** for secure, offline operation.

---

## 🎯 Overview

- **System:** Retrieval-Augmented Generation (RAG)
- **Use-case:** Conversational knowledge base for medical guidelines
- **Privacy:** Fully local, no external API calls
- **Goal:** Enable evidence-based, fast decision-making for healthcare professionals

---

##  Key Features

-  Converts static clinical PDFs into a dynamic, conversational knowledge base
-  Fully local inference using LLMs (via Llama.cpp)
-  Interactive Q&A through `eval.py`
-  Rigorous model evaluation (honesty, reasoning, faithfulness)
-  Supports vector search using Weaviate

---

## ⚙️ System Architecture

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
## Step Descriptions:

- Embed Query – Converts user questions into vector embeddings.

- Vector Search – Retrieves the most relevant document chunks from Weaviate.

- Build Prompt – Combines context with the query for the LLM.

- Generate Answer- Synthesizes a response strictly based on the provided context.

## Tech Stack:

-Languages: Python

-Frameworks & Libraries:

weaviate-client – vector database operations

llama-cpp-python – high-performance local LLM inference

Sentence Transformers – high-quality embeddings

pandas – evaluation and analysis

-Tools & Technologies:

Weaviate – vector database

Llama.cpp – local inference of GGUF models

Marker – PDF → Markdown conversion

Conda – environment management

## Installation & Setup:
```bash 
# Clone the repository
git clone https://github.com/Ojas1584/Clinical_Decision_Support_RAG
cd Clinical_Decision_Support_RAG

# Create & activate Conda environment
conda create -n rag_py311 python=3.11
conda activate rag_py311

# Install dependencies
pip install -r requirements.txt

# Start Weaviate
docker-compose up -d

# Prepare and ingest data
jupyter notebook data_preparation.ipynb
jupyter notebook data_ingestion.ipynb

# Start interacting
python eval.py

```bash
