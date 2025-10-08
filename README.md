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

