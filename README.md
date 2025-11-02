# Clinical-RAG: Intelligent Assistant for Medical Guidelines

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=yellow)](https://www.python.org/)
[![Weaviate](https://img.shields.io/badge/Weaviate-Embedded-blue?logo=weaviate&logoColor=00A98F)](https://weaviate.io/)
[![Llama.cpp](https://img.shields.io/badge/Llama.cpp-Local_Inference-blue?logo=cplusplus&logoColor=white)](https://github.com/ggerganov/llama.cpp)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://www.docker.com/)

An advanced **Retrieval-Augmented Generation (RAG)** system providing clinicians with fast, accurate, and evidence-based answers from medical guidelines. **Runs 100% locally.**

---

## Overview

Clinical-RAG is a specialized RAG system designed for the healthcare domain. Its primary use case is to serve as a conversational knowledge base, allowing clinicians and researchers to ask complex questions against a curated library of 500+ official NCCN & ICMR guidelines.

The system's core principle is **privacy and security**; it runs entirely locally without any external API calls, ensuring all query data remains confidential. The ultimate goal is to enable faster, evidence-based clinical decision-making by transforming static, complex PDFs into a dynamic and interactive resource.

---

## Core Features

- **PDF to Knowledge Base:** Converts 500+ complex clinical PDFs (NCCN, ICMR) into a dynamic, queryable vector database.
- **100% Local & Secure:** All components, including the LLM (via `Llama.cpp`) and vector database (Weaviate), run on-premise. No data ever leaves the machine.
- **High-Performance RAG Pipeline:** Implements a state-of-the-art RAG pipeline using Weaviate for fast vector search and Sentence Transformers for accurate embeddings.
- **Model Benchmarking:** Includes a robust evaluation script (`eval.py`) to benchmark LLMs (like **Llama 3.1** and **Mistral**) for reasoning, faithfulness, and honesty against the guidelines.
- **Efficient Data Processing:** Uses **Marker** to efficiently convert complex PDFs into clean Markdown, preserving tables and formatting crucial for medical data.

---

## Tech Stack

* **Core AI/ML:**
    * [Python](https://www.python.org/) (3.11)
    * [Llama.cpp](https://github.com/ggerganov/llama.cpp)
    * [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (Local LLM inference)
    * [Hugging Face sentence-transformers](https://www.sbert.net/) (Embeddings)
    * [Pandas](https://pandas.pydata.org/) (Data handling & evaluation)

* **Data Pipeline & Storage:**
    * [Weaviate](https://weaviate.io/) (Vector Database)
    * [weaviate-client](https://weaviate.io/developers/weaviate/client-libraries/python) (DB interaction)

* **Data Pre-processing:**
    * [Marker](https://github.com/VikParuchuri/marker) (PDF-to-Markdown)
    * [Jupyter](https://jupyter.org/) (Data processing notebooks)

* **Environment & Tools:**
    * [Docker & Docker Compose](https://www.docker.com/)
    * [Conda](https://docs.conda.io/en/latest/) (Environment Management)

---

## Architecture Overview

The system follows a classic, high-performance RAG pipeline, containerized for easy setup. When a user submits a query, it flows through the following steps:

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

1. Embed Query: Converts the user's question into a vector embedding using Sentence Transformers.

2. Vector Search: Performs a similarity search in the Weaviate database to retrieve the most relevant document chunks.

3. Build Prompt: Combines the retrieved context with the original query into a detailed prompt template.

4. Generate Answer: Sends the prompt to a locally-run LLM (via Llama.cpp) which synthesizes an answer grounded solely in the provided context, along with source document references.

## Getting Started

**Prerequisites**:

* Python 3.11+
* Conda
* Docker

**Installation & Setup**:

Clone the repository:
```

git clone [https://github.com/Ojas1584/Clinical_Decision_Support_RAG](https://github.com/Ojas1584/Clinical_Decision_Support_RAG)
cd Clinical_Decision_Support_RAG

```
Create & activate Conda environment:
```

conda create -n rag_py311 python=3.11
conda activate rag_py311
```
Install dependencies:
```
pip install -r requirements.txt
```

Start Weaviate: Ensure Docker Desktop is running. Then, launch the Weaviate container:
```
docker-compose up -d
```
**Prepare and ingest data**:

Place your source PDF guidelines in the project directory.

Run the data_preparation.ipynb notebook to convert PDFs to Markdown using Marker.

Run the data_ingestion.ipynb notebook to chunk, vectorize, and load the Markdown content into Weaviate.

Start interacting: Download your chosen GGUF model file (e.g., Llama 3.1 8B Instruct Q4_K_M) and place it in the project directory. Then, run the interactive script:

```

python eval.py
```
(Note: Ensure the MODEL_PATH variable in eval.py points to your downloaded model file.)

---

## Model Evaluation Results

A comparative analysis of five leading models was conducted to select the optimal LLM for this RAG system. **Llama-3.1-8B-Instruct** demonstrated superior performance in reasoning and faithfulness to the source guidelines.

| Model Name                | Q1   | Q2   | Q3  | Q4   | Q5   | Average | Rank |
| :------------------------ | :--- | :--- | :-- | :--- | :--- | :------ | :--- |
| **Llama-3.1-8B-Instruct** | 10.0 | 10.0 | 9.5 | 7.5  | 10.0 | **9.4** | 1    |
| Mistral-7B-Instruct-v0.3  | 10.0 | 9.0  | 6.0 | 10.0 | 10.0 | **9.0** | 2    |
| Gemma-1.1-7B-Instruct     | 7.0  | 1.0  | 3.0 | 10.0 | 10.0 | **6.2** | 3    |
| Nous-Hermes-2-Pro-Llama-3 | 3.0  | 10.0 | 7.0 | 2.0  | 5.0  | **5.4** | 4    |
| Qwen1.5-7B-Chat           | 8.0  | 10.0 | 2.0 | 5.0  | 2.0  | **5.4** | 4    |

*(Evaluation based on Honesty, Reasoning, and Faithfulness; see evaluation methodology for details)*

---

## Evaluation Methodology

The primary goal of the evaluation was to assess how effectively each Large Language Model (LLM) performed *within the specific constraints of the Retrieval-Augmented Generation (RAG) pipeline*. The focus was not on the LLM's general knowledge but on its ability to generate accurate, relevant, and trustworthy answers based *solely* on the information retrieved from the NCCN and ICMR guideline documents.

The evaluation centered on key **qualitative metrics** essential for a reliable clinical decision support tool:

* **Faithfulness & Honesty:**
    * Does the model strictly adhere to the provided context from the retrieved document chunks?
    * Does it accurately represent the information found in the sources?
    * Crucially, does it **refuse to answer** or state that the information is unavailable if the retrieved context does not contain the answer, thus avoiding **hallucination**?

* **Reasoning & Synthesis:**
    * Can the model understand and follow the specific instructions of the question (e.g., compare, summarize, list)?
    * Can it logically synthesize information if the answer requires combining facts from different parts of the retrieved context?

* **Relevance:**
    * Is the generated answer directly and concisely addressing the specific question asked, based *only* on the provided context?

---
## Acknowledgements & Data Sources

This project utilizes official clinical guidelines from leading medical organizations. We gratefully acknowledge their work in creating and disseminating these vital resources:

* **National Comprehensive Cancer Network (NCCN)** - [NCCN Clinical Practice Guidelines in Oncology (NCCN Guidelines®)](https://www.nccn.org/guidelines/category_1)
* **Indian Council of Medical Research (ICMR)** – [ICMR Guidelines Portal](https://www.icmr.gov.in/guidelines/)

**Note on Copyright:** The original PDF documents are **not redistributed** in this repository. The vector database must be created locally by the user from these official sources. Please download the latest guidelines directly from the NCCN and ICMR websites to ensure the use of current, evidence-based information.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
