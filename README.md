🧠 Clinical-RAG: Intelligent Assistant for Medical Guidelines
Clinical-RAG is an advanced Retrieval-Augmented Generation (RAG) system that provides clinicians and researchers with fast, accurate, and evidence-based answers to complex healthcare questions. It leverages a curated knowledge base of official clinical guidelines and runs entirely locally for secure, offline operation.

🎯 Overview
System: Retrieval-Augmented Generation (RAG)

Use-case: Conversational knowledge base for medical guidelines

Privacy: Fully local, no external API calls

Goal: Enable evidence-based, fast decision-making for healthcare professionals

💡 Key Features
✅ Converts static clinical PDFs into a dynamic, conversational knowledge base

✅ Fully local inference using LLMs (via Llama.cpp)

✅ Interactive Q&A through eval.py

✅ Rigorous model evaluation (honesty, reasoning, faithfulness)

✅ Supports vector search using Weaviate

⚙️ System Architecture
Plaintext

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
Step Descriptions:

Embed Query – Converts user questions into vector embeddings.

Vector Search – Retrieves the most relevant document chunks from Weaviate.

Build Prompt – Combines context with the query for the LLM.

Generate Answer – Synthesizes a response strictly based on the provided context.

🔧 Tech Stack
Languages:
Python

Frameworks & Libraries:
weaviate-client – For all interactions with the Weaviate vector database.

llama-cpp-python – For high-performance local inference of quantized LLMs.

Sentence Transformers – For generating high-quality text and query embeddings.

pandas – For data analysis and tabulation of model evaluation results.

Tools & Technologies:
Weaviate – The core vector database for storing and searching document embeddings.

Llama.cpp – Enables efficient inference of GGUF-quantized models on local hardware.

Marker – Used to convert complex medical PDFs into clean, LLM-ready Markdown.

Conda – For managing the project's Python environment and dependencies.

🚀 Getting Started
Prerequisites:
Python 3.11+

Conda

Docker

Installation & Setup:
Clone the repository:

Bash

git clone https://github.com/Ojas1584/Clinical_Decision_Support_RAG
cd Clinical_Decision_Support_RAG
Create & activate Conda environment:

Bash

conda create -n rag_py311 python=3.11
conda activate rag_py311
Install dependencies:

Bash

pip install -r requirements.txt
Start Weaviate:

Bash

docker-compose up -d
Prepare and ingest data:

First, prepare your source documents by converting them from PDF to Markdown using the data_preparation.ipynb notebook.

Next, run the data_ingestion.ipynb notebook to chunk, vectorize, and load the documents into Weaviate.

Start interacting:

Bash

python eval.py
🧪 Model Evaluation Results
A comparative analysis of five leading models was conducted to select the best "brain" for our RAG system. Llama-3.1-8B-Instruct was the decisive winner.

Model Name	Q1	Q2	Q3	Q4	Q5	Average	Rank
Llama-3.1-8B-Instruct	10.0	10.0	9.5	7.5	10.0	9.4	🥇 1
Mistral-7B-Instruct-v0.3	10.0	9.0	6.0	10.0	10.0	9.0	🥈 2
Gemma-1.1-7B-Instruct	7.0	1.0	3.0	10.0	10.0	6.2	3
Nous-Hermes-2-Pro-Llama-3	3.0	10.0	7.0	2.0	5.0	5.4	4
Qwen1.5-7B-Chat	8.0	10.0	2.0	5.0	2.0	5.4	4

Export to Sheets
🔮 Future Work
[ ] Implement self-critique mechanisms (Self-RAG) for improved reliability.

[ ] Build a simple web UI using Streamlit or Flask.

[ ] Expand the knowledge base with additional clinical guidelines.

🙏 Acknowledgements & Data Sources
This project relies on official clinical guidelines from:

American Diabetes Association (ADA) – Standards of Care in Diabetes

Indian Council of Medical Research (ICMR) – ICMR Guidelines Portal

⚠️ Note: Original PDFs are not redistributed. The vector database is created locally from these official sources. Please download the latest guidelines directly from the ADA and ICMR websites.
