import weaviate
import weaviate.classes as wvc
import os
import time
import json
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
MODEL_PATH = "C:/Users/ojass/Downloads/My_Weaviate_RAG/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SYSTEM_MESSAGE = "You are an expert clinical advisor AI. Your role is to provide a clear and concise answer to the user's QUESTION based *only* on the provided CONTEXT. If the context is insufficient, state that you cannot answer."
RESULTS_FILE = "results.jsonl" 

# --- 2. INITIALIZATION ---
client = None
llm = None
try:
    print("Initializing the RAG system...")
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found: '{MODEL_PATH}'")

    print(f"Loading LLM: {os.path.basename(MODEL_PATH)}...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=8192   , n_gpu_layers=-1, verbose=False)
    
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    client = weaviate.connect_to_local(port=8083, grpc_port=50052)
    my_collection = client.collections.get("MyDocumentChunk")
    print(" System is ready. You can now ask a question.")

except Exception as e:
    print(f"Initialization failed: {e}")
    if client and client.is_connected(): client.close()
    exit()

# --- 3. RAG QUERY FUNCTION ---
def get_rag_response(question: str):
    query_vector = embedder.encode(question).tolist()
    response_objects = my_collection.query.near_vector(
        near_vector=query_vector, limit=7, return_properties=["text", "source_file"]
    )
    context = "".join([f"- {item.properties['text']}\n" for item in response_objects.objects])
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_MESSAGE}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    output = llm(prompt, max_tokens=512, stop=["<|eot_id|>", "<|end_of_text|>"], echo=False, temperature=0.0)
    answer = output['choices'][0]['text'].strip()
    return answer, response_objects.objects

# --- 4. MAIN LOOP ---
try:
    while True:
        user_question = input("\n\033[94m> Ask a question (or type 'exit'): \033[0m")
        if user_question.strip().lower() == 'exit': break
        if not user_question.strip(): continue
        
        start_time = time.time()
        answer, sources = get_rag_response(user_question)
        duration = time.time() - start_time
        
        try: terminal_width = os.get_terminal_size().columns
        except OSError: terminal_width = 80
        print("\n" + "\033[92m" + "--- ANSWER (from Llama 3.1) ---".center(terminal_width) + "\033[0m")
        print(f"\033[92m{answer}\033[0m")
        
        print("\n" + "\033[93m" + "--- RETRIEVED SOURCES ---".center(terminal_width) + "\033[0m")
        unique_sources = {source.properties.get('source_file', 'N/A') for source in sources}
        for i, filename in enumerate(unique_sources): print(f"  - {filename}")
        print("\033[93m" + "─" * terminal_width + "\033[0m")
            
        print(f"\n\033[90mTime taken: {duration:.2f} seconds\033[0m")
        
        # Save the results to a file
        result_record = {
            "question": user_question,
            "answer": answer,
            "contexts": [s.properties.get('text', '') for s in sources]
        }
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record) + "\n")

finally:
    if client and client.is_connected():
        client.close()
    print("\nConnection closed.")