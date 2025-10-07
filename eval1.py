import weaviate
import weaviate.classes as wvc
import os
import time
import csv
from datetime import datetime
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Define the models and their specific configurations for the evaluation run.
MODELS_TO_TEST = [
    {
        "name": "Llama-3.1-8B-Instruct",
        "path": "C:/Users/ojass/Downloads/My_Weaviate_RAG/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "prompt_format": "llama3",
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "path": "C:/Users/ojass/Downloads/My_Weaviate_RAG/mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        "prompt_format": "mistral",
    },
    {
        "name": "Nous-Hermes-2-Pro-Llama-3",
        "path": "C:/Users/ojass/Downloads/My_Weaviate_RAG/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
        "prompt_format": "chatml",
    },
    {
        "name": "Qwen1.5-7B-Chat",
        "path": "C:/Users/ojass/Downloads/My_Weaviate_RAG/Qwen1.5-7B-Chat-Q3_K_L.gguf",
        "prompt_format": "chatml",
    },
    {
        "name": "Gemma-1.1-7B-Instruct",
        "path": "C:/Users/ojass/Downloads/My_Weaviate_RAG/gemma-1.1-7b-it.Q4_K_M.gguf",
        "prompt_format": "gemma",
    }
]

# Define the set of standardized questions for the benchmark.
TEST_QUESTIONS = [
    "difference between type 1 and type 2 diabetes",
    "What specific BMI is mentioned as being associated with Type 1 diabetes at diagnosis?",
    "Based on the documents, compare the typical patient profile and disease progression for a child with Type 1 versus a youth with Type 2 diabetes.",
    "What are the recommended treatments for gestational diabetes?",
    "Why is Type 1 diabetes caused by a poor diet?",
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SYSTEM_MESSAGE = "You are an expert clinical advisor AI. Your role is to provide a clear and concise answer to the user's QUESTION based *only* on the provided CONTEXT. If the context is insufficient, state that you cannot answer."


def format_prompt(model_format, system_msg, context, question):
    """Constructs the full prompt string based on the model's required format."""
    if model_format == "llama3":
        return f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_format == "mistral":
        return f"<s>[INST] {system_msg}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question} [/INST]"
    elif model_format == "chatml":
        return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nCONTEXT:\n{context}\n\nQUESTION:\n{question}<|im_end|>\n<|im_start|>assistant\n"
    elif model_format == "gemma":
        return f"<start_of_turn>user\n{system_msg}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}<end_of_turn>\n<start_of_turn>model\n"
    else:
        raise ValueError(f"Unknown prompt format: {model_format}")


def get_stop_tokens(model_format):
    """Returns the appropriate stop tokens for a given model format."""
    if model_format == "llama3":
        return ["<|eot_id|>", "<|end_of_text|>"]
    elif model_format == "mistral":
        return ["</s>"]
    elif model_format == "chatml":
        return ["<|im_end|>"]
    elif model_format == "gemma":
        return ["<end_of_turn>"]
    else:
        raise ValueError(f"Unknown prompt format: {model_format}")


def run_benchmark_for_model(model_config, questions, weaviate_collection, embedder_model, run_timestamp):
    """Loads a single model, runs it against all test questions, and saves results to a dedicated CSV."""
    model_name_sanitized = model_config['name'].replace('.', '_')
    results_filename = f"{model_name_sanitized}_{run_timestamp}.csv"

    print(f"\n--- Processing Model: {model_config['name']} ---")
    if not os.path.exists(model_config['path']):
        print(f"Warning: Model file not found, skipping: {model_config['path']}")
        return

    with open(results_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer", "Retrieved Contexts"])
    print(f"Output file for this model: '{results_filename}'")

    llm = Llama(model_path=model_config['path'], n_ctx=2048, n_gpu_layers=-1, verbose=False)

    for i, question in enumerate(questions):
        print(f"  -> Processing Question {i+1}/{len(questions)}: '{question[:40]}...'")

        query_vector = embedder_model.encode(question).tolist()
        response_objects = weaviate_collection.query.near_vector(
            near_vector=query_vector, limit=7, return_properties=["text"]
        )
        contexts_list = [item.properties['text'] for item in response_objects.objects]
        context_str = "\n".join(contexts_list)

        prompt = format_prompt(model_config['prompt_format'], SYSTEM_MESSAGE, context_str, question)
        stop_tokens = get_stop_tokens(model_config['prompt_format'])

        output = llm(prompt, max_tokens=512, stop=stop_tokens, echo=False, temperature=0.0)
        answer = output['choices'][0]['text'].strip()

        with open(results_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            contexts_for_csv = "\n---\n".join(contexts_list)
            writer.writerow([question, answer, contexts_for_csv])

    del llm


def main():
    """Main function to orchestrate the RAG model evaluation benchmark."""
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting automated evaluation run (Timestamp: {run_timestamp})...")

    client = None
    try:
        client = weaviate.connect_to_local(port=8083, grpc_port=50052)
        my_collection = client.collections.get("MyDocumentChunk")
        embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        total_models = len(MODELS_TO_TEST)
        for i, model_config in enumerate(MODELS_TO_TEST):
            print(f"\nProcessing Model {i+1}/{total_models}...")
            run_benchmark_for_model(model_config, TEST_QUESTIONS, my_collection, embedder, run_timestamp)

    except Exception as e:
        print(f"An error occurred during the benchmark run: {e}")
    finally:
        if client and client.is_connected():
            client.close()

    print(f"\nEvaluation run complete. Check the folder for your new CSV files.")


if __name__ == "__main__":
    main()