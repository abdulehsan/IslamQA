import os
import json
import shutil
from dotenv import load_dotenv
from typing import List, Dict, Any

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

FAISS_INDEX_PATH = "vectorDB"
CSV_FILES_CONFIG = [
    {"path": "sahih_bukhari.csv", "source_column": "hadithEnglish", "source_type": "Sahih Bukhari", "encoding": "utf-8"},
    {"path": "sahih_muslim.csv", "source_column": "hadithEnglish", "source_type": "Sahih Muslim", "encoding": "utf-8"},
    {"path": "merged_quran.csv", "source_column": "Ayah Translation", "source_type": "Quran", "encoding": "utf-8"}
]

def load_and_chunk_documents(csv_configs: List[Dict[str, Any]]) -> List[Document]:
    all_documents = []
    for config in csv_configs:
        if not os.path.exists(config["path"]):
            print(f"Warning: CSV not found: {config['path']}")
            continue
        loader = CSVLoader(file_path=config["path"], source_column=config["source_column"], encoding=config["encoding"])
        documents = loader.load()
        for doc in documents:
            if doc.page_content.strip():
                doc.metadata.update({"source_type": config["source_type"], "original_file": config["path"]})
                all_documents.append(doc)
    if not all_documents:
        raise ValueError("No documents loaded from CSVs.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_documents(all_documents)

def get_or_create_vector_db(index_path: str, embeddings_model, csv_configs, force_recreate=False):
    if force_recreate and os.path.exists(index_path):
        shutil.rmtree(index_path)
    
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    
    docs = load_and_chunk_documents(csv_configs)
    vector_db = FAISS.from_documents(docs, embeddings_model)
    vector_db.save_local(index_path)
    return vector_db

def call_custom_llm(client, context_chunks: List[str], user_question: str):
    context = "\n\n".join(context_chunks)
    prompt = f"""Context:\n{context}\n\nQuestion:\n{user_question}\n\nAnswer:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mixtral-8x7b-32768",
    )
    return response.choices[0].message.content.strip()

def load_jsonl_records(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl_records(records: List[Dict[str, Any]], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\nResults written to: {output_path}")

def batch_rag(jsonl_records: List[Dict[str, Any]], retriever, llm_client) -> List[Dict[str, Any]]:
    enriched_records = []

    for idx, record in enumerate(jsonl_records):
        question = record.get("Question", "").strip()
        if not question:
            print(f"Skipping record {idx} (no question field)")
            continue

        print(f"\n[{idx+1}/{len(jsonl_records)}] Question: {question[:80]}")

        retrieved_docs = retriever.get_relevant_documents(question)
        top_chunks = [doc.page_content for doc in retrieved_docs]
        answer = call_custom_llm(llm_client, top_chunks, question)

        new_record = dict(record)
        new_record["Generated_Answer"] = answer
        enriched_records.append(new_record)

    return enriched_records

def main_batch(force_recreate=False):
    client = Groq(api_key=os.getenv("groq"))
    embeddings = 
    vector_db = get_or_create_vector_db(FAISS_INDEX_PATH, embeddings, CSV_FILES_CONFIG, force_recreate)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    input = 
    output = 
    records = load_jsonl_records(input)
    print(f"Loaded {len(records)} records from {input}")

    updated_records = batch_rag(records, retriever, client)
    save_jsonl_records(updated_records, output)

if __name__ == "__main__":
    main_batch()
