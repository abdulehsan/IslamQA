import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

input_file = 'test.jsonl'
output_file = 'llama17b_maverick(rag).jsonl'
path = Path("D:\\Abdullah Files\\Programmming\\python\\IslamQA\\Merged_DB")
embedder = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=embedder)

groq_keys = [
    os.getenv("GROQ_KEY_1"),
    os.getenv("GROQ_KEY_2"),
    os.getenv("GROQ_KEY_3"),
    os.getenv("GROQ_KEY_4"),
    os.getenv("GROQ_KEY_5"),
]

system_prompt = """You are an AI Islamic assistant designed to answer user questions about Islam using only the verified Islamic sources retrieved for each query.
These sources include translations of the Quran, authentic Hadith collections, and classical/recognized scholarly rulings (e.g., Fiqh, Sharia).
Your responsibilities:
1. Ground every answer in the provided context. Do not answer anything outside it.
2. Do not infer or guess answers. If the retrieved context does not adequately support an answer, respond with:
3. “I’m sorry, I cannot answer this question based on the provided Islamic sources.”
4. Maintain a respectful and neutral tone in line with Islamic ethics.
5. When context is available, structure the answer clearly, possibly including references (e.g., Surah name, Hadith number) if they exist in the context.
6. If multiple views are shown in the context (e.g., different Madhahib), mention that with clarity and neutrality.
7. Avoid issuing any new rulings or personal interpretations. You are not a scholar.
Remember: You are not a general-purpose assistant. You only answer based on retrieved Islamic texts. You do not have opinions or beliefs."""

def load_db(embedder):
    return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)

def call_groq_with_rotation(messages, model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.5, max_tokens=512):
    for key in groq_keys:
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[!] Key failed: {key[:8]}... — {type(e).__name__}: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print("[!] Rate limited. Trying next key...")
                time.sleep(1)
                continue
            else:
                continue
    raise RuntimeError("All Groq API keys failed or quota exceeded.")

vector_db = load_db(embedding)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

results = []

with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        if idx >= 100:
            break
        try:
            data = json.loads(line)
            question = data.get("Question") or data.get("question")
            if not question:
                continue

            context = retriever.invoke(question)
            reference = [f"{doc.metadata.get('source')}" for doc in context]
            retrieved_texts = [f"{doc.page_content}" for doc in context]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:{context}\nUser Query: {question}"}
            ]

            generated_answer = call_groq_with_rotation(messages)

            output_record = {
                "Question": data.get("Question") or data.get("question"),
                "Reference_Answer": data.get("Answer") or data.get("answer"),
                "Generated_Answer": generated_answer,
                "Retrieved_Docs": reference,
                "Retrieved_Texts": retrieved_texts,
            }

            results.append(output_record)
            print(f"Processed {idx + 1}/100")

        except Exception as e:
            print(f"Error at record {idx + 1}: {type(e).__name__}: {e}")
            continue

with open(output_file, 'w', encoding='utf-8') as out_file:
    for item in results:
        out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nDone! Generated answers saved to {output_file}")
