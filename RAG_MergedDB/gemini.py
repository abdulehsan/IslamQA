import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

load_dotenv()
input_file = 'test.jsonl'
output_file = 'gemini(rag).jsonl'
db_path = Path("D:\\Abdullah Files\\Programmming\\python\\IslamQA\\Merged_DB")
embedder_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=embedder_name)

gemini_keys = [
    os.getenv("GEMINI_KEY_1"),
    os.getenv("GEMINI_KEY_2"),
    os.getenv("GEMINI_KEY_3"),
    os.getenv("GEMINI_KEY_4"),
    os.getenv("GEMINI_KEY_5"),
    os.getenv("GEMINI_KEY_6"),
]

system_prompt = """
You are an AI Islamic assistant. Your only job is to answer user questions using the provided Islamic context. You are not allowed to generate answers beyond what the context explicitly supports.

You must not include any inner thoughts, step-by-step thinking, or reasoning tags such as <think>. Your answers should be clean, direct, and based only on the provided context.

Your strict operating principles:

1. Do not answer anything not clearly supported by the provided context. No inferences, no assumptions.
2. If the context is incomplete, vague, or irrelevant to the question, simply respond: “I’m sorry, I cannot answer this question based on the provided Islamic sources.”
3. Do not offer personal opinions, reasoning, or interpretations under any circumstances. You are not a scholar.
4. Never generalize from unrelated or ambiguous context. Only use what is clearly and explicitly relevant.
5. If multiple views are in the context (e.g., different Madhahib), present them fairly and without bias.
6. Maintain an ethical, respectful tone in accordance with Islamic principles.
7. Cite references when available in the context (e.g., Surah name, verse number, Hadith ID).
8. Keep your entire response concise and limited to 512 tokens maximum.
9. Prioritize accuracy, Islamic ethics, and clarity. Avoid repetition or filler.

Remember: You are not a general-purpose assistant. You are a domain-restricted AI grounded only in retrieved Islamic sources. If the answer is not in the context, you do not have it.
"""

def load_vectorstore(embedder):
    return FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)

def call_gemini_with_failover(question, context, temperature=0.5, max_tokens=512):
    for api_key in gemini_keys:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt_parts = [
                system_prompt,
                f"\n\n### Retrieved Islamic Context:\n{context}",
                f"\n\n### User Question:\n{question}",
                f"\n\n### Answer (max 512 tokens):"
            ]

            response = model.generate_content(
                prompt_parts,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )

            return response.text
        except Exception as e:
            print(f"[!] Key failed: {api_key[:8]}... | {type(e).__name__}: {e}")
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                print("[!] Rate limit hit. Trying next key...")
                time.sleep(100)
            continue

    raise RuntimeError("All Gemini API keys failed or quota exhausted.")

vector_db = load_vectorstore(embedding)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
results = []

with open(input_file, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx >= 100:
            break
        try:
            data = json.loads(line)
            question = data.get("Question") or data.get("question")
            if not question:
                continue

            retrieved_docs = retriever.invoke(question)
            references = [doc.metadata.get("source") for doc in retrieved_docs]
            context_texts = [doc.page_content for doc in retrieved_docs]
            joined_context = "\n\n".join(context_texts)

            generated_answer = call_gemini_with_failover(
                question=question,
                context=joined_context
            )

            results.append({
                "Question": question,
                "Reference_Answer": data.get("Answer") or data.get("answer"),
                "Generated_Answer": generated_answer,
                "Retrieved_Docs": references,
                "Retrieved_Texts": context_texts,
            })

            print(f"Processed {idx+1}/100")

        except Exception as e:
            print(f"Error at record {idx+1}: {type(e).__name__}: {e}")
            continue

with open(output_file, 'w', encoding='utf-8') as f:
    for record in results:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"\nDone! Output saved to {output_file}")
