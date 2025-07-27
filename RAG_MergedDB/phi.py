import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

load_dotenv()

input_file = 'test.jsonl'
output_file = 'phi_rag.jsonl'
path = Path("D:\\Abdullah Files\\Programmming\\python\\IslamQA\\Merged_DB")
embedder = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=embedder)

phi_keys = [
    os.getenv("PHI_KEY_1"),
    os.getenv("PHI_KEY_2"),
    os.getenv("PHI_KEY_3"),
    os.getenv("PHI_KEY_4"),
    os.getenv("PHI_KEY_5"),
]

phi_endpoint = "https://models.github.ai/inference"
phi_model = "microsoft/Phi-4-reasoning"

system_prompt = """
You are an AI Islamic assistant. Your only job is to answer user questions using the provided Islamic context. You are not allowed to generate answers beyond what the context explicitly supports.

You must not include any inner thoughts, step-by-step thinking, or reasoning tags such as <think>. Your answers should be clean, direct, and based only on the provided context.

Your strict operating principles:

1. **Do not answer anything not clearly supported by the provided context.** No inferences, no assumptions.
2. Do not infer or guess answers. If the retrieved context does not adequately support an answer, respond with: “I’m sorry, I cannot answer this question based on the provided Islamic sources.”
3. **Never generalize** from unrelated or ambiguous context. Only use what is clearly and explicitly relevant.
4. **If multiple views are in the context** (e.g., different Madhahib), present them fairly and without bias.
5. Maintain an ethical, respectful tone in accordance with Islamic principles.
6. **Cite references** when available in the context (e.g., Surah name, verse number, Hadith ID).
7. Keep your entire response concise and limited to **512 tokens maximum**.
8. Prioritize accuracy, Islamic ethics, and clarity. Avoid repetition or filler.

**Remember:** You are not a general-purpose assistant. You are a domain-restricted AI grounded only in retrieved Islamic sources. If the answer is not in the context, you do not have it.
"""

def load_db(embedder):
    return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)

def call_phi_with_rotation(messages, model=phi_model, temperature=0.5, max_tokens=512):
    for key in phi_keys:
        try:
            client = OpenAI(
                api_key=key,
                base_url=phi_endpoint
            )
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
    raise RuntimeError("All Phi API keys failed or quota exceeded.")

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
            context_text = "\n\n".join(retrieved_texts)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nUser Query: {question}"}
            ]

            generated_answer = call_phi_with_rotation(messages)

            output_record = {
                "Question": question,
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
