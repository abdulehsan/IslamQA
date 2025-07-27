import json
import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    # context_relevancy,
)
from ragas import evaluate
import os
import openai
from dotenv import load_dotenv
import re

load_dotenv()

openai.api_key = os.getenv("openai_api")

def clean_text(text):
    if not isinstance(text, str):
        return text
    CUSTOM_STOPWORDS = {
    "narrated","Narrated","<D>","<think>"
}
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.lower()
    text = re.sub(r"[\\\/\[\]\(\)\{\}]", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    text = re.sub(r"([.,!?])\1+", r"\1", text)
    text = re.sub(r"(^|\s)[.,!?]+", " ", text)
    text = re.sub(r"[.,!?]+(\s|$)", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

candidate = load_jsonl("deepseek_r1_distill_llama_70b.jsonl")
Documents = [cand["Document"] for cand in candidate]

combined_data = []
for cand in candidate:
    if not cand.get("Answer"):
        continue
    combined_data.append({
        "question": cand["Question"],
        "answer": cand["Generated_Answer"],
        "contexts": Documents,
        "ground_truth": cand["Answer"]
    })

df = pd.DataFrame(combined_data)
df = df.applymap(clean_text)
dataset = Dataset.from_pandas(df)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        # context_relevancy,
    ],
)
df = results.to_pandas()
average_scores = df.mean(numeric_only=True)
print("Average Scores:")
print(average_scores)

print("RAGAS Metrics:")
print(results)
