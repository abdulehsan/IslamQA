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

load_dotenv()

openai.api_key = os.getenv("openai_api")

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

candidate = load_jsonl("mistral_7b_instruct_v0.1.jsonl")
reference = load_jsonl("test.jsonl")

combined_data = []
for ref, cand in zip(reference, candidate):
    if not ref.get("Answer"):
        continue
    combined_data.append({
        "question": ref["Question"],
        "answer": cand["Answer"],
        "contexts": cand["Reference"],
        "ground_truth": ref["Answer"]
    })

df = pd.DataFrame(combined_data)
dataset = Dataset.from_pandas(df)

# Evaluate using RAGAS
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
