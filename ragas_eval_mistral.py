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

ground_truth_data = load_jsonl("gemini.jsonl")
rag_output_data = load_jsonl("rag_mistral_output.jsonl")

combined_data = []
for gt, rag in zip(ground_truth_data, rag_output_data):
    if not gt.get("Answer"):
        continue
    combined_data.append({
        "question": gt["Question"],
        "answer": rag["Answer"],
        "contexts": rag["Reference"],  # must be a list
        "ground_truth": gt["Answer"]
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
