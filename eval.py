import json
import bert_score
from rouge_score import rouge_scorer
from typing import List
from pathlib import Path


def load_pairs(file_path: str, ref_key: str, gen_key: str) -> (List[str], List[str]):
    references, candidates = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ref = data.get(ref_key)
            gen = data.get(gen_key)
            if ref and gen:
                references.append(ref.strip())
                candidates.append(gen.strip())
    return references, candidates

def evaluate_scores(candidates: List[str], references: List[str], model_name="en"):
    P, R, F1 = bert_score.score(candidates, references, lang=model_name)
    print(f"\n--- BERTScore ---")
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall:    {R.mean().item():.4f}")
    print(f"F1 Score:  {F1.mean().item():.4f}")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []

    for ref, gen in zip(references, candidates):
        scores = scorer.score(ref, gen)
        rouge1_list.append(scores['rouge1'].fmeasure)
        rouge2_list.append(scores['rouge2'].fmeasure)
        rougeL_list.append(scores['rougeL'].fmeasure)

    print(f"\n--- ROUGE Scores ---")
    print(f"ROUGE-1 F1: {sum(rouge1_list)/len(rouge1_list):.4f}")
    print(f"ROUGE-2 F1: {sum(rouge2_list)/len(rouge2_list):.4f}")
    print(f"ROUGE-L F1: {sum(rougeL_list)/len(rougeL_list):.4f}")
    print()

# File and key config
file_path = Path("gemini(rag).jsonl")
ref_key = "Reference_Answer"
gen_key = "Generated_Answer"

# Load and evaluate
references, candidates = load_pairs(file_path, ref_key, gen_key)
min_len = min(len(references), len(candidates))
evaluate_scores(candidates[:min_len], references[:min_len])
