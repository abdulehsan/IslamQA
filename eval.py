import json
import bert_score
from rouge_score import rouge_scorer
from typing import List

def load_answers(file_path: str, answer_key: str) -> List[str]:
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            answer = data.get(answer_key)
            if answer:
                answers.append(answer.strip())
    return answers

def evaluate_scores(candidates: List[str], references: List[str], model_name="en"):
    P, R, F1 = bert_score.score(candidates, references, lang=model_name)
    print(f"--- BERTScore ---")
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

    print(f"--- ROUGE Scores ---")
    print(f"ROUGE-1 F1: {sum(rouge1_list)/len(rouge1_list):.4f}")
    print(f"ROUGE-2 F1: {sum(rouge2_list)/len(rouge2_list):.4f}")
    print(f"ROUGE-L F1: {sum(rougeL_list)/len(rougeL_list):.4f}")
    print()

reference_file = "gemini.jsonl"
reference_answers = load_answers(reference_file, answer_key="Answer")  # or "Generated_Answer" if that's the key

candidate_files = [
    "rag_mistral_output.jsonl",
    ]

for candidate_file in candidate_files:
    print(f"Evaluating: {candidate_file}")
    candidate_answers = load_answers(candidate_file, answer_key="Answer")
    
    min_len = min(len(candidate_answers), len(reference_answers))
    evaluate_scores(candidate_answers[:min_len], reference_answers[:min_len])
