import json
from rouge_score import rouge_scorer

input_file = 'mistral_7b_instruct_v0.1.jsonl'

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file, 1):
        data = json.loads(line)
        answer = data.get("Answer")
        gen_answer = data.get("Generated_Answer")

        if not answer or not gen_answer:
            continue

        scores = scorer.score(answer, gen_answer)

        print(f"Question {idx}:")
        print(f"  ROUGE-1 F1 Score: {scores['rouge1'].fmeasure:.4f}")
        print(f"  ROUGE-L F1 Score: {scores['rougeL'].fmeasure:.4f}")

