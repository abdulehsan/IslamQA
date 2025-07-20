import json
import bert_score
from rouge_score import rouge_scorer
import transformers
transformers.logging.set_verbosity_error()

generated_answers = []
actual_answers = []

input_file = 'gemini.jsonl'
with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        data = json.loads(line)
        answer = data.get("Answer")
        generated_answer = data.get("Generated_Answer")

        if generated_answer and answer:
            generated_answers.append(generated_answer.strip())
            actual_answers.append(answer.strip())
        else:
            continue

P, R, F1 = bert_score.score(generated_answers, actual_answers, lang="en")

print(f"Average BERTScore Precision: {P.mean().item():.4f}")
print(f"Average BERTScore Recall: {R.mean().item():.4f}")
print(f"Average BERTScore F1: {F1.mean().item():.4f}")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_list, rouge2_list, rougeL_list = [], [], []

for ref, gen in zip(actual_answers, generated_answers):
    scores = scorer.score(ref, gen)
    rouge1_list.append(scores['rouge1'].fmeasure)
    rouge2_list.append(scores['rouge2'].fmeasure)
    rougeL_list.append(scores['rougeL'].fmeasure)

print(f"Average ROUGE-1 F1: {sum(rouge1_list)/len(rouge1_list):.4f}")
print(f"Average ROUGE-2 F1: {sum(rouge2_list)/len(rouge2_list):.4f}")
print(f"Average ROUGE-L F1: {sum(rougeL_list)/len(rougeL_list):.4f}")
