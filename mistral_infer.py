import json
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from huggingface_hub import login
from os import getenv
import traceback

hf_tokens = getenv("hf_hub_token")
login(hf_tokens)


API_URL = "mistralai/Mistral-7B-Instruct-v0.1"  # or v0.1
client = InferenceClient(model=API_URL)
tokenizer = AutoTokenizer.from_pretrained(API_URL)

input_file = 'test.jsonl'
output_file = 'mistral_7b_instruct_v0.1.jsonl'

TEMPERATURE = 0.1
MAX_NEW_TOKENS = 512

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

            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)

            response = client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
            )
            generated_answer = response["choices"][0]["message"]["content"].strip()
            output_record = {
                "Question": data.get("Question") or data.get("question"),
                "Answer": data.get("Answer") or data.get("answer"),
                "Document": data.get("Document") or data.get("document"),
                "Generated_Answer": generated_answer
            }

            results.append(output_record)
            print(f"Processed {idx+1}/100")

        except Exception as e:
            print(f"Error at record {idx+1}:  {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

with open(output_file, 'w', encoding='utf-8') as out_file:
    for item in results:
        out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nDone! Generated answers saved to {output_file}")

