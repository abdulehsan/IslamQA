import json
from groq import Groq
from os import getenv
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=getenv("groq"),
)

input_file = 'test.jsonl'
output_file = 'deepseek_v3_0324.jsonl'


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

            response = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                 max_completion_tokens=512,
                stream=False
            )

            generated_answer = response.choices[0].message.content.strip()

            output_record = {
                "Question": data.get("Question") or data.get("question"),
                "Answer": data.get("Answer") or data.get("answer"),
                "Document": data.get("Document") or data.get("document"),
                "Generated_Answer": generated_answer
            }

            results.append(output_record)
            print(f"Processed {idx+1}/100")

        except Exception as e:
            print(f"Error at record {idx+1}: {type(e).__name__}: {e}")
            continue

with open(output_file, 'w', encoding='utf-8') as out_file:
    for item in results:
        out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nDone! Generated answers saved to {output_file}")
