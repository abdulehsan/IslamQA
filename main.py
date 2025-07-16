import json
from langchain_openai import ChatOpenAI
from os import getenv

# Initialize LLM
op_api_key = getenv("openrouter")
if not op_api_key:
    raise ValueError("OPENROUTER API key missing.")

llm = ChatOpenAI(
    openai_api_key=op_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="tngtech/deepseek-r1t2-chimera:free",
    temperature=0.7
)

# Load questions and process
input_file = 'test.jsonl'
output_file = 'deepseek-r1t2-chimera.jsonl'
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

            # Directly call the model
            response = llm.invoke(question).content.strip()

            # Store result
            output_record = {
                "Question": data.get("Question") or data.get("question"),
                "Answer": data.get("Answer") or data.get("answer"),
                "Document": data.get("Document") or data.get("document"),
                "Generated_Answer": response
            }

            results.append(output_record)
            print(f"Processed {idx+1}/100")

        except Exception as e:
            print(f"Skipping record {idx+1}: {e}")
            continue

# Save results
with open(output_file, 'w', encoding='utf-8') as out_file:
    for item in results:
        out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nGenerated answers saved to {output_file}")
