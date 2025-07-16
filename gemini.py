import google.generativeai as genai
import json
from os import getenv
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=getenv("gemini_api"))

# Load Gemini 1.5 Pro model
model = genai.GenerativeModel("gemini-2.5-flash")

# question = "What is todays date in pakistan?"
# response = model.generate_content(question)
# generated_answer = response.text.strip()

# print(f"Generated Answer: {generated_answer}" )
# # Input/output files

input_file = 'test.jsonl'
output_file = 'gemini.jsonl'

results = []

# Process first 100 questions
with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        if idx >= 100:
            break
        try:
            data = json.loads(line)
            question = data.get("Question") or data.get("question")
            if not question:
                continue

            # Simple API call to Gemini
            response = model.generate_content(question)
            generated_answer = response.text.strip()

            # Prepare output
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

# Save to JSONL
with open(output_file, 'w', encoding='utf-8') as out_file:
    for item in results:
        out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nDone! Generated answers saved to {output_file}")

