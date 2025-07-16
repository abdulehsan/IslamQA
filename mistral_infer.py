# import json
# from huggingface_hub import InferenceClient
# from transformers import AutoTokenizer
# from huggingface_hub import login
# from os import getenv
# import traceback

# hf_tokens = getenv("hf_hub_token")
# login(hf_tokens)


# # HF API Setup
# API_URL = "mistralai/Mistral-7B-Instruct-v0.1"  # or v0.1
# client = InferenceClient(model=API_URL)
# tokenizer = AutoTokenizer.from_pretrained(API_URL)

# # File Paths
# input_file = 'test.jsonl'
# output_file = 'mistral_7b_instruct_v0.1.jsonl'

# # Parameters
# TEMPERATURE = 0.7
# MAX_NEW_TOKENS = 512

# # Processing
# results = []

# with open(input_file, 'r', encoding='utf-8') as file:
#     for idx, line in enumerate(file):
#         if idx >= 100:
#             break
#         try:
#             data = json.loads(line)
#             question = data.get("Question") or data.get("question")
#             if not question:
#                 continue

#             # Create chat template (HF handles the prompt formatting)
#             messages = [{"role": "user", "content": question}]
#             prompt = tokenizer.apply_chat_template(messages, tokenize=False)

#             # Call HF Inference API
#             response = client.text_generation(
#                 prompt,
#                 max_new_tokens=512,
#                 temperature=0.7,
#             )
#             generated_answer = response["choices"][0]["message"]["content"].strip()
#             # Save full record
#             output_record = {
#                 "Question": data.get("Question") or data.get("question"),
#                 "Answer": data.get("Answer") or data.get("answer"),
#                 "Document": data.get("Document") or data.get("document"),
#                 "Generated_Answer": generated_answer
#             }

#             results.append(output_record)
#             print(f"Processed {idx+1}/100")

#         except Exception as e:
#             print(f"Error at record {idx+1}:  {type(e).__name__}: {e}")
#             traceback.print_exc()
#             continue

# # Write to JSONL
# with open(output_file, 'w', encoding='utf-8') as out_file:
#     for item in results:
#         out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

# print(f"\nDone! Generated answers saved to {output_file}")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load v0.1 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Example messages
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# Format prompt using chat template
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

# Generate response
generated_ids = model.generate(
    model_inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(output)
