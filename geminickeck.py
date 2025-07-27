import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def try_key(key, prompt):
    if not key:
        raise RuntimeError("Empty or missing API key.")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Key failed: {e}")

gemini_keys = [
    os.getenv("GEMINI_KEY_1"),
    os.getenv("GEMINI_KEY_2"),
    os.getenv("GEMINI_KEY_3"),
    os.getenv("GEMINI_KEY_4"),
    os.getenv("GEMINI_KEY_5"),
    os.getenv("GEMINI_KEY_6"),
]

result = None

for key in gemini_keys:
    try:
        result = try_key(key, "Say hello")
        print("SUCCESS:", result)
    except RuntimeError as e:
        print(f"{key} failed:", e)

if result is None:
    raise RuntimeError("All Gemini API keys failed or quota exhausted.")
