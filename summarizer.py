import os
import google.generativeai as genai
from dotenv import load_dotenv

# Force load the hidden .env variables before authentication so the API key doesn't return None
load_dotenv()

# Securely configure the API key imported from the backend environment
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize the lightning fast Gemini 2.5 Flash model specifically designed for massive context handling
model = genai.GenerativeModel("gemini-2.5-flash")

WORD_TARGETS = {
    "concise":  "strictly between 100 and 110 words",
    "midsize":  "approximately 200 words",
    "detailed": "between 300 and 350 words",
}

def generate_summary(text, mode="concise"):
    if len(text.split()) < 40:
        return text

    safe_text = text[:500000]
    target = WORD_TARGETS.get(mode, WORD_TARGETS["concise"])
    prompt = f"Read the following document carefully and write a clear, intelligent summary covering all main points in {target}. Return only the summary text.\n\nDOCUMENT CONTENT:\n{safe_text}"

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "A cloud error occurred while communicating with the Gemini AI. Please check your network or API limits and try again."