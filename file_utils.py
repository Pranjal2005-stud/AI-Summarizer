import pdfplumber
import docx
import os
import google.generativeai as genai
from dotenv import load_dotenv
import PIL.Image

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def extract_text(file_path):

    text = ""
    ext = os.path.splitext(file_path)[1].lower()

    try:

        # ---------- PDF ----------
        if ext == ".pdf":

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

        # ---------- DOCX ----------
        elif ext == ".docx":

            document = docx.Document(file_path)
            for para in document.paragraphs:
                text += para.text + "\n"

        # ---------- TXT ----------
        elif ext == ".txt":

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # ---------- IMAGE (Gemini Vision) ----------
        elif ext in [".png", ".jpg", ".jpeg"]:

            image = PIL.Image.open(file_path)
            response = model.generate_content(["Extract all the text from this image. Return only the extracted text, nothing else.", image])
            text = response.text.strip()

        else:
            text = ""

    except Exception as e:
        print("Extraction error:", e)
        return ""

    return text.strip()