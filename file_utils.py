import pdfplumber
import docx
import easyocr
import os

# Initialize the EasyOCR reader globally. This loads the English language model into memory 
# once when the server starts, making subsequent image uploads significantly faster!
ocr_reader = easyocr.Reader(['en'])

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

        # ---------- IMAGE OCR ----------
        elif ext in [".png", ".jpg", ".jpeg"]:

            # EasyOCR natively supports reading directly from the file path
            result = ocr_reader.readtext(file_path, detail=0)
            text = " ".join(result)

        else:
            text = ""

    except Exception as e:
        print("Extraction error:", e)
        return ""

    return text.strip()