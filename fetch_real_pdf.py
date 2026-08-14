import sys
sys.path.insert(0, ".")
from tools.pdf_tool import fetch_pdf_base64, decode_base64_to_text, clean_pdf_text
import base64
import PyPDF2
import io
from db.connection import get_policy_credentials_by_no

credentials = get_policy_credentials_by_no('DTPS26402347')
access_token = credentials["access_token"]
base64_string = fetch_pdf_base64('DTPS26402347', access_token)

pdf_bytes = base64.b64decode(base64_string)
pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
full_text = ""
for i, page in enumerate(pdf_reader.pages):
    full_text += page.extract_text() + "\n"

with open('raw_pdf_text.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

cleaned_text = clean_pdf_text(full_text)
with open('cleaned_pdf_text.txt', 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

print("Saved raw and cleaned text.")
