import requests
import base64
import PyPDF2
import io
from typing import Optional
from openai import OpenAI
from config.settings import (
    HARDCODED_POLICY_NO,
    HARDCODED_ACCESS_TOKEN,
    POLICY_DOCUMENT_API_URL,
    OPENAI_API_KEY
)

client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_schedule_base64(policy_no: str, access_token: str) -> Optional[str]:
    """Call API and get Base64 encoded policy schedule PDF"""
    try:
        url = f"{POLICY_DOCUMENT_API_URL}?policy={policy_no}&token={access_token}&template=schedule"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("document")
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return None

def decode_base64_to_text(base64_string: str) -> Optional[str]:
    """Decode Base64 string to PDF text"""
    try:
        pdf_bytes = base64.b64decode(base64_string)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        return full_text
    except Exception as e:
        print(f"Error decoding schedule PDF: {e}")
        return None

def get_schedule_text(policy_no: str, access_token: str) -> Optional[str]:
    """Fetch and decode policy schedule"""
    base64_string = fetch_schedule_base64(policy_no, access_token)
    if not base64_string:
        return None
    return decode_base64_to_text(base64_string)

def answer_from_schedule(question: str) -> str:
    """
    Main function called by agent
    Uses hardcoded policy_no and token for demo
    TODO: Replace with DB query after demo
    """
    policy_no = HARDCODED_POLICY_NO
    access_token = HARDCODED_ACCESS_TOKEN

    # Fetch schedule text
    schedule_text = get_schedule_text(policy_no, access_token)
    if not schedule_text:
        return "Sorry, I could not retrieve the policy schedule."

    # Generate answer using OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful insurance assistant.
                Answer the user's question using only the provided
                policy schedule context. Be clear and concise."""
            },
            {
                "role": "user",
                "content": f"""
                Policy Schedule Content:
                {schedule_text}

                Question: {question}

                Answer based only on the schedule content provided.
                """
            }
        ]
    )

    return response.choices[0].message.content