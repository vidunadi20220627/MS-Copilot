import requests
import base64
import PyPDF2
import io
import logging
from typing import Optional, Any
from openai import OpenAI
from config.settings import (
    POLICY_DOCUMENT_API_URL,
    OPENAI_API_KEY
)
from db.connection import get_policy_credentials_by_no
from tools.pdf_tool import clean_pdf_text

logger = logging.getLogger("policy_schedule_tool")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=20.0
)

def fetch_schedule_base64(policy_no: str, access_token: str) -> Optional[str]:
    """Call API and get Base64 encoded policy schedule"""
    try:
        url = f"{POLICY_DOCUMENT_API_URL}?policy={policy_no}&token={access_token}&template=schedule"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("document")
    except Exception as e:
        logger.exception("Error fetching schedule: %s", e)
        return None

def decode_base64_to_text(base64_string: str) -> Optional[str]:
    """Decode Base64 to PDF text, then clean it."""
    try:
        pdf_bytes = base64.b64decode(base64_string)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        # Apply text cleaning to fix broken words from PyPDF2
        full_text = clean_pdf_text(full_text)

        return full_text
    except Exception as e:
        logger.exception("Error decoding schedule: %s", e)
        return None

def answer_from_schedule(question: str, policy_no: str, return_metadata: bool = False) -> Any:
    """
    Main function called by agent
    Only called when user provides a policy number
    Gets credentials for that specific policy from DB
    """
    # Get credentials for specific policy
    credentials = get_policy_credentials_by_no(policy_no)
    if not credentials:
        if return_metadata:
            return {"answer": f"Sorry, I could not find policy {policy_no} in the system. Please check the policy number and try again.", "schedule_text": None}
        return f"Sorry, I could not find policy {policy_no} in the system. Please check the policy number and try again."

    access_token = credentials["access_token"]

    # Fetch schedule
    base64_string = fetch_schedule_base64(policy_no, access_token)
    if not base64_string:
        if return_metadata:
            return {"answer": f"Sorry, I could not retrieve the policy schedule for {policy_no}.", "schedule_text": None}
        return f"Sorry, I could not retrieve the policy schedule for {policy_no}."

    # Decode to text
    schedule_text = decode_base64_to_text(base64_string)
    if not schedule_text:
        if return_metadata:
            return {"answer": "Sorry, I could not read the policy schedule.", "schedule_text": None}
        return "Sorry, I could not read the policy schedule."

    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an insurance assistant for brokers.
                Answer using ONLY the provided policy schedule content.

                Format rules:
                - If the answer has more than one distinct point (e.g. multiple benefits, multiple dates), use short bullet points (start each with "- ")
                - If it's a single fact, answer in one short sentence — no bullets needed
                - No preamble, no repeating the question
                - Each bullet should be one short phrase, not a paragraph
                - If the answer is not in the schedule, say so in one line"""
            },
            {
                "role": "user",
                "content": f"""
                Policy Schedule for {policy_no}:
                {schedule_text}

                Question: {question}
                """
            }
        ]
    )

    answer = response.choices[0].message.content
    if return_metadata:
        return {"answer": answer, "schedule_text": schedule_text}
    return answer