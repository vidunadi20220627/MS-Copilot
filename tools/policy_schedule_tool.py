import requests
import base64
import PyPDF2
import io
import logging
from typing import Optional
from openai import OpenAI
from config.settings import (
    POLICY_DOCUMENT_API_URL,
    OPENAI_API_KEY
)
from db.connection import get_policy_credentials_by_no

# ── Logging Setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/schedule_tool.log")
    ]
)
logger = logging.getLogger("schedule_tool")

client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_schedule_base64(policy_no: str, access_token: str) -> Optional[str]:
    """Call API and get Base64 encoded policy schedule"""
    url = f"{POLICY_DOCUMENT_API_URL}?policy={policy_no}&token={access_token}&template=schedule"
    logger.info(f"[FETCH SCHEDULE] Calling schedule API for policy: {policy_no}")
    logger.info(f"[FETCH SCHEDULE] URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        logger.info(f"[FETCH SCHEDULE] API response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        doc = data.get("document")

        if doc:
            logger.info(f"[FETCH SCHEDULE] Base64 document received — length: {len(doc)} chars")
        else:
            logger.warning("[FETCH SCHEDULE] API returned response but no 'document' field found")
            logger.warning(f"[FETCH SCHEDULE] Response keys: {list(data.keys())}")

        return doc

    except requests.exceptions.Timeout:
        logger.error("[FETCH SCHEDULE] API request timed out after 30 seconds")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"[FETCH SCHEDULE] HTTP error: {e} | Status: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"[FETCH SCHEDULE] Unexpected error: {e}")
        return None

def decode_base64_to_text(base64_string: str) -> Optional[str]:
    """Decode Base64 to PDF text"""
    logger.info("[DECODE SCHEDULE] Decoding Base64 to PDF text")

    try:
        pdf_bytes = base64.b64decode(base64_string)
        logger.info(f"[DECODE SCHEDULE] PDF bytes size: {len(pdf_bytes)} bytes")

        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        logger.info(f"[DECODE SCHEDULE] Total pages in PDF: {total_pages}")

        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            full_text += page_text + "\n"
            logger.info(f"[DECODE SCHEDULE] Page {i + 1}/{total_pages} — {len(page_text)} chars")

        logger.info(f"[DECODE SCHEDULE] Total text extracted: {len(full_text)} chars")

        if len(full_text.strip()) == 0:
            logger.warning("[DECODE SCHEDULE] Extracted text is empty — PDF may be image based")

        return full_text

    except Exception as e:
        logger.error(f"[DECODE SCHEDULE] Error decoding PDF: {e}")
        return None

def answer_from_schedule(question: str, policy_no: str) -> str:
    """
    Main function called by agent
    Only called when user provides a policy number
    """
    logger.info(f"[SCHEDULE TOOL] answer_from_schedule called")
    logger.info(f"[SCHEDULE TOOL] Question: {question}")
    logger.info(f"[SCHEDULE TOOL] Policy no: {policy_no}")

    # Get credentials for specific policy
    logger.info(f"[SCHEDULE TOOL] Fetching credentials from DB for policy: {policy_no}")
    credentials = get_policy_credentials_by_no(policy_no)

    if not credentials:
        logger.error(f"[SCHEDULE TOOL] Policy {policy_no} not found in DB view")
        return f"Sorry, I could not find policy {policy_no} in the system. Please check the policy number and try again."

    access_token = credentials["access_token"]
    logger.info(f"[SCHEDULE TOOL] Credentials found — token (first 8): {access_token[:8]}...")

    # Fetch schedule
    base64_string = fetch_schedule_base64(policy_no, access_token)
    if not base64_string:
        logger.error(f"[SCHEDULE TOOL] Failed to fetch schedule for policy: {policy_no}")
        return f"Sorry, I could not retrieve the policy schedule for {policy_no}."

    # Decode to text
    schedule_text = decode_base64_to_text(base64_string)
    if not schedule_text:
        logger.error("[SCHEDULE TOOL] Failed to decode schedule PDF")
        return "Sorry, I could not read the policy schedule."

    logger.info(f"[SCHEDULE TOOL] Schedule text ready — {len(schedule_text)} chars")
    logger.info("[SCHEDULE TOOL] Sending schedule to GPT for answer generation")

    # Generate answer
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful insurance assistant.
                    Answer the user question using only the provided
                    policy schedule content. Be clear and concise.
                    If the answer is not in the schedule say so."""
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
        logger.info(f"[SCHEDULE TOOL] GPT answer: {answer[:200]}...")
        return answer

    except Exception as e:
        logger.error(f"[SCHEDULE TOOL] GPT error: {e}")
        return "Sorry, I encountered an error generating the answer."