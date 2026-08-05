import requests
import base64
import PyPDF2
import io
import os
import logging
from typing import Optional, List, Any
from openai import OpenAI
from vector_store.chroma import (
    get_or_create_collection,
    delete_collection,
    collection_exists
)
from config.settings import (
    POLICY_DOCUMENT_API_URL,
    OPENAI_API_KEY
)
from db.connection import (
    get_latest_policy_wording_credentials,
    get_policy_credentials_by_no
)

# ── Logging Setup ────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pdf_tool.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("pdf_tool")

client = OpenAI(api_key=OPENAI_API_KEY)

# Cache to track indexed tokens
indexed_tokens: dict = {}

def fetch_pdf_base64(policy_no: str, access_token: str) -> Optional[str]:
    """Call API and get Base64 encoded PDF"""
    url = f"{POLICY_DOCUMENT_API_URL}?policy={policy_no}&token={access_token}&template=wording"
    logger.info(f"[FETCH PDF] Calling wording API for policy: {policy_no}")

    try:
        response = requests.get(url, timeout=30)
        logger.info(f"[FETCH PDF] API response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        doc = data.get("document")

        if doc:
            logger.info(f"[FETCH PDF] Base64 document received - length: {len(doc)} chars")
        else:
            logger.warning("[FETCH PDF] No 'document' field in API response")
            logger.warning(f"[FETCH PDF] Response keys: {list(data.keys())}")

        return doc

    except requests.exceptions.Timeout:
        logger.error("[FETCH PDF] API request timed out after 30 seconds")
        return None
    except requests.exceptions.HTTPError as e:
        logger.exception("[FETCH PDF] HTTP error")
        return None
    except Exception as e:
        logger.exception("[FETCH PDF] Unexpected error")
        return None

def decode_base64_to_text(base64_string: str) -> Optional[str]:
    """Decode Base64 string to PDF text"""
    logger.info("[DECODE PDF] Decoding Base64 to PDF text")

    try:
        pdf_bytes = base64.b64decode(base64_string)
        logger.info(f"[DECODE PDF] PDF bytes size: {len(pdf_bytes)} bytes")

        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        logger.info(f"[DECODE PDF] Total pages: {total_pages}")

        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            full_text += page_text + "\n"
            logger.info(f"[DECODE PDF] Page {i + 1}/{total_pages} - {len(page_text)} chars")

        logger.info(f"[DECODE PDF] Total text: {len(full_text)} chars")

        if len(full_text.strip()) == 0:
            logger.warning("[DECODE PDF] Extracted text is empty")

        return full_text

    except Exception as e:
        logger.exception("[DECODE PDF] Error")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    logger.info(f"[CHUNK] Total words: {len(words)} | Chunks: {len(chunks)}")
    return chunks

def get_embedding(text: str) -> list:
    """Get embedding from OpenAI"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def index_pdf(policy_no: str, access_token: str) -> bool:
    """Fetch PDF, extract text, chunk, embed and store in ChromaDB"""
    logger.info(f"[INDEX PDF] Starting indexing for policy: {policy_no}")

    base64_string = fetch_pdf_base64(policy_no, access_token)
    if not base64_string:
        logger.error("[INDEX PDF] Failed to fetch Base64 PDF")
        return False

    text = decode_base64_to_text(base64_string)
    if not text:
        logger.error("[INDEX PDF] Failed to decode PDF text")
        return False

    chunks = chunk_text(text)
    if not chunks:
        logger.error("[INDEX PDF] No chunks created")
        return False

    collection_name = f"policy_wording_{policy_no}"
    logger.info(f"[INDEX PDF] Deleting old collection: {collection_name}")
    delete_collection(collection_name)

    collection = get_or_create_collection(collection_name)
    logger.info(f"[INDEX PDF] Embedding {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )
        if (i + 1) % 10 == 0:
            logger.info(f"[INDEX PDF] Progress: {i + 1}/{len(chunks)}")

    indexed_tokens[policy_no] = access_token
    logger.info(f"[INDEX PDF] Indexing complete - {len(chunks)} chunks stored")
    return True

def should_reindex(policy_no: str, access_token: str) -> bool:
    """Check if PDF needs re-indexing"""
    collection_name = f"policy_wording_{policy_no}"

    if not collection_exists(collection_name):
        logger.info("[REINDEX CHECK] Collection not found - reindex needed")
        return True

    cached_token = indexed_tokens.get(policy_no)
    if cached_token != access_token:
        logger.info("[REINDEX CHECK] Token changed - reindex needed")
        return True

    logger.info("[REINDEX CHECK] Using existing cache - no reindex needed")
    return False

def resolve_question_with_history(
    question: str,
    history: List[dict]
) -> str:
    """
    KEY FIX: Rewrite vague questions using conversation history
    Turns "tell me more about it" into a specific searchable question
    Uses GPT to understand what "it" refers to from history
    """
    if not history:
        return question

    question_lower = question.lower().strip()

    # Check if question is vague (contains pronouns or is short)
    vague_indicators = [
        "it", "that", "this", "those", "these",
        "tell me more", "explain", "elaborate",
        "what about", "more about", "go on"
    ]

    is_vague = any(indicator in question_lower for indicator in vague_indicators)
    is_short = len(question.split()) <= 6

    if not (is_vague or is_short):
        # Question is specific enough — no need to rewrite
        return question

    logger.info(f"[RESOLVE QUESTION] Vague question detected: '{question}'")
    logger.info(f"[RESOLVE QUESTION] Resolving using {len(history)} history messages")

    try:
        # Build history context for GPT
        recent_history = history[-6:] if len(history) > 6 else history
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in recent_history
        ])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a question resolver for an insurance assistant.
                    
                    Given a conversation history and a vague follow-up question,
                    rewrite the follow-up question to be specific and searchable.
                    
                    Rules:
                    - Replace pronouns like "it", "that", "this" with the actual topic
                    - Make the question self-contained and specific
                    - Keep it as a question about insurance
                    - Return ONLY the rewritten question, nothing else
                    - If you cannot determine what the question refers to,
                      return the original question unchanged
                    
                    Example:
                    History: User asked about exclusions, bot explained war exclusions
                    Follow-up: "tell me more about it"
                    Rewritten: "Tell me more about war exclusions in travel insurance"
                    """
                },
                {
                    "role": "user",
                    "content": f"""Conversation history:
{history_text}

Follow-up question: {question}

Rewrite this follow-up question to be specific:"""
                }
            ],
            max_tokens=100
        )

        resolved = response.choices[0].message.content.strip()
        logger.info(f"[RESOLVE QUESTION] Original: '{question}'")
        logger.info(f"[RESOLVE QUESTION] Resolved: '{resolved}'")
        return resolved

    except Exception as e:
        logger.exception("[RESOLVE QUESTION] Error resolving question")
        return question

def search_pdf(
    policy_no: str,
    question: str,
    top_k: int = 3,
    return_metadata: bool = False
) -> Optional[Any]:
    """Search ChromaDB for relevant chunks"""
    collection_name = f"policy_wording_{policy_no}"
    logger.info(f"[SEARCH PDF] Searching: {collection_name}")
    logger.info(f"[SEARCH PDF] Question: {question}")

    collection = get_or_create_collection(collection_name)
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    if results and results["documents"] and results["documents"][0]:
        chunks = results["documents"][0]
        distances = results.get("distances", [[]])[0]
        logger.info(f"[SEARCH PDF] Found {len(chunks)} relevant chunks")
        for i, (chunk, dist) in enumerate(zip(chunks, distances)):
            logger.info(f"[SEARCH PDF] Chunk {i + 1} distance: {dist:.4f}")
            logger.info(f"[SEARCH PDF] Chunk {i + 1} preview: {chunk[:150]}...")
        if return_metadata:
            return [
                {
                    "chunk_id": i + 1,
                    "content": chunk,
                    "distance": round(float(dist), 4) if dist is not None else None
                }
                for i, (chunk, dist) in enumerate(zip(chunks, distances))
            ]
        return "\n\n".join(chunks)

    logger.warning("[SEARCH PDF] No relevant chunks found")
    return None

def answer_from_pdf(
    question: str,
    policy_no: Optional[str] = None,
    conversation_history: Optional[List[dict]] = None,
    return_metadata: bool = False
) -> Any:
    """
    Main function called by agent

    Two scenarios:
    1. policy_no is None - get latest wording from DB
    2. policy_no provided - get wording for that specific policy

    Now also accepts conversation_history to resolve vague questions
    """
    if conversation_history is None:
        conversation_history = []

    logger.info(f"[PDF TOOL] Question: {question}")
    logger.info(f"[PDF TOOL] Policy: {policy_no if policy_no else 'latest'}")
    logger.info(f"[PDF TOOL] History: {len(conversation_history)} messages")

    # ── Step 1: Resolve vague question using history ──────────────
    resolved_question = resolve_question_with_history(
        question,
        conversation_history
    )

    if policy_no is None:
        logger.info("[PDF TOOL] Fetching latest policy wording from DB")
        credentials = get_latest_policy_wording_credentials()
        if not credentials:
            logger.error("[PDF TOOL] No active policy wording found in DB")
            return "Sorry, I could not find any active policy wording in the system."
        policy_no = credentials["policy_no"]
        access_token = credentials["access_token"]
        logger.info(f"[PDF TOOL] Latest policy: {policy_no}")
    else:
        logger.info(f"[PDF TOOL] Fetching credentials for policy: {policy_no}")
        credentials = get_policy_credentials_by_no(policy_no)
        if not credentials:
            logger.error(f"[PDF TOOL] Policy {policy_no} not found")
            return f"Sorry, I could not find policy {policy_no} in the system."
        access_token = credentials["access_token"]

    logger.info(f"[PDF TOOL] Token (first 8): {access_token[:8]}...")

    # ── Step 2: Re-index if needed ────────────────────────────────
    if should_reindex(policy_no, access_token):
        logger.info(f"[PDF TOOL] Re-indexing PDF for: {policy_no}")
        success = index_pdf(policy_no, access_token)
        if not success:
            return "Sorry, I could not retrieve the policy wording document."
    else:
        logger.info(f"[PDF TOOL] Using cached index for: {policy_no}")

    # ── Step 3: Search using RESOLVED question ────────────────────
    logger.info(f"[PDF TOOL] Searching with resolved question: {resolved_question}")
    relevant_chunks = search_pdf(
        policy_no,
        resolved_question,
        return_metadata=return_metadata
    )

    if not relevant_chunks:
        logger.warning("[PDF TOOL] No relevant chunks found")
        if return_metadata:
            return {
                "answer": "Sorry, I could not find relevant information in the policy wording.",
                "resolved_question": resolved_question,
                "wording_chunks": []
            }
        return "Sorry, I could not find relevant information in the policy wording."

    logger.info("[PDF TOOL] Sending to GPT for answer generation")

    # ── Step 4: Generate answer ───────────────────────────────────
    try:
        context_text = relevant_chunks
        if return_metadata and isinstance(relevant_chunks, list):
            context_text = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful insurance assistant.
                    Answer the user question using only the provided
                    policy wording context. Be clear and concise.
                    If the answer is not in the context say so."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Context from policy wording:
                    {context_text}

                    Question: {resolved_question}
                    """
                }
            ]
        )

        answer = response.choices[0].message.content
        logger.info(f"[PDF TOOL] Answer: {answer[:200]}...")
        return answer

    except Exception as e:
        logger.error(f"[PDF TOOL] GPT error: {e}")
        return "Sorry, I encountered an error generating the answer."