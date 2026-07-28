import requests
import base64
import PyPDF2
import io
import logging
from typing import Optional
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pdf_tool.log")
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
    logger.info(f"[FETCH PDF] URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        logger.info(f"[FETCH PDF] API response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        doc = data.get("document")

        if doc:
            logger.info(f"[FETCH PDF] Base64 document received — length: {len(doc)} chars")
        else:
            logger.warning("[FETCH PDF] API returned response but no 'document' field found")
            logger.warning(f"[FETCH PDF] Response keys: {list(data.keys())}")

        return doc

    except requests.exceptions.Timeout:
        logger.error("[FETCH PDF] API request timed out after 30 seconds")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"[FETCH PDF] HTTP error: {e} | Status: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"[FETCH PDF] Unexpected error: {e}")
        return None

def decode_base64_to_text(base64_string: str) -> Optional[str]:
    """Decode Base64 string to PDF text"""
    logger.info("[DECODE PDF] Decoding Base64 to PDF text")

    try:
        pdf_bytes = base64.b64decode(base64_string)
        logger.info(f"[DECODE PDF] PDF bytes size: {len(pdf_bytes)} bytes")

        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        logger.info(f"[DECODE PDF] Total pages in PDF: {total_pages}")

        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            full_text += page_text + "\n"
            logger.info(f"[DECODE PDF] Page {i + 1}/{total_pages} extracted — {len(page_text)} chars")

        logger.info(f"[DECODE PDF] Total text extracted: {len(full_text)} chars")

        if len(full_text.strip()) == 0:
            logger.warning("[DECODE PDF] Extracted text is empty — PDF may be image based")

        return full_text

    except Exception as e:
        logger.error(f"[DECODE PDF] Error decoding PDF: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    logger.info(f"[CHUNK TEXT] Total words: {len(words)} | Chunks created: {len(chunks)}")
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
        logger.error("[INDEX PDF] Failed to fetch Base64 PDF — indexing aborted")
        return False

    text = decode_base64_to_text(base64_string)
    if not text:
        logger.error("[INDEX PDF] Failed to decode PDF text — indexing aborted")
        return False

    chunks = chunk_text(text)

    if not chunks:
        logger.error("[INDEX PDF] No chunks created — text may be empty")
        return False

    collection_name = f"policy_wording_{policy_no}"
    logger.info(f"[INDEX PDF] Deleting old ChromaDB collection: {collection_name}")
    delete_collection(collection_name)

    collection = get_or_create_collection(collection_name)
    logger.info(f"[INDEX PDF] Created new ChromaDB collection: {collection_name}")
    logger.info(f"[INDEX PDF] Embedding {len(chunks)} chunks into ChromaDB...")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )
        if (i + 1) % 10 == 0:
            logger.info(f"[INDEX PDF] Progress: {i + 1}/{len(chunks)} chunks embedded")

    indexed_tokens[policy_no] = access_token
    logger.info(f"[INDEX PDF] Indexing complete ✅ — {len(chunks)} chunks stored")
    return True

def should_reindex(policy_no: str, access_token: str) -> bool:
    """Check if PDF needs re-indexing"""
    collection_name = f"policy_wording_{policy_no}"

    if not collection_exists(collection_name):
        logger.info(f"[REINDEX CHECK] Collection '{collection_name}' not found — reindex needed")
        return True

    cached_token = indexed_tokens.get(policy_no)
    if cached_token != access_token:
        logger.info(f"[REINDEX CHECK] Token changed for {policy_no} — reindex needed")
        logger.info(f"[REINDEX CHECK] Cached token: {cached_token}")
        logger.info(f"[REINDEX CHECK] Current token: {access_token}")
        return True

    logger.info(f"[REINDEX CHECK] Collection exists and token unchanged — using cache ✅")
    return False

def search_pdf(policy_no: str, question: str, top_k: int = 3) -> Optional[str]:
    """Search ChromaDB for relevant chunks"""
    collection_name = f"policy_wording_{policy_no}"
    logger.info(f"[SEARCH PDF] Searching ChromaDB collection: {collection_name}")
    logger.info(f"[SEARCH PDF] Question: {question}")
    logger.info(f"[SEARCH PDF] Retrieving top {top_k} chunks")

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
            logger.info(f"[SEARCH PDF] Chunk {i + 1} — similarity distance: {dist:.4f}")
            logger.info(f"[SEARCH PDF] Chunk {i + 1} preview: {chunk[:150]}...")
        return "\n\n".join(chunks)

    logger.warning("[SEARCH PDF] No relevant chunks found in ChromaDB")
    return None

def answer_from_pdf(question: str, policy_no: Optional[str] = None) -> str:
    """
    Main function called by agent

    Two scenarios:
    1. policy_no is None → get latest wording from DB
    2. policy_no provided → get wording for that specific policy
    """
    logger.info(f"[PDF TOOL] answer_from_pdf called")
    logger.info(f"[PDF TOOL] Question: {question}")
    logger.info(f"[PDF TOOL] Policy no: {policy_no if policy_no else 'Not provided — will use latest'}")

    if policy_no is None:
        logger.info("[PDF TOOL] Fetching LATEST policy wording credentials from DB")
        credentials = get_latest_policy_wording_credentials()
        if not credentials:
            logger.error("[PDF TOOL] No active policy wording found in DB")
            return "Sorry, I could not find any active policy wording in the system."
        policy_no = credentials["policy_no"]
        access_token = credentials["access_token"]
        logger.info(f"[PDF TOOL] Latest policy from DB: {policy_no}")
    else:
        logger.info(f"[PDF TOOL] Fetching credentials for specific policy: {policy_no}")
        credentials = get_policy_credentials_by_no(policy_no)
        if not credentials:
            logger.error(f"[PDF TOOL] Policy {policy_no} not found in DB view")
            return f"Sorry, I could not find policy {policy_no} in the system. Please check the policy number and try again."
        access_token = credentials["access_token"]
        logger.info(f"[PDF TOOL] Credentials found for policy: {policy_no}")

    logger.info(f"[PDF TOOL] Access token (first 8 chars): {access_token[:8]}...")

    # Re-index if needed
    if should_reindex(policy_no, access_token):
        logger.info(f"[PDF TOOL] Re-indexing PDF for policy: {policy_no}")
        success = index_pdf(policy_no, access_token)
        if not success:
            logger.error(f"[PDF TOOL] Failed to index PDF for policy: {policy_no}")
            return "Sorry, I could not retrieve the policy wording document."
    else:
        logger.info(f"[PDF TOOL] Using cached ChromaDB index for policy: {policy_no}")

    # Search for relevant chunks
    relevant_chunks = search_pdf(policy_no, question)
    if not relevant_chunks:
        logger.warning(f"[PDF TOOL] No relevant chunks found for question: {question}")
        return "Sorry, I could not find relevant information in the policy wording."

    logger.info("[PDF TOOL] Sending relevant chunks to GPT for answer generation")

    # Generate answer
    try:
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
                    {relevant_chunks}

                    Question: {question}
                    """
                }
            ]
        )

        answer = response.choices[0].message.content
        logger.info(f"[PDF TOOL] GPT answer generated: {answer[:200]}...")
        return answer

    except Exception as e:
        logger.error(f"[PDF TOOL] GPT error: {e}")
        return "Sorry, I encountered an error generating the answer."