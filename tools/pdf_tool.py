import requests
import base64
import PyPDF2
import io
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

client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=20.0
)

# Cache to track indexed tokens
# Key: policy_no, Value: last indexed token
indexed_tokens: dict = {}

def fetch_pdf_base64(policy_no: str, access_token: str) -> Optional[str]:
    """Call API and get Base64 encoded PDF"""
    try:
        url = f"{POLICY_DOCUMENT_API_URL}?policy={policy_no}&token={access_token}&template=wording"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("document")
    except Exception as e:
        print(f"Error fetching PDF: {e}")
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
        print(f"Error decoding PDF: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
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
    print(f"Indexing PDF for policy {policy_no}...")

    base64_string = fetch_pdf_base64(policy_no, access_token)
    if not base64_string:
        return False

    text = decode_base64_to_text(base64_string)
    if not text:
        return False

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    collection_name = f"policy_wording_{policy_no}"
    delete_collection(collection_name)
    collection = get_or_create_collection(collection_name)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )

    indexed_tokens[policy_no] = access_token
    print(f"PDF indexed successfully ✅ ({len(chunks)} chunks)")
    return True

def should_reindex(policy_no: str, access_token: str) -> bool:
    """Check if PDF needs re-indexing"""
    collection_name = f"policy_wording_{policy_no}"

    if not collection_exists(collection_name):
        return True

    if indexed_tokens.get(policy_no) != access_token:
        return True

    return False

def search_pdf(policy_no: str, question: str, top_k: int = 3) -> Optional[str]:
    """Search ChromaDB for relevant chunks"""
    collection_name = f"policy_wording_{policy_no}"
    collection = get_or_create_collection(collection_name)

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    if results and results["documents"]:
        return "\n\n".join(results["documents"][0])

    return None

def answer_from_pdf(question: str, policy_no: Optional[str] = None) -> str:
    """
    Main function called by agent

    Two scenarios:
    1. policy_no is None
       → No policy number in question
       → Get LATEST policy wording from DB
       → Use that policy_no and token

    2. policy_no is provided
       → User gave specific policy number
       → Get credentials for THAT policy from DB
       → Use that specific policy wording
    """

    if policy_no is None:
        # Scenario 1 — no policy number — get latest
        print("No policy number provided — fetching latest wording")
        credentials = get_latest_policy_wording_credentials()
        if not credentials:
            return "Sorry, I could not find any active policy wording in the system."
        policy_no = credentials["policy_no"]
        access_token = credentials["access_token"]
    else:
        # Scenario 2 — specific policy number given
        print(f"Policy number provided — fetching wording for {policy_no}")
        credentials = get_policy_credentials_by_no(policy_no)
        if not credentials:
            return f"Sorry, I could not find policy {policy_no} in the system. Please check the policy number and try again."
        access_token = credentials["access_token"]

    # Re-index if needed
    if should_reindex(policy_no, access_token):
        success = index_pdf(policy_no, access_token)
        if not success:
            return "Sorry, I could not retrieve the policy wording document."

    # Search for relevant chunks
    relevant_chunks = search_pdf(policy_no, question)
    if not relevant_chunks:
        return "Sorry, I could not find relevant information in the policy wording."

    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an insurance assistant for brokers.
                Answer using ONLY the provided policy wording context.

                Format rules:
                - If the answer has more than one distinct point, use short bullet points (start each with "- ")
                - If it's a single fact (one date, one amount, one yes/no), answer in one short sentence — no bullets needed
                - No preamble, no "based on the document", no repeating the question
                - Each bullet should be one short phrase or sentence, not a paragraph
                - If the context doesn't contain the answer, say so in one line — do not guess"""
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

    return response.choices[0].message.content