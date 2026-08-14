import requests
import base64
import PyPDF2
import io
import os
import re
import logging
from typing import Optional, List, Any, Dict, Tuple
from openai import OpenAI
from vector_store.chroma import (
    get_or_create_collection,
    delete_collection,
    collection_exists,
    get_collection_metadata
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

# ── Schema Version ───────────────────────────────────────────────────
# Bump this whenever chunking logic, metadata schema, or embedding model
# changes. Forces automatic re-indexing of all cached collections.
CHUNK_SCHEMA_VERSION = "v3"

# ── Retrieval Tuning Constants ───────────────────────────────────────
DEFAULT_TOP_K = 5               # Final chunks returned to GPT
CANDIDATE_MULTIPLIER = 3        # Retrieve top_k * this many candidates
RELEVANCE_THRESHOLD = 0.75      # Cosine distance cutoff (lower = better match)
KEYWORD_BOOST_WEIGHT = 0.15     # Distance reduction for keyword matches
CATEGORY_BOOST_WEIGHT = 0.05    # Extra distance reduction for category matches

# Shared literal for exclusion detection (S1192)
_KW_NOT_PAY = "not pay"

# ── Section Detection Patterns ───────────────────────────────────────
# These patterns identify structural boundaries in ERGO policy wordings.
# Ordered by specificity — more specific patterns first.

SECTION_HEADER_PATTERNS = [
    # "Section 1 – Definitions", "Section 4 – General Exclusions"
    re.compile(r"^(Section\s+\d+\s*[–—-]\s*\S.*)$", re.IGNORECASE | re.MULTILINE),
    # "Part 1 – Personal Accident", "Part 17 – Trip Cancellation"
    # Also matches "Part 3 7" (broken text) via optional space
    re.compile(r"^(Part\s+\d+(?:\s+\d+)?\s*[–—-]\s*\S.*)$", re.IGNORECASE | re.MULTILINE),
    # Roman numeral sub-sections: "I. Limits of coverage", "II. Policy extension"
    re.compile(r"^((?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+\S.*)$", re.MULTILINE),
]

# Pattern for numbered definitions/clauses: "1)", "2)", "32)", etc.
NUMBERED_CLAUSE_PATTERN = re.compile(r"^\d{1,3}\)\s+", re.MULTILINE)

# ── Stopwords for keyword extraction ─────────────────────────────────
# IMPORTANT: Domain terms like "cover", "policy", "insurance" are NOT
# included here — they are meaningful search signals in insurance context.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "his", "her",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "or", "and", "but", "not", "no", "nor", "so", "too",
    "very", "just", "about", "up", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "than", "also",
    "of", "in", "to", "for", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "between", "any",
    "tell", "explain", "describe", "know", "mean",
    "please", "thanks", "thank", "hi", "hello",
})

# ── Category Detection Rules ─────────────────────────────────────────
# Dynamically detect category from section header text.
# Each rule: (keywords_to_match_in_header, category_name)
# Order matters — first match wins.
CATEGORY_RULES = [
    # Definitions
    (["definition"], "definitions"),
    # Conditions & eligibility
    (["important condition"], "eligibility_conditions"),
    # Scope and limits
    (["scope", "limits of cover"], "scope_and_limits"),
    # Exclusions
    (["general exclusion", "exclusion"], "exclusions"),
    # Policy admin
    (["special condition", "cancellation", "refund"], "policy_admin"),
    (["general condition", "claims provision", "payment of claims",
      "arbitration", "subrogation", "governing law", "data protection",
      "modification", "currency", "interest"], "policy_admin"),
    # Medical coverage
    (["medical expense", "sickness", "injury", "pregnancy",
      "physician", "tcm", "chiropractor", "hospital allowance",
      "evacuation", "repatriation", "mortal remains", "funeral",
      "dental treatment", "automatic extension"], "medical_coverage"),
    # Assistance & support
    (["compassionate visit", "hospital visitation", "child transfer",
      "telephone", "internet expense"], "assistance_and_support"),
    # Accidental death & disability
    (["accidental death", "permanent disablement", "common carrier",
      "double cover", "child education"], "accidental_death_and_disability"),
    # Trip disruption
    (["trip cancellation", "trip postponement", "insolvency",
      "trip curtailment", "flight diversion", "flight overbooking",
      "trip delay", "trip misconnection", "cruise"], "trip_disruption"),
    # Baggage & documents
    (["baggage", "passport", "travel document", "money"], "baggage_and_documents"),
    # Financial & liability
    (["credit card", "personal liability"], "financial_and_liability"),
    # Security incidents
    (["kidnap", "hostage", "hijack", "terrorism"], "security_incidents"),
    # Lifestyle & activities
    (["sports equipment", "home protection", "rental vehicle",
      "pet care", "adventure activity"], "lifestyle_and_activities"),
    # COVID-19 special extension
    (["covid"], "special_extensions"),
    # Benefits summary
    (["summary of benefits", "benefit summary"], "benefits_summary"),
]

# ── Category-to-question mapping for category boosting ───────────────
# When user question contains these terms, boost chunks from the category.
CATEGORY_BOOST_MAP = {
    "definitions": ["definition", "meaning", "defined", "means"],
    "exclusions": ["exclusion", "excluded", "not covered", "not cover", _KW_NOT_PAY],
    "medical_coverage": ["medical", "hospital", "doctor", "sickness", "injury",
                         "dental", "evacuation", "repatriation"],
    "trip_disruption": ["cancellation", "cancel", "delay", "postpone", "curtail",
                        "diversion", "overbooking", "misconnection"],
    "security_incidents": ["terrorism", "terrorist", "kidnap", "hostage", "hijack"],
    "special_extensions": ["covid", "covid-19", "coronavirus", "pandemic"],
    "baggage_and_documents": ["baggage", "luggage", "passport", "document"],
    "accidental_death_and_disability": ["death", "disablement", "disability", "accident"],
    "financial_and_liability": ["liability", "credit card"],
    "benefits_summary": ["summary", "benefit", "benefits", "table", "limit"],
    "eligibility_conditions": ["eligibility", "eligible", "condition", "requirement"],
    "scope_and_limits": ["scope", "limit", "period", "extension", "duration"],
    "policy_admin": ["cancellation policy", "refund", "premium"],
    "lifestyle_and_activities": ["adventure", "sports", "pet", "rental", "home"],
    "assistance_and_support": ["visit", "compassionate", "telephone", "internet"],
}


# ══════════════════════════════════════════════════════════════════════
#                    PDF TEXT CLEANING (Step 1)
# ══════════════════════════════════════════════════════════════════════



# Page footer/header patterns to strip
_PAGE_NOISE_PATTERNS = [
    # "Page | 1", "Page | 12"
    re.compile(r'^Page\s*\|\s*\d+\s*$', re.MULTILINE),
    # "V7 Mar 2026"
    re.compile(r'^V\d+\s+\w+\s+\d{4}\s*$', re.MULTILINE),
    # "Version No. ETP – 007    ©ERGO Insurance Pte. Ltd."
    re.compile(r'Version\s*No\.\s*ETP\s*[–—-]\s*\d+\s*©?ERGO\s+Insurance\s+Pte\.?\s+Ltd\.?', re.IGNORECASE),
    # Standalone page numbers
    re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE),
    # "17\n" style page numbers at start of line followed by newline
    re.compile(r'^\d{1,3}\s*\n(?=[A-Z])', re.MULTILINE),
]


def clean_pdf_text(text: str) -> str:
    """
    Clean text extracted from PDF by PyPDF2.
    Fixes broken words, removes page noise, and normalizes whitespace.

    This is the MOST CRITICAL fix — without it, terms like 'COVID-19'
    may be stored as 'COVID -19' and never match keyword searches.
    """
    if not text:
        return text

    original_len = len(text)

    # 1. Fix specific known broken terms first
    text = re.sub(r'COVID\s*[-–]\s*19', 'COVID-19', text, flags=re.IGNORECASE)
    text = re.sub(r'pre\s*[-–]\s*existing', 'pre-existing', text, flags=re.IGNORECASE)

    # 2. Fix general word breaks: "sightseei ng" → "sightseeing"
    # Pattern: lowercase letters + space + 1-3 lowercase letters at word boundary
    # This catches PyPDF2's character-spacing splits
    def _fix_word_break(match):
        return match.group(1) + match.group(2)

    # Iteratively fix word breaks (some words may have multiple breaks)
    for _ in range(3):
        new_text = re.sub(
            r'\b([a-z]{2,})\s([a-z]{1,3})\b',
            _fix_word_break,
            text
        )
        if new_text == text:
            break
        text = new_text

    # 3. Fix breaks with capital continuation: "Polic y" → "Policy"
    text = re.sub(
        r'\b([A-Z][a-z]+)\s([a-z]{1,3})\b',
        _fix_word_break,
        text
    )

    # 4. Remove page noise (footers, headers, version strings)
    for pattern in _PAGE_NOISE_PATTERNS:
        text = pattern.sub('', text)

    # 5. Normalize whitespace
    # Collapse multiple blank lines into single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces into single space (but preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Strip trailing whitespace on each line without regex (S8786)
    text = '\n'.join(line.rstrip(' \t') for line in text.split('\n'))

    logger.info(
        "[CLEAN TEXT] Cleaned %s → %s chars (removed %s chars of noise)",
        original_len, len(text), original_len - len(text)
    )

    return text


# ══════════════════════════════════════════════════════════════════════
#                       PDF FETCH & DECODE
# ══════════════════════════════════════════════════════════════════════

def fetch_pdf_base64(policy_no: str, access_token: str) -> Optional[str]:
    """Call API and get Base64 encoded PDF"""
    url = f"{POLICY_DOCUMENT_API_URL}?policy={policy_no}&token={access_token}&template=wording"
    logger.info("[FETCH PDF] Calling wording API for policy: %s", policy_no)

    try:
        response = requests.get(url, timeout=30)
        logger.info("[FETCH PDF] API response status: %s", response.status_code)
        response.raise_for_status()
        data = response.json()
        doc = data.get("document")

        if doc:
            logger.info("[FETCH PDF] Base64 document received - length: %s chars", len(doc))
        else:
            logger.warning("[FETCH PDF] No 'document' field in API response")
            logger.warning("[FETCH PDF] Response keys: %s", list(data.keys()))

        return doc

    except requests.exceptions.Timeout:
        logger.error("[FETCH PDF] API request timed out after 30 seconds")
        return None
    except requests.exceptions.HTTPError:
        logger.exception("[FETCH PDF] HTTP error")
        return None
    except Exception:  # noqa: BLE001
        logger.exception("[FETCH PDF] Unexpected error")
        return None

def decode_base64_to_text(base64_string: str) -> Optional[str]:
    """Decode Base64 string to PDF text, then clean it."""
    logger.info("[DECODE PDF] Decoding Base64 to PDF text")

    try:
        pdf_bytes = base64.b64decode(base64_string)
        logger.info("[DECODE PDF] PDF bytes size: %s bytes", len(pdf_bytes))

        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        logger.info("[DECODE PDF] Total pages: %s", total_pages)

        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            full_text += page_text + "\n"
            logger.info("[DECODE PDF] Page %s/%s - %s chars", i + 1, total_pages, len(page_text))

        logger.info("[DECODE PDF] Total raw text: %s chars", len(full_text))

        if len(full_text.strip()) == 0:
            logger.warning("[DECODE PDF] Extracted text is empty")
            return full_text

        # Apply text cleaning to fix broken words and remove noise
        full_text = clean_pdf_text(full_text)

        return full_text

    except (base64.binascii.Error, PyPDF2.errors.PdfReadError) as exc:
        logger.exception("[DECODE PDF] PDF decode/parse error: %s", exc)
        return None
    except Exception:  # noqa: BLE001
        logger.exception("[DECODE PDF] Unexpected error")
        return None


# ══════════════════════════════════════════════════════════════════════
#            CATEGORY-AWARE CHUNKING (Step 2)
# ══════════════════════════════════════════════════════════════════════

def _detect_section_name(line: str) -> Optional[str]:
    """
    Check if a line is a section/part header.
    Returns the header text if matched, None otherwise.
    """
    stripped = line.strip()
    if not stripped:
        return None

    for pattern in SECTION_HEADER_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()

    return None


def _detect_category(section_name: str) -> str:
    """
    Dynamically detect category from section header text.
    Uses keyword matching against CATEGORY_RULES — not hardcoded Part numbers.
    Falls back to 'general' if no rule matches.
    """
    if not section_name:
        return "general"

    header_lower = section_name.lower()

    for keywords, category in CATEGORY_RULES:
        if any(kw in header_lower for kw in keywords):
            return category

    return "general"


def _extract_cap_phrases(text: str, keywords: set) -> None:
    cap_phrases = re.findall(
        r'\b[A-Z][a-z]+(?:\s+(?:of|and|or|the|for|in|to|by|on|with|due)\s+)?'
        r'(?:[A-Z][a-z]+(?:\s+(?:of|and|or|the|for|in|to|by|on|with|due)\s+)?)*',
        text
    )
    for phrase in cap_phrases:
        phrase = phrase.strip()
        if len(phrase) > 2 and phrase.lower() not in STOPWORDS:
            keywords.add(phrase)


def _extract_upper_terms(text: str, keywords: set) -> None:
    upper_terms = re.findall(r'\b[A-Z][A-Z0-9-]+\b', text)
    for term in upper_terms:
        if len(term) >= 2:
            keywords.add(term)


def _extract_hyphenated(text: str, keywords: set) -> None:
    hyphenated = re.findall(r'\b[A-Za-z]+-[A-Za-z0-9]+\b', text)
    for term in hyphenated:
        if len(term) > 3:
            keywords.add(term)


def _extract_quoted(text: str, keywords: set) -> None:
    quoted = re.findall(r'"([^"]+)"', text)
    for term in quoted:
        term = term.strip()
        if len(term) > 1 and len(term) < 60:
            keywords.add(term)


def _extract_domain_terms(text: str, keywords: set) -> None:
    content_lower = text.lower()
    domain_terms = [
        "COVID-19", "coronavirus", "pandemic", "epidemic",
        "terrorism", "terrorist", "kidnap", "hostage",
        "baggage", "passport", "cancellation", "curtailment",
        "evacuation", "repatriation", "accidental death",
        "personal liability", "trip delay", "flight delay",
    ]
    for term in domain_terms:
        if term.lower() in content_lower:
            keywords.add(term)


def _extract_keywords_from_chunk(text: str) -> List[str]:
    """
    Extract meaningful keywords from a chunk for metadata tagging.
    Focuses on domain-specific terms, capitalized phrases, and
    terms in ALL CAPS or with special formatting.
    """
    keywords = set()

    _extract_cap_phrases(text, keywords)
    _extract_upper_terms(text, keywords)
    _extract_hyphenated(text, keywords)
    _extract_quoted(text, keywords)
    _extract_domain_terms(text, keywords)

    # Limit to avoid excessively large metadata
    return sorted(keywords)[:40]


def _classify_chunk_type_by_section(section_lower: str) -> str:
    if "definition" in section_lower:
        return "definition"
    if "exclusion" in section_lower:
        return "exclusion"
    if any(cond in section_lower for cond in ["general condition", "special condition", "important condition"]):
        return "condition"
    if "scope and limits" in section_lower:
        return "scope"
    if "claim" in section_lower:
        return "claims"
    if "benefit" in section_lower or "summary of benefit" in section_lower:
        return "benefit"
    return ""


def _classify_chunk_type_by_part(content_lower: str) -> str:
    if any(kw in content_lower for kw in ["exclusion", "not cover", _KW_NOT_PAY, "shall not"]):
        return "exclusion"
    if any(kw in content_lower for kw in ["benefit", "cover", "pay", "reimburse"]):
        return "benefit"
    return "coverage"


def _classify_chunk_type_by_content(content_lower: str) -> str:
    if any(kw in content_lower for kw in ["will not cover", _KW_NOT_PAY, "exclude", "exclusion"]):
        return "exclusion"
    if any(kw in content_lower for kw in ["means ", "refers to", "is defined as"]):
        return "definition"
    return "general"


def _classify_chunk_type(section_name: str, content: str) -> str:
    """
    Classify a chunk's type based on its section and content.
    Used for metadata tagging.
    """
    section_lower = section_name.lower() if section_name else ""
    content_lower = content.lower()[:300]  # Check start of content

    sec_type = _classify_chunk_type_by_section(section_lower)
    if sec_type:
        return sec_type

    # Check content patterns for Parts (Part 1 - Personal Accident, etc.)
    if "part" in section_lower:
        return _classify_chunk_type_by_part(content_lower)

    # Fallback content-based detection
    return _classify_chunk_type_by_content(content_lower)


def _split_at_sentence_boundary(text: str, max_words: int) -> List[str]:
    """
    Split a long text block into sub-chunks at sentence boundaries,
    keeping each sub-chunk under max_words.
    """
    sentences = re.split(r'(?<=[.;])\s+', text)
    sub_chunks = []
    current = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > max_words and current:
            sub_chunks.append(" ".join(current))
            current = [sentence]
            current_word_count = sentence_words
        else:
            current.append(sentence)
            current_word_count += sentence_words

    if current:
        sub_chunks.append(" ".join(current))

    return sub_chunks


def _flush_block(current_section: str, current_block_lines: List[str], raw_segments: List[Tuple[str, str]]) -> None:
    if current_block_lines:
        block_text = "\n".join(current_block_lines).strip()
        if block_text:
            raw_segments.append((current_section, block_text))
        current_block_lines.clear()


def _parse_raw_segments(lines: List[str]) -> List[Tuple[str, str]]:
    raw_segments = []
    current_section = "Preamble"
    current_block_lines = []

    for line in lines:
        detected_section = _detect_section_name(line)
        if detected_section:
            _flush_block(current_section, current_block_lines, raw_segments)
            current_section = detected_section
            continue

        if NUMBERED_CLAUSE_PATTERN.match(line.strip()):
            _flush_block(current_section, current_block_lines, raw_segments)

        current_block_lines.append(line)

    _flush_block(current_section, current_block_lines, raw_segments)
    return raw_segments


def _merge_small_segments(raw_segments: List[Tuple[str, str]], min_chunk_words: int) -> List[Tuple[str, str]]:
    merged_segments = []
    i = 0
    while i < len(raw_segments):
        section, text_block = raw_segments[i]
        word_count = len(text_block.split())

        while (word_count < min_chunk_words
               and i + 1 < len(raw_segments)
               and raw_segments[i + 1][0] == section):
            i += 1
            next_text = raw_segments[i][1]
            text_block = text_block + "\n\n" + next_text
            word_count = len(text_block.split())

        merged_segments.append((section, text_block))
        i += 1
    return merged_segments


def _split_oversized_segments(merged_segments: List[Tuple[str, str]], max_chunk_words: int) -> List[Tuple[str, str]]:
    final_chunks = []
    for section, text_block in merged_segments:
        word_count = len(text_block.split())

        if word_count > max_chunk_words:
            sub_chunks = _split_at_sentence_boundary(text_block, max_chunk_words)
            for sub in sub_chunks:
                if sub.strip():
                    final_chunks.append((section, sub.strip()))
        else:
            if text_block.strip():
                final_chunks.append((section, text_block.strip()))
    return final_chunks


def chunk_text_with_sections(
    text: str,
    max_chunk_words: int = 600,
    min_chunk_words: int = 80
) -> List[Dict[str, Any]]:
    """
    Section-aware chunking with category tagging.

    Strategy:
    1. Split text into lines
    2. Detect section headers to track current section
    3. Split on numbered clauses within sections (definitions, exclusions)
    4. Merge small adjacent chunks from the same section
    5. Split oversized chunks at sentence boundaries
    6. Tag each chunk with section name, category, type, and keywords

    Returns list of dicts:
    [
        {
            "content": "...",
            "section": "Section 1 – Definitions",
            "category": "definitions",
            "chunk_type": "definition",
            "keywords": ["COVID-19", "WHO", "Coronavirus"]
        },
        ...
    ]
    """
    lines = text.split("\n")
    raw_segments = _parse_raw_segments(lines)
    logger.info("[CHUNK] Raw segments detected: %s", len(raw_segments))

    merged_segments = _merge_small_segments(raw_segments, min_chunk_words)
    logger.info("[CHUNK] After merging small segments: %s", len(merged_segments))

    final_chunks = _split_oversized_segments(merged_segments, max_chunk_words)
    logger.info("[CHUNK] Final chunks after splitting oversized: %s", len(final_chunks))

    # ── Build chunk dicts with metadata ──────────────────────────────
    result = []
    for section, content in final_chunks:
        chunk_type = _classify_chunk_type(section, content)
        category = _detect_category(section)
        keywords = _extract_keywords_from_chunk(content)

        result.append({
            "content": content,
            "section": section,
            "category": category,
            "chunk_type": chunk_type,
            "keywords": keywords
        })

    logger.info(
        "[CHUNK] Total chunks: %s | Types: %s | Categories: %s",
        len(result), dict(_count_types(result)), dict(_count_categories(result))
    )
    return result


def _count_types(chunks: List[Dict]) -> List[tuple]:
    """Count chunk types for logging."""
    counts: Dict[str, int] = {}
    for c in chunks:
        t = c.get("chunk_type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items())


def _count_categories(chunks: List[Dict]) -> List[tuple]:
    """Count chunk categories for logging."""
    counts: Dict[str, int] = {}
    for c in chunks:
        cat = c.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return sorted(counts.items())


# Keep old function signature as an alias for backward compatibility
def chunk_text(text: str, _chunk_size: int = 500, _overlap: int = 50) -> list:
    """
    DEPRECATED: Use chunk_text_with_sections() instead.
    Kept for backward compatibility — now delegates to the new function
    and returns just the content strings.
    """
    logger.warning("[CHUNK] chunk_text() is deprecated — using section-aware chunking")
    chunks = chunk_text_with_sections(text)
    return [c["content"] for c in chunks]


# ---- Embeddings ----

def get_embedding(text: str) -> list:
    """Get embedding from OpenAI"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# ══════════════════════════════════════════════════════════════════════
#                   INDEXING WITH METADATA
# ══════════════════════════════════════════════════════════════════════

def index_pdf(policy_no: str, access_token: str) -> bool:
    """
    Fetch PDF, extract text, clean it, chunk with section/category awareness,
    embed and store in ChromaDB with metadata.
    """
    logger.info("[INDEX PDF] Starting indexing for policy: %s", policy_no)

    base64_string = fetch_pdf_base64(policy_no, access_token)
    if not base64_string:
        logger.error("[INDEX PDF] Failed to fetch Base64 PDF")
        return False

    text = decode_base64_to_text(base64_string)
    if not text:
        logger.error("[INDEX PDF] Failed to decode PDF text")
        return False

    # Use section-aware chunking (text already cleaned in decode step)
    chunks = chunk_text_with_sections(text)
    if not chunks:
        logger.error("[INDEX PDF] No chunks created")
        return False

    collection_name = f"policy_wording_{policy_no}"
    logger.info("[INDEX PDF] Deleting old collection: %s", collection_name)
    delete_collection(collection_name)

    collection = get_or_create_collection(
        collection_name,
        metadata={"schema_version": CHUNK_SCHEMA_VERSION}
    )
    logger.info("[INDEX PDF] Embedding %s chunks (schema %s)...", len(chunks), CHUNK_SCHEMA_VERSION)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["content"])

        # Store rich metadata alongside the document
        chunk_metadata = {
            "section": chunk["section"],
            "category": chunk["category"],
            "chunk_type": chunk["chunk_type"],
            "keywords": ",".join(chunk["keywords"]),
            "schema_version": CHUNK_SCHEMA_VERSION,
        }

        collection.add(
            documents=[chunk["content"]],
            embeddings=[embedding],
            metadatas=[chunk_metadata],
            ids=[f"chunk_{i}"]
        )
        if (i + 1) % 10 == 0:
            logger.info("[INDEX PDF] Progress: %s/%s", i + 1, len(chunks))

    indexed_tokens[policy_no] = access_token
    logger.info("[INDEX PDF] Indexing complete - %s chunks stored", len(chunks))
    return True


def should_reindex(policy_no: str, access_token: str) -> bool:
    """
    Check if PDF needs re-indexing.
    Triggers reindex if:
    - Collection doesn't exist
    - Access token changed
    - Schema version is outdated (chunking logic changed)
    """
    collection_name = f"policy_wording_{policy_no}"

    if not collection_exists(collection_name):
        logger.info("[REINDEX CHECK] Collection not found - reindex needed")
        return True

    cached_token = indexed_tokens.get(policy_no)
    if cached_token != access_token:
        logger.info("[REINDEX CHECK] Token changed - reindex needed")
        return True

    # Check schema version — force reindex if chunking logic changed
    stored_meta = get_collection_metadata(collection_name)
    stored_version = stored_meta.get("schema_version") if stored_meta else None
    if stored_version != CHUNK_SCHEMA_VERSION:
        logger.info(
            "[REINDEX CHECK] Schema version mismatch "
            "(stored=%s, current=%s) - reindex needed",
            stored_version, CHUNK_SCHEMA_VERSION
        )
        return True

    logger.info("[REINDEX CHECK] Using existing cache - no reindex needed")
    return False


# ══════════════════════════════════════════════════════════════════════
#              KEYWORD EXTRACTION FOR SEARCH (Step 3)
# ══════════════════════════════════════════════════════════════════════

def _process_search_token(token_lower: str, terms: List[str]) -> None:
    if token_lower in STOPWORDS or len(token_lower) <= 1:
        return

    if token_lower.count('-') > 1:
        parts = token_lower.rsplit('-', 1)
        if len(parts) == 2:
            terms.append(parts[0])
            if parts[1] not in STOPWORDS and len(parts[1]) > 1:
                terms.append(parts[1])
    else:
        terms.append(token_lower)


def _extract_search_terms(question: str) -> List[str]:
    """
    Extract meaningful search terms from a user question.
    Used for the keyword leg of hybrid search.

    Handles:
    - Hyphenated compound terms: "COVID-19-related" → ["covid-19", "related"]
    - Preserves domain terms (cover, policy, insurance — NOT stopwords)
    - Returns lowercased terms for case-insensitive matching.
    """
    terms = []
    raw_tokens = re.findall(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', question)

    for token in raw_tokens:
        _process_search_token(token.lower(), terms)

    return terms


# ══════════════════════════════════════════════════════════════════════
#     HYBRID SEARCH with Category Boosting (Steps 3, 4)
# ══════════════════════════════════════════════════════════════════════

def _keyword_match_score(keywords_csv: str, search_terms: List[str]) -> float:
    """
    Calculate how well a chunk's stored keywords match the search terms.
    Uses normalized matching — strips hyphens and lowercases both sides.
    Returns a score between 0.0 (no match) and 1.0 (perfect match).
    """
    if not keywords_csv or not search_terms:
        return 0.0

    # Normalize: lowercase and also create a hyphen-stripped version
    stored_lower = keywords_csv.lower()
    stored_normalized = stored_lower.replace("-", "")

    matches = 0
    for term in search_terms:
        term_normalized = term.replace("-", "")
        # Check both original and normalized
        if term in stored_lower or term_normalized in stored_normalized:
            matches += 1

    return matches / len(search_terms)


def _document_text_match_score(document: str, search_terms: List[str]) -> float:
    """
    Calculate how well the document text contains the search terms.
    Fallback for when keywords metadata doesn't capture everything.
    Uses normalized matching.
    Returns a score between 0.0 and 1.0.
    """
    if not document or not search_terms:
        return 0.0

    doc_lower = document.lower()
    doc_normalized = doc_lower.replace("-", "")

    matches = 0
    for term in search_terms:
        term_normalized = term.replace("-", "")
        if term in doc_lower or term_normalized in doc_normalized:
            matches += 1

    return matches / len(search_terms)


def _get_category_boost(category: str, search_terms: List[str]) -> float:
    """
    Check if the chunk's category matches the question's intent.
    Returns a boost score (0.0 or 1.0).

    Example: question "Does this cover COVID-19?" contains "covid-19"
    → matches special_extensions category → boost = 1.0
    """
    boost_terms = CATEGORY_BOOST_MAP.get(category)
    if not boost_terms:
        return 0.0

    for term in search_terms:
        if any(bt in term for bt in boost_terms):
            return 1.0

    return 0.0


def _score_chunk(doc: str, dist: float, meta: Dict[str, Any], search_terms: List[str]) -> Dict[str, Any]:
    keywords_csv = meta.get("keywords", "") if meta else ""
    kw_score = _keyword_match_score(keywords_csv, search_terms)
    text_score = _document_text_match_score(doc, search_terms)
    best_kw_score = max(kw_score, text_score)

    category = meta.get("category", "general") if meta else "general"
    cat_boost = _get_category_boost(category, search_terms)

    hybrid_distance = dist - (best_kw_score * KEYWORD_BOOST_WEIGHT) - (cat_boost * CATEGORY_BOOST_WEIGHT)

    return {
        "content": doc,
        "original_distance": round(float(dist), 4),
        "keyword_score": round(best_kw_score, 4),
        "category_boost": round(cat_boost, 4),
        "hybrid_distance": round(float(hybrid_distance), 4),
        "section": meta.get("section", "Unknown") if meta else "Unknown",
        "category": category,
        "chunk_type": meta.get("chunk_type", "general") if meta else "general",
    }


def search_pdf(
    policy_no: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    return_metadata: bool = False
) -> Optional[Any]:
    """
    Hybrid search: embedding similarity + keyword matching + category boost + relevance gate.

    Steps:
    1. Retrieve top_k * CANDIDATE_MULTIPLIER candidates by embedding similarity
    2. Extract search terms from the question
    3. Compute a hybrid score: embedding distance boosted by keyword matches + category match
    4. Apply relevance threshold — discard garbage chunks
    5. Return the best top_k results
    """
    collection_name = f"policy_wording_{policy_no}"
    logger.info("[SEARCH PDF] Searching: %s", collection_name)
    logger.info("[SEARCH PDF] Question: %s", question)

    collection = get_or_create_collection(collection_name)
    question_embedding = get_embedding(question)

    # Step 1: Retrieve extra candidates for re-ranking
    n_candidates = top_k * CANDIDATE_MULTIPLIER
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_candidates,
        include=["documents", "distances", "metadatas"]
    )

    if not results or not results["documents"] or not results["documents"][0]:
        logger.warning("[SEARCH PDF] No chunks found in collection")
        return None

    documents = results["documents"][0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Step 2: Extract search terms for keyword matching
    search_terms = _extract_search_terms(question)
    logger.info("[SEARCH PDF] Search terms for keyword matching: %s", search_terms)

    # Step 3: Compute hybrid scores and re-rank
    scored_chunks = [
        _score_chunk(doc, float(dist), meta, search_terms)
        for doc, dist, meta in zip(documents, distances, metadatas)
    ]

    # Sort by hybrid distance (ascending — lower is better)
    scored_chunks.sort(key=lambda x: x["hybrid_distance"])

    # Step 4: Apply relevance gate — discard chunks above threshold
    relevant_chunks = [
        c for c in scored_chunks
        if c["hybrid_distance"] <= RELEVANCE_THRESHOLD
    ]

    if not relevant_chunks:
        logger.warning(
            "[SEARCH PDF] All %s candidates failed relevance gate "
            "(threshold=%s). Best distance: %s",
            len(scored_chunks),
            RELEVANCE_THRESHOLD,
            scored_chunks[0]['hybrid_distance'] if scored_chunks else 'N/A'
        )
        # Fallback: return top 2 by hybrid distance with a warning,
        # rather than returning nothing for every edge case
        relevant_chunks = scored_chunks[:2]
        logger.info("[SEARCH PDF] Fallback: returning top 2 despite low relevance")

    # Step 5: Take top_k
    final_chunks = relevant_chunks[:top_k]

    logger.info("[SEARCH PDF] Returning %s chunks (from %s candidates)", len(final_chunks), len(scored_chunks))
    for i, chunk in enumerate(final_chunks):
        logger.info(
            "[SEARCH PDF] Chunk %s: hybrid_dist=%.4f | orig_dist=%.4f | "
            "kw_score=%.4f | cat_boost=%.1f | cat=%s | section=%s | type=%s",
            i + 1,
            chunk['hybrid_distance'],
            chunk['original_distance'],
            chunk['keyword_score'],
            chunk['category_boost'],
            chunk['category'],
            chunk['section'][:40],
            chunk['chunk_type']
        )
        logger.info("[SEARCH PDF] Chunk %s preview: %s...", i + 1, chunk['content'][:150])

    if return_metadata:
        return [
            {
                "chunk_id": i + 1,
                "content": chunk["content"],
                "distance": chunk["hybrid_distance"],
                "original_distance": chunk["original_distance"],
                "keyword_score": chunk["keyword_score"],
                "section": chunk["section"],
                "category": chunk["category"],
                "chunk_type": chunk["chunk_type"],
            }
            for i, chunk in enumerate(final_chunks)
        ]

    return "\n\n".join(chunk["content"] for chunk in final_chunks)


# ══════════════════════════════════════════════════════════════════════
#                    QUESTION RESOLUTION
# ══════════════════════════════════════════════════════════════════════

def resolve_question_with_history(
    question: str,
    history: List[dict]
) -> str:
    """
    Rewrite vague questions using conversation history
    Turns "tell me more about it" into a specific searchable question
    Uses GPT to understand what "it" refers to from history
    """
    if not history:
        return question

    question_lower = question.lower().strip()
    word_count = len(question.split())

    # Only treat as vague if the question STARTS with a pronoun/vague
    # opener, or is very short. A plain substring match on words like
    # "this"/"it" fires on complete, self-contained questions too
    # (e.g. "what is not covered by this travel insurance?"), causing
    # an unnecessary extra GPT call on almost every question.
    vague_openers = [
        "it", "that", "this", "those", "these",
        "tell me more", "explain", "elaborate",
        "what about", "more about", "go on"
    ]

    starts_vague = any(question_lower.startswith(opener) for opener in vague_openers)
    is_short = word_count <= 5

    if not (starts_vague or is_short):
        return question

    logger.info("[RESOLVE QUESTION] Vague question detected: '%s'", question)
    logger.info("[RESOLVE QUESTION] Resolving using %s history messages", len(history))

    try:
        recent_history = history[-6:] if len(history) > 6 else history
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in recent_history
        ])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        logger.info("[RESOLVE QUESTION] Original: '%s'", question)
        logger.info("[RESOLVE QUESTION] Resolved: '%s'", resolved)
        return resolved

    except (KeyError, IndexError):
        logger.exception("[RESOLVE QUESTION] Error parsing GPT response")
        return question
    except Exception:  # noqa: BLE001
        logger.exception("[RESOLVE QUESTION] Error resolving question")
        return question


# ══════════════════════════════════════════════════════════════════════
#                     CREDENTIAL RESOLUTION
# ══════════════════════════════════════════════════════════════════════

def _resolve_credentials(policy_no: Optional[str]) -> Optional[dict]:
    """Fetch policy_no + access_token, either the latest or for a specific policy."""
    if policy_no is None:
        logger.info("[PDF TOOL] Fetching latest policy wording from DB")
        credentials = get_latest_policy_wording_credentials()
        if not credentials:
            logger.error("[PDF TOOL] No active policy wording found in DB")
            return None
        logger.info("[PDF TOOL] Latest policy: %s", credentials['policy_no'])
        return credentials

    logger.info("[PDF TOOL] Fetching credentials for policy: %s", policy_no)
    credentials = get_policy_credentials_by_no(policy_no)
    if not credentials:
        logger.error("[PDF TOOL] Policy %s not found", policy_no)
        return None
    return credentials

def _ensure_pdf_indexed(policy_no: str, access_token: str) -> bool:
    """Reindex the policy PDF if needed. Returns True on success."""
    if should_reindex(policy_no, access_token):
        logger.info("[PDF TOOL] Re-indexing PDF for: %s", policy_no)
        return index_pdf(policy_no, access_token)
    logger.info("[PDF TOOL] Using cached index for: %s", policy_no)
    return True

def _build_context_text(relevant_chunks: Any, return_metadata: bool) -> str:
    """Flatten chunk metadata (if present) into plain context text for GPT."""
    if return_metadata and isinstance(relevant_chunks, list):
        return "\n\n".join(chunk["content"] for chunk in relevant_chunks)
    return relevant_chunks

def _generate_answer_from_context(context_text: str, resolved_question: str) -> str:
    """Call GPT to answer the resolved question using the given context."""
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
                    {context_text}

                    Question: {resolved_question}
                    """
                }
            ]
        )

        answer = response.choices[0].message.content
        logger.info("[PDF TOOL] Answer: %s...", answer[:200])
        return answer

    except (KeyError, IndexError):
        logger.exception("[PDF TOOL] GPT response parsing error")
        return "Sorry, I encountered an error generating the answer."
    except Exception:  # noqa: BLE001
        logger.exception("[PDF TOOL] GPT error")
        return "Sorry, I encountered an error generating the answer."


# ══════════════════════════════════════════════════════════════════════
#                     MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

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

    logger.info("[PDF TOOL] Question: %s", question)
    logger.info("[PDF TOOL] Policy: %s", policy_no if policy_no else 'latest')
    logger.info("[PDF TOOL] History: %s messages", len(conversation_history))

    resolved_question = resolve_question_with_history(question, conversation_history)

    credentials = _resolve_credentials(policy_no)
    if not credentials:
        if policy_no is None:
            return "Sorry, I could not find any active policy wording in the system."
        return f"Sorry, I could not find policy {policy_no} in the system."

    policy_no = credentials["policy_no"]
    access_token = credentials["access_token"]
    logger.info("[PDF TOOL] Token (first 8): %s...", access_token[:8])

    if not _ensure_pdf_indexed(policy_no, access_token):
        return "Sorry, I could not retrieve the policy wording document."

    logger.info("[PDF TOOL] Searching with resolved question: %s", resolved_question)
    relevant_chunks = search_pdf(policy_no, resolved_question, return_metadata=return_metadata)

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
    context_text = _build_context_text(relevant_chunks, return_metadata)
    answer = _generate_answer_from_context(context_text, resolved_question)

    if return_metadata:
        return {
            "answer": answer,
            "resolved_question": resolved_question,
            "wording_chunks": relevant_chunks if isinstance(relevant_chunks, list) else []
        }
    return answer