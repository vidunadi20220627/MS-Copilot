import sys
sys.path.insert(0, ".")
from vector_store.chroma import get_or_create_collection

coll = get_or_create_collection('policy_wording_DTPS26402347')
data = coll.get(include=['documents', 'metadatas'])
print(f"Total chunks in DB: {len(data['ids'])}")

covid_chunks = []
for doc, meta in zip(data['documents'], data['metadatas']):
    if "covid" in doc.lower() or "covid" in str(meta).lower():
        covid_chunks.append((meta, doc[:200]))

print(f"Chunks mentioning COVID: {len(covid_chunks)}")
for c in covid_chunks:
    print(c)
