import os
import io
import uuid
import json
import pandas as pd
from datetime import datetime

from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
#from azure.search.documents.indexes.models import SearchDocument
from azure.core.credentials import AzureKeyCredential

from PyPDF2 import PdfReader
from openai import OpenAI

# ============ ENV ============
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "kb-files")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "kb-index")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI embedding model
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

client = OpenAI(api_key=OPENAI_API_KEY)

# Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

# ============ Helpers ============
def get_embedding(text: str):
    text = text.replace("\n", " ")
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding  # list[float]

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

def normalize(s):
    return str(s).strip() if s is not None else ""

# ============ Extractors ============
def extract_from_excel(file_bytes: bytes, filename: str):
    df = pd.read_excel(io.BytesIO(file_bytes))
    docs = []
    for idx, row in df.iterrows():
        row_text = " | ".join([f"{c}:{normalize(row[c])}" for c in df.columns])
        docs.append({
            "source_type": "excel",
            "source_file": filename,
            "page": 0,
            "content": row_text
        })
    return docs

def extract_from_pdf(file_bytes: bytes, filename: str):
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in chunk_text(text):
            docs.append({
                "source_type": "pdf",
                "source_file": filename,
                "page": page_no,
                "content": chunk
            })
    return docs

# ============ Main Ingestion ============
def ingest_all_blobs():
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(BLOB_CONTAINER)

    all_docs = []
    for blob in container_client.list_blobs():
        name = blob.name.lower()
        blob_client = container_client.get_blob_client(blob.name)
        data = blob_client.download_blob().readall()

        if name.endswith(".xlsx") or name.endswith(".xls"):
            all_docs.extend(extract_from_excel(data, blob.name))
        elif name.endswith(".pdf"):
            all_docs.extend(extract_from_pdf(data, blob.name))

    print(f"Extracted raw docs/chunks: {len(all_docs)}")

    # Create Search Documents
    search_docs = []
    for d in all_docs:
        emb = get_embedding(d["content"])

        search_docs.append({
            "id": str(uuid.uuid4()),
            "content": d["content"],
            "source_type": d["source_type"],
            "source_file": d["source_file"],
            "page": int(d["page"]),
            "contentVector": emb,
            "ingested_at": datetime.utcnow().isoformat()
        })

    # Upload in batches
    batch_size = 200
    for i in range(0, len(search_docs), batch_size):
        batch = search_docs[i:i+batch_size]
        result = search_client.upload_documents(documents=batch)
        print(f"Uploaded batch {i//batch_size + 1}, success={sum([r.succeeded for r in result])}")

    return {"status": "success", "total_chunks": len(search_docs)}

if __name__ == "__main__":
    print(ingest_all_blobs())

