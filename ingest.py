import os
import io
import uuid
import pandas as pd
from datetime import datetime

from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ========= ENV =========
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "kb-files")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "kb-index")

VECTOR_DIM = 384  # MiniLM

# Load embedding model once
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

def get_embedding(text: str):
    text = (text or "").replace("\n", " ").strip()
    if not text:
        text = "empty"
    return embedder.encode([text])[0].tolist()

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150):
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return chunks

def normalize(v):
    return "" if v is None else str(v).strip()

def extract_from_excel(file_bytes: bytes, filename: str):
    df = pd.read_excel(io.BytesIO(file_bytes))
    docs = []
    for _, row in df.iterrows():
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

def ingest_all_blobs():
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING missing")
    if not (AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY and AZURE_SEARCH_INDEX):
        raise ValueError("Azure Search env vars missing")

    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(BLOB_CONTAINER)

    extracted_docs = []
    for blob in container_client.list_blobs():
        blob_name = blob.name
        lower = blob_name.lower()

        blob_client = container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().readall()

        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            extracted_docs.extend(extract_from_excel(data, blob_name))
        elif lower.endswith(".pdf"):
            extracted_docs.extend(extract_from_pdf(data, blob_name))

    search_docs = []
    for d in extracted_docs:
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

    batch_size = 200
    for i in range(0, len(search_docs), batch_size):
        batch = search_docs[i:i + batch_size]
        search_client.upload_documents(documents=batch)

    return {"status": "success", "chunks_uploaded": len(search_docs)}
