import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from botbuilder.core import BotFrameworkAdapterSettings, BotFrameworkAdapter, TurnContext
from botbuilder.schema import Activity

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer

from ingest import ingest_all_blobs

# ============ ENV ============
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "kb-index")

MICROSOFT_APP_ID = os.getenv("MicrosoftAppId", "")
MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword", "")

# ============ Clients ============
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

# ============ Bot Adapter ============
settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

app = FastAPI()

# ============ Helpers ============
def get_embedding(text: str):
    text = (text or "").replace("\n", " ").strip()
    if not text:
        text = "empty"
    return embedder.encode([text])[0].tolist()

def search_top_k(question: str, k: int = 5):
    qvec = get_embedding(question)

    results = search_client.search(
        search_text=question,   # hybrid search = better
        vector_queries=[{
            "vector": qvec,
            "k": k,
            "fields": "contentVector"
        }],
        select=["content", "source_type", "source_file", "page"],
        top=k
    )

    contexts = []
    sources = []
    for r in results:
        contexts.append(r["content"])
        sources.append({
            "source_type": r.get("source_type"),
            "source_file": r.get("source_file"),
            "page": r.get("page", 0)
        })
    return contexts, sources

def format_answer(question: str, contexts, sources):
    if not contexts:
        return "Not available in uploaded Excel/PDF data."

    # Fast extractive answer
    answer = "✅ I found these relevant details in KB:\n\n"
    for i, c in enumerate(contexts[:3], start=1):
        answer += f"{i}) {c}\n\n"

    src_lines = []
    for s in sources[:3]:
        if s.get("source_type") == "pdf":
            src_lines.append(f"- {s.get('source_file')} (page {s.get('page')})")
        else:
            src_lines.append(f"- {s.get('source_file')} (excel)")
    if src_lines:
        answer += "📌 Sources:\n" + "\n".join(src_lines)

    return answer

async def on_message_activity(turn_context: TurnContext):
    question = (turn_context.activity.text or "").strip()
    if not question:
        await turn_context.send_activity("Ask a question based on uploaded Excel/PDF files.")
        return

    contexts, sources = search_top_k(question, k=5)
    answer = format_answer(question, contexts, sources)

    await turn_context.send_activity(answer)

# ============ ROUTES ============
@app.post("/api/messages")
async def messages(request: Request):
    body = await request.json()
    activity = Activity().deserialize(body)
    auth_header = request.headers.get("Authorization", "")

    async def aux_func(turn_context: TurnContext):
        if turn_context.activity.type == "message":
            await on_message_activity(turn_context)

    await adapter.process_activity(activity, auth_header, aux_func)
    return JSONResponse(status_code=200, content={})

@app.post("/ingest")
async def ingest():
    return ingest_all_blobs()

@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG bot is running (FREE mode)"}
