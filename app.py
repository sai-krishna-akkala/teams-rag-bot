import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from botbuilder.core import BotFrameworkAdapterSettings, BotFrameworkAdapter, TurnContext
from botbuilder.schema import Activity

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from openai import OpenAI
from ingest import ingest_all_blobs

# ============ ENV ============
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "kb-index")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MICROSOFT_APP_ID = os.getenv("MicrosoftAppId", "")
MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword", "")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ============ Clients ============
client = OpenAI(api_key=OPENAI_API_KEY)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

# ============ Bot Adapter ============
settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

# ============ FastAPI app ============
app = FastAPI()

# ============ Helpers ============
def get_embedding(text: str):
    text = (text or "").replace("\n", " ").strip()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def search_top_k(question: str, k: int = 5):
    qvec = get_embedding(question)

    results = search_client.search(
        search_text=None,
        vector_queries=[{
            "vector": qvec,
            "k": k,
            "fields": "contentVector"
        }],
        select=["content", "source_type", "source_file", "page"]
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

async def answer_with_rag(question: str):
    contexts, sources = search_top_k(question, k=5)
    context_text = "\n\n".join([f"[CTX{i+1}] {c}" for i, c in enumerate(contexts)])

    prompt = f"""
You are a Teams bot assistant.
Answer ONLY using the context below (from Excel/PDF).
If answer not found, say: Not available in uploaded Excel/PDF data.

Context:
{context_text}

Question: {question}
Answer:
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()

    # Attach sources
    src_lines = []
    for s in sources[:3]:
        if s.get("source_type") == "pdf":
            src_lines.append(f"- {s.get('source_file')} (page {s.get('page')})")
        else:
            src_lines.append(f"- {s.get('source_file')} (excel)")
    if src_lines:
        answer += "\n\n📌 Sources:\n" + "\n".join(src_lines)

    return answer

async def on_message_activity(turn_context: TurnContext):
    question = (turn_context.activity.text or "").strip()
    if not question:
        await turn_context.send_activity("Ask a question based on uploaded Excel/PDF files.")
        return

    answer = await answer_with_rag(question)
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
    return JSONResponse(status_code=201, content={})

@app.post("/ingest")
async def ingest():
    result = ingest_all_blobs()
    return result

@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG bot is running"}
