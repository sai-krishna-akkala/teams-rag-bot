import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from botbuilder.core import BotFrameworkAdapterSettings, BotFrameworkAdapter, TurnContext
from botbuilder.schema import Activity
from botframework.connector.auth import MicrosoftAppCredentials

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv

load_dotenv()

# ============ ENV ============
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "kb-index")

MICROSOFT_APP_ID = os.getenv("MicrosoftAppId", "")
MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword", "")
MICROSOFT_APP_TENANT_ID = os.getenv("MicrosoftAppTenantId", "")

# ============ Validation ============
if not AZURE_SEARCH_ENDPOINT:
    raise ValueError("AZURE_SEARCH_ENDPOINT missing")
if not AZURE_SEARCH_KEY:
    raise ValueError("AZURE_SEARCH_KEY missing")
if not AZURE_SEARCH_INDEX:
    raise ValueError("AZURE_SEARCH_INDEX missing")

if MICROSOFT_APP_ID and not MICROSOFT_APP_PASSWORD:
    raise ValueError("MicrosoftAppPassword missing")
if MICROSOFT_APP_ID and not MICROSOFT_APP_TENANT_ID:
    raise ValueError("MicrosoftAppTenantId missing (single tenant bot)")

# ============ Azure Search ============
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

# ============ Bot Adapter ============
if MICROSOFT_APP_TENANT_ID:
    MicrosoftAppCredentials.tenant_id = MICROSOFT_APP_TENANT_ID

settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

# ============ FastAPI ============
app = FastAPI()

# ============ Helpers ============
def search_top_k(question: str, k: int = 5):
    results = search_client.search(
        search_text=question,
        select=["content", "source_type", "source", "page"],
        top=k
    )

    contexts = []
    sources = []
    for r in results:
        contexts.append(r.get("content", ""))
        sources.append({
            "source_type": r.get("source_type"),
            "source": r.get("source"),
            "page": r.get("page", 0)
        })

    return contexts, sources

def format_answer(contexts, sources):
    if not contexts:
        return "Not available in uploaded Excel/PDF data."

    answer = "✅ I found these relevant details in KB:\n\n"
    for i, c in enumerate(contexts[:3], start=1):
        answer += f"{i}) {c}\n\n"

    src_lines = []
    for s in sources[:3]:
        if s.get("source_type") == "pdf":
            src_lines.append(f"- {s.get('source')} (page {s.get('page')})")
        else:
            src_lines.append(f"- {s.get('source')} (excel)")

    if src_lines:
        answer += "📌 Sources:\n" + "\n".join(src_lines)

    return answer

async def on_message_activity(turn_context: TurnContext):
    question = (turn_context.activity.text or "").strip()
    if not question:
        await turn_context.send_activity("Ask a question based on uploaded Excel/PDF files.")
        return

    try:
        contexts, sources = search_top_k(question, k=5)
        answer = format_answer(contexts, sources)
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    await turn_context.send_activity(answer)

# ============ Routes ============
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

@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG bot running (Codespaces TEXT mode)"}
