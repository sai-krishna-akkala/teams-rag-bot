import os
import asyncio
from aiohttp import web

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

# OpenAI models
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)

# Azure AI Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

# Bot adapter
settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

# ============ Helpers ============
def get_embedding(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text.replace("\n", " "))
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
Answer ONLY using the context below from Excel/PDF files.
If the answer is not in context, reply: "Not available in the uploaded Excel/PDF data."

Context:
{context_text}

Question: {question}

Answer in short and clear:
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    final_answer = resp.choices[0].message.content.strip()

    # Append sources
    src_lines = []
    for s in sources[:3]:
        if s["source_type"] == "pdf":
            src_lines.append(f"- {s['source_file']} (page {s['page']})")
        else:
            src_lines.append(f"- {s['source_file']} (excel row/chunk)")
    if src_lines:
        final_answer += "\n\n📌 Sources:\n" + "\n".join(src_lines)

    return final_answer

# ============ Bot logic ============
async def on_message_activity(turn_context: TurnContext):
    question = (turn_context.activity.text or "").strip()
    if not question:
        await turn_context.send_activity("Please ask a question based on the uploaded Excel/PDF files.")
        return

    answer = await answer_with_rag(question)
    await turn_context.send_activity(answer)

# ============ HTTP endpoints ============
async def messages(req: web.Request) -> web.Response:
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    async def aux_func(turn_context: TurnContext):
        if turn_context.activity.type == "message":
            await on_message_activity(turn_context)

    await adapter.process_activity(activity, auth_header, aux_func)
    return web.Response(status=201)

async def ingest(req: web.Request) -> web.Response:
    # Rebuild index when you upload new Excel/PDFs
    result = ingest_all_blobs()
    return web.json_response(result)

app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_post("/ingest", ingest)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
