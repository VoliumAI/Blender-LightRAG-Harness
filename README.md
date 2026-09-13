# Blender Documentation — LightRAG Retrieval Harness

A prebuilt retrieval corpus over Blender's official documentation (Manual + Python API): a FAISS
vector index + a NetworkX knowledge graph + grounding images, built from `docs.blender.org`. This
repo is **retrieval only** — no fine-tuned model, no SFT data, nothing hosted. You connect it to
whatever model or agent you already use, and it hands back grounded Blender/`bpy` context for
that model to answer from. Everything runs on CPU; the embedding model is ~146 MB.

## What's in this repo
<img width="1678" height="937" alt="image" src="https://github.com/user-attachments/assets/1c48c79f-d173-438e-bb5e-59b8810b393b" />

| File | What it is |
|---|---|
| `blender_lightrag/blender_docs.faiss` | FAISS vector index of embedded documentation chunks |
| `blender_lightrag/chunk_metadata.json` | The text/metadata for each chunk, in the same order as the FAISS index |
| `blender_lightrag/knowledge_graph.graphml` | NetworkX graph connecting `bpy` modules, operators, classes, and functions |
| `blender_lightrag/image_metadata.json` | Which local image file backs which source-doc image, its caption, and every page that references it — a lookup table, not read by the retrieval code below |

The embedding model and the reference images are hosted separately (see Setup below) — this
keeps the GitHub repo small, since neither fits comfortably in a git repo.

Keep everything in the same folder once you've downloaded it all — chunk records reference the
images folder by relative path, and the FAISS index only lines up with `chunk_metadata.json` if
both are loaded together, as saved.

## Setup (do this once, regardless of what you connect it to)

**1. Get the embedding model.** Every vector in `blender_docs.faiss` was built with **Snowflake
Arctic Embed M Long, Q8_0, GGUF**. You must embed queries with this exact model — a different
model puts queries in a different vector space and retrieval will return nonsense.

Download: **https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF**
(direct file, 146 MB: [`.../resolve/main/snowflake-arctic-embed-m-long-q8_0.gguf`](https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF/resolve/main/snowflake-arctic-embed-m-long-q8_0.gguf))

Put it in a `models/` folder next to your code.

**2. Get the reference images.** Diagrams/screenshots from the source docs ship as
`Blender_images.rar` on the same Hugging Face repo as the embedding model:
**https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF**

The archive itself is flat — the image files sit at its root, with no folder already inside it.
`chunk_metadata.json` expects them to be found at `blender_lightrag/images/<filename>` relative to
wherever you run your code, so **you need to create that folder yourself and extract into it** —
don't just "Extract Here" next to the other data files, or the images end up in the wrong place
and every image path in the chunk data will fail to resolve.

```bash
# from the folder containing blender_lightrag/
mkdir -p blender_lightrag/images
# Linux/macOS with unrar:
unrar x Blender_images.rar blender_lightrag/images/
# Windows with WinRAR/7-Zip: right-click Blender_images.rar -> "Extract to..." -> browse to (or
# create) the blender_lightrag/images folder as the destination, rather than using "Extract Here"
```

Check afterward that images landed directly inside `blender_lightrag/images/` (e.g.
`blender_lightrag/images/<some-hash>.png`) and not nested another level deeper in something like
`blender_lightrag/images/Blender_images/<some-hash>.png` — some GUI archive tools default to
creating an extra subfolder named after the archive itself, which would need deleting/flattening.

If you don't need image grounding, you can skip this entirely — everything else works without
it, you'll just get an empty image list from that part of the API.

**3. Install dependencies.**

```bash
pip install faiss-cpu llama-cpp-python numpy networkx
```

**4. Save the retrieval helper.** This repo intentionally doesn't ship a scraper or any serving
code — just the three data files above. Save this as `blender_retrieve.py` in the same folder:

```python
import json
import os
import re
import numpy as np
import faiss
import networkx as nx
from llama_cpp import Llama

MODEL_PATH       = "models/snowflake-arctic-embed-m-long-q8_0.gguf"   # point at your download
FAISS_INDEX_FILE = "blender_lightrag/blender_docs.faiss"
CHUNK_JSON_FILE  = "blender_lightrag/chunk_metadata.json"
GRAPH_FILE       = "blender_lightrag/knowledge_graph.graphml"

_model = Llama(model_path=MODEL_PATH, embedding=True, n_ctx=2048, n_gpu_layers=0, verbose=False)
_index = faiss.read_index(FAISS_INDEX_FILE)
with open(CHUNK_JSON_FILE, "r", encoding="utf-8") as f:
    _chunks = json.load(f)
_graph = nx.read_graphml(GRAPH_FILE)

# Snowflake Arctic Embed's documented asymmetric-retrieval convention: this
# exact prefix belongs on QUERY text only, never on indexed document text --
# the corpus was built without it, so retrieval has to add it back at query
# time to land in the same space the index was built in. Skipping this
# doesn't error, it just quietly makes every result somewhat worse.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed(text: str, is_query: bool = False) -> np.ndarray:
    if is_query:
        text = _QUERY_PREFIX + text
    vec = np.array(_model.embed(text), dtype=np.float32)
    if vec.ndim > 1:
        vec = vec.mean(axis=0)          # some pooling configs return one row per token
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


_PASCAL_WORD = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b')


def graph_context(hit_chunks: list, query_text: str, hops: int = 2) -> list:
    seeds = set()
    for c in hit_chunks:
        title = c.get("source_title", "")
        if title and _graph.has_node(title):
            seeds.add(title)
    for m in _PASCAL_WORD.finditer(query_text):
        if _graph.has_node(m.group(1)):
            seeds.add(m.group(1))

    neighbors = set(seeds)
    for s in seeds:
        neighbors.update(nx.ego_graph(_graph, s, radius=hops).nodes())

    lines = []
    sub = _graph.subgraph(neighbors)
    for u, v, data in sub.edges(data=True):
        lines.append(f"{u} -[{data.get('relation', 'related_to')}]-> {v}")
    return lines


def retrieve(query: str, top_k: int = 5, graph_hops: int = 2, with_images: bool = False):
    q = embed(query[:1000], is_query=True).reshape(1, -1)
    scores, idxs = _index.search(q, top_k)
    hits = [_chunks[i] for i in idxs[0] if 0 <= i < len(_chunks)]

    parts = [f"[{i + 1}] ({h['source_url']}) {h['text']}" for i, h in enumerate(hits)]
    lines = graph_context(hits, query, hops=graph_hops)
    if lines:
        parts.append("## Connected Concepts:\n" + "\n".join(lines))
    ctx = "\n\n".join(parts)

    if not with_images:
        return ctx

    image_paths = []
    for h in hits:
        for img in h.get("image_refs", []):
            p = os.path.join("blender_lightrag", img["local_path"])
            if p not in image_paths:
                image_paths.append(p)
    return ctx, image_paths
```

**5. Confirm it works.**

```python
from blender_retrieve import retrieve
print(retrieve("How do I add a cube using Python?", top_k=3))
```

⚠️ **If step 1 wasn't done correctly, this raises an error from `llama_cpp` (bad model path)
rather than silently returning nothing** — that's the first thing to check if this doesn't work.

That's the whole retrieval API. Everything below is just different ways to put that string in
front of a model.

---

## Connect it to your setup

Pick one:

- Already using **Claude Desktop, Claude Code, or Cursor** → **Option A (MCP)**
- Calling a **hosted model's API directly** (Anthropic, OpenAI, or similar) → **Option B**
- Running a **local model or your own agent loop** → **Option C**

### Option A — MCP (Claude Desktop, Claude Code, Cursor, or any MCP client)

Set this up once and the model can call it whenever it decides Blender context would help,
instead of you pasting context in by hand every time.

```bash
pip install fastmcp
```

Save this as `blender_mcp_server.py` in the same folder as `blender_retrieve.py`:

```python
from fastmcp import FastMCP
from blender_retrieve import retrieve

mcp = FastMCP("blender-lightrag")

@mcp.tool()
def search_blender_docs(query: str, top_k: int = 5) -> str:
    """Search Blender documentation and return grounded context (relevant
    chunks + connected knowledge-graph concepts) for the query."""
    return retrieve(query, top_k=top_k, graph_hops=2)

@mcp.tool()
def search_blender_docs_with_images(query: str, top_k: int = 5) -> dict:
    """Same as search_blender_docs, plus local file paths of any diagrams/
    screenshots attached to the retrieved chunks."""
    ctx, image_paths = retrieve(query, top_k=top_k, with_images=True)
    return {"context": ctx, "image_paths": image_paths}

if __name__ == "__main__":
    mcp.run()   # stdio transport — the default, and what Claude Desktop/Code expect
```

Register it. For **Claude Code**:

```bash
claude mcp add blender-lightrag python /absolute/path/to/blender_mcp_server.py
```

For **Claude Desktop** (or Cursor, same shape), add this to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows: `%APPDATA%\Claude\claude_desktop_config.json`) and restart the app:

```json
{
  "mcpServers": {
    "blender-lightrag": {
      "command": "python",
      "args": ["/absolute/path/to/blender_mcp_server.py"]
    }
  }
}
```

Use **absolute paths** everywhere — for the script, and for `MODEL_PATH` inside
`blender_retrieve.py` — MCP launches your script as a subprocess from wherever the client lives,
not from this folder, so relative paths silently fail to resolve.

### Option B — A cloud API (Anthropic, OpenAI, or any chat-completions endpoint)

Retrieve first, then put the result in the system prompt (or prepend it to the user message for
endpoints with no separate system field) before sending the request — standard RAG wiring.

**Anthropic:**

```python
from blender_retrieve import retrieve
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
user_query = "How do I add a cube using Python?"
ctx = retrieve(user_query, top_k=5, graph_hops=2)

response = client.messages.create(
    model="claude-sonnet-5",
    system=f"You are a Blender automation assistant. Use this retrieved context when relevant:\n\n{ctx}",
    messages=[{"role": "user", "content": user_query}],
    max_tokens=1024,
)
print(response.content[0].text)
```

**OpenAI (or any OpenAI-compatible endpoint):**

```python
from blender_retrieve import retrieve
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from the environment
user_query = "How do I add a cube using Python?"
ctx = retrieve(user_query, top_k=5, graph_hops=2)

response = client.chat.completions.create(
    model="<your model>",   # whichever model you currently have access to
    messages=[
        {"role": "system", "content": f"You are a Blender automation assistant. Use this retrieved context when relevant:\n\n{ctx}"},
        {"role": "user", "content": user_query},
    ],
)
print(response.choices[0].message.content)
```

Any other provider follows one of these two shapes (separate `system` field, or a
`role: "system"` entry inside `messages`) — check which your SDK uses and drop `ctx` in there.

### Option C — A local model or your own agent framework

If you're running a local llama.cpp server, Ollama, or a custom agent loop, the integration is
the same one line — `retrieve()` returns plain text (or text + image paths), so it doesn't care
what consumes it:

```python
from blender_retrieve import retrieve

ctx, image_paths = retrieve(user_query, top_k=5, with_images=True)
# ctx          -> paste into your prompt template wherever "context" or "retrieved docs" goes
# image_paths  -> local file paths; load and attach these if your model accepts image input
```

For a vision-capable model, load each path in `image_paths` and attach it as an image content
block alongside `ctx`.

---

## Reference

**Chunk fields** (`chunk_metadata.json`): `chunk_id`, `text` (embedded content), `source_title`,
`source_url` (source page on `docs.blender.org`), `section_kind` (`"prose"` or `"code"`),
`chunk_index`, `image_refs` (list of `{"local_path", "alt"}`, path relative to
`blender_lightrag/`).

**Image metadata** (`image_metadata.json`): a dict keyed by `local_path` (the same path used in
`image_refs` above), each entry `{"source_urls": [...], "alt": "...", "pages": [...]}` — every
source-doc URL that image appeared under, and every page that references it. This is a
supplementary lookup, not something the retrieval code above reads — useful if you want to
browse "what diagrams exist for topic X" independent of any specific query, but not required for
`retrieve()` to work.

**Knowledge graph**: a `networkx.DiGraph` in `knowledge_graph.graphml`. Entity/relation
extraction is regex-based pattern matching over `bpy` module/operator/class/function names in the
doc text, not a real parser of Blender's actual C/Python bindings — useful as a "what else is this
connected to" hint, not ground truth. Graph nodes are keyed by their own name directly (no
separate ID layer), which is what `graph_context()` above relies on.

**This is a static snapshot.** It reflects Blender's documentation as of whenever this corpus was
built and won't pick up anything published after that. The tooling that generated it isn't part
of this repo.

**Source content**: text and images are derived from Blender's own documentation
(`docs.blender.org`) — short excerpts and a modest number of diagrams, not a full mirror. Check
Blender's documentation license (CC-BY-SA) yourself before redistributing the corpus further.
