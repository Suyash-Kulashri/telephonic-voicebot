# AI Voicebot POC — Build Guide (From Scratch)

**Goal of this POC:** a working, self-hosted, real-time voice agent that you can talk to in a browser tab — you speak, it detects when you're done talking, transcribes you, thinks with a self-hosted LLM (optionally grounded with RAG), and speaks back — with every core component (STT, LLM, TTS, VAD, media server) running on infrastructure you control, not a managed API.

**Scope for v1 POC:** one agent, one room, browser client, no telephony, no Kubernetes. Get the loop working end-to-end first — scale and phone integration come after.

---

## 0. Prerequisites

| Requirement | Why |
|---|---|
| Linux or WSL2 (Ubuntu 22.04+ recommended) | Docker + GPU drivers are smoothest here |
| Python 3.10+ | `livekit-agents` SDK target |
| Docker + Docker Compose | Self-hosting LiveKit server, Qdrant, Ollama |
| Node.js 18+ | For the LiveKit web test client |
| NVIDIA GPU (8GB+ VRAM) *recommended* | Whisper + LLM + TTS all run much faster on GPU. CPU works for the POC but expect multi-second latency. |
| Git | Cloning example repos |

If you don't have a GPU, that's fine for proving the pipeline works — just use smaller models (`whisper-base`, a 3B-parameter LLM, Piper for TTS) and expect higher latency until you move to GPU hardware.

---

## 1. Project Structure

Set this up first so every later step has a home:

```
voicebot-poc/
├── docker-compose.yml          # LiveKit server, Qdrant, Ollama
├── livekit.yaml                 # LiveKit server config
├── agent/
│   ├── main.py                  # The LiveKit agent entrypoint
│   ├── stt.py                   # Whisper wrapper
│   ├── llm.py                   # Ollama wrapper + RAG hook
│   ├── tts.py                   # Piper/XTTS wrapper
│   ├── rag.py                   # Embedding + retrieval logic
│   └── requirements.txt
├── knowledge_base/              # Sample docs for RAG
│   └── faq.md
└── web-client/                  # Minimal LiveKit React test client
```

```bash
mkdir -p voicebot-poc/agent voicebot-poc/knowledge_base
cd voicebot-poc
git init
```

---

## 2. Step 1 — Self-Host the LiveKit Server

This is the WebRTC media server (SFU) that all audio flows through.

**`docker-compose.yml`:**
```yaml
version: "3.9"
services:
  livekit:
    image: livekit/livekit-server:latest
    command: --config /etc/livekit.yaml
    ports:
      - "7880:7880"       # HTTP/WS signaling
      - "7881:7881"       # RTC TCP
      - "50000-50100:50000-50100/udp"  # RTC media
    volumes:
      - ./livekit.yaml:/etc/livekit.yaml
```

**`livekit.yaml`:**
```yaml
port: 7880
rtc:
  tcp_port: 7881
  port_range_start: 50000
  port_range_end: 50100
  use_external_ip: false   # set true + configure NAT mapping if not on localhost
keys:
  devkey: devsecret        # API key/secret pair — use real random values outside local dev
```

```bash
docker compose up -d livekit
```

Verify it's up:
```bash
curl http://localhost:7880
```

**Milestone check:** LiveKit server responds. You now have a place for audio to flow through.

---

## 3. Step 2 — Python Agent Environment

```bash
cd agent
python3 -m venv venv
source venv/bin/activate
pip install "livekit-agents[silero,turn-detector]~=1.0" livekit-plugins-openai python-dotenv
```

**`.env`** (in `agent/`):
```
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret
```

**Minimal `main.py`** — a bot that joins a room and echoes with a placeholder pipeline (confirms wiring works before you plug in real models):
```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import silero

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        vad=silero.VAD.load(),   # Step 3 lives here
        # stt=..., llm=..., tts=...  # filled in during Steps 5-8
    )
    agent = Agent(instructions="You are a helpful voice assistant.")
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

```bash
python main.py dev
```

**Milestone check:** the agent process connects to your LiveKit server and shows "registered worker" in logs.

---

## 4. Step 3 — Voice Activity Detection (VAD)

Already wired above via `silero.VAD.load()` — Silero VAD ships as a LiveKit plugin and runs locally on CPU, no GPU needed. This is what tells the agent "someone is speaking now" vs. silence.

**What to verify:** join with a mic (Step 9's test client) and check agent logs show speech-start/speech-end events as you talk and pause.

---

## 5. Step 4 — Turn Detection (Endpointing)

VAD alone just detects *silence*, not "the user is actually finished." Add the turn-detector plugin so the agent doesn't jump in during a mid-sentence pause:

```bash
pip install "livekit-agents[turn-detector]"
```

```python
from livekit.plugins.turn_detector.multilingual import MultilingualModel

session = AgentSession(
    vad=silero.VAD.load(),
    turn_detection=MultilingualModel(),
    ...
)
```

**Milestone check:** pause mid-sentence with "um..." — the agent should wait; finish a sentence — it should respond promptly.

---

## 6. Step 5 — Self-Hosted STT (Whisper)

For a POC, `faster-whisper` (CTranslate2-based, much faster than vanilla Whisper) is the pragmatic choice.

```bash
pip install faster-whisper
```

**`stt.py`:**
```python
from faster_whisper import WhisperModel

model = WhisperModel("base.en", device="cuda", compute_type="float16")
# CPU fallback: device="cpu", compute_type="int8"

def transcribe(audio_path: str) -> str:
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text for seg in segments)
```

For real streaming (not just file-based), use LiveKit's built-in local Whisper integration or wrap `faster-whisper`'s streaming support directly in an `STT` plugin class — the framework expects an `stt.STT` subclass with a `recognize`/`stream` method. Start with file-based transcription to validate accuracy, then move to streaming once the loop works.

**Milestone check:** speak a sentence, see accurate text logged.

---

## 7. Step 6 — Self-Hosted LLM

Easiest self-hosted path for a POC: **Ollama**.

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull llama3.1:8b
```

**`llm.py`:**
```python
import httpx

async def generate(prompt: str, context: str = "") -> str:
    full_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAssistant:"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.1:8b", "prompt": full_prompt, "stream": False},
        )
        return resp.json()["response"]
```

For production-grade streaming token output (needed to reduce latency into TTS later), swap to **vLLM** which exposes an OpenAI-compatible `/v1/chat/completions` streaming endpoint — LiveKit's `openai` plugin can point straight at it via `base_url`.

**Milestone check:** send a test prompt via curl/Python, get a coherent response back.

---

## 8. Step 7 — RAG Layer

**Stand up a vector DB:**
```bash
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

**Embed and index your knowledge base:**
```bash
pip install sentence-transformers qdrant-client
```

```python
# rag.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)

def index_docs(chunks: list[str]):
    qdrant.recreate_collection("kb", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    vectors = embedder.encode(chunks)
    points = [PointStruct(id=i, vector=v.tolist(), payload={"text": c}) for i, (v, c) in enumerate(zip(vectors, chunks))]
    qdrant.upsert(collection_name="kb", points=points)

def retrieve(query: str, top_k: int = 3) -> str:
    qvec = embedder.encode(query).tolist()
    hits = qdrant.search(collection_name="kb", query_vector=qvec, limit=top_k)
    return "\n".join(h.payload["text"] for h in hits)
```

Put a few Q&A pairs or short paragraphs in `knowledge_base/faq.md`, chunk them (split by paragraph is fine for a POC), and run `index_docs()` once at startup.

Wire retrieval into `llm.py`: call `retrieve(user_query)` and pass the result as `context` into `generate()`.

**Milestone check:** ask something only answerable from your knowledge base — the LLM's answer should reflect it, and asking the same question with RAG disabled should give a generic/wrong answer.

---

## 9. Step 8 — Self-Hosted TTS

**Fastest to get running: Piper** (lightweight, CPU-friendly, good enough for a POC).

```bash
pip install piper-tts
# download a voice model, e.g. en_US-lessac-medium
```

```python
# tts.py
import subprocess

def synthesize(text: str, out_path: str = "out.wav"):
    subprocess.run(
        ["piper", "--model", "en_US-lessac-medium.onnx", "--output_file", out_path],
        input=text.encode(), check=True,
    )
    return out_path
```

For higher-quality, more natural voice (at higher compute cost), swap in **Coqui XTTS v2** or **Kokoro** later — same interface, just a different backend.

**Milestone check:** generated `.wav` sounds intelligible and plays back correctly.

---

## 10. Step 9 — Wire the Full Pipeline into the LiveKit Agent

Now replace the placeholder session in `main.py` with real STT/LLM/TTS plugin instances (either using LiveKit's built-in plugin interfaces pointed at your self-hosted endpoints, or custom wrapper classes implementing the `STT`/`LLM`/`TTS` base classes from `livekit.agents`):

```python
from livekit.plugins import silero, openai  # openai plugin works with any OpenAI-compatible endpoint (vLLM, Ollama's OpenAI-compat mode)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

session = AgentSession(
    vad=silero.VAD.load(),
    turn_detection=MultilingualModel(),
    stt=my_whisper_stt_plugin,       # from Step 5, adapted to STT interface
    llm=openai.LLM(base_url="http://localhost:11434/v1", model="llama3.1:8b"),  # or vLLM
    tts=my_piper_tts_plugin,         # from Step 8, adapted to TTS interface
)
```

Hook RAG in as a pre-processing step on the user's transcribed text before it reaches the LLM call (or as a "tool" the LLM can call — cleaner but more setup for a POC).

**Milestone check:** full loop — you speak, it transcribes, retrieves context, generates a reply, and speaks back — all in one continuous session.

---

## 11. Step 10 — Test with a Web Client

Fastest path: LiveKit's hosted **Agents Playground** (works against a self-hosted server too) or a minimal custom client:

```bash
npx create-livekit-app web-client
cd web-client
npm install
```

Point it at `ws://localhost:7880` with a generated access token (LiveKit CLI: `lk token create --join --room test-room --identity tester`), open in browser, allow mic access, and talk to your agent.

**Milestone check:** natural back-and-forth conversation with acceptable latency (aim for under ~2s response start for a POC; production targets are much tighter).

---

## 12. Stretch Goal — Telephony (SIP)

Once the browser loop works:
```bash
docker compose -f livekit-sip-compose.yml up -d   # LiveKit's SIP bridge
```
Configure a SIP trunk (a provider like Telnyx or a self-hosted Asterisk/FreeSWITCH instance) to route calls into your LiveKit room, so the same agent code answers phone calls with no changes.

---

## 13. Stretch Goal — Containerize Each Service

Once each piece works standalone, split into its own Dockerfile (agent, STT service, TTS service) so they can scale independently later — this is what makes the "MLOps & microservices" part of the JD real rather than theoretical.

---

## 14. Testing & Validation Checklist

- [ ] LiveKit server reachable and agent registers as a worker
- [ ] VAD correctly detects speech start/stop
- [ ] Turn detector doesn't cut off mid-sentence pauses
- [ ] STT transcribes clearly with <1s lag for short utterances
- [ ] LLM response is coherent and (with RAG) grounded in your knowledge base
- [ ] TTS output is intelligible and starts playing quickly after LLM begins responding
- [ ] Full round trip (you finish speaking → bot starts speaking) feels conversational, not robotic
- [ ] Interrupting the bot mid-response (barge-in) is handled without crashing

---

## 15. Common Pitfalls

- **Latency stacks up silently** — measure each stage (STT time, LLM time-to-first-token, TTS time-to-first-audio) separately; don't just eyeball the total.
- **Blocking calls in async code** — running Whisper/TTS synchronously inside the agent's event loop will stall the whole session; run heavy inference in a thread pool or separate service.
- **CPU-only testing gives misleading latency numbers** — validate logic on CPU, but benchmark real latency targets on GPU.
- **RAG without chunking strategy** — dumping whole documents into context works for a 3-line FAQ; anything bigger needs real chunking + top-k tuning.

---

## 16. After the POC

Once this loop works end-to-end, the next real steps toward the full JD scope are: streaming STT/TTS (instead of file-based) to cut latency further, splitting components into separately-scaled microservices, adding proper MLOps (model versioning, GPU inference serving via vLLM/Triton, monitoring), and telephony hardening for production call volumes.
