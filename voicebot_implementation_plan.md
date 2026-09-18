# AI Voicebot POC — Week-by-Week / Day-by-Day Implementation Plan

Assumes full-time focus (5 days/week). Maps directly onto the steps in the POC Build Guide, with what to study before/alongside each build day.

## Week 1 — Foundations + Transport Layer

**Day 1–2: Python async + audio basics (study-heavy)**
- Study: `asyncio` (async/await, event loops, tasks), basic audio concepts (sample rate, PCM vs Opus, chunking).
- Build: nothing yet — write a couple of small async scripts (e.g., a WebSocket echo server) to get comfortable with async I/O, since the whole pipeline runs on it.

**Day 3–4: WebRTC + LiveKit fundamentals**
- Study: WebRTC basics (ICE/STUN/TURN, SDP, why it's used over plain WebSockets), LiveKit's docs on rooms/tracks/participants/agent jobs.
- Build: self-host the LiveKit server via Docker Compose, verify it's reachable.

**Day 5: Agent environment**
- Study: `livekit-agents` SDK docs — `AgentSession`, `Agent`, worker lifecycle.
- Build: set up the Python agent env, get the placeholder agent connecting and registering as a worker.

## Week 2 — Voice Pipeline Front Half

**Day 6: VAD**
- Study: how Silero VAD works at a high level (a small model classifying speech vs. silence frame-by-frame).
- Build: wire in `silero.VAD.load()`, confirm speech start/stop events fire correctly.

**Day 7: Turn detection**
- Study: the difference between VAD (silence detection) and turn detection (semantic "are they actually done talking").
- Build: add the turn-detector plugin, test with mid-sentence pauses vs. finished sentences.

**Day 8–9: Self-hosted STT**
- Study: Whisper architecture basics (encoder-decoder, why streaming Whisper is harder than batch), `faster-whisper`/CTranslate2.
- Build: get `faster-whisper` transcribing file-based audio accurately, then look at streaming integration.

**Day 10: Self-hosted LLM**
- Study: how Ollama serves local models, GGUF quantization basics (why 8B models run on consumer GPUs).
- Build: stand up Ollama, pull a model, get responses returning coherently via HTTP.

## Week 3 — Brains + Voice Output

**Day 11–12: RAG**
- Study: embeddings (what a vector represents), cosine similarity search, basic chunking strategies.
- Build: stand up Qdrant, embed a small knowledge base with `sentence-transformers`, get retrieval returning relevant chunks, wire into the LLM prompt.

**Day 13–14: Self-hosted TTS**
- Study: how neural TTS models (Piper/XTTS) turn text into waveform, streaming synthesis vs. full-sentence generation.
- Build: get Piper generating intelligible audio from text.

**Day 15: Full pipeline wiring**
- Study: LiveKit's `STT`/`LLM`/`TTS` plugin base classes (interfaces your wrappers need to implement).
- Build: replace the placeholder session with real STT → RAG-augmented LLM → TTS, all inside one `AgentSession`.

## Week 4 — Test, Debug, Stretch Goals

**Day 16: Web client + first full conversation**
- Study: LiveKit access tokens, the Agents Playground.
- Build: connect a browser client, have your first real spoken conversation with the bot.

**Day 17–18: Latency debugging (this is where real skill shows)**
- Study: where latency hides in streaming pipelines — time-to-first-token, time-to-first-audio, blocking calls in async code.
- Build: instrument each stage (STT time, LLM TTFT, TTS TTFA) and fix the worst offender — usually blocking inference calls that need to move to a thread pool.

**Day 19: Telephony (stretch)**
- Study: SIP basics, how a SIP trunk provider routes calls into a LiveKit room.
- Build: wire up LiveKit's SIP bridge with a test trunk.

**Day 20: Containerization (stretch)**
- Study: multi-container orchestration basics, why STT/TTS need independent GPU scaling from the API layer.
- Build: split agent/STT/TTS into separate Dockerfiles.

---

This is an aggressive but realistic timeline if you're full-time and already comfortable with Python — expect Week 2–3 (STT/LLM/RAG/TTS) to eat the most time if any of those are new to you.