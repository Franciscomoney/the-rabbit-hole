# The Rabbit Hole — The Infinite Interactive Audiobook

> **NVIDIA GTC 2026 Golden Ticket Submission**

Turn any book into a fully voiced, interactive choose-your-own-adventure experience — powered entirely by NVIDIA AI.

Upload a PDF or TXT, and Rabbit Hole transforms it into an immersive audio narrative where **you** make the choices. Every playthrough is unique. Every decision reshapes the story.

**Live demo:** [rabbithole.experiment.franciscocordobaotalora.com](https://rabbithole.experiment.franciscocordobaotalora.com)

---

## How It Works

```
Upload Book → Chunk & Embed → Play Adventure → Listen & Choose → Story Branches
     │              │                │                │               │
   PDF/TXT    NVIDIA NV-EmbedQA   Nemotron 30B    Edge TTS      Your choices
              semantic vectors    generates story   narrates     shape the path
```

1. **Upload** any book (PDF or plain text) — drag & drop or tap to select
2. **Nemotron extracts** the real book title and author from the text automatically
3. The text is split into semantic chunks and embedded using **NVIDIA NV-EmbedQA-E5-v5**
4. Hit play — **NVIDIA Nemotron 30B** reads the relevant passages and generates an interactive narrative in second person
5. **Edge TTS** narrates the story with expressive voices
6. You pick from 3 choices at every turn — the AI weaves your decision into the next passage
7. The adventure continues, drawing from the actual book content, until you've explored the whole story

---

## NVIDIA Models Used

| Model | Role | API |
|-------|------|-----|
| **[nvidia/nemotron-3-nano-30b-a3b](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b)** | Story generation, interactive narrative, title/author extraction, book summaries, tag extraction | OpenRouter |
| **[nvidia/llama-3.3-nemotron-super-49b-v1.5](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1.5)** | Choice enhancement — rewrites adventure options into vivid, actionable prompts | OpenRouter |
| **[nvidia/nv-embedqa-e5-v5](https://build.nvidia.com/nvidia/nv-embedqa-e5-v5)** | Semantic text embeddings for RAG-based passage retrieval | NVIDIA NIM API |

### Why These Models?

- **Nemotron 30B** generates rich, contextual narratives at high speed. It handles story generation, title extraction, and summaries — efficient enough for real-time interactive use.
- **Nemotron Super 49B** enhances the 3 adventure choices with vivid, actionable language ("You charge through the door..."). A quick second pass that makes choices feel like a real choose-your-own-adventure book.
- **NV-EmbedQA-E5-v5** provides high-quality semantic embeddings that power the retrieval system. As the user progresses through the adventure, the system retrieves the most relevant book passages to feed into the narrative generation — ensuring the AI stays faithful to the original text.

---

## Why Nemotron? — Technical Deep Dive

We evaluated multiple open-weight LLMs for real-time interactive narration. The constraint was strict: **the model must generate a full narrative turn (3 paragraphs + 3 branching choices) fast enough that a human doesn't lose immersion.** That means under 5 seconds end-to-end, including network latency.

### The Real-Time Narration Problem

Interactive audiobooks are fundamentally different from chatbots. A chatbot user waits for a response. An audiobook listener **expects continuous flow** — any gap longer than a few seconds breaks the spell. This constrains model choice in ways that benchmarks don't capture:

| Requirement | Why It Matters | How Nemotron Solves It |
|-------------|---------------|----------------------|
| **< 5s generation** | User is listening, not reading — pauses kill immersion | Nemotron 30B's Hybrid Mamba-Transformer MoE architecture activates only 3.6B of 31.6B params per token, delivering fast inference |
| **Faithful to source** | RAG-grounded narration must reference real characters, places, events | Nemotron's 256K context window ingests full book passages without truncation artifacts |
| **Structured output** | Must produce exactly 3 numbered choices after narrative — every turn, reliably | Nemotron follows constrained formatting instructions more consistently than similarly-sized models we tested |
| **Second-person voice** | "You walk into the room..." requires sustained stylistic control | Nemotron maintains persona across 20+ conversation turns without drift |

### Measured Performance (Production)

These are real numbers from the live deployment, not synthetic benchmarks:

| Metric | Value | Notes |
|--------|-------|-------|
| Narrative generation | **2–5s** | 900–1000 chars per turn (3 paragraphs + 3 choices) |
| TTS synthesis | **2.1–2.6s** | ~700 chars of narration → spoken audio |
| Semantic embedding | **< 1s** | Single query embedding via NV-EmbedQA for branching |
| Bulk embedding (upload) | **~2s per batch of 50** | Book ingestion: 50 chunks per NIM API call |
| Full turn (perceived) | **~3s** | Staggered UI: narrative + audio start immediately, choices appear 2s later |

The perceived latency is the key number. By starting TTS as soon as the narrative arrives (while choices populate in the background), the user hears the story within ~3 seconds of making a choice. This is fast enough to feel like turning a page.

### Why Not Other Models?

We tested alternatives during development:

- **Llama 3.3 70B**: Higher quality narratives but 2–3x slower generation. Broke the real-time constraint.
- **Mistral/Mixtral**: Competitive speed, but less consistent at maintaining structured output (3 numbered choices) across long sessions. Would frequently drop to 2 choices or merge narrative with choices.
- **Nemotron Super 49B** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`): We use this as a second pass for choice enhancement — it rewrites the 3 adventure options into vivid, actionable prompts. Worth the extra latency because it only processes ~50 tokens (the choices), not the full narrative.
- **Smaller models (7B–13B)**: Fast but couldn't sustain coherent RAG-grounded narration. Would hallucinate characters not in the book passage.

Nemotron 30B hits the sweet spot: fast enough for real-time, smart enough for faithful storytelling.

### The RAG Architecture Decision

Standard RAG retrieves passages before generation. We do something different — **we use RAG for branching, not just context**.

```
Standard RAG:    User query → Retrieve → Generate
Rabbit Hole:     User choice → Embed choice → Cosine similarity against all future chunks
                              → Route to most semantically relevant passage → Generate from there
```

When a user picks "You grab the lantern and rush toward the screams in the cellar," we:
1. Embed that choice text using **NV-EmbedQA-E5-v5**
2. Compare against all book chunks *ahead* of the current position
3. Jump to the chunk with the highest semantic similarity (e.g., chunk 0 → chunk 56)
4. Generate the next narrative from that passage

This means **the same book produces different adventures every playthrough**. Three users making three different choices at the same decision point will each land in a different part of the book. The embedding quality of NV-EmbedQA is critical here — low-quality embeddings would route users to irrelevant passages, breaking the story.

### Nemotron Across the Pipeline

We don't just use Nemotron for narration. It handles **5 distinct tasks** across the application lifecycle:

| Task | Model | Why This Model |
|------|-------|---------------|
| Interactive narrative generation | Nemotron 30B | Speed + quality balance for real-time |
| Choice enhancement | Nemotron Super 49B | Higher quality for short-form rewriting |
| Book title/author extraction | Nemotron 30B | Reliable JSON extraction from first page |
| Adventure summaries | Nemotron 30B | Creative hook-style descriptions |
| Topic tag extraction | Nemotron 30B | Concise theme identification |

Each task uses different temperature settings, token limits, and system prompts — but the same underlying model family. This demonstrates Nemotron's versatility: one model ecosystem handling structured extraction, creative generation, and style transfer.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (SPA)                  │
│            Single HTML + CSS + JS                │
│     Mobile-first · Tab navigation · Audio player │
└──────────────────────┬──────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────┐
│               FastAPI Backend                    │
│                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Chunker  │  │ Embeddings│  │  Adventure   │  │
│  │ (PyPDF2) │  │ (NV-Embed)│  │  (Nemotron)  │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
│                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ TTS      │  │ Cover Gen │  │  Summary Gen │  │
│  │(Edge TTS)│  │ (Gemini)  │  │  (Nemotron)  │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
│                                                  │
│              SQLite Database                     │
│         (books, chunks, embeddings, sessions)    │
└──────────────────────────────────────────────────┘
```

### Key Components

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server — upload, adventure, TTS, cover/summary generation endpoints |
| `adventure.py` | Nemotron-powered interactive narrative engine with RAG |
| `embeddings.py` | NVIDIA NV-EmbedQA integration for semantic search |
| `chunker.py` | Semantic text chunking with sentence-boundary awareness |
| `database.py` | SQLite storage for books, chunks, embeddings, and sessions |
| `static/index.html` | Full mobile-first SPA (single file, no build step) |

---

## Features

- **Upload any book** — PDF or TXT, drag & drop or tap to select
- **Smart title extraction** — Nemotron reads the first page and extracts the real book title and author
- **AI-generated covers** — beautiful book covers created automatically on upload
- **AI-generated summaries** — hook-style descriptions that make you the protagonist
- **Smart topic tags** — 3 AI-extracted themes per book
- **Interactive audio playback** — narrated with synchronized text highlighting (Apple Music lyrics style)
- **Choose-your-own-adventure** — 3 choices at every turn, each routed to different book passages via semantic search
- **Real branching** — choices use embedding similarity to find the most relevant passage, so every path is unique
- **Staggered UI** — narrative appears instantly with TTS, choice buttons pop in 2s later while the user reads
- **Page-turn transitions** — 3D CSS perspective animations between screens, like turning a real book page
- **App-shell layout** — fixed header with frosted glass, independent container scroll with CSS scroll snap
- **Session persistence** — continue where you left off
- **Voice input** — speak your choices (optional, requires Whisper STT endpoint)
- **Search** — real-time search across titles, authors, and tags
- **Share** — native Web Share API on mobile, clipboard fallback on desktop
- **Custom loading animations** — book scanner (upload) and neural weave spinner (adventure)
- **Dark glassmorphism UI** — NVIDIA green accent, frosted glass cards, smooth animations
- **No build step** — pure HTML/CSS/JS frontend, just serve it
- **Zero dependencies frontend** — no React, no npm, no bundler

---

## Open Source Stack

| Component | Technology | Open Source? | License |
|-----------|-----------|:------------:|---------|
| LLM | NVIDIA Nemotron 3 Nano 30B | **Yes** | Open weights (NVIDIA AI Foundation License) |
| Embeddings | NVIDIA NV-EmbedQA-E5-v5 | **Yes** | Open weights (CC-BY-4.0) |
| Backend | FastAPI + Python | **Yes** | MIT |
| Database | SQLite | **Yes** | Public Domain |
| STT | Whisper Large V3 Turbo | **Yes** | MIT |
| TTS | edge-tts (Microsoft Edge) | **Yes** | MIT (library) |
| Text extraction | PyPDF2 | **Yes** | BSD |
| Frontend | Vanilla HTML/CSS/JS | **Yes** | No frameworks |

**Both core AI models (Nemotron and NV-EmbedQA) are open-weight NVIDIA models.** The entire stack can be self-hosted — no proprietary API lock-in required. We use OpenRouter and NVIDIA NIM as hosted endpoints for convenience, but you can point these at your own inference servers.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Franciscomoney/the-rabbit-hole.git
cd the-rabbit-hole
```

### 2. Set up Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys
```

You need:
- **NVIDIA API Key** (free) — [build.nvidia.com](https://build.nvidia.com/) — for NV-EmbedQA embeddings
- **OpenRouter API Key** — [openrouter.ai](https://openrouter.ai/) — for Nemotron LLM + cover generation
- **OVH AI Token** (optional) — for voice input via Whisper STT

### 4. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 7501
```

Open `http://localhost:7501` — upload a book and start your adventure.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the SPA |
| `POST` | `/api/upload` | Upload a book (streaming progress) |
| `GET` | `/api/books` | List all books in the library |
| `POST` | `/api/books/{id}/start` | Start a new adventure session |
| `POST` | `/api/adventure` | Send a choice, get next narrative + choices |
| `POST` | `/api/tts` | Text-to-speech synthesis |
| `POST` | `/api/voice` | Voice input transcription (Whisper) |
| `PATCH` | `/api/books/{id}` | Edit book title/author |
| `DELETE` | `/api/books/{id}` | Delete a book |
| `POST` | `/api/books/{id}/generate-cover` | Generate AI cover art |
| `POST` | `/api/books/{id}/generate-summary` | Generate AI summary + tags |
| `GET` | `/api/models` | List all NVIDIA models powering the app |

---

## License

MIT — use it, fork it, build on it.

---

Built with NVIDIA Nemotron for the [NVIDIA GTC 2026 Golden Ticket Contest](https://www.nvidia.com/gtc/).
