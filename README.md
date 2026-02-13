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
| **[nvidia/nv-embedqa-e5-v5](https://build.nvidia.com/nvidia/nv-embedqa-e5-v5)** | Semantic text embeddings for RAG-based passage retrieval | NVIDIA NIM API |

### Why These Models?

- **Nemotron 30B** is a reasoning-capable model that generates rich, contextual narratives grounded in the source material. It understands story structure, creates branching choices, and maintains coherence across turns — all while being efficient enough for real-time interactive use.
- **NV-EmbedQA-E5-v5** provides high-quality semantic embeddings that power the retrieval system. As the user progresses through the adventure, the system retrieves the most relevant book passages to feed into the narrative generation — ensuring the AI stays faithful to the original text.

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
- **Choose-your-own-adventure** — 3 choices at every turn, confirmed via slide-up modal
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
