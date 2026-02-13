import os
import uuid
import json
import asyncio
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, AsyncGenerator
from dotenv import load_dotenv

from database import (
    save_book, save_chunk, get_book_info, create_session,
    get_session, update_session, get_chunk, get_all_books,
    update_book, delete_book, update_book_cover, update_book_summary
)
from chunker import process_book
from embeddings import get_embeddings_batch
from adventure import generate_adventure_response
import edge_tts

load_dotenv()

# Edge TTS Configuration (Microsoft voices - free and fast)
# Voice mapping for different character types
VOICE_MAPPING = {
    "narrator": "en-US-GuyNeural",       # Deep male narrator
    "male": "en-US-ChristopherNeural",   # Male character
    "female": "en-US-JennyNeural",       # Female character
    "old_male": "en-US-RogerNeural",     # Older male
    "young_female": "en-US-AriaNeural",  # Young female
    "british": "en-GB-RyanNeural",       # British male
    "default": "en-US-GuyNeural"         # Fallback
}

# Available Edge TTS voices (English):
# en-US-GuyNeural, en-US-JennyNeural, en-US-AriaNeural, en-US-ChristopherNeural
# en-US-RogerNeural, en-GB-RyanNeural, en-GB-SoniaNeural, en-AU-WilliamNeural

app = FastAPI(title="The Rabbit Hole - Powered by NVIDIA Nemotron")

# Whimsical loading messages
LOADING_MESSAGES = {
    "reading": [
        "The White Rabbit is examining your manuscript...",
        "Curiouser and curiouser! Reading the pages...",
        "The Caterpillar puffs smoke rings while reading...",
    ],
    "chunking": [
        "The Mad Hatter is cutting the story into tea-time portions...",
        "Tweedledee and Tweedledum are dividing chapters...",
        "The Queen's cards are shuffling your paragraphs...",
    ],
    "embedding": [
        "The Cheshire Cat is memorizing passage {current} of {total}...",
        "Painting the roses red... I mean, encoding passage {current}...",
        "Down the rabbit hole with passage {current} of {total}...",
        "The Dormouse dreams of passage {current}...",
    ],
    "saving": [
        "The March Hare is filing everything in the tea cabinet...",
        "Storing memories in the Queen's garden...",
        "The looking glass remembers all...",
    ],
    "complete": [
        "Why, you're nothing but a pack of cards! ...Just kidding, we're ready!",
        "Off with their... wait, no. Your adventure awaits!",
        "The rabbit hole is open. Time for tea... I mean, adventure!",
    ]
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ensure covers directory exists
os.makedirs("static/covers", exist_ok=True)


async def generate_book_cover(book_id: str, title: str, first_chunk: str = ""):
    """Generate a professional book cover using Gemini via OpenRouter."""
    import base64

    if not OPENROUTER_API_KEY:
        logger.warning("No OPENROUTER_API_KEY set, skipping cover generation")
        return

    logger.info(f"=== GENERATING COVER for '{title}' ===")

    # Create a rich prompt for a professional book cover
    snippet = first_chunk[:300] if first_chunk else ""
    prompt = f"""Create a beautiful, professional book cover design for a book titled "{title}".

The book content begins: "{snippet}..."

Design requirements:
- Professional publishing quality, like edited by a top book designer
- Rich, atmospheric illustration that captures the book's mood and theme
- The title "{title}" should be prominently displayed in elegant typography
- Dark, moody color palette with rich contrast
- No author name needed
- Portrait orientation (book cover proportions)
- Think Penguin Classics or Folio Society quality"""

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-2.5-flash-image",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )

        if response.status_code != 200:
            logger.error(f"Cover generation failed: {response.status_code} - {response.text[:200]}")
            return

        data = response.json()

        # Extract image from response
        images = data.get("choices", [{}])[0].get("message", {}).get("images", [])
        if not images:
            logger.warning("No images in Gemini response")
            return

        image_url = images[0].get("image_url", {}).get("url", "")
        if not image_url or "," not in image_url:
            logger.warning("Invalid image URL format")
            return

        # Decode base64 image
        img_data = base64.b64decode(image_url.split(",", 1)[1])
        cover_path = f"static/covers/{book_id}.png"
        with open(cover_path, "wb") as f:
            f.write(img_data)

        # Update database with cover URL
        cover_url = f"/static/covers/{book_id}.png"
        update_book_cover(book_id, cover_url)

        logger.info(f"=== COVER GENERATED === Size: {len(img_data)} bytes, Path: {cover_path}")

    except Exception as e:
        logger.error(f"Cover generation error: {e}")


async def generate_book_summary(book_id: str, title: str, chunks: list):
    """Generate an engaging adventure summary and topic tags."""
    if not OPENROUTER_API_KEY:
        return

    logger.info(f"=== GENERATING SUMMARY + TAGS for '{title}' ===")

    # Grab beginning, middle and end to understand the full story
    sample_text = ""
    if chunks:
        sample_text += chunks[0][:600]
        if len(chunks) > 5:
            mid = len(chunks) // 2
            sample_text += "\n...\n" + chunks[mid][:600]
        if len(chunks) > 2:
            sample_text += "\n...\n" + chunks[-1][:600]

    prompt = f"""Analyze this book and respond with EXACTLY this JSON format, nothing else:

{{"summary": "your 2-sentence hook here", "tags": ["topic1", "topic2", "topic3"]}}

The book is "{title}". Here are excerpts:

{sample_text}

Rules for summary:
- Start with "Become..." or "Step into..." or "You are..."
- 2 sentences that make the reader the protagonist
- Vivid, exciting, like a movie trailer

Rules for tags:
- 3 main topics/themes of the book (e.g. "Sea Voyages", "Ancient Persia", "Treasure Hunting")
- Short, 1-3 words each
- About the book's actual content, not generic labels

Respond with ONLY the JSON object."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "nvidia/nemotron-3-nano-30b-a3b",
                    "messages": [
                        {"role": "system", "content": "You are a creative writing assistant. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500,
                    "min_tokens": 50
                }
            )

        if response.status_code != 200:
            logger.error(f"Summary generation failed: {response.status_code} - {response.text[:200]}")
            return

        data = response.json()
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning") or "").strip()
        logger.info(f"Content: {repr(raw[:200])}")
        logger.info(f"Reasoning (last 300): {repr(reasoning[-300:])}")

        # Try content first, then extract JSON from reasoning
        json_str = raw
        if not json_str:
            # Find JSON in reasoning
            import re
            json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', reasoning)
            if json_match:
                json_str = json_match.group(0)

        if not json_str:
            logger.warning("No JSON found in response")
            return

        # Clean markdown code blocks
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        parsed = json.loads(json_str)
        summary = parsed.get("summary", "").strip().strip('"')
        tags = parsed.get("tags", [])
        tags_str = json.dumps(tags) if tags else None

        if summary:
            update_book_summary(book_id, summary, tags_str)
            logger.info(f"=== SUMMARY + TAGS === {summary[:80]}... | Tags: {tags}")
        else:
            logger.warning("Summary was empty")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from summary response: {e}")
    except Exception as e:
        logger.error(f"Summary generation error: {e}")


async def extract_book_title(first_chunk: str, filename: str) -> dict:
    """Use Nemotron to extract the real book title and author from the text."""
    if not OPENROUTER_API_KEY:
        return {"title": os.path.splitext(filename)[0], "author": ""}

    # Use first ~1500 chars which typically contain title page info
    sample = first_chunk[:1500]

    prompt = f"""Look at this text from the beginning of a book file named "{filename}".
Extract the actual book title and author name.

Text:
{sample}

Respond with ONLY this JSON format, nothing else:
{{"title": "The Actual Book Title", "author": "Author Name"}}

Rules:
- Extract the REAL title from the text content, not the filename
- If you can identify the author, include it. If not, use empty string ""
- Clean up the title: proper capitalization, no extra whitespace
- If you truly cannot determine the title from the text, use the filename without extension"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "nvidia/nemotron-3-nano-30b-a3b",
                    "messages": [
                        {"role": "system", "content": "You extract book metadata. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "min_tokens": 10
                }
            )

        if response.status_code != 200:
            logger.error(f"Title extraction failed: {response.status_code}")
            return {"title": os.path.splitext(filename)[0], "author": ""}

        data = response.json()
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning") or "").strip()

        json_str = raw
        if not json_str:
            import re
            json_match = re.search(r'\{[^{}]*"title"[^{}]*\}', reasoning)
            if json_match:
                json_str = json_match.group(0)

        if json_str:
            result = json.loads(json_str)
            title = result.get("title", "").strip()
            author = result.get("author", "").strip()
            if title:
                logger.info(f"Extracted title: '{title}', author: '{author}'")
                return {"title": title, "author": author}

    except Exception as e:
        logger.error(f"Title extraction error: {e}")

    return {"title": os.path.splitext(filename)[0], "author": ""}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

class AdventureRequest(BaseModel):
    session_id: str
    user_input: Optional[str] = None

class TranscribeRequest(BaseModel):
    audio_data: str  # Base64 encoded audio

# OVH Whisper STT
OVH_WHISPER_URL = os.getenv("OVH_WHISPER_URL")
OVH_AI_TOKEN = os.getenv("OVH_AI_TOKEN")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_message(stage: str, current: int = 0, total: int = 0) -> str:
    """Get a random whimsical message for the stage."""
    messages = LOADING_MESSAGES.get(stage, ["Processing..."])
    msg = random.choice(messages)
    return msg.format(current=current, total=total)


@app.post("/api/upload")
async def upload_book(file: UploadFile = File(...)):
    """Upload and process a book file with streaming progress."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    content = await file.read()
    filename = file.filename

    async def generate_progress() -> AsyncGenerator[str, None]:
        try:
            # Stage 1: Reading
            logger.info("Stage: Reading file")
            yield json.dumps({
                "stage": "reading",
                "progress": 10,
                "message": get_message("reading"),
                "icon": "🐰"
            }) + "\n"
            await asyncio.sleep(0.3)

            # Generate book ID
            book_id = str(uuid.uuid4())[:8]
            title = os.path.splitext(filename)[0]
            author = ""

            # Stage 2: Chunking
            logger.info("Stage: Chunking")
            yield json.dumps({
                "stage": "chunking",
                "progress": 20,
                "message": get_message("chunking"),
                "icon": "🎩"
            }) + "\n"

            chunks = process_book(content, filename)
            logger.info(f"Created {len(chunks)} chunks")

            if not chunks:
                yield json.dumps({
                    "stage": "error",
                    "message": "Oh dear! The pages are blank!",
                    "icon": "😿"
                }) + "\n"
                return

            # Extract real title and author from the text
            yield json.dumps({
                "stage": "extracting_title",
                "progress": 25,
                "message": "Reading the title page...",
                "icon": "📖"
            }) + "\n"

            title_info = await extract_book_title(chunks[0], filename)
            title = title_info["title"]
            author = title_info["author"]
            logger.info(f"Book title: '{title}', author: '{author}'")

            yield json.dumps({
                "stage": "chunking_done",
                "progress": 30,
                "message": f"Found {len(chunks)} delicious story morsels!",
                "icon": "🍰",
                "chunks": len(chunks)
            }) + "\n"
            await asyncio.sleep(0.2)

            # Stage 3: Embeddings (with progress updates)
            logger.info("Stage: Embeddings")
            yield json.dumps({
                "stage": "embedding_start",
                "progress": 35,
                "message": "The Cheshire Cat begins memorizing...",
                "icon": "😸",
                "total": len(chunks)
            }) + "\n"

            # Process embeddings in batches and report progress
            batch_size = 5  # Smaller batches for faster feedback
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size

                progress = 35 + int((i / len(chunks)) * 45)

                logger.info(f"Embedding batch {batch_num}/{total_batches}")

                yield json.dumps({
                    "stage": "embedding",
                    "progress": progress,
                    "message": get_message("embedding", current=min(i + batch_size, len(chunks)), total=len(chunks)),
                    "icon": "😸" if batch_num % 2 == 0 else "🐱",
                    "current": min(i + batch_size, len(chunks)),
                    "total": len(chunks)
                }) + "\n"

                # Get embeddings for this batch
                batch_embeddings = await get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Batch {batch_num} complete")

            # Stage 4: Saving
            yield json.dumps({
                "stage": "saving",
                "progress": 85,
                "message": get_message("saving"),
                "icon": "🐇"
            }) + "\n"

            # Save to database
            save_book(book_id, title, filename, len(chunks), author=author)

            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                save_chunk(book_id, i, chunk, embedding)

            # Create session
            session_id = str(uuid.uuid4())[:8]
            create_session(session_id, book_id)

            yield json.dumps({
                "stage": "saving_done",
                "progress": 95,
                "message": "All memories safely stored in the looking glass!",
                "icon": "🪞"
            }) + "\n"
            await asyncio.sleep(0.3)

            # Stage 5: Complete
            yield json.dumps({
                "stage": "complete",
                "progress": 100,
                "message": get_message("complete"),
                "icon": "🎉",
                "success": True,
                "book_id": book_id,
                "session_id": session_id,
                "title": title,
                "author": author,
                "total_chunks": len(chunks)
            }) + "\n"

            # Fire and forget: generate book cover and summary in background
            asyncio.create_task(generate_book_cover(book_id, title, chunks[0] if chunks else ""))
            asyncio.create_task(generate_book_summary(book_id, title, chunks))

        except Exception as e:
            yield json.dumps({
                "stage": "error",
                "message": f"Oh my ears and whiskers! {str(e)}",
                "icon": "😱"
            }) + "\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/adventure")
async def adventure(request: AdventureRequest):
    """Handle adventure interactions."""
    session = get_session(request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    book_id = session["book_id"]
    current_chunk = session["current_chunk_index"]
    history = session["conversation_history"]

    is_start = len(history) == 0

    # Generate response
    result = await generate_adventure_response(
        book_id=book_id,
        current_chunk_index=current_chunk,
        user_input=request.user_input,
        conversation_history=history,
        is_start=is_start
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Update conversation history
    if request.user_input:
        history.append({"role": "user", "content": request.user_input})
    history.append({"role": "assistant", "content": result["response"]})

    # Keep history manageable
    if len(history) > 20:
        history = history[-20:]

    # Update session
    update_session(request.session_id, result["new_chunk_index"], history)

    return {
        "response": result["response"],
        "current_section": result["new_chunk_index"] + 1,
        "total_sections": result["total_chunks"],
        "book_title": result["book_title"]
    }


@app.post("/api/transcribe")
async def transcribe_audio(audio_data: str = Form(...)):
    """Transcribe audio using OVH Whisper."""
    import base64
    import tempfile
    import subprocess

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)

        if len(audio_bytes) < 100:
            return {"text": "", "error": "No audio recorded"}

        # Save webm to temp file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name

        # Convert webm to wav using ffmpeg (OVH Whisper expects wav)
        wav_path = webm_path.replace(".webm", ".wav")
        process = subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True,
            timeout=30,
        )

        # Clean up webm
        os.unlink(webm_path)

        if process.returncode != 0:
            return {"text": "", "error": "Audio conversion failed"}

        # Read wav file
        with open(wav_path, "rb") as f:
            wav_data = f.read()

        # Clean up wav
        os.unlink(wav_path)

        if len(wav_data) < 1000:
            return {"text": "", "error": "Audio too short"}

        # Send to OVH Whisper
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OVH_WHISPER_URL}/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {OVH_AI_TOKEN}",
                    "accept": "application/json",
                },
                files={"file": ("audio.wav", wav_data, "audio/wav")},
            )

        if response.status_code != 200:
            logger.error(f"OVH Whisper error: {response.status_code} - {response.text}")
            return {"text": "", "error": f"Transcription failed: {response.status_code}"}

        result = response.json()
        return {"text": result.get("text", "").strip()}

    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return {"text": "", "error": str(e)}


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    book_info = get_book_info(session["book_id"])

    return {
        "session_id": session_id,
        "book_id": session["book_id"],
        "book_title": book_info["title"] if book_info else "Unknown",
        "current_section": session["current_chunk_index"] + 1,
        "total_sections": book_info["total_chunks"] if book_info else 0
    }


@app.get("/api/books")
async def list_books():
    """Get all books in the library."""
    books = get_all_books()
    return {"books": books}


@app.post("/api/books/{book_id}/generate-cover")
async def generate_cover_endpoint(book_id: str):
    """Trigger cover generation for an existing book."""
    book_info = get_book_info(book_id)
    if not book_info:
        raise HTTPException(status_code=404, detail="Book not found")
    first_chunk = ""
    chunk = get_chunk(book_id, 0)
    if chunk:
        first_chunk = chunk.content
    asyncio.create_task(generate_book_cover(book_id, book_info["title"], first_chunk))
    return {"status": "generating", "message": "Cover generation started"}


@app.post("/api/books/{book_id}/generate-summary")
async def generate_summary_endpoint(book_id: str):
    """Trigger summary generation for an existing book."""
    from database import get_all_chunks
    book_info = get_book_info(book_id)
    if not book_info:
        raise HTTPException(status_code=404, detail="Book not found")
    all_chunks = get_all_chunks(book_id)
    chunk_texts = [c.content for c in all_chunks]
    asyncio.create_task(generate_book_summary(book_id, book_info["title"], chunk_texts))
    return {"status": "generating", "message": "Summary generation started"}


@app.post("/api/books/{book_id}/start")
async def start_book_session(book_id: str):
    """Start a new adventure session with an existing book."""
    book_info = get_book_info(book_id)
    if not book_info:
        raise HTTPException(status_code=404, detail="Book not found")

    session_id = str(uuid.uuid4())[:8]
    create_session(session_id, book_id)

    return {
        "session_id": session_id,
        "book_id": book_id,
        "title": book_info["title"],
        "total_chunks": book_info["total_chunks"]
    }


class BookUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "narrator"  # narrator, male, female, elder, young, mysterious
    emotion: Optional[str] = None  # Optional emotion/style prompt


@app.patch("/api/books/{book_id}")
async def update_book_info(book_id: str, data: BookUpdate):
    """Update book title or author."""
    book_info = get_book_info(book_id)
    if not book_info:
        raise HTTPException(status_code=404, detail="Book not found")

    update_book(book_id, title=data.title, author=data.author)
    return {"success": True, "book_id": book_id}


@app.delete("/api/books/{book_id}")
async def delete_book_endpoint(book_id: str):
    """Delete a book and all associated data."""
    book_info = get_book_info(book_id)
    if not book_info:
        raise HTTPException(status_code=404, detail="Book not found")

    delete_book(book_id)
    return {"success": True, "deleted": book_id}


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Edge TTS (Microsoft)."""
    import time
    import base64

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    # Get voice from mapping
    voice = VOICE_MAPPING.get(request.voice, VOICE_MAPPING["default"])

    char_count = len(request.text)
    logger.info(f"=== TTS REQUEST (Edge TTS) ===")
    logger.info(f"Voice: {voice}")
    logger.info(f"Characters: {char_count}")
    logger.info(f"Text preview: {request.text[:100]}...")

    start_time = time.time()

    try:
        # Generate audio using Edge TTS
        communicate = edge_tts.Communicate(request.text, voice)
        audio_data = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        elapsed = time.time() - start_time

        # Convert to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        logger.info(f"=== TTS COMPLETE ===")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Audio size: {len(audio_data)} bytes")

        return {
            "audio_data": audio_base64,
            "voice": voice,
            "text_length": char_count,
            "generation_time": round(elapsed, 2)
        }

    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.get("/api/voices")
async def list_voices():
    """List available TTS voices."""
    return {
        "voices": VOICE_MAPPING,
        "available": [
            {"id": "en-US-GuyNeural", "name": "Guy", "desc": "Deep male narrator"},
            {"id": "en-US-ChristopherNeural", "name": "Christopher", "desc": "Male character"},
            {"id": "en-US-JennyNeural", "name": "Jenny", "desc": "Female character"},
            {"id": "en-US-AriaNeural", "name": "Aria", "desc": "Young female"},
            {"id": "en-US-RogerNeural", "name": "Roger", "desc": "Older male"},
            {"id": "en-GB-RyanNeural", "name": "Ryan", "desc": "British male"}
        ]
    }


@app.get("/api/models")
async def list_models():
    """List all NVIDIA models powering this application."""
    return {
        "models": [
            {
                "component": "Story Generation (LLM)",
                "model": "nvidia/nemotron-3-nano-30b-a3b",
                "provider": "NVIDIA via OpenRouter",
                "params": "31.6B total, 3.6B active (MoE)",
                "context": "256K tokens",
                "architecture": "Hybrid Mamba-Transformer MoE"
            },
            {
                "component": "Semantic Embeddings",
                "model": "nvidia/nv-embedqa-e5-v5",
                "provider": "NVIDIA NIM API",
                "description": "Text embedding for semantic book chunk retrieval"
            },
            {
                "component": "Speech-to-Text",
                "model": "Whisper Large V3 Turbo",
                "provider": "OVH AI Endpoints",
                "description": "Open-source voice input transcription"
            },
            {
                "component": "Text-to-Speech",
                "model": "Microsoft Edge TTS",
                "provider": "edge-tts (open-source library)",
                "description": "High-quality voice narration"
            }
        ]
    }


# Mount static files (must be after API routes)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7501)
