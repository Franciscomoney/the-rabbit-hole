import httpx
import os
import json
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from database import get_chunks_range, get_book_info, get_all_chunks
from embeddings import get_embedding, cosine_similarity

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NEMOTRON_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

SYSTEM_PROMPT = """You are the Narrator of a choose-your-own-adventure book. You MUST use the book content provided to create the story.

RULES:
- Read the BOOK CONTENT provided and create narrative BASED ON it
- Use characters, settings, and events FROM THE BOOK
- Write in second person ("You...")
- Each choice you offer must lead to a DIFFERENT direction in the story
- Choice 1 should push the story forward (action/courage)
- Choice 2 should explore sideways (investigation/curiosity)
- Choice 3 should pull back or take a cautious/unexpected path

OUTPUT FORMAT (follow this exactly):
Write 3 paragraphs of narrative (3-4 sentences each), then exactly 3 numbered choices.

1. [bold action — advances the main plot]
2. [explore/investigate — uncovers details or side content]
3. [cautious/retreat — takes a different path through the story]

IMPORTANT: Your narrative MUST reference specific elements from the book content. Each choice must be meaningfully different — they determine which part of the book the reader experiences next."""

async def find_branching_chunks(book_id: str, choice_text: str, current_chunk: int, total_chunks: int) -> Dict[str, int]:
    """Use semantic search to find the best chunk for a given choice.

    Returns the chunk index that best matches the choice semantically,
    ensuring we don't go backwards and picks from different parts of the book.
    """
    choice_embedding = await get_embedding(choice_text)
    all_chunks = get_all_chunks(book_id)

    if not all_chunks:
        return current_chunk + 1

    # Score all chunks ahead of current position
    scored = []
    for chunk in all_chunks:
        # Only consider chunks ahead (or at least not too far behind)
        if chunk.chunk_index <= current_chunk:
            continue
        score = cosine_similarity(choice_embedding, chunk.embedding)
        scored.append((chunk.chunk_index, score))

    if not scored:
        # We're at the end, wrap or stay
        return min(current_chunk + 1, total_chunks - 1)

    # Sort by similarity score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return the best matching chunk
    return scored[0][0]


async def find_three_paths(book_id: str, choices: List[str], current_chunk: int, total_chunks: int) -> List[int]:
    """Find 3 different chunk destinations for the 3 choices.

    Each choice maps to a different part of the book via semantic similarity.
    Ensures the 3 paths don't all land on the same chunk.
    """
    if not choices or total_chunks <= 1:
        return [min(current_chunk + 1, total_chunks - 1)] * 3

    all_chunks = get_all_chunks(book_id)
    if not all_chunks:
        return [min(current_chunk + 1, total_chunks - 1)] * 3

    # Only consider chunks ahead of current position
    future_chunks = [c for c in all_chunks if c.chunk_index > current_chunk]

    if not future_chunks:
        return [min(current_chunk + 1, total_chunks - 1)] * 3

    # Get embeddings for all 3 choices
    choice_embeddings = []
    for choice in choices[:3]:
        emb = await get_embedding(choice)
        choice_embeddings.append(emb)

    # For each choice, score all future chunks
    paths = []
    used_chunks = set()

    for i, choice_emb in enumerate(choice_embeddings):
        scored = []
        for chunk in future_chunks:
            score = cosine_similarity(choice_emb, chunk.embedding)
            scored.append((chunk.chunk_index, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Pick the best chunk that hasn't been used by another choice
        picked = None
        for chunk_idx, score in scored:
            if chunk_idx not in used_chunks:
                picked = chunk_idx
                used_chunks.add(chunk_idx)
                break

        if picked is None:
            # All good chunks taken, just go sequential
            picked = min(current_chunk + 1 + i, total_chunks - 1)

        paths.append(picked)

    logger.info(f"=== BRANCHING PATHS === Current: {current_chunk}, Paths: {paths}")
    return paths


async def generate_adventure_response(
    book_id: str,
    current_chunk_index: int,
    user_input: Optional[str],
    conversation_history: List[Dict],
    is_start: bool = False
) -> Dict:
    """Generate the next part of the adventure with real branching."""
    book_info = get_book_info(book_id)
    if not book_info:
        return {"error": "Book not found"}

    total_chunks = book_info["total_chunks"]

    # Determine which chunk to use based on user's choice
    new_chunk_index = current_chunk_index

    if user_input and not is_start:
        # Extract the choice text (format: "1. choice text" or "2. choice text")
        choice_text = user_input

        # Use semantic search to find the most relevant passage for this choice
        logger.info(f"=== BRANCHING === Searching for passage matching: {choice_text[:80]}...")
        target_chunk = await find_branching_chunks(
            book_id, choice_text, current_chunk_index, total_chunks
        )
        new_chunk_index = target_chunk
        logger.info(f"=== BRANCHING === Choice led to chunk {new_chunk_index} (was at {current_chunk_index})")

    # Get context from the target chunk (and neighbors for richer context)
    context_chunks = get_chunks_range(book_id, new_chunk_index, count=2)

    if not context_chunks:
        return {"error": "No content available"}

    book_context = "\n\n".join([chunk.content for chunk in context_chunks])

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    context_message = f"""Here is the BOOK CONTENT you must use (Section {new_chunk_index + 1} of {total_chunks}):

---
{book_context}
---

INSTRUCTIONS: Create your narrative using the characters, settings, and events from this passage. Reference specific details from the text above. Your 3 choices must lead to genuinely different directions in the story."""

    messages.append({"role": "user", "content": context_message})
    messages.append({"role": "assistant", "content": "I understand. I will create an immersive choose-your-own-adventure narrative based on this book content, with 3 meaningfully different choices."})

    # Add conversation history (last 6 exchanges max)
    for msg in conversation_history[-12:]:
        messages.append(msg)

    if is_start:
        messages.append({
            "role": "user",
            "content": "Begin the adventure! Set the scene and give me my first 3 choices. Remember: each choice should lead to a different part of the story."
        })
    elif user_input:
        messages.append({"role": "user", "content": f"I chose: {user_input}\n\nContinue the story from this choice. Show me what happens and give me 3 new choices."})
    else:
        messages.append({
            "role": "user",
            "content": "Continue the story and give me 3 new choices."
        })

    # Call Nemotron Super 49B via OpenRouter
    logger.info(f"=== NEMOTRON SUPER REQUEST ===")
    logger.info(f"Model: {NEMOTRON_MODEL}")
    logger.info(f"User input: {user_input}")
    logger.info(f"Is start: {is_start}")
    logger.info(f"Current chunk: {current_chunk_index} -> {new_chunk_index}")
    logger.info(f"History length: {len(conversation_history)}")

    MIN_RESPONSE_LENGTH = 400
    MAX_RETRIES = 2
    assistant_response = None

    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(MAX_RETRIES + 1):
            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": NEMOTRON_MODEL,
                    "messages": messages,
                    "temperature": 0.75,
                    "max_tokens": 1500
                }
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            logger.info(f"=== NEMOTRON RESPONSE (attempt {attempt + 1}) ===")
            logger.info(f"Content length: {len(content) if content else 0} chars")
            logger.info(f"Content preview: {content[:200] if content else 'EMPTY'}...")

            if content and len(content.strip()) >= MIN_RESPONSE_LENGTH:
                assistant_response = content
                break
            elif attempt < MAX_RETRIES:
                logger.warning(f"Response too short ({len(content) if content else 0} chars), retrying...")
            else:
                assistant_response = content

    if not assistant_response or len(assistant_response.strip()) < 100:
        logger.warning("Response still too short after retries, using fallback")
        assistant_response = """You stand at a crossroads in the story. The pages of this tale stretch out before you in every direction, each path promising a different adventure. The world described in these pages pulses with life, waiting for your decision.

The air is thick with possibility. Characters you've met linger at the edges of your vision, their fates intertwined with yours. Every choice you make here will send ripples through the narrative, changing what comes next.

Three paths diverge before you. Each leads deeper into the story, but through entirely different doors. Where will you go?

1. Push forward boldly into the heart of the action
2. Investigate the mysteries lurking in the shadows
3. Take the unexpected path — retreat and find another way"""

    return {
        "response": assistant_response,
        "new_chunk_index": new_chunk_index,
        "total_chunks": total_chunks,
        "book_title": book_info["title"]
    }


async def search_relevant_passage(book_id: str, query: str) -> Optional[int]:
    """Search for a relevant passage based on user query."""
    query_embedding = await get_embedding(query)
    chunks = get_all_chunks(book_id)

    if not chunks:
        return None

    best_match = None
    best_score = -1

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.embedding)
        if score > best_score:
            best_score = score
            best_match = chunk.chunk_index

    return best_match if best_score > 0.3 else None
