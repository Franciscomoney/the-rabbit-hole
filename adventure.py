import httpx
import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from database import get_chunks_range, get_book_info, get_all_chunks
from embeddings import get_embedding, cosine_similarity

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NEMOTRON_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

SYSTEM_PROMPT = """You are the Narrator of an interactive story adventure. You MUST use the book content provided to create the story.

YOUR TASK:
- Read the BOOK CONTENT provided
- Create an immersive narrative BASED ON that book content
- Use characters, settings, and events FROM THE BOOK
- Write in second person ("You...")

OUTPUT FORMAT (you MUST follow this exactly):
Write 3 paragraphs of narrative (3-4 sentences each), then 3 numbered choices.

Paragraph 1: Describe the scene using details from the book.
Paragraph 2: Build atmosphere and tension from the book's themes.
Paragraph 3: Present the situation the reader faces.

Then give exactly 3 choices:
1. [action choice]
2. [action choice]
3. [action choice]

IMPORTANT: Your narrative MUST reference specific elements from the book content. Do not make up unrelated stories."""

async def generate_adventure_response(
    book_id: str,
    current_chunk_index: int,
    user_input: Optional[str],
    conversation_history: List[Dict],
    is_start: bool = False
) -> Dict:
    """
    Generate the next part of the adventure.

    Returns:
        Dict with 'response', 'new_chunk_index', and 'choices'
    """
    book_info = get_book_info(book_id)
    if not book_info:
        return {"error": "Book not found"}

    total_chunks = book_info["total_chunks"]

    # Get relevant chunks for context
    context_chunks = get_chunks_range(book_id, current_chunk_index, count=3)

    if not context_chunks:
        return {"error": "No content available"}

    # Build context from chunks
    book_context = "\n\n".join([chunk.content for chunk in context_chunks])

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add book context as user message for better attention
    context_message = f"""Here is the BOOK CONTENT you must use (Section {current_chunk_index + 1} of {total_chunks}):

---
{book_context}
---

INSTRUCTIONS: Create your narrative using the characters, settings, and events from this passage. Reference specific details from the text above."""

    messages.append({"role": "user", "content": context_message})
    messages.append({"role": "assistant", "content": "I understand. I will create an immersive narrative based on this book content, using its characters and settings."})

    # Add conversation history (last 6 exchanges max)
    for msg in conversation_history[-12:]:
        messages.append(msg)

    # Add user input or start prompt
    if is_start:
        messages.append({
            "role": "user",
            "content": "Begin the adventure! Introduce the story and give me my first choices."
        })
    elif user_input:
        messages.append({"role": "user", "content": user_input})
    else:
        messages.append({
            "role": "user",
            "content": "Continue the story and give me new choices."
        })

    # Call Nemotron via OpenRouter with retry for short responses
    import logging
    logger = logging.getLogger(__name__)

    # Log what we're sending
    logger.info(f"=== NEMOTRON REQUEST ===")
    logger.info(f"User input: {user_input}")
    logger.info(f"Is start: {is_start}")
    logger.info(f"Current chunk: {current_chunk_index}")
    logger.info(f"History length: {len(conversation_history)}")
    logger.info(f"Book context length: {len(book_context)} chars")
    logger.info(f"Book context preview: {book_context[:200]}...")

    MIN_RESPONSE_LENGTH = 500  # Minimum acceptable response length
    MAX_RETRIES = 2
    assistant_response = None

    async with httpx.AsyncClient(timeout=60.0) as client:
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
                    "temperature": 0.7,
                    "max_tokens": 1500,
                    "min_tokens": 400  # Request minimum tokens
                }
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            logger.info(f"=== NEMOTRON RESPONSE (attempt {attempt + 1}) ===")
            logger.info(f"Content length: {len(content) if content else 0} chars")
            logger.info(f"Content: {content[:200] if content else 'EMPTY'}...")

            if content and len(content.strip()) >= MIN_RESPONSE_LENGTH:
                assistant_response = content
                break
            elif attempt < MAX_RETRIES:
                logger.warning(f"Response too short ({len(content) if content else 0} chars), retrying...")
            else:
                # Use whatever we got on last attempt
                assistant_response = content

    # Handle empty or still too short response with fallback
    if not assistant_response or len(assistant_response.strip()) < 100:
        logger.warning("Response still too short after retries, using fallback")
        assistant_response = """You stand at the beginning of a mysterious adventure. The pages of the story beckon you forward into the unknown. The air is thick with anticipation as you prepare to step into this world of wonder and possibility.

The atmosphere around you shifts and changes, filled with the promise of discovery. Every detail seems significant, every shadow holds a secret waiting to be revealed. You sense that great adventures await those brave enough to seek them.

Before you lie multiple paths, each leading to different fates and fortunes. The choice is yours to make, and yours alone. What will you do?

1. Step forward into the unknown
2. Look around and observe your surroundings
3. Listen carefully for any sounds or clues"""

    # Determine if we should advance to next chunk
    new_chunk_index = current_chunk_index

    # Advance chunk based on user choices (simplified logic)
    if user_input and not is_start:
        # Check if user wants to move forward
        forward_keywords = ["continue", "forward", "next", "proceed", "go on", "1", "2", "3"]
        if any(kw in user_input.lower() for kw in forward_keywords):
            if new_chunk_index < total_chunks - 1:
                new_chunk_index = min(current_chunk_index + 1, total_chunks - 1)

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

    # Find most similar chunk
    best_match = None
    best_score = -1

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.embedding)
        if score > best_score:
            best_score = score
            best_match = chunk.chunk_index

    return best_match if best_score > 0.3 else None
