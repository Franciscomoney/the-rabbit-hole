import sqlite3
import json
import os
from typing import List, Optional
from dataclasses import dataclass

DB_PATH = "wonderland.db"

@dataclass
class BookChunk:
    id: int
    book_id: str
    chunk_index: int
    content: str
    embedding: List[float]

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            total_chunks INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            author TEXT DEFAULT 'Unknown'
        )
    ''')

    # Migration: add author column if it doesn't exist
    try:
        c.execute('ALTER TABLE books ADD COLUMN author TEXT DEFAULT "Unknown"')
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add cover_url column if it doesn't exist
    try:
        c.execute('ALTER TABLE books ADD COLUMN cover_url TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add summary column if it doesn't exist
    try:
        c.execute('ALTER TABLE books ADD COLUMN summary TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add tags column if it doesn't exist
    try:
        c.execute('ALTER TABLE books ADD COLUMN tags TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists

    c.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding TEXT,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            book_id TEXT,
            current_chunk_index INTEGER DEFAULT 0,
            conversation_history TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    ''')

    c.execute('CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(book_id, chunk_index)')

    conn.commit()
    conn.close()

def save_book(book_id: str, title: str, filename: str, total_chunks: int, author: str = ""):
    """Save book metadata."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT OR REPLACE INTO books (id, title, filename, total_chunks, author) VALUES (?, ?, ?, ?, ?)',
        (book_id, title, filename, total_chunks, author or "")
    )
    conn.commit()
    conn.close()

def save_chunk(book_id: str, chunk_index: int, content: str, embedding: List[float]):
    """Save a text chunk with its embedding."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT INTO chunks (book_id, chunk_index, content, embedding) VALUES (?, ?, ?, ?)',
        (book_id, chunk_index, content, json.dumps(embedding))
    )
    conn.commit()
    conn.close()

def get_chunk(book_id: str, chunk_index: int) -> Optional[BookChunk]:
    """Get a specific chunk by index."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT id, book_id, chunk_index, content, embedding FROM chunks WHERE book_id = ? AND chunk_index = ?',
        (book_id, chunk_index)
    )
    row = c.fetchone()
    conn.close()

    if row:
        return BookChunk(
            id=row[0],
            book_id=row[1],
            chunk_index=row[2],
            content=row[3],
            embedding=json.loads(row[4])
        )
    return None

def get_chunks_range(book_id: str, start_index: int, count: int = 3) -> List[BookChunk]:
    """Get a range of chunks for context."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT id, book_id, chunk_index, content, embedding FROM chunks WHERE book_id = ? AND chunk_index >= ? AND chunk_index < ? ORDER BY chunk_index',
        (book_id, start_index, start_index + count)
    )
    rows = c.fetchall()
    conn.close()

    return [
        BookChunk(id=r[0], book_id=r[1], chunk_index=r[2], content=r[3], embedding=json.loads(r[4]))
        for r in rows
    ]

def get_all_chunks(book_id: str) -> List[BookChunk]:
    """Get all chunks for a book."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT id, book_id, chunk_index, content, embedding FROM chunks WHERE book_id = ? ORDER BY chunk_index',
        (book_id,)
    )
    rows = c.fetchall()
    conn.close()

    return [
        BookChunk(id=r[0], book_id=r[1], chunk_index=r[2], content=r[3], embedding=json.loads(r[4]))
        for r in rows
    ]

def get_book_info(book_id: str):
    """Get book metadata."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, title, filename, total_chunks, author, cover_url, summary, tags FROM books WHERE id = ?', (book_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {"id": row[0], "title": row[1], "filename": row[2], "total_chunks": row[3], "author": row[4], "cover_url": row[5], "summary": row[6], "tags": row[7]}
    return None

def get_all_books():
    """Get all books in the library."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, title, filename, total_chunks, created_at, author, cover_url, summary, tags FROM books ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()

    return [
        {"id": r[0], "title": r[1], "filename": r[2], "total_chunks": r[3], "created_at": r[4], "author": r[5], "cover_url": r[6], "summary": r[7], "tags": r[8]}
        for r in rows
    ]

def update_book_cover(book_id: str, cover_url: str):
    """Update book cover URL."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE books SET cover_url = ? WHERE id = ?', (cover_url, book_id))
    conn.commit()
    conn.close()

def update_book_summary(book_id: str, summary: str, tags: str = None):
    """Update book summary and tags."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if tags:
        c.execute('UPDATE books SET summary = ?, tags = ? WHERE id = ?', (summary, tags, book_id))
    else:
        c.execute('UPDATE books SET summary = ? WHERE id = ?', (summary, book_id))
    conn.commit()
    conn.close()

def update_book(book_id: str, title: str = None, author: str = None):
    """Update book metadata."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if title and author:
        c.execute('UPDATE books SET title = ?, author = ? WHERE id = ?', (title, author, book_id))
    elif title:
        c.execute('UPDATE books SET title = ? WHERE id = ?', (title, book_id))
    elif author:
        c.execute('UPDATE books SET author = ? WHERE id = ?', (author, book_id))

    conn.commit()
    conn.close()

def delete_book(book_id: str):
    """Delete a book and all its chunks and sessions."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM chunks WHERE book_id = ?', (book_id,))
    c.execute('DELETE FROM sessions WHERE book_id = ?', (book_id,))
    c.execute('DELETE FROM books WHERE id = ?', (book_id,))
    conn.commit()
    conn.close()

def create_session(session_id: str, book_id: str):
    """Create a new adventure session."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT INTO sessions (id, book_id, current_chunk_index, conversation_history) VALUES (?, ?, 0, ?)',
        (session_id, book_id, '[]')
    )
    conn.commit()
    conn.close()

def get_session(session_id: str):
    """Get session data."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT id, book_id, current_chunk_index, conversation_history FROM sessions WHERE id = ?',
        (session_id,)
    )
    row = c.fetchone()
    conn.close()

    if row:
        return {
            "id": row[0],
            "book_id": row[1],
            "current_chunk_index": row[2],
            "conversation_history": json.loads(row[3])
        }
    return None

def update_session(session_id: str, chunk_index: int, conversation_history: list):
    """Update session progress."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'UPDATE sessions SET current_chunk_index = ?, conversation_history = ? WHERE id = ?',
        (chunk_index, json.dumps(conversation_history), session_id)
    )
    conn.commit()
    conn.close()

# Initialize database on import
init_db()
