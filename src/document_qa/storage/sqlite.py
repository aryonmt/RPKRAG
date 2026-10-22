import sqlite3
from pathlib import Path
from document_qa.core.config import settings

def get_db_connection() -> sqlite3.Connection:
    """Establishes a connection to SQLite with required PRAGMAs."""
    # Ensure the directory exists
    db_path = Path(settings.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def init_db() -> None:
    """Creates the required tables and FTS5 virtual table if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check FTS5 support before proceeding
    try:
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS test_fts USING fts5(a);")
        cursor.execute("DROP TABLE IF EXISTS test_fts;")
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"FTS5 is not supported by this SQLite build: {e}")

    # Create tables
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            title TEXT,
            file_type TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            generation INTEGER NOT NULL DEFAULT 1,
            page_count INTEGER,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            warnings TEXT NOT NULL DEFAULT '[]',
            error_code TEXT,
            error_message TEXT,
            cleanup_pending INTEGER NOT NULL DEFAULT 0,
            raw_file_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        -- A document hash should be unique within a conversation if it's not deleting
        CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_hash ON documents(conversation_id, file_hash) WHERE status != 'deleting';

        CREATE TABLE IF NOT EXISTS chunks (
            node_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            breadcrumb TEXT,
            original_text TEXT NOT NULL,
            search_text TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            node_id UNINDEXED,
            body_search,
            breadcrumb_search,
            title_search,
            tokenize='unicode61'
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            standalone_question TEXT,
            status TEXT,
            answer_text TEXT,
            missing TEXT,
            citations TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully.")