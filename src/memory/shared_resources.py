import os
from pathlib import Path
from dotenv import load_dotenv
from .sqlite_store import SQLiteStore
from qdrant_client.async_qdrant_client import AsyncQdrantClient

# Load environment variables
load_dotenv()

# Database configuration
DEFAULT_DB_PATH = os.path.join(os.getcwd(), "data", "story_memory.db")

# Ensure data directory exists
Path(DEFAULT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

async def get_sqlite_store(table: str, user_id: str, story_id: str, db_path: str = None) -> SQLiteStore:
    """Factory function to create SQLite store instances."""
    return SQLiteStore(
        db_path=db_path or DEFAULT_DB_PATH,
        table=table,
        user_id=user_id,
        story_id=story_id
    )

SHARED_QDRANT = AsyncQdrantClient(
    url="http://localhost:6333", timeout=10.0
)


# # Initialize shared Supabase client
# SUPABASE_URL_ACQ = os.getenv("SUPABASE_URL")
# SUPABASE_KEY_ACQ = os.getenv("SUPABASE_KEY")
# if not SUPABASE_URL_ACQ or not SUPABASE_KEY_ACQ:
#     raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
# SHARED_SUPABASE: Client = create_client(SUPABASE_URL_ACQ, SUPABASE_KEY_ACQ)


# # Keep Qdrant if you're still using it for episodic memory
# QDRANT_URL_ACQ = os.getenv("QDRANT_URL")
# QDRANT_KEY_ACQ = os.getenv("QDRANT_API_KEY")

# SHARED_QDRANT = None
# if QDRANT_URL_ACQ and QDRANT_KEY_ACQ:
#     SHARED_QDRANT = QdrantClient(
#         url=QDRANT_URL_ACQ,
#         api_key=QDRANT_KEY_ACQ
#     )
# else:
#     print("Warning: Qdrant configuration not found. Episodic memory will not be available.")

