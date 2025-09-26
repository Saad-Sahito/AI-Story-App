import os
from .shared_resources import SHARED_QDRANT


def delete_sqlite_db(db_path: str) -> bool:
    """
    Deletes the entire SQLite database file from disk.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        bool: True if deleted successfully, False otherwise.
    """
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"✅ Database deleted: {db_path}")
            return True
        else:
            print(f"⚠️ Database file not found: {db_path}")
            return False
    except Exception as e:
        print(f"❌ Error deleting database: {e}")
        return False
    
def delete_all_qdrant_collections() -> None:
    """
    Deletes all collections in the connected Qdrant instance.

    Args:
        qdrant_client: Qdrant client instance (e.g. SHARED_QDRANT).
    """
    try:
        collections = SHARED_QDRANT.get_collections().collections
        if not collections:
            print("⚠️ No collections found in Qdrant.")
            return

        for coll in collections:
            SHARED_QDRANT.delete_collection(coll.name)
            print(f"🗑️ Deleted collection: {coll.name}")
        print("✅ All Qdrant collections deleted.")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting Qdrant collections: {e}")
        return False
