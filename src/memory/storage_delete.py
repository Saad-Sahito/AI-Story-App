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



async def delete_all_qdrant_collections() -> bool:
    """
    Deletes all collections in the connected Qdrant instance.
    """
    try:
        collections_response = await SHARED_QDRANT.get_collections()
        collections = collections_response.collections

        if not collections:
            print("⚠️ No collections found in Qdrant.")
            return False

        for coll in collections:
            await SHARED_QDRANT.delete_collection(coll.name)
            print(f"🗑️ Deleted collection: {coll.name}")

        print("✅ All Qdrant collections deleted.")
        return True

    except Exception as e:
        print(f"❌ Error deleting Qdrant collections: {e}")
        return False
