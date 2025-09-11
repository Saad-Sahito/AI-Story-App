from typing import Dict, Any, Optional, List
from pymongo import MongoClient
import uuid


class MongoStore:
    def __init__(
        self,
        db_name: str = "story_db",
        collection: str = "Long Form",
        host: str = "localhost",
        port: int = 27017,
    ):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection_name = collection
        self.collection = self.db[collection]

    # def with_namespace(self, namespace: str):
    #     """Return a new MongoStore with the same DB but a different collection."""
    #     return MongoStore(
    #         db_name=self.db.name,
    #         collection=f"{self.collection_name}_{namespace}",
    #         host=self.client.HOST,
    #         port=self.client.PORT,
    #     )

    # ---------- PUT ----------
    def put_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Append generic story text to a chapter (scene not used)."""
        chapter_id = (metadata or {}).get("chapter_id", "")

        self.collection.update_one(
            {"chapter_id": chapter_id},  # group by chapter only
            [
                {
                    "$set": {
                        "chapter_id": chapter_id,
                        "text": {
                            "$concat": [
                                {"$ifNull": ["$text", ""]},  # keep old or start empty
                                " ",
                                text
                            ]
                        }
                    }
                }
            ],
            upsert=True,  # create if doesn't exist
        )

    
    def put_characters_or_world(
    self,
    details_dict: Dict[str, str],
    scene_id: str,
    chapter_id: str,
):
        """
        Append new details to the existing details string,
        but always replace metadata with the latest.
        """
        for name, details in details_dict.items():
            self.collection.update_one(
                {"name": name},
                [
                    {
                        "$set": {
                            "metadata": {"scene_id": scene_id, "chapter_id": chapter_id},
                            "details": {
                                "$concat": [
                                    {"$ifNull": ["$details", ""]},  # keep old or start empty
                                    " ",
                                    details
                                ]
                            }
                        }
                    }
                ],
                upsert=True,
            )


    # ---------- GET ----------
    def get_texts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch generic story texts."""
        return list(self.collection.find({},).limit(limit))
    
    def get_text(self, name: str):
        return self.collection.find_one({"chapter_id": name})

    def get_character_or_world(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a specific character/world by name."""
        return self.collection.find_one({"name": name}, {"_id": 0})

    def get_all_characters_or_worlds(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all as a dict {name -> details}."""
        cursor = self.collection.find({}, {"_id": 0})
        return {doc["name"]: doc["details"] for doc in cursor if "name" in doc}
