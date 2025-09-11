from typing import List, Dict 
from sentence_transformers import SentenceTransformer 
from qdrant_client import QdrantClient 
from qdrant_client.http import models 
import uuid 
import json


class QdrantStore:
    def __init__(
        self,
        collection: str = "Episodic Form",
        user_id: str = None,
        story_id: str = None,
        model_name: str = r"C:\Users\saadn\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        host: str = "localhost",
        port: int = 6333
    ):
        self.collection = collection
        self.user_id = user_id
        self.story_id = story_id
        self.host = host
        self.port = port

        self.client = QdrantClient(host=host, port=port)

        # Reuse model if provided
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Only create if doesn't exist
        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )

    def with_namespace(self, namespace: str):
        """Return a new QdrantStore with same model but different collection."""
        return QdrantStore(
            collection=f"{self.collection}_{namespace}",
            user_id=self.user_id,
            story_id=self.story_id,
            model_name=self.model_name,
            host=self.host,
            port=self.port,
        )

    def _embed_text(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    # ---------- Upserts ----------
    def put(self, text: str, metadata: Dict = None):
        vec = self._embed_text(text)
        payload = metadata.copy() if metadata else {}
        payload.update({
            "text": text,
            "user_id": self.user_id,
            "story_id": self.story_id
        })

        self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)],
        )

    def put_dict_replace_character(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        for k, v in data.items():
            # Filter by user + story + character
            filter_conds = [
                models.FieldCondition(key="character_name", match=models.MatchValue(value=k)),
                models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
            ]
            search_results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(must=filter_conds),
                limit=1,
            )

            point_id = search_results[0].id if search_results else str(uuid.uuid4())
            vec = self._embed_text(v)

            payload = (metadata.copy() if metadata else {})
            payload.update({
                "key": k,
                "value": v,
                "text": v,
                "character_name": k,
                "user_id": self.user_id,
                "story_id": self.story_id
            })

            self.client.upsert(collection_name=self.collection, points=[models.PointStruct(id=point_id, vector=vec, payload=payload)])

    def put_dict_replace_world(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        for k, v in data.items():
            filter_conds = [
                models.FieldCondition(key="world_element", match=models.MatchValue(value=k)),
                models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
            ]
            search_results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(must=filter_conds),
                limit=1,
            )

            point_id = search_results[0].id if search_results else str(uuid.uuid4())
            vec = self._embed_text(v)

            payload = (metadata.copy() if metadata else {})
            payload.update({
                "key": k,
                "value": v,
                "text": v,
                "world_element": k,
                "user_id": self.user_id,
                "story_id": self.story_id
            })

            self.client.upsert(collection_name=self.collection, points=[models.PointStruct(id=point_id, vector=vec, payload=payload)])

    # ---------- Search ----------
    def search(self, query: str, k: int = 5, metadata: Dict[str, str] = None):
        vec = self._embed_text(query)

        # Always filter by current user + story
        must_conds = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
            models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id))
        ]

        # Exclude additional metadata if provided
        must_not = []
        if metadata:
            for key, val in metadata.items():
                must_not.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))

        search_filter = models.Filter(must=must_conds, must_not=must_not if must_not else None)

        results = self.client.search(collection_name=self.collection, query_vector=vec, limit=k, query_filter=search_filter)

        hits = []
        for r in results:
            payload = r.payload or {}
            base_text = payload.get("value") or payload.get("text") or ""
            merged = f"{payload.get('key', '')}: {base_text}" if "key" in payload else base_text
            ignore_keys = {"text", "value", "key", "user_id", "story_id"}
            if metadata:
                ignore_keys.update(metadata.keys())
            meta_parts = [f"{k}={v}" for k, v in payload.items() if k not in ignore_keys]
            if meta_parts:
                merged = f"{merged} | {'; '.join(meta_parts)}"
            merged = f"{merged} (score={r.score:.3f})"
            hits.append(merged)
        return hits

    # ---------- Chapter retrieval ----------
    def get_chapter_content(self, chapter_number: int) -> list[str]:
        all_texts = []
        offset = None

        while True:
            scroll_results, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    should=[  # OR condition for int/str chapter
                        models.FieldCondition(key="chapter_id", match=models.MatchValue(value=chapter_number)),
                        models.FieldCondition(key="chapter_id", match=models.MatchValue(value=str(chapter_number))),
                    ],
                    must=[
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                        models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id))
                    ]
                ),
                limit=100,
                offset=offset,
            )

            all_texts.extend(r.payload["text"] for r in scroll_results if "text" in r.payload)
            if next_offset is None:
                break
            offset = next_offset

        return all_texts
