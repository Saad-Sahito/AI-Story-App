from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import os


class QdrantStore:
    def __init__(
        self,
        collection: str = "episodic_story_memory",
        user_id: str = None,
        story_id: str = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        namespace: str = None,
    ):
        self.collection = collection
        self.user_id = user_id
        self.story_id = story_id
        self.namespace = namespace  # ✅ store namespace

        load_dotenv()

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Reuse model if provided
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # --- Ensure collection exists ---
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                on_disk_payload=True,
            )

        # --- Ensure payload indexes exist ---
        required_indexes = {
            "user_id": models.PayloadSchemaType.KEYWORD,
            "story_id": models.PayloadSchemaType.KEYWORD,
            "namespace": models.PayloadSchemaType.KEYWORD,
            "character_name": models.PayloadSchemaType.KEYWORD,
            "world_element": models.PayloadSchemaType.KEYWORD,
            "chapter_id": models.PayloadSchemaType.INTEGER,
        }

        for field, schema in required_indexes.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as e:
                # Skip if index already exists
                if "already exists" not in str(e):
                    raise

    def with_namespace(self, namespace: str):
        """Return a new store bound to a namespace (same collection)."""
        return QdrantStore(
            collection=self.collection,
            user_id=self.user_id,
            story_id=self.story_id,
            model_name=self.model_name,
            namespace=namespace,
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
            "story_id": self.story_id,
            "namespace": self.namespace,   # ✅ add namespace
        })

        self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)],
        )

    def put_dict_replace_character(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        for k, v in data.items():
            filter_conds = [
                models.FieldCondition(key="character_name", match=models.MatchValue(value=k)),
                models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
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
                "story_id": self.story_id,
                "namespace": self.namespace,
            })

            self.client.upsert(collection_name=self.collection, points=[models.PointStruct(id=point_id, vector=vec, payload=payload)])

    def put_dict_replace_world(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        for k, v in data.items():
            filter_conds = [
                models.FieldCondition(key="world_element", match=models.MatchValue(value=k)),
                models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
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
                "story_id": self.story_id,
                "namespace": self.namespace,
            })

            self.client.upsert(collection_name=self.collection, points=[models.PointStruct(id=point_id, vector=vec, payload=payload)])

    # ---------- Search ----------(brings all other points apart from the excluded metadata)
    def search(self, query: str, k: int = 5, metadata: Dict[str, str] = None):
        vec = self._embed_text(query)

        must_conds = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
            models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
            models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),  # ✅ namespace filter
        ]

        must_not_conds = []  # ❌ Exclusion conditions

        if metadata:
            for key, val in metadata.items():
                must_not_conds.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))

        search_filter = models.Filter(
            must=must_conds,
            must_not=must_not_conds  # ✅ Exclude given metadata instead of including it
        )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=k,
            query_filter=search_filter
        )

        hits = []
        for r in results:
            payload = r.payload or {}
            base_text = payload.get("value") or payload.get("text") or ""
            merged = f"{payload.get('key', '')}: {base_text}" if "key" in payload else base_text

            ignore_keys = {"text", "value", "key", "user_id", "story_id", "namespace"}
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
                    must=[
                        models.FieldCondition(key="chapter_id", match=models.MatchValue(value=chapter_number)),  # ✅ works now
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                        models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                        models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
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
