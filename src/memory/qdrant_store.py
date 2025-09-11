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
        model_name: str = r"C:\Users\saadn\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        host: str = "localhost",
        port: int = 6333,
        model: SentenceTransformer = None,
    ):
        self.collection = collection
        self.host = host
        self.port = port

        self.client = QdrantClient(host=host, port=port)

        # Either reuse a provided model or create a new one
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # only create if it doesn't already exist
        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )

    def with_namespace(self, namespace: str):
        """Return a new QdrantStore with same model but different collection."""
        return QdrantStore(
            collection=f"{self.collection}_{namespace}",
            model_name=self.model_name,
            host=self.host,
            port=self.port,
            model=self.model,   # reuse the same loaded model instance
        )

    def _embed_text(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    def put(self, text: str, metadata: Dict = None):
        vec = self._embed_text(text)
        payload = metadata.copy() if metadata else {}
        payload["text"] = text

        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                )
            ],
        )

    def put_dict_replace_character(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        data_dict = self._ensure_dict(data)
        for k, v in data_dict.items():
            search_results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="character_name", match=models.MatchValue(value=k))]
                ),
                limit=1,
            )

            point_id = search_results[0].id if search_results else str(uuid.uuid4())
            vec = self._embed_text(v)

            payload = (metadata.copy() if metadata else {})
            payload.update({
                "key": k,
                "value": v,
                "text": v,
                "character_name": k
            })

            point = models.PointStruct(id=point_id, vector=vec, payload=payload)
            self.client.upsert(collection_name=self.collection, points=[point])
    
    def put_dict_replace_world(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        data_dict = self._ensure_dict(data)
        for k, v in data_dict.items():
            search_results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="world_element", match=models.MatchValue(value=k))]
                ),
                limit=1,
            )

            point_id = search_results[0].id if search_results else str(uuid.uuid4())
            vec = self._embed_text(v)

            payload = (metadata.copy() if metadata else {})
            payload.update({
                "key": k,
                "value": v,
                "text": v,
                "world_element": k
            })

            point = models.PointStruct(id=point_id, vector=vec, payload=payload)
            self.client.upsert(collection_name=self.collection, points=[point])


    def _ensure_dict(self, data):
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            return json.loads(data)
        else:
            raise TypeError(f"Unsupported type for data: {type(data)}")

    def search(self, query: str, k: int = 5, metadata: Dict[str, str] = None):
        """
        Semantic search in this collection, returning merged strings.
        Excludes results matching metadata (e.g. chapter_id).
        """
        vec = self._embed_text(query)

        # Build filter if metadata provided (exclude matches)
        search_filter = None
        if metadata:
            conditions = []
            for key, val in metadata.items():
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=val))
                )
            search_filter = models.Filter(must_not=conditions)

        # Run vector search
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=k,
            query_filter=search_filter,
        )

        hits = []
        for r in results:
            payload = r.payload or {}

            # Prefer "value" if present (characters/world), else fallback to "text"
            base_text = payload.get("value") or payload.get("text") or ""

            # If a key exists, prepend it
            if "key" in payload:
                merged = f"{payload['key']}: {base_text}"
            else:
                merged = base_text

            # Add metadata except ignored keys
            ignore_keys = {"text", "value", "key"}
            if metadata:
                ignore_keys.update(metadata.keys())  # also ignore excluded fields

            meta_parts = []
            for k, v in payload.items():
                if k not in ignore_keys:
                    meta_parts.append(f"{k}={v}")

            if meta_parts:
                merged = f"{merged} | {'; '.join(meta_parts)}"

            merged = f"{merged} (score={r.score:.3f})"
            hits.append(merged)

        return hits



    def get_chapter_content(self, chapter_number: int) -> list[str]:
        """Return all 'text' fields from payloads where 'chapter_id' matches,
        handling both int and str types in Qdrant.
        """
        all_texts = []
        offset = None

        while True:
            scroll_results, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    should=[  # OR condition → try int and str match
                        models.FieldCondition(
                            key="chapter_id",
                            match=models.MatchValue(value=chapter_number)
                        ),
                        models.FieldCondition(
                            key="chapter_id",
                            match=models.MatchValue(value=str(chapter_number))
                        ),
                    ]
                ),
                limit=100,
                offset=offset,
            )

            # collect only 'text' fields
            all_texts.extend(
                r.payload["text"] for r in scroll_results if "text" in r.payload
            )
            #print(f"SCROLLED {len(scroll_results)} POINTS...")
            if next_offset is None:
                break
            offset = next_offset
        #print(f"FOUND {len(all_texts)} TEXTS FOR CHAPTER {chapter_number}")
        return all_texts

