from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
import uuid
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
#from qdrant_client.async_qdrant_client import AsyncQdrantClient
from .shared_resources import SHARED_QDRANT  # Import the shared client
from contextlib import asynccontextmanager
from setup.shared_redis_pool import get_redis_client
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio import Semaphore
import asyncio
import time
from fastapi import HTTPException
import redis.asyncio as redis
#from src.utilities.redis_lock import with_redis_lock



@asynccontextmanager
async def redis_lock(client, lock_key, timeout=30, retries=10, retry_delay=1.0):
    lock_value = str(time.time())
    acquired = False
    try:
        for attempt in range(retries):
            try:
                acquired = await client.set(lock_key, lock_value, nx=True, ex=timeout)
                if acquired:
                    print(f"✅ Acquired lock for {lock_key} at {time.time()}")
                    break
                ttl = await client.ttl(lock_key)
                if ttl == -1:  # Stale lock with no TTL
                    print(f"🔍 Detected stale lock with no TTL for {lock_key}, removing")
                    await client.delete(lock_key)
                elif ttl == -2:  # Lock doesn't exist
                    print(f"🔍 Lock {lock_key} does not exist, retrying")
                else:
                    print(f"🔍 Lock acquisition failed for {lock_key} at {time.time()}, attempt {attempt + 1}/{retries}, TTL={ttl}")
                await asyncio.sleep(retry_delay)
            except redis.RedisError as e:
                print(f"❌ Redis error during lock acquisition for {lock_key}: {e}")
                await asyncio.sleep(retry_delay)
        if not acquired:
            raise HTTPException(status_code=503, detail=f"Could not acquire lock for {lock_key} after {retries} attempts")
        
        yield
        
    finally:
        if acquired:
            lua = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            try:
                result = await client.eval(lua, 1, lock_key, lock_value)
                if result == 1:
                    print(f"✅ Released lock for {lock_key} at {time.time()}")
                else:
                    print(f"🔍 Lock {lock_key} not released: different lock value or already expired")
            except redis.RedisError as e:
                print(f"❌ Failed to release lock for {lock_key}: {e}")
                # Attempt to remove stale lock if it matches our value
                try:
                    current_value = await client.get(lock_key)
                    if current_value == lock_value.encode():
                        await client.delete(lock_key)
                        print(f"✅ Forcibly released stale lock for {lock_key}")
                    else:
                        print(f"🔍 Lock {lock_key} not forcibly released: different lock value")
                except redis.RedisError as e:
                    print(f"❌ Failed to forcibly release lock for {lock_key}: {e}")
    
class QdrantStore:
    _shared_model = None  # Global singleton for SentenceTransformer
    collection_init_lock = asyncio.Lock()
    def __init__(self, collection: str = "episodic_story_memory", user_id: str = None, 
                   story_id: str = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                   namespace: str = None, client=None):
        self.collection = collection
        self.user_id = user_id
        self.story_id = story_id
        self.namespace = namespace or "default"
        self.client = client or SHARED_QDRANT
        if QdrantStore._shared_model is None:
            QdrantStore._shared_model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.model = QdrantStore._shared_model
        self.dim = self.model.get_sentence_embedding_dimension()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.request_semaphore = Semaphore(20)

    async def async_init(self):
        # Ensure collection exists
        async with QdrantStore.collection_init_lock:
            collections = await self.client.get_collections()
            existing_collections = [c.name for c in collections.collections]
            if self.collection not in existing_collections:
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                    on_disk_payload=True,
                )

            # Ensure payload indexes exist
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
                    await self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception as e:
                    if "already exists" not in str(e):
                        raise

    # @classmethod
    # def clear_shared_models(cls):
    #     """Force clear all cached models to free memory."""
    #     cls._shared_models.clear()
    #     import gc; gc.collect()

    def with_namespace(self, namespace: str):
        store = QdrantStore(
            collection=self.collection,
            user_id=self.user_id,
            story_id=self.story_id,
            model_name=self.model_name,
            namespace=namespace,
            client=self.client,
        )
        store.model = self.model
        return store

    async def _embed_text(self, text: str) -> List[float]:
        truncated = text[:512]
        loop = get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode([truncated], convert_to_numpy=True)[0].tolist()
        )

    # ---------- Upserts ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def put(self, text: str, metadata: Dict = None):
        async with self.request_semaphore:
            vec = await self._embed_text(text)
            #print(f"Embedding length: {len(vec)}")  # should match self.dim
            #print(f"Embedding dimension: {self.dim}")
            payload = metadata.copy() if metadata else {}
            payload.update({
                "text": text,
                "user_id": self.user_id,
                "story_id": self.story_id,
                "namespace": self.namespace,   # ✅ add namespace
            })

            await self.client.upsert(
                collection_name=self.collection,
                points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)],
            )



    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def put_dict_replace_character(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        client = await get_redis_client()  # Assume this returns an async Redis client
        async with self.request_semaphore:
            for k, v in data.items():
                lock_key = f"lock:qdrant:{self.user_id}:{self.story_id}:{self.namespace}:character:{k}"
                
                async with redis_lock(client=client, lock_key=lock_key):
                    filter_conds = [
                        models.FieldCondition(key="character_name", match=models.MatchValue(value=k)),
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                        models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                        models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
                    ]
                    search_results, _ = await self.client.scroll(
                        collection_name=self.collection,
                        scroll_filter=models.Filter(must=filter_conds),
                        limit=1,
                    )
                    point_id = search_results[0].id if search_results else str(uuid.uuid4())
                    vec = await self._embed_text(v)
                    payload = (metadata.copy() if metadata else {})
                    payload.update({
                        "key": k, "value": v, "text": v, "character_name": k,
                        "user_id": self.user_id, "story_id": self.story_id, "namespace": self.namespace,
                    })
                    await self.client.upsert(
                        collection_name=self.collection,
                        points=[models.PointStruct(id=point_id, vector=vec, payload=payload)]
                    )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def put_dict_replace_world(self, data: Dict[str, str], metadata: Dict[str, int] = None):
        client = await get_redis_client()
        async with self.request_semaphore:
            for k, v in data.items():
                lock_key = f"lock:qdrant:{self.user_id}:{self.story_id}:{self.namespace}:world_element:{k}"
                async with redis_lock(client=client, lock_key=lock_key):
                    filter_conds = [
                        models.FieldCondition(key="world_element", match=models.MatchValue(value=k)),
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                        models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                        models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
                    ]
                    search_results, _ = await self.client.scroll(
                        collection_name=self.collection,
                        scroll_filter=models.Filter(must=filter_conds),
                        limit=1,
                    )

                    point_id = search_results[0].id if search_results else str(uuid.uuid4())
                    vec = await self._embed_text(v)

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

                    await self.client.upsert(collection_name=self.collection, points=[models.PointStruct(id=point_id, vector=vec, payload=payload)])

    # ---------- Search ----------(brings all other points apart from the excluded metadata)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def search(self, query: str, k: int = 5, metadata: Dict[str, str] = None):
        async with self.request_semaphore:
            vec = await self._embed_text(query)  # Now async
            must_conds = [
                models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
            ]
            must_not_conds = []
            if metadata:
                for key, val in metadata.items():
                    must_not_conds.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
            search_filter = models.Filter(must=must_conds, must_not=must_not_conds)
            results = await self.client.search(  # Now async
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def get_chapter_content(self, chapter_number: int) -> list[str]:
        async with self.request_semaphore:
            all_texts = []
            offset = None
            while True:
                scroll_results, next_offset = await self.client.scroll(  # Now async
                    collection_name=self.collection,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="chapter_id", match=models.MatchValue(value=chapter_number)),
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
    
    async def close(self):
        if self.client is not SHARED_QDRANT:
            try:
                await self.client.close()
            except Exception:
                pass
        self.client = None
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass