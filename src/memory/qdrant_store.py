from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
import uuid
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
from .shared_resources import SHARED_QDRANT
from contextlib import asynccontextmanager
from setup.shared_redis_pool import get_redis_client
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio import Semaphore
import asyncio
import json
import time
from fastapi import HTTPException
import redis.asyncio as redis


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
                if ttl == -1:
                    print(f"🔍 Detected stale lock with no TTL for {lock_key}, removing")
                    await client.delete(lock_key)
                elif ttl == -2:
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
    _shared_model = None
    collection_init_lock = asyncio.Lock()
    _initialized_collections = set()  # Track initialized collections
    
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
        """
        Initialize Qdrant collection and indexes with proper locking and idempotency.
        Uses both in-memory tracking and Redis locks to prevent concurrent initialization.
        """
        # Quick check: if already initialized in this process, skip
        if self.collection in QdrantStore._initialized_collections:
            print(f"✅ Collection {self.collection} already initialized in this process")
            return
        
        # Use Redis lock to coordinate across multiple processes/instances
        redis_client = await get_redis_client()
        collection_lock_key = f"lock:qdrant_init:{self.collection}"
        
        async with redis_lock(redis_client, collection_lock_key, timeout=120, retries=30, retry_delay=2.0):
            # Double-check after acquiring lock
            if self.collection in QdrantStore._initialized_collections:
                print(f"✅ Collection {self.collection} already initialized by another request")
                return
            
            try:
                # Step 1: Check and create collection if needed
                collections = await self.client.get_collections()
                existing_collections = [c.name for c in collections.collections]
                
                if self.collection not in existing_collections:
                    print(f"🔄 Creating collection {self.collection}")
                    await self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                        on_disk_payload=True,
                    )
                    print(f"✅ Created collection {self.collection}")
                else:
                    print(f"✅ Collection {self.collection} already exists")

                # Step 2: Get existing indexes to check what needs to be created
                collection_info = await self.client.get_collection(self.collection)
                existing_indexes = set()
                
                if collection_info.payload_schema:
                    existing_indexes = set(collection_info.payload_schema.keys())
                
                print(f"🔍 Existing indexes for {self.collection}: {existing_indexes}")

                # Step 3: Create only missing indexes
                required_indexes = {
                    "user_id": models.PayloadSchemaType.KEYWORD,
                    "story_id": models.PayloadSchemaType.KEYWORD,
                    "namespace": models.PayloadSchemaType.KEYWORD,
                    "character_name": models.PayloadSchemaType.KEYWORD,
                    "world_element": models.PayloadSchemaType.KEYWORD,
                    "chapter_id": models.PayloadSchemaType.INTEGER,
                }
                
                for field, schema in required_indexes.items():
                    if field in existing_indexes:
                        print(f"✅ Index {field} already exists")
                        continue
                    
                    try:
                        print(f"🔄 Creating index {field} with schema {schema}")
                        await self.client.create_payload_index(
                            collection_name=self.collection,
                            field_name=field,
                            field_schema=schema,
                            wait=True,  # Wait for index creation to complete
                        )
                        print(f"✅ Created index {field}")
                        
                        # Small delay between index creations to reduce load
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "already exists" in error_msg or "exist" in error_msg:
                            print(f"✅ Index {field} already exists (caught during creation)")
                        else:
                            print(f"❌ Failed to create index {field}: {e}")
                            raise

                # Mark as initialized
                QdrantStore._initialized_collections.add(self.collection)
                print(f"✅ Completed initialization for collection {self.collection}")
                
            except Exception as e:
                print(f"❌ Critical error during Qdrant initialization: {e}")
                import traceback
                traceback.print_exc()
                raise

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
        loop = get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode([text], convert_to_numpy=True)[0].tolist()
        )

    # ---------- Upserts ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def put(self, text: str, metadata: Dict = None):
        async with self.request_semaphore:
            vec = await self._embed_text(text)
            payload = metadata.copy() if metadata else {}
            payload.update({
                "text": text,
                "user_id": self.user_id,
                "story_id": self.story_id,
                "namespace": self.namespace,
            })

            await self.client.upsert(
                collection_name=self.collection,
                points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)],
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def put_dict_replace_character(self, data: Dict[str, dict], metadata: Dict[str, Any] = None):
        client = await get_redis_client()
        async with self.request_semaphore:
            for name, info in data.items():
                lock_key = f"lock:qdrant:{self.user_id}:{self.story_id}:{self.namespace}:character:{name}"
                async with redis_lock(client=client, lock_key=lock_key):
                    # --- 1️⃣ Find existing point ---
                    filter_conds = [
                        models.FieldCondition(key="character_name", match=models.MatchValue(value=name)),
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

                    # --- 2️⃣ Prepare embedding text ---
                    # Convert structured info into a text summary for embeddings
                    embed_text = (
                        f"Name: {info.get('name')}\n"
                        f"Summary: {info.get('summary', '')}\n"
                        f"Traits: {', '.join(info.get('traits', []))}\n"
                        f"Relationships: {json.dumps(info.get('relationships', {}))}\n"
                        f"Emotional State: {info.get('emotional_state', '')}\n"
                        f"Goals: {info.get('goals', '')}\n"
                        f"Status Changes: {info.get('status_changes', '')}"
                    )
                    vec = await self._embed_text(embed_text)

                    # --- 3️⃣ Create full payload ---
                    payload = metadata.copy() if metadata else {}
                    payload.update({
                        "key": name,
                        "value": info["summary"],  # human-readable search value
                        "text": embed_text,        # full text used for vector embedding
                        "character_name": name,
                        "structured_data": info,   # <-- full structured object here
                        "user_id": self.user_id,
                        "story_id": self.story_id,
                        "namespace": self.namespace,
                        "type": "character"
                    })

                    # --- 4️⃣ Upsert into Qdrant ---
                    await self.client.upsert(
                        collection_name=self.collection,
                        points=[models.PointStruct(id=point_id, vector=vec, payload=payload)]
                    )


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def put_dict_replace_world(self, data: Dict[str, dict], metadata: Dict[str, Any] = None):
        client = await get_redis_client()
        async with self.request_semaphore:
            for name, info in data.items():
                lock_key = f"lock:qdrant:{self.user_id}:{self.story_id}:{self.namespace}:world:{name}"
                async with redis_lock(client=client, lock_key=lock_key):
                    filter_conds = [
                        models.FieldCondition(key="world_element", match=models.MatchValue(value=name)),
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

                    embed_text = (
                        f"Name: {info.get('name')}\n"
                        f"Summary: {info.get('summary', '')}\n"
                        f"Atmosphere: {info.get('atmosphere', '')}\n"
                        f"Culture: {info.get('culture', '')}\n"
                        f"Events: {info.get('events', '')}\n"
                        f"Connections: {json.dumps(info.get('connections', {}))}"
                    )
                    vec = await self._embed_text(embed_text)

                    payload = metadata.copy() if metadata else {}
                    payload.update({
                        "key": name,
                        "value": info["summary"],
                        "text": embed_text,
                        "world_element": name,
                        "structured_data": info,
                        "user_id": self.user_id,
                        "story_id": self.story_id,
                        "namespace": self.namespace,
                        "type": "world"
                    })

                    await self.client.upsert(
                        collection_name=self.collection,
                        points=[models.PointStruct(id=point_id, vector=vec, payload=payload)]
                    )

    # ---------- Search ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def search(self, query: str, k: int = 5, metadata: Dict[str, Any] = None):
        async with self.request_semaphore:
            vec = await self._embed_text(query)

            # Base conditions (always required)
            must_conds = [
                models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
            ]
            must_not_conds = []

            # Optional filters based on metadata
            if metadata:
                # If "type" exists — include it as a positive filter
                if "type" in metadata:
                    must_conds.append(
                        models.FieldCondition(key="type", match=models.MatchValue(value=metadata["type"]))
                    )

                # If "chapter_id" exists — exclude it (avoid current chapter)
                if "chapter_id" in metadata:
                    must_not_conds.append(
                        models.FieldCondition(key="chapter_id", match=models.MatchValue(value=metadata["chapter_id"]))
                    )

            # Build final filter
            search_filter = models.Filter(must=must_conds, must_not=must_not_conds)

            # Execute the vector search
            results = await self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=k,
                query_filter=search_filter
            )

            # Format results
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
    async def get_chapter_content(self, metadata: Dict[str, Any] = None) -> list[str]:
        async with self.request_semaphore:
            all_texts = []
            offset = None

            # Build base must conditions
            must_conds = [
                models.FieldCondition(key="chapter_id", match=models.MatchValue(value=metadata["chapter_number"])),
                models.FieldCondition(key="user_id", match=models.MatchValue(value=self.user_id)),
                models.FieldCondition(key="story_id", match=models.MatchValue(value=self.story_id)),
                models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace)),
            ]

            # Add type filter if provided
            if metadata and "type" in metadata:
                must_conds.append(
                    models.FieldCondition(key="type", match=models.MatchValue(value=metadata["type"]))
                )

            while True:
                scroll_results, next_offset = await self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=models.Filter(must=must_conds),
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