import redis.asyncio as redis
from fastapi import HTTPException
from os import environ
import asyncio

# Initialize Redis connection pool
REDIS_POOL = redis.ConnectionPool(
    host=environ.get('REDIS_HOST', 'localhost'),
    port=int(environ.get('REDIS_PORT', 6379)),
    db=int(environ.get('REDIS_DB', 0)),
    decode_responses=True,
    max_connections=int(environ.get('REDIS_MAX_CONNECTIONS', 50)),  # Increased default for higher concurrency
    retry_on_timeout=True
)

async def get_redis_client(max_retries=3, retry_delay=1):
    """Get a Redis client with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            client = redis.Redis(connection_pool=REDIS_POOL)
            await client.ping()
            # Check pool usage
            pool = REDIS_POOL
            if len(pool._in_use_connections) >= pool.max_connections:
                print(f"⚠️ Warning: Redis connection pool exhausted ({pool._in_use_connections}/{pool.max_connections})")
                raise redis.ConnectionError("Connection pool exhausted")
            return client
        except redis.ConnectionError as e:
            print(f"❌ Redis connection error, attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            continue
        except redis.RedisError as e:
            print(f"❌ Redis error in get_redis_client: {e}")
            raise HTTPException(status_code=500, detail="Failed to connect to session storage")
    raise HTTPException(status_code=500, detail="Failed to connect to Redis after retries")