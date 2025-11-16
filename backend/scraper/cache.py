import redis
import json
import pickle
import logging
from typing import Any, Optional, Union
import time

logger = logging.getLogger(__name__)

class RedisCache:
    """
    Redis caching wrapper for scraped results with serialization
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, expire_seconds: int = 600, 
                 encoding: str = 'utf-8'):
        self.host = host
        self.port = port
        self.db = db
        self.expire_seconds = expire_seconds
        self.encoding = encoding
        self.redis_client = None
        self._connect()
    
    def _connect(self) -> bool:
        """Establish connection to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,  # We handle encoding manually
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None
            return False
    
    def is_connected(self) -> bool:
        """Check if Redis connection is active"""
        if self.redis_client is None:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            self.redis_client = None
            return False
    
    def _ensure_connection(self) -> bool:
        """Ensure we have an active Redis connection"""
        if not self.is_connected():
            return self._connect()
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache by key
        Returns None if key doesn't exist or connection fails
        """
        if not self._ensure_connection():
            return None
        
        try:
            serialized_value = self.redis_client.get(key)
            if serialized_value is None:
                return None
            
            # Try JSON deserialization first, then pickle as fallback
            try:
                # Decode bytes to string for JSON
                value_str = serialized_value.decode(self.encoding)
                return json.loads(value_str)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to pickle
                return pickle.loads(serialized_value)
                
        except Exception as e:
            logger.error(f"Error getting key '{key}' from Redis: {e}")
            return None
    
    def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """
        Set value in cache with expiration
        Returns True if successful, False otherwise
        """
        if not self._ensure_connection():
            return False
        
        expire = expire_seconds if expire_seconds is not None else self.expire_seconds
        
        try:
            # Try JSON serialization first, then pickle as fallback
            try:
                serialized_value = json.dumps(value).encode(self.encoding)
            except (TypeError, ValueError):
                serialized_value = pickle.dumps(value)
            
            if expire > 0:
                self.redis_client.setex(key, expire, serialized_value)
            else:
                self.redis_client.set(key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting key '{key}' in Redis: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._ensure_connection():
            return False
        
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting key '{key}' from Redis: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._ensure_connection():
            return False
        
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking existence of key '{key}': {e}")
            return False
    
    def get_keys(self, pattern: str = "*") -> list:
        """Get all keys matching pattern"""
        if not self._ensure_connection():
            return []
        
        try:
            return self.redis_client.keys(pattern)
        except Exception as e:
            logger.error(f"Error getting keys with pattern '{pattern}': {e}")
            return []
    
    def clear_pattern(self, pattern: str = "scrape:*") -> int:
        """Clear all keys matching pattern, returns number of deleted keys"""
        if not self._ensure_connection():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing pattern '{pattern}': {e}")
            return 0
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining time to live for key in seconds"""
        if not self._ensure_connection():
            return None
        
        try:
            ttl = self.redis_client.ttl(key)
            return ttl if ttl >= 0 else None
        except Exception as e:
            logger.error(f"Error getting TTL for key '{key}': {e}")
            return None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics and information"""
        if not self._ensure_connection():
            return {"connected": False}
        
        try:
            info = self.redis_client.info()
            cache_stats = {
                "connected": True,
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "expire_seconds": self.expire_seconds,
                "keys_count": info.get('db0', {}).get('keys', 0),
                "used_memory": info.get('used_memory', 0),
                "used_memory_human": info.get('used_memory_human', '0B'),
                "scrape_keys_count": len(self.get_keys("scrape:*"))
            }
            return cache_stats
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {"connected": False, "error": str(e)}


# API-ready factory functions
def create_redis_cache(host: str = 'localhost', port: int = 6379, 
                      expire_seconds: int = 600) -> RedisCache:
    """Create RedisCache instance with specified parameters"""
    return RedisCache(host=host, port=port, expire_seconds=expire_seconds)

def create_redis_cache_from_url(url: str, expire_seconds: int = 600) -> RedisCache:
    """Create RedisCache instance from Redis URL"""
    try:
        import redis as redis_lib
        redis_client = redis_lib.from_url(url)
        host = redis_client.connection_pool.connection_kwargs.get('host', 'localhost')
        port = redis_client.connection_pool.connection_kwargs.get('port', 6379)
        db = redis_client.connection_pool.connection_kwargs.get('db', 0)
        redis_client.close()
        
        return RedisCache(host=host, port=port, db=db, expire_seconds=expire_seconds)
    except Exception as e:
        logger.error(f"Error creating Redis cache from URL: {e}")
        # Fallback to default
        return RedisCache(expire_seconds=expire_seconds)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test cache
    cache = create_redis_cache(expire_seconds=60)
    
    # Test set and get
    test_data = {"name": "test", "value": 123}
    cache.set("test_key", test_data)
    
    retrieved = cache.get("test_key")
    print("Retrieved:", retrieved)
    
    # Test cache info
    info = cache.get_cache_info()
    print("Cache info:", info)