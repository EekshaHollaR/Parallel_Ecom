from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time

@dataclass
class APIResponse:
    """
    Standard API response format
    """
    status: str  # "success" or "error"
    data: Optional[Any] = None
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "status": self.status,
            "timestamp": self.timestamp
        }
        
        if self.data is not None:
            result["data"] = self.data
        
        if self.message is not None:
            result["message"] = self.message
        
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        
        return result
    
    @classmethod
    def success(cls, data: Any = None, message: str = None, latency_ms: float = None) -> 'APIResponse':
        """Create a success response"""
        return cls(
            status="success",
            data=data,
            message=message,
            latency_ms=latency_ms
        )
    
    @classmethod
    def error(cls, message: str, latency_ms: float = None) -> 'APIResponse':
        """Create an error response"""
        return cls(
            status="error",
            message=message,
            latency_ms=latency_ms
        )

@dataclass
class HealthResponse:
    """
    Health check response schema
    """
    status: str
    timestamp: float
    version: str
    services: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RecommendationResponse:
    """
    Recommendation response schema
    """
    user_id: str
    recommendations: List[Dict[str, Any]]
    count: int
    exclude_rated: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PriceComparisonResponse:
    """
    Price comparison response schema
    """
    product: str
    total_results: int
    price_stats: Dict[str, float]
    results: List[Dict[str, Any]]
    sites_searched: List[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BatchResponse:
    """
    Batch operation response schema
    """
    batch_results: Dict[str, Any]
    total_items: int
    cached_items: int
    computed_items: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Validation schemas (for request validation)
class ValidationSchemas:
    """Request validation schemas"""
    
    @staticmethod
    def recommendations_request():
        return {
            'user_id': {'type': 'string', 'required': True},
            'n_recommendations': {'type': 'integer', 'required': False, 'min': 1, 'max': 100},
            'exclude_rated': {'type': 'boolean', 'required': False}
        }
    
    @staticmethod
    def price_compare_request():
        return {
            'product': {'type': 'string', 'required': True},
            'sites': {'type': 'string', 'required': False},
            'use_cache': {'type': 'boolean', 'required': False}
        }
    
    @staticmethod
    def batch_recommendations_request():
        return {
            'user_ids': {
                'type': 'list',
                'required': True,
                'schema': {'type': 'string'},
                'maxlength': 100
            }
        }