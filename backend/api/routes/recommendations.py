from flask import Blueprint, request, jsonify, current_app
import logging
import time
import random
from typing import List, Dict, Any

from ..utils.schemas import APIResponse
from ..utils.timing import timer

logger = logging.getLogger(__name__)

recommendations_bp = Blueprint('recommendations', __name__)

# Global variables for model and cache (initialized on first use)
_recommender = None
_cache = None

def get_recommender():
    """Lazy initialization of recommender"""
    global _recommender
    if _recommender is None:
        try:
            from backend.recommender.als_ncg import create_als_recommender
            
            _recommender = create_als_recommender(
                rank=current_app.config['ALS_RANK'],
                regularization=current_app.config['ALS_REGULARIZATION'],
                use_gpu=current_app.config['USE_GPU'],
                n_workers=current_app.config['MAX_WORKERS']
            )
            logger.info("Recommender initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize recommender: {e}")
            _recommender = None
    
    return _recommender

def get_cache():
    """Lazy initialization of cache"""
    global _cache
    if _cache is None:
        try:
            from backend.scraper.cache import create_redis_cache
            
            _cache = create_redis_cache(
                host=current_app.config['REDIS_HOST'],
                port=current_app.config['REDIS_PORT'],
                expire_seconds=current_app.config['CACHE_EXPIRY']
            )
            logger.info("Cache initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            _cache = None
    
    return _cache

def get_celery():
    """Lazy initialization of Celery"""
    try:
        from backend.api.tasks.celery_app import celery
        return celery
    except ImportError as e:
        logger.error(f"Failed to import Celery: {e}")
        return None

def is_heavy_load() -> bool:
    """
    Determine if we should use async processing due to heavy load
    This is a simplified implementation - in production, you'd check:
    - System load average
    - Current request queue length
    - Database connection pool usage
    - etc.
    """
    # For demo purposes, use a random condition or request parameter
    force_async = request.args.get('async', 'false').lower() == 'true'
    
    # Simulate heavy load detection (20% chance)
    simulated_heavy_load = random.random() < 0.2
    
    return force_async or simulated_heavy_load

@recommendations_bp.route('/recommendations', methods=['GET'])
@timer
def get_recommendations():
    """
    Get recommendations for a user - with async support
    """
    start_time = time.time()
    
    # Get query parameters
    user_id = request.args.get('user_id', type=str)
    n_recommendations = request.args.get('n_recommendations', 10, type=int)
    exclude_rated = request.args.get('exclude_rated', 'true').lower() == 'true'
    force_async = request.args.get('async', 'false').lower() == 'true'
    
    # Validate parameters
    if not user_id:
        return APIResponse.error("Missing required parameter: user_id", 400).to_dict(), 400
    
    if n_recommendations <= 0 or n_recommendations > 100:
        return APIResponse.error("n_recommendations must be between 1 and 100", 400).to_dict(), 400
    
    # Check if we should use async processing
    if force_async or is_heavy_load():
        celery = get_celery()
        if not celery:
            return APIResponse.error("Async processing not available", 503).to_dict(), 503
        
        # Submit async task
        task = celery.send_task(
            'async_recommend',
            args=[user_id, n_recommendations, exclude_rated]
        )
        
        response_data = {
            "job_id": task.id,
            "status": "PENDING",
            "message": "Recommendation task submitted for async processing",
            "polling_endpoint": f"/task_status/{task.id}"
        }
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
    
    # Synchronous processing (existing logic)
    cache_key = f"recommendations:{user_id}:{n_recommendations}:{exclude_rated}"
    cache = get_cache()
    
    if cache:
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for recommendations user_id={user_id}")
            return APIResponse.success(
                data=cached_result,
                latency_ms=(time.time() - start_time) * 1000
            ).to_dict()
    
    try:
        recommender = get_recommender()
        if not recommender:
            return APIResponse.error("Recommender not available", 503).to_dict(), 503
        
        # Convert user_id to integer index
        try:
            user_idx = int(user_id) % 1000
        except ValueError:
            user_idx = hash(user_id) % 1000
        
        # Get recommendations
        recommendations = recommender.recommend_for_user(
            user_idx=user_idx,
            n_recommendations=n_recommendations,
            exclude_rated=exclude_rated
        )
        
        # Format response
        recommendations_data = []
        for item_idx, score in recommendations:
            recommendations_data.append({
                "item_id": str(item_idx),
                "score": float(score),
                "item_name": f"Product {item_idx}",
                "confidence": min(1.0, max(0.0, score / 5.0))
            })
        
        response_data = {
            "user_id": user_id,
            "recommendations": recommendations_data,
            "count": len(recommendations_data),
            "exclude_rated": exclude_rated
        }
        
        # Cache the result
        if cache:
            cache.set(cache_key, response_data)
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        return APIResponse.error(f"Failed to get recommendations: {str(e)}", 500).to_dict(), 500

@recommendations_bp.route('/recommendations/batch', methods=['POST'])
@timer
def get_batch_recommendations():
    """
    Get recommendations for multiple users in batch - with async support
    """
    start_time = time.time()
    
    try:
        user_ids = request.get_json()
        if not isinstance(user_ids, list):
            return APIResponse.error("Request body must be a JSON array of user IDs", 400).to_dict(), 400
        
        if len(user_ids) > 100:
            return APIResponse.error("Maximum 100 users per batch request", 400).to_dict(), 400
        
        n_recommendations = request.args.get('n_recommendations', 10, type=int)
        force_async = request.args.get('async', 'false').lower() == 'true'
        
        # For batch operations, always use async if more than 10 users
        # or if explicitly requested
        if force_async or len(user_ids) > 10:
            celery = get_celery()
            if not celery:
                return APIResponse.error("Async processing not available", 503).to_dict(), 503
            
            # Submit async batch task
            task = celery.send_task(
                'async_batch_recommend',
                args=[user_ids, n_recommendations]
            )
            
            response_data = {
                "job_id": task.id,
                "status": "PENDING",
                "message": f"Batch recommendation task for {len(user_ids)} users submitted for async processing",
                "polling_endpoint": f"/task_status/{task.id}"
            }
            
            return APIResponse.success(
                data=response_data,
                latency_ms=(time.time() - start_time) * 1000
            ).to_dict()
        
        # Synchronous batch processing (existing logic)
        cache = get_cache()
        results = {}
        uncached_users = []
        
        for user_id in user_ids:
            cache_key = f"recommendations:{user_id}:{n_recommendations}:true"
            if cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    results[user_id] = cached_result
                else:
                    uncached_users.append(user_id)
            else:
                uncached_users.append(user_id)
        
        # Generate recommendations for uncached users
        if uncached_users:
            recommender = get_recommender()
            if not recommender:
                return APIResponse.error("Recommender not available", 503).to_dict(), 503
            
            for user_id in uncached_users:
                try:
                    user_idx = int(user_id) % 1000
                    recommendations = recommender.recommend_for_user(
                        user_idx=user_idx,
                        n_recommendations=n_recommendations,
                        exclude_rated=True
                    )
                    
                    user_recommendations = []
                    for item_idx, score in recommendations:
                        user_recommendations.append({
                            "item_id": str(item_idx),
                            "score": float(score),
                            "item_name": f"Product {item_idx}",
                            "confidence": min(1.0, max(0.0, score / 5.0))
                        })
                    
                    result_data = {
                        "user_id": user_id,
                        "recommendations": user_recommendations,
                        "count": len(user_recommendations)
                    }
                    
                    results[user_id] = result_data
                    
                    # Cache the result
                    if cache:
                        cache_key = f"recommendations:{user_id}:{n_recommendations}:true"
                        cache.set(cache_key, result_data)
                        
                except Exception as e:
                    logger.error(f"Error processing user {user_id}: {e}")
                    results[user_id] = {"error": str(e)}
        
        response_data = {
            "batch_results": results,
            "total_users": len(user_ids),
            "cached_users": len(user_ids) - len(uncached_users),
            "computed_users": len(uncached_users)
        }
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}")
        return APIResponse.error(f"Batch recommendations failed: {str(e)}", 500).to_dict(), 500

# ... (keep the existing similar_items endpoint unchanged)