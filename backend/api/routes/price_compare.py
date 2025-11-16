from flask import Blueprint, request, jsonify, current_app
import logging
import time
import random
from typing import List, Dict, Any

from ..utils.schemas import APIResponse
from ..utils.timing import timer

logger = logging.getLogger(__name__)

price_compare_bp = Blueprint('price_compare', __name__)

# Global variables for scraper and cache
_scraper = None
_cache = None

# # def get_scraper():
#     """Lazy initialization of price scraper"""
#     global _scraper
#     if _scraper is None:
#         try:
#             from backend.scraper.scraper import create_price_scraper
            
#             _scraper = create_price_scraper(
#                 redis_host=current_app.config['REDIS_HOST'],
#                 redis_port=current_app.config['REDIS_PORT'],
#                 max_workers=current_app.config['MAX_WORKERS']
#             )
#             logger.info("Price scraper initialized")
            
#         except Exception as e:
#             logger.error(f"Failed to initialize price scraper: {e}")
#             _scraper = None
    
#     return _scraper


def get_scraper():
    """Lazy initialization of price scraper"""
    global _scraper
    if _scraper is None:
        try:
            from backend.scraper.scraper import create_price_scraper
            
            # Try to create scraper with Redis cache
            _scraper = create_price_scraper(
                redis_host=current_app.config['REDIS_HOST'],
                redis_port=current_app.config['REDIS_PORT'],
                max_workers=current_app.config['MAX_WORKERS']
            )
            logger.info("Price scraper initialized with Redis cache")
            
        except Exception as e:
            logger.warning(f"Failed to initialize price scraper with Redis: {e}")
            try:
                # Fallback: create scraper without cache
                from backend.scraper.scraper import create_scraper_without_cache
                _scraper = create_scraper_without_cache(max_workers=4)
                logger.info("Price scraper initialized without cache")
            except Exception as e2:
                logger.error(f"Failed to initialize price scraper: {e2}")
                _scraper = None
    
    return _scraper

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
        # Test if Celery is properly configured
        celery.control.inspect().active()
        return celery
    except Exception as e:
        logger.warning(f"Celery not available: {e}")
        return None    

# def get_celery():
    # """Lazy initialization of Celery"""
    # try:
    #     from backend.api.tasks.celery_app import celery
    #     return celery
    # except ImportError as e:
    #     logger.error(f"Failed to import Celery: {e}")
    #     return None

def is_heavy_load() -> bool:
    """
    Determine if we should use async processing due to heavy load
    For price comparison, consider product complexity and current load
    """
    force_async = request.args.get('async', 'false').lower() == 'true'
    
    # For price scraping, use async for complex product names or high load
    product_name = request.args.get('product', '')
    product_complexity = len(product_name.split()) > 3
    
    # Simulate heavy load detection
    simulated_heavy_load = random.random() < 0.3
    
    return force_async or product_complexity or simulated_heavy_load

@price_compare_bp.route('/compare_price', methods=['GET'])
@timer
def compare_price():
    """
    Compare prices for a product across multiple sites - with async support
    """
    start_time = time.time()
    
    # Get query parameters
    product_name = request.args.get('product', type=str)
    sites_param = request.args.get('sites', '')
    use_cache = request.args.get('use_cache', 'true').lower() == 'true'
    force_async = request.args.get('async', 'false').lower() == 'true'
    
    # Validate parameters
    if not product_name or not product_name.strip():
        return APIResponse.error("Missing required parameter: product", 400).to_dict(), 400
    
    product_name = product_name.strip()
    
    # Parse sites filter
    sites_filter = []
    if sites_param:
        sites_filter = [site.strip().lower() for site in sites_param.split(',')]
    
    # Check if we should use async processing
    if force_async or is_heavy_load():
        celery = get_celery()
        if not celery:
            return APIResponse.error("Async processing not available", 503).to_dict(), 503
        
        # Submit async task
        task = celery.send_task(
            'async_compare_price',
            args=[product_name, sites_filter, use_cache]
        )
        
        response_data = {
            "job_id": task.id,
            "status": "PENDING",
            "message": "Price comparison task submitted for async processing",
            "polling_endpoint": f"/task_status/{task.id}"
        }
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
    
    # Synchronous processing (existing logic)
    cache_key = f"price_compare:{product_name.lower()}"
    if sites_filter:
        cache_key += f":{','.join(sorted(sites_filter))}"
    
    cache = get_cache()
    
    if use_cache and cache:
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for price comparison: {product_name}")
            return APIResponse.success(
                data=cached_result,
                latency_ms=(time.time() - start_time) * 1000
            ).to_dict()
    
    try:
        scraper = get_scraper()
        if not scraper:
            return APIResponse.error("Price scraper not available", 503).to_dict(), 503
        
        # Scrape prices
        scraped_results = scraper.scrape_single_product(
            product_name=product_name,
            use_cache=use_cache
        )
        
        # Apply sites filter if specified
        if sites_filter:
            filtered_results = [
                result for result in scraped_results 
                if result.get('site', '').lower() in sites_filter
            ]
        else:
            filtered_results = scraped_results
        
        # Calculate price statistics
        prices = [result['price'] for result in filtered_results if 'price' in result]
        price_stats = {}
        if prices:
            price_stats = {
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "total_offers": len(prices)
            }
        
        # Sort by price (lowest first)
        filtered_results.sort(key=lambda x: x.get('price', float('inf')))
        
        # Format response
        response_data = {
            "product": product_name,
            "total_results": len(filtered_results),
            "price_stats": price_stats,
            "results": filtered_results,
            "sites_searched": sites_filter if sites_filter else "all",
            "timestamp": time.time()
        }
        
        # Cache the result
        if cache and use_cache:
            cache.set(cache_key, response_data)
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Error comparing prices for product '{product_name}': {e}")
        return APIResponse.error(f"Failed to compare prices: {str(e)}", 500).to_dict(), 500

@price_compare_bp.route('/compare_price/batch', methods=['POST'])
@timer
def compare_price_batch():
    """
    Compare prices for multiple products in batch - with async support
    """
    start_time = time.time()
    
    try:
        products = request.get_json()
        if not isinstance(products, list):
            return APIResponse.error("Request body must be a JSON array of product names", 400).to_dict(), 400
        
        if len(products) > 20:
            return APIResponse.error("Maximum 20 products per batch request", 400).to_dict(), 400
        
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        sites_param = request.args.get('sites', '')
        sites_filter = [site.strip().lower() for site in sites_param.split(',')] if sites_param else []
        force_async = request.args.get('async', 'false').lower() == 'true'
        
        # For batch operations, always use async if more than 5 products
        # or if explicitly requested
        if force_async or len(products) > 5:
            celery = get_celery()
            if not celery:
                return APIResponse.error("Async processing not available", 503).to_dict(), 503
            
            # Submit async batch task
            task = celery.send_task(
                'async_batch_compare_price',
                args=[products, sites_filter]
            )
            
            response_data = {
                "job_id": task.id,
                "status": "PENDING",
                "message": f"Batch price comparison task for {len(products)} products submitted for async processing",
                "polling_endpoint": f"/task_status/{task.id}"
            }
            
            return APIResponse.success(
                data=response_data,
                latency_ms=(time.time() - start_time) * 1000
            ).to_dict()
        
        # Synchronous batch processing (existing logic)
        scraper = get_scraper()
        if not scraper:
            return APIResponse.error("Price scraper not available", 503).to_dict(), 503
        
        # Scrape all products
        batch_results = scraper.scrape_multiple_products(
            product_names=products,
            use_cache=use_cache
        )
        
        # Process results
        processed_results = {}
        for product, results in batch_results.items():
            # Apply sites filter if specified
            if sites_filter:
                filtered_results = [
                    result for result in results 
                    if result.get('site', '').lower() in sites_filter
                ]
            else:
                filtered_results = results
            
            # Calculate price statistics
            prices = [result['price'] for result in filtered_results if 'price' in result]
            price_stats = {}
            if prices:
                price_stats = {
                    "min_price": min(prices),
                    "max_price": max(prices),
                    "avg_price": sum(prices) / len(prices),
                    "total_offers": len(prices)
                }
            
            # Sort by price
            filtered_results.sort(key=lambda x: x.get('price', float('inf')))
            
            processed_results[product] = {
                "total_results": len(filtered_results),
                "price_stats": price_stats,
                "results": filtered_results
            }
        
        response_data = {
            "batch_results": processed_results,
            "total_products": len(products),
            "sites_searched": sites_filter if sites_filter else "all",
            "timestamp": time.time()
        }
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Error in batch price comparison: {e}")
        return APIResponse.error(f"Batch price comparison failed: {str(e)}", 500).to_dict(), 500

# ... (keep the existing get_available_sites endpoint unchanged)