from .celery_app import celery
import logging
import time
from typing import Dict, List, Any, Optional
import random

logger = logging.getLogger(__name__)

@celery.task(bind=True, name='async_recommend')
def async_recommend(self, user_id: str, n_recommendations: int = 10, 
                   exclude_rated: bool = True) -> Dict[str, Any]:
    """
    Celery task for generating recommendations asynchronously
    """
    task_id = self.request.id
    
    try:
        logger.info(f"Starting async recommendation task {task_id} for user {user_id}")
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'status': 'Initializing recommender...',
                'user_id': user_id
            }
        )
        
        # Simulate some processing time for demonstration
        time.sleep(2)
        
        # Import recommender (lazy import to avoid circular imports)
        from backend.recommender.als_ncg import create_als_recommender
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'status': 'Loading recommender model...',
                'user_id': user_id
            }
        )
        
        # Initialize recommender
        recommender = create_als_recommender(
            rank=50,
            regularization=0.1,
            use_gpu=False,
            n_workers=4
        )
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 60,
                'total': 100,
                'status': 'Generating recommendations...',
                'user_id': user_id
            }
        )
        
        # Convert user_id to integer index (mock mapping)
        try:
            user_idx = int(user_id) % 1000
        except ValueError:
            user_idx = hash(user_id) % 1000
        
        # Generate recommendations
        recommendations = recommender.recommend_for_user(
            user_idx=user_idx,
            n_recommendations=n_recommendations,
            exclude_rated=exclude_rated
        )
        
        # Format results
        recommendations_data = []
        for item_idx, score in recommendations:
            recommendations_data.append({
                "item_id": str(item_idx),
                "score": float(score),
                "item_name": f"Product {item_idx}",
                "confidence": min(1.0, max(0.0, score / 5.0))
            })
        
        result = {
            "user_id": user_id,
            "recommendations": recommendations_data,
            "count": len(recommendations_data),
            "exclude_rated": exclude_rated,
            "task_id": task_id,
            "completed_at": time.time()
        }
        
        logger.info(f"Async recommendation task {task_id} completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Async recommendation task {task_id} failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={
                'current': 100,
                'total': 100,
                'status': f'Task failed: {str(e)}',
                'user_id': user_id
            }
        )
        raise

@celery.task(bind=True, name='async_compare_price')
def async_compare_price(self, product_name: str, sites: List[str] = None, 
                       use_cache: bool = True) -> Dict[str, Any]:
    """
    Celery task for price comparison asynchronously
    """
    task_id = self.request.id
    
    try:
        logger.info(f"Starting async price comparison task {task_id} for product: {product_name}")
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'status': 'Initializing scraper...',
                'product_name': product_name
            }
        )
        
        # Import scraper (lazy import to avoid circular imports)
        from backend.scraper.scraper import create_price_scraper
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Scraping prices from sites...',
                'product_name': product_name
            }
        )
        
        # Initialize scraper
        scraper = create_price_scraper(max_workers=6)
        
        # Scrape prices
        scraped_results = scraper.scrape_single_product(
            product_name=product_name,
            use_cache=use_cache
        )
        
        # Apply sites filter if specified
        if sites:
            filtered_results = [
                result for result in scraped_results 
                if result.get('site', '').lower() in [s.lower() for s in sites]
            ]
        else:
            filtered_results = scraped_results
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 80,
                'total': 100,
                'status': 'Processing results...',
                'product_name': product_name
            }
        )
        
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
        
        result = {
            "product": product_name,
            "total_results": len(filtered_results),
            "price_stats": price_stats,
            "results": filtered_results,
            "sites_searched": sites if sites else "all",
            "task_id": task_id,
            "completed_at": time.time()
        }
        
        logger.info(f"Async price comparison task {task_id} completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Async price comparison task {task_id} failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={
                'current': 100,
                'total': 100,
                'status': f'Task failed: {str(e)}',
                'product_name': product_name
            }
        )
        raise

@celery.task(bind=True, name='async_batch_recommend')
def async_batch_recommend(self, user_ids: List[str], n_recommendations: int = 10) -> Dict[str, Any]:
    """
    Celery task for batch recommendations asynchronously
    """
    task_id = self.request.id
    
    try:
        logger.info(f"Starting async batch recommendation task {task_id} for {len(user_ids)} users")
        
        results = {}
        total_users = len(user_ids)
        
        for i, user_id in enumerate(user_ids):
            # Update progress
            progress = int((i / total_users) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': progress,
                    'total': 100,
                    'status': f'Processing user {i+1}/{total_users}',
                    'current_user': user_id,
                    'processed': i,
                    'total_users': total_users
                }
            )
            
            # Generate recommendations for this user (simplified)
            try:
                user_idx = int(user_id) % 1000  # Mock mapping
                
                # In real implementation, you'd use the actual recommender
                # For demo, generate mock recommendations
                mock_recommendations = []
                for j in range(n_recommendations):
                    mock_recommendations.append({
                        "item_id": str(j + 1000),
                        "score": float(random.uniform(3.0, 5.0)),
                        "item_name": f"Product {j + 1000}",
                        "confidence": float(random.uniform(0.6, 0.95))
                    })
                
                results[user_id] = {
                    "user_id": user_id,
                    "recommendations": mock_recommendations,
                    "count": len(mock_recommendations)
                }
                
            except Exception as e:
                logger.error(f"Failed to process user {user_id}: {e}")
                results[user_id] = {"error": str(e)}
            
            # Small delay to simulate processing
            time.sleep(0.5)
        
        result = {
            "batch_results": results,
            "total_users": total_users,
            "successful_users": len([r for r in results.values() if 'error' not in r]),
            "task_id": task_id,
            "completed_at": time.time()
        }
        
        logger.info(f"Async batch recommendation task {task_id} completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Async batch recommendation task {task_id} failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={
                'current': 100,
                'total': 100,
                'status': f'Task failed: {str(e)}',
                'total_users': len(user_ids)
            }
        )
        raise

@celery.task(bind=True, name='async_batch_compare_price')
def async_batch_compare_price(self, products: List[str], sites: List[str] = None) -> Dict[str, Any]:
    """
    Celery task for batch price comparison asynchronously
    """
    task_id = self.request.id
    
    try:
        logger.info(f"Starting async batch price comparison task {task_id} for {len(products)} products")
        
        # Import scraper
        from backend.scraper.scraper import create_price_scraper
        
        # Initialize scraper
        scraper = create_price_scraper(max_workers=4)
        
        # Scrape all products
        batch_results = scraper.scrape_multiple_products(
            product_names=products,
            use_cache=True
        )
        
        # Process results with progress updates
        processed_results = {}
        total_products = len(products)
        
        for i, (product, results) in enumerate(batch_results.items()):
            # Update progress
            progress = int((i / total_products) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': progress,
                    'total': 100,
                    'status': f'Processing product {i+1}/{total_products}',
                    'current_product': product,
                    'processed': i,
                    'total_products': total_products
                }
            )
            
            # Apply sites filter if specified
            if sites:
                filtered_results = [
                    result for result in results 
                    if result.get('site', '').lower() in [s.lower() for s in sites]
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
        
        result = {
            "batch_results": processed_results,
            "total_products": total_products,
            "sites_searched": sites if sites else "all",
            "task_id": task_id,
            "completed_at": time.time()
        }
        
        logger.info(f"Async batch price comparison task {task_id} completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Async batch price comparison task {task_id} failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={
                'current': 100,
                'total': 100,
                'status': f'Task failed: {str(e)}',
                'total_products': len(products)
            }
        )
        raise