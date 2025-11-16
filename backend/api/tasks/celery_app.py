from celery import Celery
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def make_celery():
    """
    Create and configure Celery instance
    """
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = os.environ.get('REDIS_PORT', 6379)
    
    celery = Celery(
        'ecommerce_tasks',
        broker=f'redis://{redis_host}:{redis_port}/0',
        backend=f'redis://{redis_host}:{redis_port}/1',
        include=['backend.api.tasks.async_tasks']
    )
    
    # Celery configuration
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,  # 5 minutes
        result_expires=3600,  # 1 hour
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        broker_connection_retry_on_startup=True,
    )
    
    # Development settings
    if os.environ.get('FLASK_ENV') == 'development':
        celery.conf.update(
            task_always_eager=False,  # Don't execute tasks locally unless testing
            task_eager_propagates=True,
        )
    
    logger.info(f"Celery configured with Redis: {redis_host}:{redis_port}")
    return celery

# Create Celery instance
celery = make_celery()