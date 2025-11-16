from flask import Blueprint, jsonify, request
import logging
from ..utils.schemas import APIResponse, HealthResponse
import time

logger = logging.getLogger(__name__)

common_bp = Blueprint('common', __name__)

@common_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    from flask import current_app
    
    health_data = HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        services={
            "api": "healthy",
            "redis": "unknown",  # Would check Redis connection in real implementation
            "recommender": "unknown",  # Would check model status
            "scraper": "unknown"  # Would check scraper status
        }
    )
    
    return APIResponse.success(health_data.to_dict()).to_dict()

@common_bp.route('/status', methods=['GET'])
def status():
    """
    Detailed status endpoint
    """
    from flask import current_app
    
    status_data = {
        "environment": "development" if current_app.debug else "production",
        "debug_mode": current_app.debug,
        "cache_enabled": True,
        "max_workers": current_app.config['MAX_WORKERS'],
        "use_gpu": current_app.config['USE_GPU'],
        "als_rank": current_app.config['ALS_RANK']
    }
    
    return APIResponse.success(status_data).to_dict()

@common_bp.route('/config', methods=['GET'])
def get_config():
    """
    Get current configuration (without sensitive data)
    """
    from flask import current_app
    
    config_data = {
        "redis_host": current_app.config['REDIS_HOST'],
        "redis_port": current_app.config['REDIS_PORT'],
        "cache_expiry": current_app.config['CACHE_EXPIRY'],
        "max_workers": current_app.config['MAX_WORKERS'],
        "als_rank": current_app.config['ALS_RANK'],
        "als_regularization": current_app.config['ALS_REGULARIZATION'],
        "use_gpu": current_app.config['USE_GPU']
    }
    
    return APIResponse.success(config_data).to_dict()