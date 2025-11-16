from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import time
import os
from typing import Dict, Any, Optional

from .routes.recommendations import recommendations_bp
from .routes.price_compare import price_compare_bp
from .routes.common import common_bp
from .routes.tasks import tasks_bp  # Add tasks blueprint
from .utils.timing import timer
from .utils.schemas import APIResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(test_config: Optional[Dict] = None) -> Flask:
    """
    Create and configure the Flask application
    """
    app = Flask(__name__)
    
    # Configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key'),
        REDIS_HOST=os.environ.get('REDIS_HOST', 'localhost'),
        REDIS_PORT=int(os.environ.get('REDIS_PORT', 6379)),
        CACHE_EXPIRY=int(os.environ.get('CACHE_EXPIRY', 600)),
        MAX_WORKERS=int(os.environ.get('MAX_WORKERS', 8)),
        ALS_RANK=int(os.environ.get('ALS_RANK', 50)),
        ALS_REGULARIZATION=float(os.environ.get('ALS_REGULARIZATION', 0.1)),
        USE_GPU=os.environ.get('USE_GPU', 'false').lower() == 'true',
        CELERY_BROKER_URL=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        CELERY_RESULT_BACKEND=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
    )
    
    if test_config:
        app.config.update(test_config)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(price_compare_bp)
    app.register_blueprint(common_bp)
    app.register_blueprint(tasks_bp)  # Register tasks blueprint
    
    # Add timing middleware
    @app.before_request
    def start_timer():
        request.start_time = time.time()
    
    @app.after_request
    def add_header(response):
        # Add latency to response headers
        if hasattr(request, 'start_time'):
            latency_ms = (time.time() - request.start_time) * 1000
            response.headers['X-Latency-MS'] = f'{latency_ms:.2f}'
        return response
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return APIResponse.error("Endpoint not found", 404).to_dict(), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return APIResponse.error("Internal server error", 500).to_dict(), 500
    
    @app.errorhandler(400)
    def bad_request(error):
        return APIResponse.error("Bad request", 400).to_dict(), 400
    
    logger.info("Flask application initialized successfully with Celery support")
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)