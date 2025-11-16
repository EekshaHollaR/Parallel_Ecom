from flask import Blueprint, request, jsonify, current_app
import logging
import time
from typing import Dict, Any

from ..utils.schemas import APIResponse
from ..utils.timing import timer

logger = logging.getLogger(__name__)

tasks_bp = Blueprint('tasks', __name__)

def get_celery():
    """Lazy initialization of Celery"""
    try:
        from backend.api.tasks.celery_app import celery
        return celery
    except ImportError as e:
        logger.error(f"Failed to import Celery: {e}")
        return None

@tasks_bp.route('/task_status/<task_id>', methods=['GET'])
@timer
def task_status(task_id: str):
    """
    Get the status of an asynchronous task
    """
    start_time = time.time()
    
    celery = get_celery()
    if not celery:
        return APIResponse.error("Task queue not available", 503).to_dict(), 503
    
    try:
        task = celery.AsyncResult(task_id)
        
        response_data = {
            "task_id": task_id,
            "status": task.status,
            "task_name": getattr(task, 'name', 'unknown')
        }
        
        # Add task-specific data based on status
        if task.status == 'PENDING':
            response_data["message"] = "Task is pending execution"
        elif task.status == 'PROGRESS':
            response_data.update(task.result or {})
            response_data["message"] = "Task is in progress"
        elif task.status == 'SUCCESS':
            response_data["result"] = task.result
            response_data["message"] = "Task completed successfully"
        elif task.status == 'FAILURE':
            response_data["error"] = str(task.result)
            response_data["message"] = "Task failed"
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        return APIResponse.error(f"Failed to get task status: {str(e)}", 500).to_dict(), 500

@tasks_bp.route('/tasks', methods=['GET'])
@timer
def list_tasks():
    """
    Get list of recent tasks (basic implementation)
    In production, you'd use Celery monitoring tools
    """
    start_time = time.time()
    
    celery = get_celery()
    if not celery:
        return APIResponse.error("Task queue not available", 503).to_dict(), 503
    
    try:
        # This is a simplified implementation
        # In production, use Celery's inspection features
        inspector = celery.control.inspect()
        
        active_tasks = inspector.active() or {}
        scheduled_tasks = inspector.scheduled() or {}
        reserved_tasks = inspector.reserved() or {}
        
        task_counts = {
            "active": sum(len(tasks) for tasks in active_tasks.values()),
            "scheduled": sum(len(tasks) for tasks in scheduled_tasks.values()),
            "reserved": sum(len(tasks) for tasks in reserved_tasks.values()),
        }
        
        response_data = {
            "task_counts": task_counts,
            "workers_available": bool(active_tasks),  # Simple check
            "timestamp": time.time()
        }
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return APIResponse.error(f"Failed to list tasks: {str(e)}", 500).to_dict(), 500

@tasks_bp.route('/tasks/<task_id>/cancel', methods=['POST'])
@timer
def cancel_task(task_id: str):
    """
    Cancel a running task
    """
    start_time = time.time()
    
    celery = get_celery()
    if not celery:
        return APIResponse.error("Task queue not available", 503).to_dict(), 503
    
    try:
        task = celery.AsyncResult(task_id)
        
        if task.status in ['PENDING', 'PROGRESS']:
            task.revoke(terminate=True)
            message = "Task cancellation requested"
        else:
            message = f"Task already in terminal state: {task.status}"
        
        response_data = {
            "task_id": task_id,
            "status": task.status,
            "message": message
        }
        
        return APIResponse.success(
            data=response_data,
            latency_ms=(time.time() - start_time) * 1000
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return APIResponse.error(f"Failed to cancel task: {str(e)}", 500).to_dict(), 500