import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict, List, Optional, Any
import logging
import time
from dataclasses import dataclass

try:
    from .parallel_engine import ParallelEngine
    from .gpu_engine import GPUEngine
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPU engine not available, falling back to CPU")

logger = logging.getLogger(__name__)

@dataclass
class ALSConfig:
    """Configuration for ALS-NCG training"""
    rank: int = 50
    max_epochs: int = 20
    regularization: float = 0.1
    convergence_tol: float = 1e-4
    ncg_steps: int = 5
    ncg_tol: float = 1e-6
    use_gpu: bool = False
    n_workers: int = 4
    random_state: int = 42

class ALS_NCG_Recommender:
    """
    ALS with Nonlinear Conjugate Gradient optimization
    Supports multiprocessing and optional GPU acceleration
    """
    
    def __init__(self, config: ALSConfig):
        self.config = config
        self.parallel_engine = ParallelEngine(n_workers=config.n_workers)
        
        # Initialize GPU engine if available and requested
        self.gpu_engine = None
        if config.use_gpu and GPU_AVAILABLE:
            try:
                self.gpu_engine = GPUEngine()
                logger.info("GPU engine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU engine: {e}. Using CPU.")
        
        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        
        # Training history
        self.training_history = {
            'epoch_times': [],
            'train_rmse': [],
            'test_rmse': [],
            'convergence': []
        }
    
    def initialize_parameters(self, n_users: int, n_items: int):
        """Initialize factor matrices and biases"""
        rng = np.random.RandomState(self.config.random_state)
        
        # Initialize factors with small random values
        self.user_factors = rng.normal(0, 0.1, (n_users, self.config.rank))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.config.rank))
        
        # Initialize biases
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0
        
        logger.info(f"Initialized parameters for {n_users} users, {n_items} items")
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair"""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained yet")
        
        prediction = self.global_bias + self.user_bias[user_idx] + self.item_bias[item_idx]
        prediction += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return prediction
    
    def predict_batch(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """Predict ratings for batch of user-item pairs"""
        if self.gpu_engine and self.config.use_gpu:
            return self.gpu_engine.predict_batch(
                self.user_factors, self.item_factors,
                self.user_bias, self.item_bias, self.global_bias,
                user_indices, item_indices
            )
        else:
            user_factors_batch = self.user_factors[user_indices]
            item_factors_batch = self.item_factors[item_indices]
            user_bias_batch = self.user_bias[user_indices]
            item_bias_batch = self.item_bias[item_indices]
            
            predictions = (self.global_bias + user_bias_batch + item_bias_batch + 
                         np.sum(user_factors_batch * item_factors_batch, axis=1))
            return predictions
    
    def calculate_rmse(self, matrix: sp.csr_matrix, mask: Optional[np.ndarray] = None) -> float:
        """Calculate RMSE for given sparse matrix"""
        if matrix.nnz == 0:
            return 0.0
        
        user_indices, item_indices = matrix.nonzero()
        actual_ratings = matrix.data
        
        if mask is not None:
            # Apply mask if provided
            mask_values = mask[user_indices, item_indices]
            valid_indices = mask_values > 0
            user_indices = user_indices[valid_indices]
            item_indices = item_indices[valid_indices]
            actual_ratings = actual_ratings[valid_indices]
        
        predicted_ratings = self.predict_batch(user_indices, item_indices)
        
        mse = np.mean((actual_ratings - predicted_ratings) ** 2)
        return np.sqrt(mse)
    
    def ncg_optimize(self, factors: np.ndarray, fixed_factors: np.ndarray, 
                    ratings: sp.csr_matrix, is_user: bool) -> np.ndarray:
        """
        Nonlinear Conjugate Gradient optimization for factors
        """
        n_entities = factors.shape[0]
        improved_factors = factors.copy()
        
        for i in range(n_entities):
            if is_user:
                # Get items rated by this user
                user_ratings = ratings[i]
                if user_ratings.nnz == 0:
                    continue
                    
                item_indices = user_ratings.indices
                actual_ratings = user_ratings.data
                fixed_vectors = fixed_factors[item_indices]
            else:
                # Get users who rated this item
                item_ratings = ratings[:, i]
                if item_ratings.nnz == 0:
                    continue
                    
                user_indices = item_ratings.indices
                actual_ratings = item_ratings.data
                fixed_vectors = fixed_factors[user_indices]
            
            # Current factors for this entity
            current_f = improved_factors[i]
            
            # NCG optimization
            x = current_f.copy()
            r = self._compute_gradient(x, fixed_vectors, actual_ratings, is_user)
            p = -r
            r_old = r.dot(r)
            
            for _ in range(self.config.ncg_steps):
                if np.linalg.norm(p) < self.config.ncg_tol:
                    break
                    
                Ap = self._compute_hessian_vector_product(p, fixed_vectors, is_user)
                alpha = r_old / p.dot(Ap)
                x = x + alpha * p
                r = r + alpha * Ap
                r_new = r.dot(r)
                
                if r_new < self.config.ncg_tol:
                    break
                    
                beta = r_new / r_old
                p = -r + beta * p
                r_old = r_new
            
            improved_factors[i] = x
        
        return improved_factors
    
    def _compute_gradient(self, x: np.ndarray, fixed_vectors: np.ndarray, 
                         ratings: np.ndarray, is_user: bool) -> np.ndarray:
        """Compute gradient for NCG"""
        predictions = fixed_vectors.dot(x)
        errors = predictions - ratings
        
        gradient = 2 * fixed_vectors.T.dot(errors) + 2 * self.config.regularization * x
        return gradient
    
    def _compute_hessian_vector_product(self, p: np.ndarray, fixed_vectors: np.ndarray, 
                                       is_user: bool) -> np.ndarray:
        """Compute Hessian-vector product for NCG"""
        return 2 * fixed_vectors.T.dot(fixed_vectors.dot(p)) + 2 * self.config.regularization * p
    
    def update_user_factors(self, train_matrix: sp.csr_matrix) -> np.ndarray:
        """Update user factors using ALS with NCG optimization"""
        start_time = time.time()
        
        if self.gpu_engine and self.config.use_gpu:
            new_user_factors = self.gpu_engine.update_factors(
                self.item_factors, train_matrix, self.config.regularization, is_user=True
            )
        else:
            # Use parallel processing for CPU
            new_user_factors = self.parallel_engine.parallel_factor_update(
                self.user_factors, self.item_factors, train_matrix, 
                self.config.regularization, is_user=True, ncg_optimize=True
            )
        
        logger.debug(f"User factors update completed in {time.time() - start_time:.2f}s")
        return new_user_factors
    
    def update_item_factors(self, train_matrix: sp.csr_matrix) -> np.ndarray:
        """Update item factors using ALS with NCG optimization"""
        start_time = time.time()
        
        if self.gpu_engine and self.config.use_gpu:
            new_item_factors = self.gpu_engine.update_factors(
                self.user_factors, train_matrix.T, self.config.regularization, is_user=False
            )
        else:
            # Use parallel processing for CPU
            new_item_factors = self.parallel_engine.parallel_factor_update(
                self.item_factors, self.user_factors, train_matrix.T, 
                self.config.regularization, is_user=False, ncg_optimize=True
            )
        
        logger.debug(f"Item factors update completed in {time.time() - start_time:.2f}s")
        return new_item_factors
    
    def fit(self, train_matrix: sp.csr_matrix, test_matrix: Optional[sp.csr_matrix] = None,
            user_mapping: Optional[Dict] = None, item_mapping: Optional[Dict] = None):
        """
        Train ALS-NCG model
        """
        n_users, n_items = train_matrix.shape
        
        # Initialize parameters if not already done
        if self.user_factors is None:
            self.initialize_parameters(n_users, n_items)
        
        logger.info(f"Starting ALS-NCG training for {n_users} users, {n_items} items")
        logger.info(f"Configuration: rank={self.config.rank}, epochs={self.config.max_epochs}")
        
        prev_rmse = float('inf')
        
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            
            # ALS step: alternate between updating user and item factors
            self.user_factors = self.update_user_factors(train_matrix)
            self.item_factors = self.update_item_factors(train_matrix)
            
            # Calculate metrics
            train_rmse = self.calculate_rmse(train_matrix)
            test_rmse = self.calculate_rmse(test_matrix) if test_matrix is not None else 0.0
            
            epoch_time = time.time() - epoch_start
            convergence = prev_rmse - train_rmse
            
            # Store history
            self.training_history['epoch_times'].append(epoch_time)
            self.training_history['train_rmse'].append(train_rmse)
            self.training_history['test_rmse'].append(test_rmse)
            self.training_history['convergence'].append(convergence)
            
            logger.info(f"Epoch {epoch+1}/{self.config.max_epochs}: "
                       f"Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}, "
                       f"Time={epoch_time:.2f}s, Δ={convergence:.6f}")
            
            # Check convergence
            if convergence < self.config.convergence_tol and epoch > 0:
                logger.info(f"Convergence reached at epoch {epoch+1}")
                break
            
            prev_rmse = train_rmse
        
        total_time = sum(self.training_history['epoch_times'])
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.training_history
    
    def recommend_for_user(self, user_idx: int, n_recommendations: int = 10, 
                          exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained yet")
        
        # Get user factor vector
        user_vector = self.user_factors[user_idx]
        user_bias_val = self.user_bias[user_idx]
        
        # Compute scores for all items
        if self.gpu_engine and self.config.use_gpu:
            scores = self.gpu_engine.compute_user_scores(
                user_vector, user_bias_val, self.item_factors, 
                self.item_bias, self.global_bias
            )
        else:
            scores = (self.global_bias + user_bias_val + self.item_bias + 
                     np.dot(self.item_factors, user_vector))
        
        # Exclude already rated items if requested
        if exclude_rated:
            # This would require the original ratings matrix
            # For API use, this should be handled by the calling code
            pass
        
        # Get top-N recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(idx, float(scores[idx])) for idx in top_indices]
        
        return recommendations
    
    def get_similar_items(self, item_idx: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find similar items using cosine similarity in factor space"""
        if self.item_factors is None:
            raise ValueError("Model not trained yet")
        
        item_vector = self.item_factors[item_idx]
        similarities = self.item_factors.dot(item_vector)
        
        # Exclude the item itself
        similarities[item_idx] = -1
        
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        similar_items = [(idx, float(similarities[idx])) for idx in top_indices]
        
        return similar_items
    
    def save_model(self, filepath: str):
        """Save model to file"""
        import pickle
        
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'global_bias': self.global_bias,
            'config': self.config,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.user_bias = model_data['user_bias']
        self.item_bias = model_data['item_bias']
        self.global_bias = model_data['global_bias']
        self.config = model_data['config']
        self.training_history = model_data['training_history']
        
        logger.info(f"Model loaded from {filepath}")


# API-ready factory function
def create_als_recommender(rank: int = 50, max_epochs: int = 20, regularization: float = 0.1,
                          use_gpu: bool = False, n_workers: int = 4) -> ALS_NCG_Recommender:
    """Factory function to create ALS recommender with given parameters"""
    config = ALSConfig(
        rank=rank,
        max_epochs=max_epochs,
        regularization=regularization,
        use_gpu=use_gpu,
        n_workers=n_workers
    )
    
    return ALS_NCG_Recommender(config)