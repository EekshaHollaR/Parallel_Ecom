import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count
import logging
from typing import Tuple, Callable
import time

logger = logging.getLogger(__name__)

class ParallelEngine:
    """Multiprocessing engine for parallel ALS operations"""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.pool = None
        
    def __enter__(self):
        self.pool = Pool(self.n_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            self.pool.join()
    
    def parallel_factor_update(self, current_factors: np.ndarray, fixed_factors: np.ndarray,
                              ratings: sp.csr_matrix, regularization: float, 
                              is_user: bool = True, ncg_optimize: bool = True) -> np.ndarray:
        """
        Parallel update of factors using ALS with optional NCG optimization
        """
        n_entities = current_factors.shape[0]
        chunk_size = max(1, n_entities // self.n_workers)
        
        # Prepare chunks for parallel processing
        chunks = []
        for i in range(0, n_entities, chunk_size):
            chunk_end = min(i + chunk_size, n_entities)
            chunks.append((i, chunk_end))
        
        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            results = pool.starmap(
                self._process_factor_chunk,
                [(chunk_start, chunk_end, current_factors, fixed_factors, 
                  ratings, regularization, is_user, ncg_optimize) 
                 for chunk_start, chunk_end in chunks]
            )
        
        # Combine results
        new_factors = current_factors.copy()
        for chunk_start, chunk_factors in results:
            chunk_end = chunk_start + chunk_factors.shape[0]
            new_factors[chunk_start:chunk_end] = chunk_factors
        
        return new_factors
    
    def _process_factor_chunk(self, chunk_start: int, chunk_end: int,
                             current_factors: np.ndarray, fixed_factors: np.ndarray,
                             ratings: sp.csr_matrix, regularization: float,
                             is_user: bool, ncg_optimize: bool) -> Tuple[int, np.ndarray]:
        """
        Process a chunk of factors (user or item)
        """
        chunk_size = chunk_end - chunk_start
        chunk_factors = np.zeros((chunk_size, current_factors.shape[1]))
        
        for i in range(chunk_size):
            entity_idx = chunk_start + i
            
            if is_user:
                entity_ratings = ratings[entity_idx]
            else:
                entity_ratings = ratings[:, entity_idx]
            
            if entity_ratings.nnz == 0:
                # No ratings, keep current factors
                chunk_factors[i] = current_factors[entity_idx]
                continue
            
            if is_user:
                rated_indices = entity_ratings.indices
                ratings_data = entity_ratings.data
                rated_factors = fixed_factors[rated_indices]
            else:
                rated_indices = entity_ratings.indices
                ratings_data = entity_ratings.data
                rated_factors = fixed_factors[rated_indices]
            
            if ncg_optimize:
                # Use NCG optimization
                chunk_factors[i] = self._ncg_optimize_single(
                    current_factors[entity_idx], rated_factors, 
                    ratings_data, regularization, is_user
                )
            else:
                # Standard ALS update
                chunk_factors[i] = self._als_update_single(
                    rated_factors, ratings_data, regularization
                )
        
        return chunk_start, chunk_factors
    
    def _als_update_single(self, rated_factors: np.ndarray, ratings: np.ndarray,
                          regularization: float) -> np.ndarray:
        """Standard ALS update for a single entity"""
        n_factors = rated_factors.shape[1]
        
        # Compute A = V^T * V + λI
        A = rated_factors.T.dot(rated_factors) + regularization * np.eye(n_factors)
        
        # Compute b = V^T * r
        b = rated_factors.T.dot(ratings)
        
        # Solve Ax = b
        try:
            new_factors = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            new_factors = np.linalg.pinv(A).dot(b)
        
        return new_factors
    
    def _ncg_optimize_single(self, current_factors: np.ndarray, rated_factors: np.ndarray,
                            ratings: np.ndarray, regularization: float, 
                            is_user: bool, max_iter: int = 5) -> np.ndarray:
        """NCG optimization for a single entity"""
        x = current_factors.copy()
        
        # Initial gradient
        r = self._compute_gradient_single(x, rated_factors, ratings, regularization)
        p = -r
        r_old = r.dot(r)
        
        for iteration in range(max_iter):
            if np.linalg.norm(p) < 1e-6:
                break
                
            # Hessian-vector product
            Ap = self._compute_hessian_vector_single(p, rated_factors, regularization)
            alpha = r_old / p.dot(Ap)
            x = x + alpha * p
            r = r + alpha * Ap
            r_new = r.dot(r)
            
            if r_new < 1e-6:
                break
                
            beta = r_new / r_old
            p = -r + beta * p
            r_old = r_new
        
        return x
    
    def _compute_gradient_single(self, x: np.ndarray, rated_factors: np.ndarray,
                               ratings: np.ndarray, regularization: float) -> np.ndarray:
        """Compute gradient for single entity"""
        predictions = rated_factors.dot(x)
        errors = predictions - ratings
        
        gradient = 2 * rated_factors.T.dot(errors) + 2 * regularization * x
        return gradient
    
    def _compute_hessian_vector_single(self, p: np.ndarray, rated_factors: np.ndarray,
                                     regularization: float) -> np.ndarray:
        """Compute Hessian-vector product for single entity"""
        return 2 * rated_factors.T.dot(rated_factors.dot(p)) + 2 * regularization * p
    
    def parallel_predict_batch(self, user_indices: np.ndarray, item_indices: np.ndarray,
                              user_factors: np.ndarray, item_factors: np.ndarray,
                              user_bias: np.ndarray, item_bias: np.ndarray,
                              global_bias: float) -> np.ndarray:
        """Parallel batch prediction"""
        n_predictions = len(user_indices)
        chunk_size = max(1, n_predictions // self.n_workers)
        
        chunks = []
        for i in range(0, n_predictions, chunk_size):
            chunk_end = min(i + chunk_size, n_predictions)
            chunks.append((i, chunk_end))
        
        with Pool(self.n_workers) as pool:
            results = pool.starmap(
                self._predict_chunk,
                [(chunk_start, chunk_end, user_indices, item_indices,
                  user_factors, item_factors, user_bias, item_bias, global_bias)
                 for chunk_start, chunk_end in chunks]
            )
        
        # Combine predictions
        predictions = np.zeros(n_predictions)
        for chunk_start, chunk_predictions in results:
            chunk_end = chunk_start + len(chunk_predictions)
            predictions[chunk_start:chunk_end] = chunk_predictions
        
        return predictions
    
    def _predict_chunk(self, chunk_start: int, chunk_end: int,
                      user_indices: np.ndarray, item_indices: np.ndarray,
                      user_factors: np.ndarray, item_factors: np.ndarray,
                      user_bias: np.ndarray, item_bias: np.ndarray,
                      global_bias: float) -> Tuple[int, np.ndarray]:
        """Predict ratings for a chunk of user-item pairs"""
        chunk_size = chunk_end - chunk_start
        predictions = np.zeros(chunk_size)
        
        for i in range(chunk_size):
            idx = chunk_start + i
            user_idx = user_indices[idx]
            item_idx = item_indices[idx]
            
            prediction = (global_bias + user_bias[user_idx] + item_bias[item_idx] +
                         np.dot(user_factors[user_idx], item_factors[item_idx]))
            predictions[i] = prediction
        
        return chunk_start, predictions


# Utility function for API use
def create_parallel_engine(n_workers: int = None) -> ParallelEngine:
    """Factory function to create parallel engine"""
    return ParallelEngine(n_workers)