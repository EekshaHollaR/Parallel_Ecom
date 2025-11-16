import numpy as np
import scipy.sparse as sp
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import GPU acceleration libraries
try:
    import numba
    from numba import cuda, jit, float32, float64, int32
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available for GPU acceleration")

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    from pycuda.compiler import SourceModule
    PYGPU_AVAILABLE = True
except ImportError:
    PYGPU_AVAILABLE = False
    logger.warning("PyCUDA not available for GPU acceleration")

class GPUEngine:
    """GPU acceleration engine for ALS operations using Numba or PyCUDA"""
    
    def __init__(self, use_numba: bool = True):
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_pycuda = not use_numba and PYGPU_AVAILABLE
        
        if not (self.use_numba or self.use_pycuda):
            raise RuntimeError("No GPU acceleration library available")
        
        if self.use_numba:
            self._compile_numba_kernels()
            logger.info("GPU engine initialized with Numba")
        else:
            self._compile_pycuda_kernels()
            logger.info("GPU engine initialized with PyCUDA")
    
    def _compile_numba_kernels(self):
        """Compile Numba kernels for GPU execution"""
        
        @jit(nopython=True, fastmath=True)
        def predict_batch_cpu(user_factors, item_factors, user_bias, item_bias, global_bias,
                             user_indices, item_indices):
            n_predictions = len(user_indices)
            predictions = np.zeros(n_predictions)
            
            for i in range(n_predictions):
                user_idx = user_indices[i]
                item_idx = item_indices[i]
                
                pred = (global_bias + user_bias[user_idx] + item_bias[item_idx] +
                       np.dot(user_factors[user_idx], item_factors[item_idx]))
                predictions[i] = pred
            
            return predictions
        
        @jit(nopython=True, fastmath=True)
        def compute_user_scores_cpu(user_vector, user_bias, item_factors, item_bias, global_bias):
            n_items = item_factors.shape[0]
            scores = np.zeros(n_items)
            
            for i in range(n_items):
                scores[i] = (global_bias + user_bias + item_bias[i] +
                            np.dot(user_vector, item_factors[i]))
            
            return scores
        
        @jit(nopython=True, fastmath=True)
        def update_factors_cpu(fixed_factors, ratings, regularization, is_user):
            n_entities = ratings.shape[0] if is_user else ratings.shape[1]
            n_factors = fixed_factors.shape[1]
            new_factors = np.zeros((n_entities, n_factors))
            
            for i in range(n_entities):
                if is_user:
                    entity_ratings = ratings[i]
                else:
                    entity_ratings = ratings[:, i]
                
                if entity_ratings.nnz == 0:
                    continue
                
                rated_indices = entity_ratings.indices
                ratings_data = entity_ratings.data
                rated_factors = fixed_factors[rated_indices]
                
                # ALS update
                A = rated_factors.T.dot(rated_factors) + regularization * np.eye(n_factors)
                b = rated_factors.T.dot(ratings_data)
                
                try:
                    new_factors[i] = np.linalg.solve(A, b)
                except:
                    # Fallback to pinv
                    new_factors[i] = np.linalg.pinv(A).dot(b)
            
            return new_factors
        
        # Store compiled functions
        self._predict_batch_numba = predict_batch_cpu
        self._compute_user_scores_numba = compute_user_scores_cpu
        self._update_factors_numba = update_factors_cpu
    
    def _compile_pycuda_kernels(self):
        """Compile PyCUDA kernels for GPU execution"""
        # Simple matrix multiplication kernel for demonstration
        # In practice, you'd have more sophisticated kernels
        kernel_code = """
        __global__ void predict_kernel(float *predictions, float *user_factors, float *item_factors,
                                      float *user_bias, float *item_bias, float global_bias,
                                      int *user_indices, int *item_indices, int n_predictions, int rank) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < n_predictions) {
                int user_idx = user_indices[idx];
                int item_idx = item_indices[idx];
                
                float dot_product = 0.0f;
                for (int i = 0; i < rank; i++) {
                    dot_product += user_factors[user_idx * rank + i] * item_factors[item_idx * rank + i];
                }
                
                predictions[idx] = global_bias + user_bias[user_idx] + item_bias[item_idx] + dot_product;
            }
        }
        """
        
        self.mod = SourceModule(kernel_code)
        self.predict_kernel = self.mod.get_function("predict_kernel")
    
    def predict_batch(self, user_factors: np.ndarray, item_factors: np.ndarray,
                     user_bias: np.ndarray, item_bias: np.ndarray, global_bias: float,
                     user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch prediction"""
        if self.use_numba:
            return self._predict_batch_numba(
                user_factors, item_factors, user_bias, item_bias, global_bias,
                user_indices, item_indices
            )
        else:
            return self._predict_batch_pycuda(
                user_factors, item_factors, user_bias, item_bias, global_bias,
                user_indices, item_indices
            )
    
    def _predict_batch_pycuda(self, user_factors: np.ndarray, item_factors: np.ndarray,
                             user_bias: np.ndarray, item_bias: np.ndarray, global_bias: float,
                             user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """PyCUDA implementation of batch prediction"""
        n_predictions = len(user_indices)
        rank = user_factors.shape[1]
        
        # Allocate GPU memory
        predictions_gpu = gpuarray.zeros(n_predictions, np.float32)
        user_factors_gpu = gpuarray.to_gpu(user_factors.astype(np.float32))
        item_factors_gpu = gpuarray.to_gpu(item_factors.astype(np.float32))
        user_bias_gpu = gpuarray.to_gpu(user_bias.astype(np.float32))
        item_bias_gpu = gpuarray.to_gpu(item_bias.astype(np.float32))
        user_indices_gpu = gpuarray.to_gpu(user_indices.astype(np.int32))
        item_indices_gpu = gpuarray.to_gpu(item_indices.astype(np.int32))
        
        # Configure grid and block
        block_size = 256
        grid_size = (n_predictions + block_size - 1) // block_size
        
        # Execute kernel
        self.predict_kernel(
            predictions_gpu, user_factors_gpu, item_factors_gpu,
            user_bias_gpu, item_bias_gpu, np.float32(global_bias),
            user_indices_gpu, item_indices_gpu, np.int32(n_predictions), np.int32(rank),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        return predictions_gpu.get()
    
    def compute_user_scores(self, user_vector: np.ndarray, user_bias: float,
                           item_factors: np.ndarray, item_bias: np.ndarray,
                           global_bias: float) -> np.ndarray:
        """GPU-accelerated computation of user scores for all items"""
        if self.use_numba:
            return self._compute_user_scores_numba(
                user_vector, user_bias, item_factors, item_bias, global_bias
            )
        else:
            # Fallback to CPU for PyCUDA (implement similarly to predict_batch)
            n_items = item_factors.shape[0]
            scores = np.zeros(n_items)
            
            for i in range(n_items):
                scores[i] = (global_bias + user_bias + item_bias[i] +
                            np.dot(user_vector, item_factors[i]))
            
            return scores
    
    def update_factors(self, fixed_factors: np.ndarray, ratings: sp.csr_matrix,
                      regularization: float, is_user: bool) -> np.ndarray:
        """GPU-accelerated factor updates"""
        if self.use_numba:
            return self._update_factors_numba(
                fixed_factors, ratings, regularization, is_user
            )
        else:
            # Fallback to CPU for PyCUDA
            return self._update_factors_cpu_fallback(
                fixed_factors, ratings, regularization, is_user
            )
    
    def _update_factors_cpu_fallback(self, fixed_factors: np.ndarray, ratings: sp.csr_matrix,
                                   regularization: float, is_user: bool) -> np.ndarray:
        """CPU fallback for factor updates"""
        n_entities = ratings.shape[0] if is_user else ratings.shape[1]
        n_factors = fixed_factors.shape[1]
        new_factors = np.zeros((n_entities, n_factors))
        
        for i in range(n_entities):
            if is_user:
                entity_ratings = ratings[i]
            else:
                entity_ratings = ratings[:, i]
            
            if entity_ratings.nnz == 0:
                continue
            
            rated_indices = entity_ratings.indices
            ratings_data = entity_ratings.data
            rated_factors = fixed_factors[rated_indices]
            
            A = rated_factors.T.dot(rated_factors) + regularization * np.eye(n_factors)
            b = rated_factors.T.dot(ratings_data)
            
            try:
                new_factors[i] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                new_factors[i] = np.linalg.pinv(A).dot(b)
        
        return new_factors


# API-ready factory function
def create_gpu_engine(use_numba: bool = True) -> Optional[GPUEngine]:
    """Factory function to create GPU engine if available"""
    try:
        return GPUEngine(use_numba=use_numba)
    except RuntimeError as e:
        logger.warning(f"GPU engine not available: {e}")
        return None