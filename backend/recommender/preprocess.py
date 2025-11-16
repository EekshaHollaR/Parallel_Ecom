import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess recommendation data for ALS training"""
    
    def __init__(self, min_rating: float = 1.0, max_rating: float = 5.0):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.rating_range = (min_rating, max_rating)
        
    def normalize_ratings(self, ratings: np.ndarray, 
                         target_min: float = 0.0, 
                         target_max: float = 1.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Normalize ratings to target range [target_min, target_max]
        Returns normalized ratings and scaling parameters
        """
        original_min = np.min(ratings)
        original_max = np.max(ratings)
        
        # Linear scaling
        normalized = (ratings - original_min) / (original_max - original_min)
        normalized = normalized * (target_max - target_min) + target_min
        
        scaling_params = {
            'original_min': original_min,
            'original_max': original_max,
            'target_min': target_min,
            'target_max': target_max
        }
        
        logger.info(f"Normalized ratings from [{original_min}, {original_max}] to [{target_min}, {target_max}]")
        
        return normalized, scaling_params
    
    def denormalize_ratings(self, normalized_ratings: np.ndarray, 
                           scaling_params: Dict[str, float]) -> np.ndarray:
        """Reverse the normalization process"""
        original_min = scaling_params['original_min']
        original_max = scaling_params['original_max']
        target_min = scaling_params['target_min']
        target_max = scaling_params['target_max']
        
        # Reverse linear scaling
        ratings = (normalized_ratings - target_min) / (target_max - target_min)
        ratings = ratings * (original_max - original_min) + original_min
        
        return ratings
    
    def split_data_chronological(self, ratings_df: pd.DataFrame, 
                               test_size: float = 0.2,
                               time_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (by timestamp) into train/test
        """
        if time_col not in ratings_df.columns:
            logger.warning(f"Time column '{time_col}' not found. Using random split instead.")
            return self.split_data_random(ratings_df, test_size)
        
        # Sort by timestamp
        sorted_df = ratings_df.sort_values(time_col)
        
        # Split by time
        split_idx = int(len(sorted_df) * (1 - test_size))
        train_df = sorted_df.iloc[:split_idx]
        test_df = sorted_df.iloc[split_idx:]
        
        logger.info(f"Chronological split: {len(train_df)} train, {len(test_df)} test "
                   f"(split time: {sorted_df.iloc[split_idx][time_col]})")
        
        return train_df, test_df
    
    def split_data_random(self, ratings_df: pd.DataFrame, 
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data randomly into train/test
        """
        train_df, test_df = train_test_split(
            ratings_df, 
            test_size=test_size,
            random_state=random_state,
            stratify=ratings_df['user_id']  # Maintain user distribution
        )
        
        logger.info(f"Random split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
    
    def split_data_leave_one_out(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Leave-one-out split: keep last rating per user for test
        """
        if 'timestamp' not in ratings_df.columns:
            raise ValueError("Leave-one-out split requires timestamp column")
        
        # Get last rating for each user
        test_df = ratings_df.sort_values('timestamp').groupby('user_id').tail(1)
        train_df = ratings_df[~ratings_df.index.isin(test_df.index)]
        
        logger.info(f"Leave-one-out split: {len(train_df)} train, {len(test_df)} test "
                   f"({len(test_df['user_id'].unique())} users in test)")
        
        return train_df, test_df
    
    def create_train_test_matrices(self, 
                                 train_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 user_mapping: Dict[Any, int],
                                 item_mapping: Dict[Any, int]) -> Tuple[csr_matrix, csr_matrix]:
        """
        Create sparse matrices for train and test sets
        """
        n_users = len(user_mapping)
        n_items = len(item_mapping)
        
        # Train matrix
        train_user_indices = train_df['user_id'].map(user_mapping)
        train_item_indices = train_df['item_id'].map(item_mapping)
        train_ratings = train_df['rating'].values
        
        train_matrix = csr_matrix(
            (train_ratings, (train_user_indices, train_item_indices)),
            shape=(n_users, n_items)
        )
        
        # Test matrix
        test_user_indices = test_df['user_id'].map(user_mapping)
        test_item_indices = test_df['item_id'].map(item_mapping)
        test_ratings = test_df['rating'].values
        
        test_matrix = csr_matrix(
            (test_ratings, (test_user_indices, test_item_indices)),
            shape=(n_users, n_items)
        )
        
        logger.info(f"Created matrices - Train: {train_matrix.nnz} interactions, "
                   f"Test: {test_matrix.nnz} interactions")
        
        return train_matrix, test_matrix
    
    def save_processed_data(self, data_dict: Dict[str, Any], filepath: str):
        """
        Save processed data to pickle file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        logger.info(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filepath: str) -> Dict[str, Any]:
        """
        Load processed data from pickle file
        """
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        logger.info(f"Loaded processed data from {filepath}")
        return data_dict
    
    def prepare_als_data(self, ratings_df: pd.DataFrame, 
                        split_method: str = 'random',
                        test_size: float = 0.2,
                        normalize: bool = False) -> Dict[str, Any]:
        """
        Complete data preparation pipeline for ALS
        """
        # Create user and item mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        # Split data
        if split_method == 'chronological':
            train_df, test_df = self.split_data_chronological(ratings_df, test_size)
        elif split_method == 'leave_one_out':
            train_df, test_df = self.split_data_leave_one_out(ratings_df)
        else:
            train_df, test_df = self.split_data_random(ratings_df, test_size)
        
        # Normalize if requested
        scaling_params = None
        if normalize:
            train_ratings_normalized, scaling_params = self.normalize_ratings(train_df['rating'].values)
            test_ratings_normalized, _ = self.normalize_ratings(test_df['rating'].values)
            
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df['rating'] = train_ratings_normalized
            test_df['rating'] = test_ratings_normalized
        
        # Create matrices
        train_matrix, test_matrix = self.create_train_test_matrices(
            train_df, test_df, user_to_idx, item_to_idx
        )
        
        # Prepare result dictionary
        result = {
            'train_matrix': train_matrix,
            'test_matrix': test_matrix,
            'train_df': train_df,
            'test_df': test_df,
            'user_mapping': user_to_idx,
            'item_mapping': item_to_idx,
            'scaling_params': scaling_params,
            'num_users': len(unique_users),
            'num_items': len(unique_items),
            'split_method': split_method
        }
        
        logger.info(f"ALS data preparation complete: {result['num_users']} users, "
                   f"{result['num_items']} items, normalized: {normalize}")
        
        return result


# Example usage
if __name__ == "__main__":
    # Example workflow
    from dataset_loader import DatasetLoader
    
    # Load data
    loader = DatasetLoader("data/")
    ratings, products = loader.load_movielens_format("ratings.csv")
    cleaned_ratings = loader.clean_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    als_data = preprocessor.prepare_als_data(
        cleaned_ratings, 
        split_method='random',
        test_size=0.2,
        normalize=True
    )
    
    # Save processed data
    preprocessor.save_processed_data(als_data, "data/processed/als_data.pkl")
    
    print("Data preparation completed successfully!")