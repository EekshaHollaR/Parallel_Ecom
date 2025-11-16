import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import json
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Load and process recommendation datasets in MovieLens/Amazon format"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.ratings_df = None
        self.products_df = None
        
    def load_movielens_format(self, ratings_file: str, products_file: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load MovieLens format data (user_id, item_id, rating, timestamp)
        """
        try:
            # Load ratings
            ratings_path = os.path.join(self.data_dir, ratings_file)
            self.ratings_df = pd.read_csv(ratings_path)
            
            # Expected columns: user_id, item_id, rating
            required_cols = ['user_id', 'item_id', 'rating']
            if not all(col in self.ratings_df.columns for col in required_cols):
                raise ValueError(f"Ratings file must contain columns: {required_cols}")
            
            # Load products if provided
            if products_file:
                products_path = os.path.join(self.data_dir, products_file)
                self.products_df = pd.read_csv(products_path)
                logger.info(f"Loaded {len(self.products_df)} products")
            
            logger.info(f"Loaded {len(self.ratings_df)} ratings from {len(self.ratings_df['user_id'].unique())} users")
            return self.ratings_df, self.products_df
            
        except Exception as e:
            logger.error(f"Error loading MovieLens data: {e}")
            raise
    
    def load_amazon_format(self, ratings_file: str, products_file: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load Amazon format data (reviewerID, asin, overall, unixReviewTime)
        """
        try:
            # Load ratings (could be JSON or CSV)
            ratings_path = os.path.join(self.data_dir, ratings_file)
            
            if ratings_file.endswith('.json'):
                with open(ratings_path, 'r') as f:
                    data = [json.loads(line) for line in f]
                self.ratings_df = pd.DataFrame(data)
            else:
                self.ratings_df = pd.read_csv(ratings_path)
            
            # Map Amazon columns to standard names
            column_mapping = {
                'reviewerID': 'user_id',
                'asin': 'item_id', 
                'overall': 'rating',
                'unixReviewTime': 'timestamp'
            }
            
            self.ratings_df = self.ratings_df.rename(columns=column_mapping)
            
            # Load products if provided
            if products_file:
                products_path = os.path.join(self.data_dir, products_file)
                if products_file.endswith('.json'):
                    with open(products_path, 'r') as f:
                        data = [json.loads(line) for line in f]
                    self.products_df = pd.DataFrame(data)
                else:
                    self.products_df = pd.read_csv(products_path)
                
                logger.info(f"Loaded {len(self.products_df)} products")
            
            logger.info(f"Loaded {len(self.ratings_df)} ratings from {len(self.ratings_df['user_id'].unique())} users")
            return self.ratings_df, self.products_df
            
        except Exception as e:
            logger.error(f"Error loading Amazon data: {e}")
            raise
    
    def clean_data(self, min_ratings_per_user: int = 5, min_ratings_per_item: int = 2) -> pd.DataFrame:
        """
        Clean the ratings data by removing users/items with too few ratings
        """
        if self.ratings_df is None:
            raise ValueError("No ratings data loaded. Call load_*_format first.")
        
        original_shape = self.ratings_df.shape[0]
        
        # Remove duplicates
        self.ratings_df = self.ratings_df.drop_duplicates(['user_id', 'item_id'])
        
        # Filter users with minimum ratings
        user_rating_counts = self.ratings_df['user_id'].value_counts()
        valid_users = user_rating_counts[user_rating_counts >= min_ratings_per_user].index
        self.ratings_df = self.ratings_df[self.ratings_df['user_id'].isin(valid_users)]
        
        # Filter items with minimum ratings  
        item_rating_counts = self.ratings_df['item_id'].value_counts()
        valid_items = item_rating_counts[item_rating_counts >= min_ratings_per_item].index
        self.ratings_df = self.ratings_df[self.ratings_df['item_id'].isin(valid_items)]
        
        # Handle missing values
        self.ratings_df = self.ratings_df.dropna(subset=['user_id', 'item_id', 'rating'])
        
        logger.info(f"Cleaned data: {original_shape} -> {self.ratings_df.shape[0]} ratings "
                   f"({len(valid_users)} users, {len(valid_items)} items)")
        
        return self.ratings_df
    
    def create_user_item_matrix(self) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
        """
        Convert ratings to sparse user-item matrix
        Returns: (sparse_matrix, user_mapping, item_mapping)
        """
        if self.ratings_df is None:
            raise ValueError("No ratings data available")
        
        # Create mappings
        unique_users = self.ratings_df['user_id'].unique()
        unique_items = self.ratings_df['item_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        # Create sparse matrix
        user_indices = self.ratings_df['user_id'].map(user_to_idx)
        item_indices = self.ratings_df['item_id'].map(item_to_idx)
        ratings = self.ratings_df['rating'].values
        
        sparse_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items))
        )
        
        logger.info(f"Created user-item matrix: {sparse_matrix.shape} "
                   f"(sparsity: {(1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])):.3f})")
        
        return sparse_matrix, user_to_idx, item_to_idx
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return dataset statistics"""
        if self.ratings_df is None:
            return {}
        
        stats = {
            'num_ratings': len(self.ratings_df),
            'num_users': self.ratings_df['user_id'].nunique(),
            'num_items': self.ratings_df['item_id'].nunique(),
            'rating_range': (self.ratings_df['rating'].min(), self.ratings_df['rating'].max()),
            'avg_ratings_per_user': len(self.ratings_df) / self.ratings_df['user_id'].nunique(),
            'avg_ratings_per_item': len(self.ratings_df) / self.ratings_df['item_id'].nunique(),
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    loader = DatasetLoader("data/")
    
    # For MovieLens format
    ratings, products = loader.load_movielens_format("ratings.csv", "movies.csv")
    cleaned_ratings = loader.clean_data(min_ratings_per_user=5, min_ratings_per_item=2)
    sparse_matrix, user_map, item_map = loader.create_user_item_matrix()
    
    stats = loader.get_dataset_stats()
    print("Dataset stats:", stats)