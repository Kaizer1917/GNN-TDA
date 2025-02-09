import numpy as np
import gudhi
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import torch

class PersistentHomology:
    def __init__(self, max_dimension: int = 1, max_scale: float = None):
        """
        Initialize the Persistent Homology calculator.
        
        Args:
            max_dimension (int): Maximum homology dimension to compute
            max_scale (float): Maximum value for the filtration parameter
        """
        self.max_dimension = max_dimension
        self.max_scale = max_scale

    def compute_distance_matrix(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Compute the Euclidean distance matrix for a point cloud.
        
        Args:
            point_cloud (np.ndarray): Input point cloud of shape (N, d)
            
        Returns:
            np.ndarray: Distance matrix of shape (N, N)
        """
        N = point_cloud.shape[0]
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    def compute_persistence_diagram(self, 
                                 data: np.ndarray, 
                                 metric: str = 'euclidean') -> Dict[str, Any]:
        """
        Compute persistence diagram from input data.
        
        Args:
            data (np.ndarray): Input data, either point cloud or distance matrix
            metric (str): Distance metric to use if input is point cloud
            
        Returns:
            Dict containing persistence diagram and other topological features
        """
        # Compute persistence diagrams using Ripser
        diagrams = ripser(data, maxdim=self.max_dimension, 
                         metric=metric, thresh=self.max_scale)
        
        # Extract persistence pairs for each dimension
        persistence_pairs = {}
        for dim in range(self.max_dimension + 1):
            persistence_pairs[dim] = diagrams['dgms'][dim]
            
        return {
            'persistence_diagram': diagrams,
            'persistence_pairs': persistence_pairs,
            'betti_numbers': self._compute_betti_numbers(persistence_pairs)
        }

    def _compute_betti_numbers(self, 
                             persistence_pairs: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Compute Betti numbers from persistence pairs.
        
        Args:
            persistence_pairs (Dict): Dictionary of persistence pairs for each dimension
            
        Returns:
            Dict: Betti numbers for each dimension
        """
        betti_numbers = {}
        for dim, pairs in persistence_pairs.items():
            # Count features that persist (finite death time)
            betti_numbers[dim] = np.sum(np.isfinite(pairs[:, 1]))
        return betti_numbers

    def plot_persistence_diagram(self, 
                               persistence_result: Dict[str, Any],
                               title: str = 'Persistence Diagram'):
        """
        Plot the persistence diagram.
        
        Args:
            persistence_result (Dict): Output from compute_persistence_diagram
            title (str): Title for the plot
        """
        plt.figure(figsize=(8, 8))
        plot_diagrams(persistence_result['persistence_diagram']['dgms'], 
                     show=True, title=title)
        plt.grid(True)
        plt.show()

    def extract_topological_features(self, 
                                   persistence_result: Dict[str, Any],
                                   n_features: int = 10) -> np.ndarray:
        """
        Extract topological features from persistence diagram.
        
        Args:
            persistence_result (Dict): Output from compute_persistence_diagram
            n_features (int): Number of features to extract per dimension
            
        Returns:
            np.ndarray: Topological feature vector
        """
        features = []
        
        for dim in range(self.max_dimension + 1):
            pairs = persistence_result['persistence_pairs'][dim]
            
            # Calculate persistence lengths
            persistence = pairs[:, 1] - pairs[:, 0]
            persistence = persistence[np.isfinite(persistence)]
            
            if len(persistence) > 0:
                # Sort by persistence length
                persistence = np.sort(persistence)[::-1]
                
                # Take top n_features
                if len(persistence) >= n_features:
                    features.extend(persistence[:n_features])
                else:
                    # Pad with zeros if not enough features
                    features.extend(persistence)
                    features.extend([0] * (n_features - len(persistence)))
            else:
                features.extend([0] * n_features)
        
        return np.array(features)

    @staticmethod
    def time_series_to_point_cloud(time_series: np.ndarray, 
                                 embedding_dimension: int = 3,
                                 time_delay: int = 1) -> np.ndarray:
        """
        Convert time series to point cloud using time-delay embedding.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dimension (int): Embedding dimension
            time_delay (int): Time delay
            
        Returns:
            np.ndarray: Point cloud representation of time series
        """
        N = len(time_series) - (embedding_dimension - 1) * time_delay
        point_cloud = np.zeros((N, embedding_dimension))
        
        for i in range(N):
            for j in range(embedding_dimension):
                point_cloud[i, j] = time_series[i + j * time_delay]
                
        return point_cloud

    def process_time_series(self,
                          time_series: np.ndarray,
                          embedding_dimension: int = 3,
                          time_delay: int = 1) -> Dict[str, Any]:
        """
        Process time series data using persistent homology.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dimension (int): Embedding dimension for point cloud
            time_delay (int): Time delay for embedding
            
        Returns:
            Dict containing persistence diagram and topological features
        """
        # Convert time series to point cloud
        point_cloud = self.time_series_to_point_cloud(
            time_series, embedding_dimension, time_delay)
        
        # Compute persistence diagram
        persistence_result = self.compute_persistence_diagram(point_cloud)
        
        # Extract topological features
        topological_features = self.extract_topological_features(persistence_result)
        
        return {
            'point_cloud': point_cloud,
            'persistence_diagram': persistence_result,
            'topological_features': topological_features
        }
