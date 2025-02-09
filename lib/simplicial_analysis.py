import numpy as np
import gudhi
from typing import List, Dict, Tuple, Any
import networkx as nx
from scipy.sparse import csr_matrix
import torch
from scipy.spatial.distance import pdist, squareform

class SimplicialComplexAnalyzer:
    def __init__(self, max_dimension: int = 3):
        """
        Initialize Simplicial Complex Analyzer.
        
        Args:
            max_dimension (int): Maximum dimension for simplicial complexes
        """
        self.max_dimension = max_dimension
        
    def build_flag_complex(self, 
                          points: np.ndarray,
                          threshold: float) -> gudhi.SimplexTree:
        """
        Build a flag complex (clique complex) from point cloud data.
        
        Args:
            points (np.ndarray): Point cloud data
            threshold (float): Distance threshold for edge creation
            
        Returns:
            gudhi.SimplexTree: Constructed flag complex
        """
        # Compute pairwise distances
        distances = pdist(points)
        dist_matrix = squareform(distances)
        
        # Create graph from distance matrix
        n_points = len(points)
        edges = [(i, j) for i in range(n_points) for j in range(i+1, n_points)
                if dist_matrix[i, j] <= threshold]
        
        # Build flag complex
        st = gudhi.SimplexTree()
        
        # Add vertices
        for i in range(n_points):
            st.insert([i])
            
        # Add edges and compute flag complex
        for edge in edges:
            st.insert(list(edge))
        
        # Expand to flag complex
        st.expand(self.max_dimension)
        
        return st
    
    def compute_hodge_laplacian(self, 
                               simplex_tree: gudhi.SimplexTree,
                               dimension: int) -> csr_matrix:
        """
        Compute the Hodge Laplacian for a given dimension.
        
        Args:
            simplex_tree (gudhi.SimplexTree): Input simplicial complex
            dimension (int): Dimension to compute Laplacian for
            
        Returns:
            csr_matrix: Hodge Laplacian matrix
        """
        # Get boundary matrices
        boundary_k = self._get_boundary_matrix(simplex_tree, dimension)
        boundary_k_plus_1 = self._get_boundary_matrix(simplex_tree, dimension + 1)
        
        # Compute Hodge Laplacian components
        if boundary_k is not None:
            lower_laplacian = boundary_k @ boundary_k.T
        else:
            lower_laplacian = 0
            
        if boundary_k_plus_1 is not None:
            upper_laplacian = boundary_k_plus_1.T @ boundary_k_plus_1
        else:
            upper_laplacian = 0
            
        # Combine components
        hodge_laplacian = lower_laplacian + upper_laplacian
        
        return hodge_laplacian
    
    def _get_boundary_matrix(self,
                           simplex_tree: gudhi.SimplexTree,
                           dimension: int) -> csr_matrix:
        """
        Compute boundary matrix for given dimension.
        
        Args:
            simplex_tree (gudhi.SimplexTree): Input simplicial complex
            dimension (int): Dimension of simplices
            
        Returns:
            csr_matrix: Boundary matrix
        """
        if dimension < 1:
            return None
            
        # Get simplices of dimensions k and k-1
        simplices_k = [simplex for simplex, _ in simplex_tree.get_simplices()
                      if len(simplex) == dimension + 1]
        simplices_k_minus_1 = [simplex for simplex, _ in simplex_tree.get_simplices()
                             if len(simplex) == dimension]
                             
        if not simplices_k or not simplices_k_minus_1:
            return None
            
        # Create index mappings
        k_dict = {tuple(sorted(s)): i for i, s in enumerate(simplices_k)}
        k_minus_1_dict = {tuple(sorted(s)): i for i, s in enumerate(simplices_k_minus_1)}
        
        # Initialize matrix entries
        rows, cols, data = [], [], []
        
        # Compute boundary matrix entries
        for simplex in simplices_k:
            simplex_idx = k_dict[tuple(sorted(simplex))]
            
            # Generate faces of dimension k-1
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i+1:]
                face_tuple = tuple(sorted(face))
                
                if face_tuple in k_minus_1_dict:
                    face_idx = k_minus_1_dict[face_tuple]
                    # Compute orientation
                    orientation = (-1)**i
                    
                    rows.append(face_idx)
                    cols.append(simplex_idx)
                    data.append(orientation)
        
        # Create sparse matrix
        return csr_matrix((data, (rows, cols)),
                        shape=(len(simplices_k_minus_1), len(simplices_k)))
    
    def compute_spectral_features(self,
                                points: np.ndarray,
                                threshold: float) -> Dict[str, np.ndarray]:
        """
        Compute spectral features from simplicial complex.
        
        Args:
            points (np.ndarray): Input point cloud
            threshold (float): Distance threshold
            
        Returns:
            Dict containing spectral features
        """
        # Build simplicial complex
        st = self.build_flag_complex(points, threshold)
        
        features = {}
        
        # Compute Hodge Laplacian spectra for each dimension
        for dim in range(self.max_dimension):
            laplacian = self.compute_hodge_laplacian(st, dim)
            
            if isinstance(laplacian, csr_matrix):
                # Compute eigenvalues
                eigenvals = np.linalg.eigvals(laplacian.toarray())
                eigenvals = np.sort(np.real(eigenvals))
                
                features[f'hodge_spectrum_dim_{dim}'] = eigenvals
                
                # Compute additional spectral features
                features[f'spectral_gap_dim_{dim}'] = eigenvals[1] - eigenvals[0]
                features[f'spectral_entropy_dim_{dim}'] = self._compute_spectral_entropy(eigenvals)
        
        return features
    
    def _compute_spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Compute spectral entropy from eigenvalues.
        
        Args:
            eigenvalues (np.ndarray): Array of eigenvalues
            
        Returns:
            float: Spectral entropy
        """
        # Normalize eigenvalues
        eigenvalues = np.abs(eigenvalues)
        total = np.sum(eigenvalues)
        
        if total == 0:
            return 0.0
            
        probs = eigenvalues / total
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy
    
    def extract_topological_signal(self,
                                 time_series: np.ndarray,
                                 window_size: int,
                                 stride: int = 1) -> np.ndarray:
        """
        Extract topological signal features from time series.
        
        Args:
            time_series (np.ndarray): Input time series
            window_size (int): Size of sliding window
            stride (int): Stride for sliding window
            
        Returns:
            np.ndarray: Topological signal features
        """
        n_samples = len(time_series)
        n_windows = (n_samples - window_size) // stride + 1
        
        features_list = []
        
        for i in range(0, n_samples - window_size + 1, stride):
            # Extract window
            window = time_series[i:i+window_size]
            
            # Create point cloud from window using time delay embedding
            points = np.array([[window[j], window[j+1], window[j+2]] 
                             for j in range(window_size-2)])
            
            # Compute spectral features
            features = self.compute_spectral_features(points, threshold=0.1)
            
            # Extract key features
            feature_vector = np.concatenate([
                features[f'hodge_spectrum_dim_{dim}'][:5]  # Take first 5 eigenvalues
                for dim in range(self.max_dimension)
            ])
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
