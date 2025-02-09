import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from typing import List, Dict, Tuple, Any
import torch
import torch.nn as nn

class SchrodingerPCA:
    def __init__(self,
                 n_components: int = 10,
                 sigma: float = 1.0,
                 hbar: float = 1.0,
                 mass: float = 1.0,
                 dt: float = 0.01):
        """
        Schrödinger PCA for dimensionality reduction and feature extraction.
        
        Args:
            n_components (int): Number of components to keep
            sigma (float): Width of Gaussian kernel
            hbar (float): Reduced Planck constant
            mass (float): Particle mass
            dt (float): Time step for evolution
        """
        self.n_components = n_components
        self.sigma = sigma
        self.hbar = hbar
        self.mass = mass
        self.dt = dt
        self.components_ = None
        self.eigenvalues_ = None
        
    def _compute_kernel(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian kernel matrix.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Kernel matrix
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            diff = X - X[i]
            K[i] = np.exp(-np.sum(diff**2, axis=1) / (2 * self.sigma**2))
            
        return K
    
    def _compute_hamiltonian(self, K: np.ndarray) -> csr_matrix:
        """
        Compute Hamiltonian operator.
        
        Args:
            K (np.ndarray): Kernel matrix
            
        Returns:
            csr_matrix: Hamiltonian operator
        """
        n_samples = K.shape[0]
        
        # Compute Laplacian
        D = np.sum(K, axis=1)
        L = diags(D) - csr_matrix(K)
        
        # Kinetic energy term
        T = -self.hbar**2 / (2 * self.mass) * L
        
        # Potential energy term (using kernel as potential)
        V = diags(np.diag(K))
        
        # Total Hamiltonian
        H = T + V
        
        return H
    
    def _time_evolution(self, H: csr_matrix) -> np.ndarray:
        """
        Compute time evolution operator.
        
        Args:
            H (csr_matrix): Hamiltonian operator
            
        Returns:
            np.ndarray: Time evolution operator
        """
        # Use sparse eigendecomposition for efficiency
        eigenvals, eigenvecs = eigsh(H, k=self.n_components, which='SA')
        
        # Compute time evolution
        U = eigenvecs @ np.diag(np.exp(-1j * eigenvals * self.dt / self.hbar)) @ eigenvecs.T.conj()
        
        return U
    
    def fit(self, X: np.ndarray) -> 'SchrodingerPCA':
        """
        Fit the Schrödinger PCA model.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        # Center the data
        X = X - np.mean(X, axis=0)
        
        # Compute kernel
        K = self._compute_kernel(X)
        
        # Compute Hamiltonian
        H = self._compute_hamiltonian(K)
        
        # Compute time evolution
        U = self._time_evolution(H)
        
        # Perform eigendecomposition of time evolution operator
        eigenvals, eigenvecs = linalg.eigh(U)
        
        # Sort by absolute eigenvalues
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Store components and eigenvalues
        self.components_ = eigenvecs[:, :self.n_components]
        self.eigenvalues_ = eigenvals[:self.n_components]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted Schrödinger PCA.
        
        Args:
            X (np.ndarray): Data to transform of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Transformed data
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before transform")
            
        # Center the data
        X = X - np.mean(X, axis=0)
        
        # Project data onto components
        return X @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Args:
            X (np.ndarray): Training data
            
        Returns:
            np.ndarray: Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed (np.ndarray): Transformed data
            
        Returns:
            np.ndarray: Data in original space
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before inverse_transform")
            
        return X_transformed @ self.components_.T

class SchrodingerFeatureExtractor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_components: int = 10,
                 sigma: float = 1.0):
        """
        Neural network module for Schrödinger feature extraction.
        
        Args:
            input_dim (int): Input dimension
            n_components (int): Number of components
            sigma (float): Kernel width
        """
        super().__init__()
        
        self.spca = SchrodingerPCA(
            n_components=n_components,
            sigma=sigma
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_components, n_components * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_components * 2, n_components)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        # Convert to numpy for SPCA
        x_np = x.detach().cpu().numpy()
        
        # Apply Schrödinger PCA
        x_transformed = self.spca.fit_transform(x_np)
        
        # Convert back to tensor
        x_transformed = torch.FloatTensor(x_transformed).to(x.device)
        
        # Extract additional features
        features = self.feature_extractor(x_transformed)
        
        return features
