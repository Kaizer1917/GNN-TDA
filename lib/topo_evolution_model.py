import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
from .persistent_homology import PersistentHomology
from .topo_signal_processing import TopologicalSignalProcessor
from .simplicial_analysis import SimplicialComplexAnalyzer

class TopologicalAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4):
        """
        Topological attention mechanism that incorporates persistence diagram features.
        
        Args:
            feature_dim (int): Dimension of input features
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.topo_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                topo_features: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of topological attention.
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            topo_features (torch.Tensor): Topological features [batch_size, topo_dim]
            mask (torch.Tensor): Optional attention mask
            
        Returns:
            torch.Tensor: Attended features
        """
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, and values
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Incorporate topological features
        topo_proj = self.topo_proj(topo_features).unsqueeze(1)  # [batch_size, 1, feature_dim]
        k = k + topo_proj.view(batch_size, 1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(2, 3))
        scores = scores / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v.transpose(1, 2))
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.output_proj(out)

class TopoEvolutionModel(nn.Module):
    def __init__(self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                num_layers: int = 3,
                num_heads: int = 4,
                dropout: float = 0.1):
        """
        Topological Evolution Model for traffic forecasting.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
            num_layers (int): Number of model layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.topo_calculator = PersistentHomology(max_dimension=1)
        
        # Add topological signal processor
        self.signal_processor = TopologicalSignalProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_scales=4,
            max_dimension=3
        )
        
        # Add simplicial analyzer
        self.simplicial_analyzer = SimplicialComplexAnalyzer(max_dimension=3)
        
        # Topological feature processing
        self.topo_encoder = nn.Sequential(
            nn.Linear(20 + hidden_dim, hidden_dim),  # Increased input size for additional features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stack of attention layers
        self.layers = nn.ModuleList([
            TopologicalAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _compute_enhanced_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced topological features using multiple methods.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Enhanced topological features
        """
        batch_size = x.size(0)
        topo_features = []
        
        for i in range(batch_size):
            # Get basic persistence features
            series = x[i].detach().cpu().numpy()
            topo_result = self.topo_calculator.process_time_series(
                series.reshape(-1), embedding_dimension=3)
            
            # Get spectral features
            points = series.reshape(-1, 3)
            spectral_features = self.simplicial_analyzer.compute_spectral_features(
                points, threshold=0.1)
            
            # Combine features
            combined_features = np.concatenate([
                topo_result['topological_features'],
                spectral_features['hodge_spectrum_dim_0'][:5],
                spectral_features['hodge_spectrum_dim_1'][:5],
                spectral_features['hodge_spectrum_dim_2'][:5]
            ])
            
            topo_features.append(torch.from_numpy(combined_features).float())
            
        return torch.stack(topo_features).to(x.device)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Process input through topological signal processor
        processed_signal = self.signal_processor(x.view(x.size(0), -1))
        processed_signal = processed_signal.view(x.size())
        
        # Project input
        x = self.input_proj(processed_signal)
        
        # Compute enhanced topological features
        topo_features = self._compute_enhanced_features(x)
        topo_features = self.topo_encoder(topo_features)
        
        # Apply attention layers
        for layer, norm in zip(self.layers, self.layer_norms):
            attended = layer(x, topo_features, mask)
            x = norm(x + self.dropout(attended))
        
        # Project to output dimension
        return self.output_proj(x)

class TopoEvolutionLoss(nn.Module):
    def __init__(self, alpha: float = 0.1):
        """
        Custom loss function incorporating topological persistence.
        
        Args:
            alpha (float): Weight for topological regularization
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.topo_calculator = PersistentHomology(max_dimension=1)
        
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Topological regularization
        pred_topo = self._compute_topological_features(pred)
        target_topo = self._compute_topological_features(target)
        topo_loss = self.mse(pred_topo, target_topo)
        
        return mse_loss + self.alpha * topo_loss
        
    def _compute_topological_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute topological features for loss calculation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Topological features
        """
        batch_size = x.size(0)
        topo_features = []
        
        for i in range(batch_size):
            series = x[i].detach().cpu().numpy()
            topo_result = self.topo_calculator.process_time_series(
                series.reshape(-1), embedding_dimension=3)
            features = torch.from_numpy(
                topo_result['topological_features']).float().to(x.device)
            topo_features.append(features)
            
        return torch.stack(topo_features)
