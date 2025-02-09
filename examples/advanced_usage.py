import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
from lib.topo_evolution_model import TopoEvolutionModel, TopoEvolutionLoss
from lib.persistent_homology import PersistentHomology
from lib.simplicial_analysis import SimplicialComplexAnalyzer
from lib.topo_signal_processing import (
    TopologicalSignalProcessor,
    TopologicalWaveletTransform,
    TopologicalFeatureExtractor
)
from lib.schrodinger_pca import SchrodingerPCA, SchrodingerFeatureExtractor

class AnomalyDetector:
    def __init__(self, threshold: float = 2.0):
        """
        Anomaly detector using topological features.
        
        Args:
            threshold (float): Number of standard deviations for anomaly threshold
        """
        self.threshold = threshold
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray):
        """Fit the detector on normal data."""
        self.scaler.fit(features)
        
    def detect(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies in features."""
        scaled_features = self.scaler.transform(features)
        scores = np.linalg.norm(scaled_features, axis=1)
        return scores > (scores.mean() + self.threshold * scores.std())

class TopologicalForecaster:
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 forecast_horizon: int,
                 n_scales: int = 4):
        """
        Forecaster using topological features.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            forecast_horizon (int): Number of steps to forecast
            n_scales (int): Number of wavelet scales
        """
        self.model = TopoEvolutionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=forecast_horizon,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        )
        self.optimizer = Adam(self.model.parameters())
        self.loss_fn = TopoEvolutionLoss(alpha=0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, 
              X: torch.Tensor, 
              y: torch.Tensor,
              epochs: int = 100,
              batch_size: int = 32):
        """Train the forecaster."""
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size].to(self.device)
                batch_y = y[i:i+batch_size].to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Generate forecasts."""
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

class TopologicalTimeSeriesAnalyzer:
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 n_scales: int = 4,
                 max_dimension: int = 3):
        """
        Comprehensive topological time series analyzer.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            n_scales (int): Number of wavelet scales
            max_dimension (int): Maximum homology dimension
        """
        # Initialize all components
        self.model = TopoEvolutionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        )
        
        # Add Schrödinger feature extractor
        self.schrodinger_extractor = SchrodingerFeatureExtractor(
            input_dim=input_dim,
            n_components=min(20, hidden_dim),
            sigma=1.0
        )
        
        self.signal_processor = TopologicalSignalProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_scales=n_scales,
            max_dimension=max_dimension
        )
        
        self.wavelet_transform = TopologicalWaveletTransform(
            n_scales=n_scales,
            max_dimension=max_dimension
        )
        
        self.topo_calculator = PersistentHomology(max_dimension=max_dimension)
        self.simplicial_analyzer = SimplicialComplexAnalyzer(max_dimension=max_dimension)
        self.feature_extractor = TopologicalFeatureExtractor(max_dimension=max_dimension)
        
        # Initialize anomaly detector and forecaster
        self.anomaly_detector = AnomalyDetector()
        self.forecaster = TopologicalForecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            forecast_horizon=24  # 24-step ahead forecasting
        )
        
        # Initialize loss functions
        self.topo_loss = TopoEvolutionLoss(alpha=0.1)
        self.mse_loss = MSELoss()
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.signal_processor.to(self.device)
        
    def compute_change_points(self, 
                            signal: np.ndarray,
                            window_size: int = 50) -> List[int]:
        """
        Detect change points using topological features.
        
        Args:
            signal (np.ndarray): Input signal
            window_size (int): Size of sliding window
            
        Returns:
            List[int]: Indices of detected change points
        """
        # Extract features using sliding window
        features = []
        for i in range(len(signal) - window_size):
            window = signal[i:i+window_size]
            topo_result = self.topo_calculator.process_time_series(
                window, embedding_dimension=3)
            features.append(topo_result['topological_features'])
            
        features = np.array(features)
        
        # Compute feature differences
        diff = np.diff(features, axis=0)
        scores = np.linalg.norm(diff, axis=1)
        
        # Detect change points
        threshold = np.mean(scores) + 2 * np.std(scores)
        change_points = np.where(scores > threshold)[0] + 1
        
        return change_points.tolist()
    
    def compute_complexity_measures(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute various topological complexity measures.
        
        Args:
            signal (np.ndarray): Input signal
            
        Returns:
            Dict[str, float]: Complexity measures
        """
        # Get persistence features
        topo_result = self.topo_calculator.process_time_series(
            signal, embedding_dimension=3)
        
        # Compute persistence entropy
        persistence = topo_result['persistence_pairs'][1][:, 1] - topo_result['persistence_pairs'][1][:, 0]
        persistence = persistence[np.isfinite(persistence)]
        persistence = persistence / persistence.sum()
        entropy = -np.sum(persistence * np.log2(persistence + 1e-10))
        
        # Get spectral features
        points = np.array([[signal[i], signal[i+1], signal[i+2]] 
                          for i in range(len(signal)-2)])
        spectral = self.simplicial_analyzer.compute_spectral_features(points, threshold=0.1)
        
        return {
            'persistence_entropy': entropy,
            'spectral_complexity': np.mean([
                spectral[f'spectral_entropy_dim_{dim}']
                for dim in range(3)
            ]),
            'betti_numbers': np.sum([
                len(topo_result['persistence_pairs'][dim])
                for dim in range(2)
            ])
        }
    
    def analyze_periodicity(self, 
                          signal: np.ndarray,
                          max_period: int = 100) -> Dict[str, Any]:
        """
        Analyze signal periodicity using topological features.
        
        Args:
            signal (np.ndarray): Input signal
            max_period (int): Maximum period to consider
            
        Returns:
            Dict[str, Any]: Periodicity analysis results
        """
        # Compute sliding window persistence diagrams
        window_size = min(len(signal) // 4, max_period)
        persistence_features = []
        
        for i in range(len(signal) - window_size):
            window = signal[i:i+window_size]
            topo_result = self.topo_calculator.process_time_series(
                window, embedding_dimension=3)
            persistence_features.append(topo_result['topological_features'])
            
        persistence_features = np.array(persistence_features)
        
        # Compute autocorrelation of topological features
        autocorr = np.correlate(persistence_features[:, 0], 
                              persistence_features[:, 0], 
                              mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))
        
        # Sort peaks by correlation value
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'autocorrelation': autocorr,
            'top_periods': [p[0] for p in peaks[:5]],
            'period_scores': [p[1] for p in peaks[:5]]
        }
    
    def compute_quantum_features(self, 
                               signal: np.ndarray,
                               window_size: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute quantum-inspired features using Schrödinger PCA.
        
        Args:
            signal (np.ndarray): Input signal
            window_size (int): Window size for sliding window analysis
            
        Returns:
            Dict[str, np.ndarray]: Quantum features
        """
        # Create sliding windows
        windows = []
        for i in range(len(signal) - window_size + 1):
            windows.append(signal[i:i+window_size])
        windows = np.array(windows)
        
        # Initialize Schrödinger PCA
        spca = SchrodingerPCA(n_components=10, sigma=1.0)
        
        # Fit and transform data
        quantum_features = spca.fit_transform(windows)
        
        # Get eigenvalues for spectral analysis
        eigenvalues = spca.eigenvalues_
        
        # Compute quantum entropy
        probs = np.abs(eigenvalues)**2
        probs = probs / np.sum(probs)
        quantum_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return {
            'quantum_features': quantum_features,
            'eigenvalues': eigenvalues,
            'quantum_entropy': quantum_entropy
        }

    def analyze_signal(self, 
                      signal: np.ndarray, 
                      window_size: int = 50) -> Dict[str, Any]:
        """
        Perform comprehensive topological analysis of a signal.
        
        Args:
            signal (np.ndarray): Input time series
            window_size (int): Window size for sliding window analysis
            
        Returns:
            dict: Analysis results
        """
        # Basic analysis from before
        X = torch.FloatTensor(signal).reshape(1, -1, 1).to(self.device)
        
        topo_result = self.topo_calculator.process_time_series(
            signal, embedding_dimension=3)
        
        points = np.array([[signal[i], signal[i+1], signal[i+2]] 
                          for i in range(len(signal)-2)])
        spectral_features = self.simplicial_analyzer.compute_spectral_features(
            points, threshold=0.1)
        
        processed_signal = self.signal_processor(X)
        wavelet_coeffs = self.wavelet_transform(X)
        topo_features = self.feature_extractor.extract_features(
            signal, window_size=window_size)
        
        # New analyses
        change_points = self.compute_change_points(signal, window_size)
        complexity = self.compute_complexity_measures(signal)
        periodicity = self.analyze_periodicity(signal)
        
        # Quantum features
        quantum_features = self.compute_quantum_features(signal, window_size)
        
        # Extract Schrödinger features using neural network
        schrodinger_features = self.schrodinger_extractor(X)
        
        # Detect anomalies
        features = topo_features['topo_signal']
        self.anomaly_detector.fit(features[:len(features)//2])
        anomalies = self.anomaly_detector.detect(features)
        
        # Generate forecasts
        forecast_input = X[:, -100:, :]
        forecasts = self.forecaster.predict(forecast_input)
        
        return {
            'persistence_features': topo_result,
            'spectral_features': spectral_features,
            'processed_signal': processed_signal.cpu().detach().numpy(),
            'wavelet_coefficients': [c.cpu().detach().numpy() for c in wavelet_coeffs],
            'topological_features': topo_features,
            'change_points': change_points,
            'complexity_measures': complexity,
            'periodicity_analysis': periodicity,
            'anomalies': anomalies,
            'forecasts': forecasts.cpu().detach().numpy(),
            'quantum_features': quantum_features,
            'schrodinger_features': schrodinger_features.cpu().detach().numpy()
        }

    def visualize_analysis(self, 
                          results: dict,
                          original_signal: np.ndarray):
        """
        Visualize analysis results.
        
        Args:
            results (dict): Analysis results from analyze_signal
            original_signal (np.ndarray): Original input signal
        """
        plt.figure(figsize=(20, 20))
        
        # Original signal with change points and anomalies
        plt.subplot(5, 2, 1)
        plt.plot(original_signal, 'b-', label='Signal')
        for cp in results['change_points']:
            plt.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
        anomaly_indices = np.where(results['anomalies'])[0]
        plt.scatter(anomaly_indices, 
                   original_signal[anomaly_indices],
                   color='red',
                   label='Anomalies')
        plt.title('Signal with Change Points and Anomalies')
        plt.legend()
        
        # Persistence diagram
        plt.subplot(5, 2, 2)
        self.topo_calculator.plot_persistence_diagram(
            results['persistence_features'],
            title='Persistence Diagram'
        )
        
        # Processed signal
        plt.subplot(5, 2, 3)
        plt.plot(results['processed_signal'].reshape(-1))
        plt.title('Processed Signal')
        
        # Wavelet coefficients
        plt.subplot(5, 2, 4)
        for i, coeffs in enumerate(results['wavelet_coefficients']):
            plt.plot(coeffs.reshape(-1), label=f'Scale {i+1}')
        plt.title('Wavelet Coefficients')
        plt.legend()
        
        # Periodicity analysis
        plt.subplot(5, 2, 5)
        periodicity = results['periodicity_analysis']
        plt.plot(periodicity['autocorrelation'])
        for period, score in zip(periodicity['top_periods'], 
                               periodicity['period_scores']):
            plt.scatter(period, score, color='red')
        plt.title('Periodicity Analysis')
        
        # Complexity measures
        plt.subplot(5, 2, 6)
        complexity = results['complexity_measures']
        plt.bar(range(len(complexity)), 
                list(complexity.values()),
                tick_label=list(complexity.keys()))
        plt.title('Complexity Measures')
        plt.xticks(rotation=45)
        
        # Quantum features
        plt.subplot(5, 2, 7)
        quantum = results['quantum_features']
        plt.plot(quantum['quantum_features'][:, :3])  # Plot first 3 components
        plt.title(f'Quantum Features (Entropy: {quantum["quantum_entropy"]:.3f})')
        
        # Schrödinger features
        plt.subplot(5, 2, 8)
        schrodinger = results['schrodinger_features']
        plt.imshow(schrodinger.T, aspect='auto', cmap='viridis')
        plt.title('Schrödinger Features')
        plt.colorbar()
        
        # Forecasts
        plt.subplot(5, 2, 9)
        forecast_horizon = results['forecasts'].shape[1]
        plt.plot(original_signal[-100:], 'b-', label='Historical')
        plt.plot(np.arange(len(original_signal)-1, 
                          len(original_signal)+forecast_horizon-1),
                results['forecasts'].reshape(-1),
                'r--', label='Forecast')
        plt.title('Forecasts')
        plt.legend()
        
        # Topological features heatmap
        plt.subplot(5, 2, 10)
        topo_signal = results['topological_features']['topo_signal']
        plt.imshow(topo_signal.T, aspect='auto', cmap='viridis')
        plt.title('Topological Signal Features')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

def generate_complex_signal(n_samples=1000):
    """Generate a complex time series with multiple components."""
    t = np.linspace(0, 100, n_samples)
    
    # Trend component
    trend = 0.01 * t
    
    # Seasonal components
    seasonal = (np.sin(0.1 * t) + 
               0.5 * np.sin(0.5 * t) + 
               0.3 * np.sin(0.2 * t))
    
    # Cyclical component
    cyclical = 0.5 * np.sin(0.02 * t)
    
    # Noise component
    noise = 0.1 * np.random.randn(n_samples)
    
    # Combine components
    signal = trend + seasonal + cyclical + noise
    
    return signal

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate sample data
    print("Generating complex time series...")
    signal = generate_complex_signal()
    
    # Initialize analyzer
    print("Initializing topological analyzer...")
    analyzer = TopologicalTimeSeriesAnalyzer(
        input_dim=1,
        hidden_dim=64,
        n_scales=4,
        max_dimension=3
    )
    
    # Perform analysis
    print("Performing comprehensive topological analysis...")
    results = analyzer.analyze_signal(signal)
    
    # Visualize results
    print("Visualizing results...")
    analyzer.visualize_analysis(results, signal)
    
    # Print summary statistics
    print("\nAnalysis Summary:")
    print("Persistence Features:")
    print(f"- Number of features: {len(results['persistence_features']['topological_features'])}")
    
    print("\nSpectral Features:")
    for dim in range(3):
        spectrum = results['spectral_features'][f'hodge_spectrum_dim_{dim}']
        print(f"- Dimension {dim} spectral gap: {spectrum[1] - spectrum[0]:.3f}")
    
    print("\nWavelet Analysis:")
    for i, coeffs in enumerate(results['wavelet_coefficients']):
        print(f"- Scale {i+1} energy: {np.sum(coeffs**2):.3f}")

if __name__ == "__main__":
    main()
