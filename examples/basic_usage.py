import torch
import numpy as np
import matplotlib.pyplot as plt
from lib.topo_evolution_model import TopoEvolutionModel
from lib.persistent_homology import PersistentHomology

def generate_sample_data(n_samples=1000):
    """Generate synthetic time series data for demonstration."""
    t = np.linspace(0, 100, n_samples)
    # Create a signal with multiple frequencies
    signal = (np.sin(0.1 * t) + 
             0.5 * np.sin(0.5 * t) + 
             0.3 * np.sin(0.2 * t + 0.1))
    # Add some noise
    signal += 0.1 * np.random.randn(n_samples)
    return signal

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate sample data
    print("Generating sample data...")
    signal = generate_sample_data()
    
    # Convert to torch tensor and reshape for model input
    X = torch.FloatTensor(signal).reshape(1, -1, 1)  # [batch_size, seq_len, features]
    
    # Initialize models
    print("Initializing models...")
    model = TopoEvolutionModel(
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Initialize persistent homology calculator
    topo_calculator = PersistentHomology(max_dimension=1)
    
    # Compute persistence diagram
    print("Computing persistence diagram...")
    topo_result = topo_calculator.process_time_series(
        signal, 
        embedding_dimension=3,
        time_delay=1
    )
    
    # Plot original signal
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title('Original Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Plot persistence diagram
    plt.subplot(1, 2, 2)
    topo_calculator.plot_persistence_diagram(
        topo_result,
        title='Persistence Diagram'
    )
    plt.tight_layout()
    
    # Process through model
    print("Processing through model...")
    with torch.no_grad():
        output = model(X)
    
    print("Model output shape:", output.shape)
    print("Topological features shape:", topo_result['topological_features'].shape)
    
    # Print summary of topological features
    print("\nTopological Feature Summary:")
    features = topo_result['topological_features']
    print(f"Number of features: {len(features)}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Mean feature value: {features.mean():.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()
