"""
Utility functions for HEDPriv Framework
Helper functions for data manipulation, metrics, and visualization
"""

import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Union, Optional
from functools import wraps
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timer(func):
    """
    Decorator to measure execution time of functions
    
    Usage:
        @timer
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.4f} seconds")
        return result
    return wrapper


def calculate_mse(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Calculate Mean Squared Error
    
    Args:
        true_values: Ground truth values
        predicted_values: Predicted values
        
    Returns:
        MSE value
    """
    return np.mean((true_values - predicted_values) ** 2)


def calculate_mae(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        true_values: Ground truth values
        predicted_values: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(true_values - predicted_values))


def calculate_relative_error(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Calculate Relative Error (percentage)
    
    Args:
        true_values: Ground truth values
        predicted_values: Predicted values
        
    Returns:
        Relative error as percentage
    """
    return np.mean(np.abs((true_values - predicted_values) / (true_values + 1e-10))) * 100


def privacy_loss(epsilon: float, delta: float) -> Dict[str, str]:
    """
    Interpret privacy loss based on epsilon and delta values
    
    Args:
        epsilon: Privacy budget
        delta: Failure probability
        
    Returns:
        Dictionary with privacy level interpretation
    """
    if epsilon < 0.1:
        level = "Very High Privacy"
        description = "Extremely strong privacy guarantees"
    elif epsilon < 0.5:
        level = "High Privacy"
        description = "Strong privacy guarantees"
    elif epsilon < 1.0:
        level = "Medium Privacy"
        description = "Good privacy guarantees"
    elif epsilon < 5.0:
        level = "Moderate Privacy"
        description = "Reasonable privacy guarantees"
    elif epsilon < 10.0:
        level = "Low Privacy"
        description = "Weak privacy guarantees"
    else:
        level = "Very Low Privacy"
        description = "Minimal privacy guarantees"
    
    return {
        "epsilon": epsilon,
        "delta": delta,
        "level": level,
        "description": description
    }


def format_results_table(results: Dict) -> str:
    """
    Format results dictionary as a readable table
    
    Args:
        results: Results dictionary from HEDPriv pipeline
        
    Returns:
        Formatted string table
    """
    table = "\n" + "="*70 + "\n"
    table += " HEDPRIV RESULTS SUMMARY\n"
    table += "="*70 + "\n\n"
    
    # Dataset info
    table += "Dataset Information:\n"
    table += f"  Samples: {results.get('n_samples', 'N/A')}\n"
    table += f"  Features: {results.get('n_features', 'N/A')}\n"
    table += f"  Feature names: {results.get('feature_names', 'N/A')}\n\n"
    
    # Results
    table += "Mean Values:\n"
    table += f"  Plaintext:  {results.get('plaintext_mean', 'N/A')}\n"
    table += f"  HE Only:    {results.get('decrypted_mean', 'N/A')}\n"
    table += f"  HE + DP:    {results.get('private_mean', 'N/A')}\n\n"
    
    # Errors
    if 'errors' in results:
        table += "Accuracy Metrics:\n"
        table += f"  HE Error:  {results['errors'].get('he_error', 'N/A'):.6f}\n"
        table += f"  DP Error:  {results['errors'].get('dp_error', 'N/A'):.6f}\n"
        table += f"  Total:     {results['errors'].get('total_error', 'N/A'):.6f}\n\n"
    
    # Performance
    if 'metrics' in results:
        table += "Performance Metrics:\n"
        for key, value in results['metrics'].items():
            table += f"  {key.replace('_', ' ').title()}: {value:.3f}s\n"
    
    table += "\n" + "="*70 + "\n"
    return table


def save_results_json(results: Dict, filepath: str):
    """
    Save results to JSON file with proper serialization
    
    Args:
        results: Results dictionary
        filepath: Output file path
    """
    # Convert numpy arrays to lists
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")


def load_results_json(filepath: str) -> Dict:
    """
    Load results from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {filepath}")
    return results


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        signal: Original signal
        noise: Noise component
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def estimate_memory_usage(data_shape: Tuple[int, ...], dtype=np.float64) -> Dict[str, float]:
    """
    Estimate memory usage for encrypted data
    
    Args:
        data_shape: Shape of data array
        dtype: Data type
        
    Returns:
        Dictionary with memory estimates in MB
    """
    element_size = np.dtype(dtype).itemsize
    plaintext_size = np.prod(data_shape) * element_size
    
    # CKKS encryption increases size by ~100x typically
    encrypted_size = plaintext_size * 100
    
    return {
        "plaintext_mb": plaintext_size / (1024**2),
        "encrypted_mb": encrypted_size / (1024**2),
        "overhead_factor": 100
    }


def generate_privacy_report(
    epsilon: float,
    delta: float,
    n_queries: int,
    results: Dict
) -> str:
    """
    Generate a comprehensive privacy report
    
    Args:
        epsilon: Privacy budget
        delta: Failure probability
        n_queries: Number of queries performed
        results: Results dictionary
        
    Returns:
        Formatted privacy report
    """
    report = "\n" + "="*70 + "\n"
    report += " PRIVACY ANALYSIS REPORT\n"
    report += "="*70 + "\n\n"
    
    # Privacy parameters
    privacy_info = privacy_loss(epsilon, delta)
    report += "Privacy Parameters:\n"
    report += f"  Epsilon (ε): {epsilon}\n"
    report += f"  Delta (δ): {delta}\n"
    report += f"  Privacy Level: {privacy_info['level']}\n"
    report += f"  Description: {privacy_info['description']}\n\n"
    
    # Budget composition
    total_epsilon = n_queries * epsilon
    report += "Privacy Budget Composition:\n"
    report += f"  Queries Performed: {n_queries}\n"
    report += f"  Per-Query Budget: {epsilon}\n"
    report += f"  Total Budget Used: {total_epsilon}\n"
    report += f"  Budget Remaining: Depends on your limit\n\n"
    
    # Utility metrics
    if 'errors' in results:
        report += "Utility Metrics:\n"
        report += f"  HE Error: {results['errors']['he_error']:.6f}\n"
        report += f"  DP Error: {results['errors']['dp_error']:.6f}\n"
        
        # Calculate relative utility
        utility = 1 / (1 + results['errors']['dp_error'])
        report += f"  Relative Utility: {utility:.4f} (0-1 scale)\n\n"
    
    # Recommendations
    report += "Recommendations:\n"
    if epsilon < 1.0:
        report += "  ✓ Strong privacy guarantees in place\n"
    elif epsilon < 5.0:
        report += "  ⚠ Consider lowering epsilon for stronger privacy\n"
    else:
        report += "  ⚠ Weak privacy - strongly consider lowering epsilon\n"
    
    if n_queries > 10:
        report += "  ⚠ High query count - consider privacy budget management\n"
    
    report += "\n" + "="*70 + "\n"
    return report


def validate_data(data: np.ndarray, min_samples: int = 10) -> Tuple[bool, str]:
    """
    Validate input data for HEDPriv pipeline
    
    Args:
        data: Input data array
        min_samples: Minimum required samples
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, np.ndarray):
        return False, "Data must be a numpy array"
    
    if data.ndim != 2:
        return False, "Data must be 2-dimensional (samples x features)"
    
    if data.shape[0] < min_samples:
        return False, f"Insufficient samples. Need at least {min_samples}"
    
    if np.isnan(data).any():
        return False, "Data contains NaN values"
    
    if np.isinf(data).any():
        return False, "Data contains infinite values"
    
    return True, "Data validation passed"


def compare_vectors(vec1: np.ndarray, vec2: np.ndarray, tolerance: float = 1e-4) -> Dict:
    """
    Compare two vectors and return detailed comparison metrics
    
    Args:
        vec1: First vector
        vec2: Second vector
        tolerance: Tolerance for "close enough" comparison
        
    Returns:
        Dictionary with comparison metrics
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape")
    
    diff = vec1 - vec2
    
    return {
        "mse": calculate_mse(vec1, vec2),
        "mae": calculate_mae(vec1, vec2),
        "max_error": np.max(np.abs(diff)),
        "min_error": np.min(np.abs(diff)),
        "mean_error": np.mean(diff),
        "std_error": np.std(diff),
        "within_tolerance": np.all(np.abs(diff) < tolerance),
        "num_within_tolerance": np.sum(np.abs(diff) < tolerance),
        "percentage_within_tolerance": (np.sum(np.abs(diff) < tolerance) / len(diff)) * 100
    }


class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress"""
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
        
        print(f"\r{self.description}: {self.current}/{self.total} "
              f"({percentage:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
        
        if self.current >= self.total:
            print()  # New line when complete


# Example usage
if __name__ == "__main__":
    # Test some utility functions
    vec1 = np.array([1.0, 2.0, 3.0, 4.0])
    vec2 = np.array([1.1, 2.05, 2.95, 4.02])
    
    print("MSE:", calculate_mse(vec1, vec2))
    print("MAE:", calculate_mae(vec1, vec2))
    print("\nPrivacy Loss Info:", privacy_loss(1.0, 1e-5))
    print("\nVector Comparison:", compare_vectors(vec1, vec2))
