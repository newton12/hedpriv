"""
Differential Privacy Module
Implements Gaussian Mechanism for ε-DP guarantees
"""

import numpy as np
from typing import Union, Tuple
import warnings


class GaussianMechanism:
    """Gaussian Mechanism for Differential Privacy"""
    
    def __init__(
        self, 
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = None
    ):
        """
        Initialize Gaussian Mechanism
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Probability of privacy guarantee failure
            sensitivity: Global sensitivity of the query
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Calculate sigma for Gaussian noise
        self.sigma = self._calculate_sigma()
        
    def _calculate_sigma(self) -> float:
        """
        Calculate noise scale (sigma) for (ε, δ)-DP
        
        Using the formula: σ ≥ (Δf / ε) * sqrt(2 * ln(1.25/δ))
        where Δf is the global sensitivity
        """
        if self.sensitivity is None:
            warnings.warn("Sensitivity not set. Using default value of 1.0")
            self.sensitivity = 1.0
        
        sigma = (self.sensitivity / self.epsilon) * np.sqrt(2 * np.log(1.25 / self.delta))
        return sigma
    
    def set_sensitivity(self, sensitivity: float):
        """
        Update sensitivity and recalculate sigma
        
        Args:
            sensitivity: Global sensitivity of the query
        """
        self.sensitivity = sensitivity
        self.sigma = self._calculate_sigma()
        print(f"Updated sensitivity: {sensitivity:.4f}, New sigma: {self.sigma:.4f}")
    
    def add_noise(
        self, 
        value: Union[float, np.ndarray],
        sensitivity: float = None
    ) -> Union[float, np.ndarray]:
        """
        Add Gaussian noise to a value or array
        
        Args:
            value: Value(s) to add noise to
            sensitivity: Override default sensitivity (optional)
            
        Returns:
            Noisy value(s)
        """
        if sensitivity is not None:
            self.set_sensitivity(sensitivity)
        
        # Generate Gaussian noise
        if isinstance(value, np.ndarray):
            noise = np.random.normal(0, self.sigma, size=value.shape)
        else:
            noise = np.random.normal(0, self.sigma)
        
        noisy_value = value + noise
        
        return noisy_value
    
    def add_noise_to_mean(
        self, 
        mean_value: Union[float, np.ndarray],
        n_samples: int,
        data_range: Tuple[float, float] = (-3, 3)
    ) -> Union[float, np.ndarray]:
        """
        Add noise to mean query with automatic sensitivity calculation
        
        For normalized data, the sensitivity of mean is: (max - min) / n
        
        Args:
            mean_value: Computed mean
            n_samples: Number of samples used to compute mean
            data_range: Range of normalized data (default: approx. ±3σ)
            
        Returns:
            Noisy mean value
        """
        # Calculate sensitivity for mean query
        data_min, data_max = data_range
        sensitivity = (data_max - data_min) / n_samples
        
        print(f"\\nApplying Differential Privacy:")
        print(f"  Epsilon (ε): {self.epsilon}")
        print(f"  Delta (δ): {self.delta}")
        print(f"  Sensitivity: {sensitivity:.6f}")
        print(f"  Noise scale (σ): {self.sigma:.6f}")
        
        # Add noise
        noisy_value = self.add_noise(mean_value, sensitivity=sensitivity)
        
        # Calculate noise magnitude
        if isinstance(mean_value, np.ndarray):
            noise_magnitude = np.linalg.norm(noisy_value - mean_value)
            print(f"  Noise magnitude: {noise_magnitude:.6f}")
        else:
            noise_magnitude = abs(noisy_value - mean_value)
            print(f"  Noise magnitude: {noise_magnitude:.6f}")
        
        return noisy_value
    
    def add_noise_to_sum(
        self,
        sum_value: Union[float, np.ndarray],
        data_range: Tuple[float, float] = (-3, 3)
    ) -> Union[float, np.ndarray]:
        """
        Add noise to sum query
        
        For normalized data, the sensitivity of sum is: (max - min)
        
        Args:
            sum_value: Computed sum
            data_range: Range of normalized data
            
        Returns:
            Noisy sum value
        """
        data_min, data_max = data_range
        sensitivity = data_max - data_min
        
        return self.add_noise(sum_value, sensitivity=sensitivity)
    
    def get_privacy_params(self) -> dict:
        """Get current privacy parameters"""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "sigma": self.sigma
        }
    
    def calculate_composition(self, n_queries: int) -> float:
        """
        Calculate epsilon after n queries (basic composition)
        
        Args:
            n_queries: Number of queries
            
        Returns:
            Total epsilon budget consumed
        """
        # Basic composition: ε_total = n * ε
        total_epsilon = n_queries * self.epsilon
        
        print(f"\\nPrivacy Budget Composition:")
        print(f"  Single query ε: {self.epsilon}")
        print(f"  Number of queries: {n_queries}")
        print(f"  Total ε consumed: {total_epsilon}")
        
        return total_epsilon


# Example usage
if __name__ == "__main__":
    # Create mechanism
    mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5)
    
    # Example: Add noise to mean
    true_mean = np.array([0.5, 0.3, 0.7, 0.2])
    n_samples = 100
    
    private_mean = mechanism.add_noise_to_mean(
        true_mean, 
        n_samples=n_samples,
        data_range=(-3, 3)
    )
    
    print(f"\\nTrue mean: {true_mean}")
    print(f"Private mean: {private_mean}")
    print(f"Error: {np.linalg.norm(true_mean - private_mean):.6f}")
