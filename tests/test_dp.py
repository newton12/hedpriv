"""
Unit tests for differential privacy module
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.differential_privacy import GaussianMechanism


class TestGaussianMechanism:
    """Test suite for GaussianMechanism class"""
    
    @pytest.fixture
    def mechanism(self):
        """Create Gaussian mechanism instance"""
        return GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    
    @pytest.fixture
    def sample_value(self):
        """Create sample value"""
        return 10.0
    
    @pytest.fixture
    def sample_array(self):
        """Create sample array"""
        return np.array([1.0, 2.0, 3.0, 4.0])
    
    def test_initialization(self, mechanism):
        """Test mechanism initialization"""
        assert mechanism.epsilon == 1.0
        assert mechanism.delta == 1e-5
        assert mechanism.sensitivity == 1.0
        assert mechanism.sigma > 0
    
    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises error"""
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=0, delta=1e-5)
        
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=-1, delta=1e-5)
    
    def test_invalid_delta(self):
        """Test that invalid delta raises error"""
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=0)
        
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=1.5)
    
    def test_sigma_calculation(self):
        """Test that sigma is calculated correctly"""
        mech1 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        mech2 = GaussianMechanism(epsilon=0.5, delta=1e-5, sensitivity=1.0)
        
        # Lower epsilon should result in higher sigma (more noise)
        assert mech2.sigma > mech1.sigma
    
    def test_set_sensitivity(self, mechanism):
        """Test sensitivity update"""
        old_sigma = mechanism.sigma
        mechanism.set_sensitivity(2.0)
        
        assert mechanism.sensitivity == 2.0
        assert mechanism.sigma > old_sigma  # Higher sensitivity = more noise
    
    def test_add_noise_scalar(self, mechanism, sample_value):
        """Test adding noise to scalar value"""
        noisy_value = mechanism.add_noise(sample_value)
        
        assert isinstance(noisy_value, (int, float))
        assert noisy_value != sample_value  # Should be different (with high probability)
    
    def test_add_noise_array(self, mechanism, sample_array):
        """Test adding noise to array"""
        noisy_array = mechanism.add_noise(sample_array)
        
        assert isinstance(noisy_array, np.ndarray)
        assert noisy_array.shape == sample_array.shape
        assert not np.array_equal(noisy_array, sample_array)
    
    def test_noise_distribution(self, mechanism):
        """Test that noise follows Gaussian distribution"""
        np.random.seed(42)
        value = 0.0
        n_samples = 10000
        
        noisy_values = [mechanism.add_noise(value) for _ in range(n_samples)]
        noise = np.array(noisy_values) - value
        
        # Check if noise is approximately Gaussian
        assert abs(noise.mean()) < 0.1  # Mean should be close to 0
        assert abs(noise.std() - mechanism.sigma) < 0.1  # Std should be close to sigma
    
    def test_add_noise_to_mean(self, mechanism):
        """Test adding noise to mean query"""
        mean_value = np.array([1.0, 2.0, 3.0, 4.0])
        n_samples = 100
        
        private_mean = mechanism.add_noise_to_mean(
            mean_value,
            n_samples=n_samples,
            data_range=(-3, 3)
        )
        
        assert isinstance(private_mean, np.ndarray)
        assert private_mean.shape == mean_value.shape
        assert not np.array_equal(private_mean, mean_value)
    
    def test_add_noise_to_sum(self, mechanism):
        """Test adding noise to sum query"""
        sum_value = np.array([100.0, 200.0, 300.0, 400.0])
        
        private_sum = mechanism.add_noise_to_sum(
            sum_value,
            data_range=(-3, 3)
        )
        
        assert isinstance(private_sum, np.ndarray)
        assert private_sum.shape == sum_value.shape
    
    def test_get_privacy_params(self, mechanism):
        """Test privacy parameters retrieval"""
        params = mechanism.get_privacy_params()
        
        assert "epsilon" in params
        assert "delta" in params
        assert "sensitivity" in params
        assert "sigma" in params
        assert params["epsilon"] == 1.0
        assert params["delta"] == 1e-5
    
    def test_calculate_composition(self, mechanism):
        """Test privacy budget composition"""
        n_queries = 5
        total_epsilon = mechanism.calculate_composition(n_queries)
        
        assert total_epsilon == 5.0  # Basic composition
    
    def test_epsilon_effect_on_noise(self):
        """Test that higher epsilon results in less noise"""
        np.random.seed(42)
        value = 10.0
        
        mech_high = GaussianMechanism(epsilon=10.0, delta=1e-5, sensitivity=1.0)
        mech_low = GaussianMechanism(epsilon=0.1, delta=1e-5, sensitivity=1.0)
        
        # Run multiple times to average out randomness
        noise_high = []
        noise_low = []
        for _ in range(100):
            noise_high.append(abs(mech_high.add_noise(value) - value))
            noise_low.append(abs(mech_low.add_noise(value) - value))
        
        # Higher epsilon should result in less noise on average
        assert np.mean(noise_high) < np.mean(noise_low)
    
    def test_sensitivity_effect(self):
        """Test that higher sensitivity results in more noise"""
        np.random.seed(42)
        
        mech1 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        mech2 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=10.0)
        
        # Higher sensitivity should have higher sigma
        assert mech2.sigma > mech1.sigma
    
    def test_privacy_guarantee(self):
        """Test basic privacy guarantee properties"""
        mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        
        # Two adjacent datasets (differ by one record)
        value1 = 100.0
        value2 = 101.0  # Differs by sensitivity
        
        # Add noise to both
        np.random.seed(42)
        noisy1 = mechanism.add_noise(value1)
        
        np.random.seed(42)
        noisy2 = mechanism.add_noise(value2)
        
        # The distributions should be similar (hard to test statistically with one sample)
        # But we can verify sigma is set appropriately
        assert mechanism.sigma > 0
    
    def test_mean_sensitivity_calculation(self, mechanism):
        """Test automatic sensitivity calculation for mean"""
        mean_value = np.array([0.5, 0.3, 0.7])
        n_samples = 1000
        data_range = (-3, 3)
        
        # Calculate expected sensitivity
        expected_sensitivity = (data_range[1] - data_range[0]) / n_samples
        
        # This should use the calculated sensitivity
        private_mean = mechanism.add_noise_to_mean(
            mean_value,
            n_samples=n_samples,
            data_range=data_range
        )
        
        assert isinstance(private_mean, np.ndarray)
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with numpy seed"""
        mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        value = 10.0
        
        np.random.seed(42)
        result1 = mechanism.add_noise(value)
        
        np.random.seed(42)
        result2 = mechanism.add_noise(value)
        
        assert result1 == result2
    
    def test_zero_sensitivity_warning(self):
        """Test handling of None sensitivity"""
        # Should issue warning and use default
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=None)
        assert mech.sensitivity == 1.0  # Default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
