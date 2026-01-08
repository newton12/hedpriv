"""
HEDPriv Pipeline - Main Integration Module
Combines CKKS encryption and differential privacy for complete workflow
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional
import json

from .preprocessing import DataPreprocessor
from .ckks_encryption import CKKSEncryptor
from .differential_privacy import GaussianMechanism


class HEDPrivPipeline:
    """
    Complete HEDPriv pipeline integrating:
    1. Data preprocessing
    2. CKKS homomorphic encryption
    3. Encrypted computation
    4. Differential privacy
    """
    
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = None,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        random_state: int = 42
    ):
        """
        Initialize HEDPriv pipeline
        
        Args:
            poly_modulus_degree: CKKS parameter
            coeff_mod_bit_sizes: CKKS coefficient modulus sizes
            epsilon: DP privacy budget
            delta: DP failure probability
            random_state: Random seed for reproducibility
        """
        # Components
        self.preprocessor = DataPreprocessor(random_state=random_state)
        self.encryptor = CKKSEncryptor(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes or [60, 40, 40, 60]
        )
        self.dp_mechanism = GaussianMechanism(epsilon=epsilon, delta=delta)
        
        # State
        self.context_created = False
        self.metrics = {
            'preprocessing_time': 0,
            'encryption_time': 0,
            'computation_time': 0,
            'decryption_time': 0,
            'dp_time': 0,
            'total_time': 0
        }
        
    def setup(self):
        """Initialize encryption context"""
        print("\n" + "="*60)
        print(" HEDPriv Framework Initialization")
        print("="*60)
        
        self.encryptor.create_context()
        self.context_created = True
        
        print(f"\nPrivacy Parameters:")
        print(f"  Epsilon (ε): {self.dp_mechanism.epsilon}")
        print(f"  Delta (δ): {self.dp_mechanism.delta}")
        print("="*60 + "\n")
        
    def load_and_preprocess(self, filepath: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset
        
        Args:
            filepath: Path to dataset CSV
            
        Returns:
            Tuple of (training_data, test_data)
        """
        start_time = time.time()
        
        # Load data
        data = self.preprocessor.load_heart_disease_data(filepath)
        
        # Preprocess
        X_train, X_test = self.preprocessor.preprocess(data)
        
        self.metrics['preprocessing_time'] = time.time() - start_time
        
        return X_train, X_test
    
    def encrypt_data(self, data: np.ndarray):
        """
        Encrypt dataset using CKKS
        
        Args:
            data: Preprocessed numpy array
            
        Returns:
            List of encrypted vectors
        """
        if not self.context_created:
            raise RuntimeError("Call setup() before encrypting data")
        
        start_time = time.time()
        
        encrypted_data = self.encryptor.encrypt_dataset(data)
        
        self.metrics['encryption_time'] = time.time() - start_time
        
        return encrypted_data
    
    def compute_encrypted_mean(self, encrypted_data: list) -> np.ndarray:
        """
        Compute mean on encrypted data
        
        Args:
            encrypted_data: List of encrypted vectors
            
        Returns:
            Decrypted mean (before DP noise)
        """
        print("\n" + "="*60)
        print(" Computing Encrypted Mean")
        print("="*60)
        
        # Compute encrypted mean
        comp_start = time.time()
        encrypted_mean = self.encryptor.encrypted_mean(encrypted_data)
        self.metrics['computation_time'] = time.time() - comp_start
        
        # Decrypt
        dec_start = time.time()
        decrypted_mean = self.encryptor.decrypt_vector(encrypted_mean)
        self.metrics['decryption_time'] = time.time() - dec_start
        
        print(f"\nDecrypted mean (before DP): {decrypted_mean}")
        print("="*60 + "\n")
        
        return decrypted_mean
    
    def compute_encrypted_variance(
        self, 
        encrypted_data: list,
        encrypted_mean=None
    ) -> np.ndarray:
        """
        Compute variance on encrypted data
        
        Args:
            encrypted_data: List of encrypted vectors
            encrypted_mean: Pre-computed encrypted mean (optional)
            
        Returns:
            Decrypted variance (before DP noise)
        """
        print("\n" + "="*60)
        print(" Computing Encrypted Variance")
        print("="*60)
        
        comp_start = time.time()
        encrypted_var = self.encryptor.encrypted_variance(
            encrypted_data, 
            encrypted_mean
        )
        comp_time = time.time() - comp_start
        
        dec_start = time.time()
        decrypted_var = self.encryptor.decrypt_vector(encrypted_var)
        dec_time = time.time() - dec_start
        
        self.metrics['computation_time'] += comp_time
        self.metrics['decryption_time'] += dec_time
        
        print(f"\nDecrypted variance (before DP): {decrypted_var}")
        print("="*60 + "\n")
        
        return decrypted_var
    
    def add_differential_privacy(
        self,
        value: np.ndarray,
        n_samples: int,
        query_type: str = 'mean'
    ) -> np.ndarray:
        """
        Add differential privacy noise to decrypted result
        
        Args:
            value: Decrypted value
            n_samples: Number of samples in dataset
            query_type: Type of query ('mean' or 'sum')
            
        Returns:
            Private result with DP noise
        """
        start_time = time.time()
        
        if query_type == 'mean':
            private_value = self.dp_mechanism.add_noise_to_mean(
                value,
                n_samples=n_samples,
                data_range=(-3, 3)  # Assuming normalized data
            )
        elif query_type == 'sum':
            private_value = self.dp_mechanism.add_noise_to_sum(
                value,
                data_range=(-3, 3)
            )
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        
        self.metrics['dp_time'] = time.time() - start_time
        
        return private_value
    
    def run_complete_pipeline(
        self,
        filepath: Optional[str] = None,
        compute_variance: bool = False
    ) -> Dict:
        """
        Execute complete HEDPriv pipeline
        
        Args:
            filepath: Path to dataset
            compute_variance: Whether to also compute variance
            
        Returns:
            Dictionary with results and metrics
        """
        total_start = time.time()
        
        print("\n" + "="*70)
        print(" HEDPRIV COMPLETE PIPELINE EXECUTION")
        print("="*70)
        
        # Step 1: Setup
        if not self.context_created:
            self.setup()
        
        # Step 2: Load and preprocess
        print("\n[STEP 1/5] Loading and Preprocessing Data...")
        X_train, X_test = self.load_and_preprocess(filepath)
        
        # Step 3: Encrypt
        print("\n[STEP 2/5] Encrypting Data...")
        encrypted_train = self.encrypt_data(X_train)
        
        # Step 4: Compute encrypted mean
        print("\n[STEP 3/5] Computing Encrypted Mean...")
        decrypted_mean = self.compute_encrypted_mean(encrypted_train)
        
        # Step 5: Apply DP
        print("\n[STEP 4/5] Applying Differential Privacy...")
        private_mean = self.add_differential_privacy(
            decrypted_mean,
            n_samples=len(X_train),
            query_type='mean'
        )
        
        # Compare with plaintext
        plaintext_mean = X_train.mean(axis=0)
        
        # Optional: Compute variance
        results = {
            'plaintext_mean': plaintext_mean,
            'decrypted_mean': decrypted_mean,
            'private_mean': private_mean,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'feature_names': self.preprocessor.get_feature_names()
        }
        
        if compute_variance:
            print("\n[STEP 5/5] Computing Encrypted Variance...")
            decrypted_var = self.compute_encrypted_variance(encrypted_train)
            private_var = self.add_differential_privacy(
                decrypted_var,
                n_samples=len(X_train),
                query_type='mean'  # Variance uses similar sensitivity
            )
            plaintext_var = X_train.var(axis=0)
            
            results.update({
                'plaintext_variance': plaintext_var,
                'decrypted_variance': decrypted_var,
                'private_variance': private_var
            })
        
        # Calculate total time
        self.metrics['total_time'] = time.time() - total_start
        results['metrics'] = self.metrics.copy()
        
        # Calculate errors
        he_error = np.linalg.norm(plaintext_mean - decrypted_mean)
        dp_error = np.linalg.norm(plaintext_mean - private_mean)
        
        results['errors'] = {
            'he_error': he_error,
            'dp_error': dp_error,
            'total_error': np.linalg.norm(decrypted_mean - private_mean)
        }
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print execution summary"""
        print("\n" + "="*70)
        print(" EXECUTION SUMMARY")
        print("="*70)
        
        print(f"\nDataset Information:")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Features: {results['n_features']}")
        print(f"  Feature names: {results['feature_names']}")
        
        print(f"\nResults (Mean):")
        print(f"  Plaintext:  {results['plaintext_mean']}")
        print(f"  HE Only:    {results['decrypted_mean']}")
        print(f"  HE + DP:    {results['private_mean']}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  HE Error (MSE): {results['errors']['he_error']:.6f}")
        print(f"  DP Error (MSE): {results['errors']['dp_error']:.6f}")
        print(f"  Total Noise:    {results['errors']['total_error']:.6f}")
        
        print(f"\nPerformance Metrics:")
        for key, value in results['metrics'].items():
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}s")
        
        print(f"\nPrivacy Parameters:")
        print(f"  Epsilon (ε): {self.dp_mechanism.epsilon}")
        print(f"  Delta (δ): {self.dp_mechanism.delta}")
        
        print("\n" + "="*70 + "\n")
    
    def save_results(self, results: Dict, filepath: str = 'results/pipeline_results.json'):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                results_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_serializable[key][k] = v.tolist()
                    else:
                        results_serializable[key][k] = v
            else:
                results_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = HEDPrivPipeline(
        poly_modulus_degree=8192,
        epsilon=1.0,
        delta=1e-5
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(compute_variance=True)
    
    # Save results
    pipeline.save_results(results)
