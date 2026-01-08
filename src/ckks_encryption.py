"""
CKKS Homomorphic Encryption Module
Implements encryption, decryption, and homomorphic operations using TenSEAL
"""

import tenseal as ts
import numpy as np
from typing import List, Union, Optional
import time


class CKKSEncryptor:
    """CKKS encryption handler for privacy-preserving analytics"""
    
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        global_scale: int = 2**40
    ):
        """
        Initialize CKKS encryptor
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            global_scale: Global scale for encoding (2^40 is standard)
        """
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.global_scale = global_scale
        self.context = None
        
    def create_context(self):
        """Create TenSEAL context with CKKS scheme"""
        print("\\n=== Creating CKKS Context ===")
        print(f"Polynomial modulus degree: {self.poly_modulus_degree}")
        print(f"Coefficient modulus bit sizes: {self.coeff_mod_bit_sizes}")
        print(f"Global scale: 2^{int(np.log2(self.global_scale))}")
        
        # Create context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
        )
        
        # Set global scale
        self.context.global_scale = self.global_scale
        
        # Generate galois keys for operations like rotation
        self.context.generate_galois_keys()
        
        print("CKKS context created successfully")
        print(f"Security level: {self.context.security_level} bits")
        print("=== Context Setup Complete ===\\n")
        
        return self.context
    
    def encrypt_vector(self, vector: Union[List[float], np.ndarray]) -> ts.CKKSVector:
        """
        Encrypt a single vector
        
        Args:
            vector: Input vector to encrypt
            
        Returns:
            Encrypted CKKS vector
        """
        if self.context is None:
            raise ValueError("Context not initialized. Call create_context() first.")
        
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        encrypted = ts.ckks_vector(self.context, vector)
        return encrypted
    
    def encrypt_dataset(self, data: np.ndarray) -> List[ts.CKKSVector]:
        """
        Encrypt entire dataset (each row becomes an encrypted vector)
        
        Args:
            data: 2D numpy array (samples x features)
            
        Returns:
            List of encrypted vectors
        """
        if self.context is None:
            raise ValueError("Context not initialized. Call create_context() first.")
        
        print(f"\\nEncrypting dataset with shape {data.shape}...")
        start_time = time.time()
        
        encrypted_vectors = []
        for i, row in enumerate(data):
            encrypted = self.encrypt_vector(row)
            encrypted_vectors.append(encrypted)
            
            if (i + 1) % 100 == 0:
                print(f"Encrypted {i + 1}/{len(data)} samples")
        
        encryption_time = time.time() - start_time
        print(f"Encryption completed in {encryption_time:.2f} seconds")
        print(f"Average time per sample: {encryption_time/len(data)*1000:.2f} ms")
        
        return encrypted_vectors
    
    def decrypt_vector(self, encrypted_vector: ts.CKKSVector) -> np.ndarray:
        """
        Decrypt a CKKS vector
        
        Args:
            encrypted_vector: Encrypted vector
            
        Returns:
            Decrypted numpy array
        """
        decrypted = encrypted_vector.decrypt()
        return np.array(decrypted)
    
    def encrypted_sum(self, encrypted_vectors: List[ts.CKKSVector]) -> ts.CKKSVector:
        """
        Compute sum of encrypted vectors (homomorphic addition)
        
        Args:
            encrypted_vectors: List of encrypted vectors
            
        Returns:
            Encrypted sum vector
        """
        print(f"\\nComputing encrypted sum of {len(encrypted_vectors)} vectors...")
        start_time = time.time()
        
        # Start with the first vector
        result = encrypted_vectors[0]
        
        # Add remaining vectors
        for i in range(1, len(encrypted_vectors)):
            result = result + encrypted_vectors[i]
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(encrypted_vectors)} vectors")
        
        computation_time = time.time() - start_time
        print(f"Encrypted sum computed in {computation_time:.2f} seconds")
        
        return result
    
    def encrypted_mean(self, encrypted_vectors: List[ts.CKKSVector]) -> ts.CKKSVector:
        """
        Compute mean of encrypted vectors
        
        Args:
            encrypted_vectors: List of encrypted vectors
            
        Returns:
            Encrypted mean vector
        """
        # Compute sum
        encrypted_sum_vec = self.encrypted_sum(encrypted_vectors)
        
        # Divide by count (scalar multiplication)
        n = len(encrypted_vectors)
        encrypted_mean_vec = encrypted_sum_vec * (1.0 / n)
        
        print(f"Encrypted mean computed (divided by {n})")
        
        return encrypted_mean_vec
    
    def encrypted_variance(
        self, 
        encrypted_vectors: List[ts.CKKSVector],
        encrypted_mean_vec: Optional[ts.CKKSVector] = None
    ) -> ts.CKKSVector:
        """
        Compute variance of encrypted vectors
        
        Args:
            encrypted_vectors: List of encrypted vectors
            encrypted_mean_vec: Pre-computed encrypted mean (optional)
            
        Returns:
            Encrypted variance vector
        """
        if encrypted_mean_vec is None:
            encrypted_mean_vec = self.encrypted_mean(encrypted_vectors)
        
        print(f"\\nComputing encrypted variance...")
        
        # Compute squared differences
        squared_diffs = []
        for enc_vec in encrypted_vectors:
            diff = enc_vec - encrypted_mean_vec
            squared_diff = diff * diff  # Element-wise multiplication
            squared_diffs.append(squared_diff)
        
        # Mean of squared differences
        encrypted_var = self.encrypted_mean(squared_diffs)
        
        print("Encrypted variance computed")
        
        return encrypted_var
    
    def get_context_info(self) -> dict:
        """Get information about the encryption context"""
        if self.context is None:
            return {"status": "Not initialized"}
        
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": self.coeff_mod_bit_sizes,
            "global_scale": self.global_scale,
            "security_level": self.context.security_level
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 4)  # 100 samples, 4 features
    
    # Initialize encryptor
    encryptor = CKKSEncryptor()
    encryptor.create_context()
    
    # Encrypt data
    encrypted_data = encryptor.encrypt_dataset(data)
    
    # Compute encrypted mean
    encrypted_mean_vec = encryptor.encrypted_mean(encrypted_data)
    
    # Decrypt result
    decrypted_mean = encryptor.decrypt_vector(encrypted_mean_vec)
    
    # Compare with plaintext
    plaintext_mean = data.mean(axis=0)
    
    print(f"\\nPlaintext mean: {plaintext_mean}")
    print(f"Decrypted mean: {decrypted_mean}")
    print(f"Difference: {np.abs(plaintext_mean - decrypted_mean)}")
