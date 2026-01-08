"""
Performance Benchmarking for HEDPriv Framework
Evaluates computational overhead and scalability
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
import json
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hedpriv_pipeline import HEDPrivPipeline
from src.preprocessing import DataPreprocessor


class PerformanceBenchmark:
    """Benchmark HEDPriv performance characteristics"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
    
    def benchmark_dataset_sizes(
        self,
        sample_sizes: List[int] = None,
        n_features: int = 4
    ) -> Dict:
        """
        Benchmark performance for different dataset sizes
        
        Args:
            sample_sizes: List of sample sizes to test
            n_features: Number of features
            
        Returns:
            Dictionary with benchmark results
        """
        if sample_sizes is None:
            sample_sizes = [100, 250, 500, 1000, 2000]
        
        print("\n" + "="*70)
        print(" DATASET SIZE SCALABILITY BENCHMARK")
        print("="*70)
        print(f"\nSample sizes: {sample_sizes}")
        print(f"Features: {n_features}")
        print("\n" + "="*70 + "\n")
        
        results = {
            'sample_sizes': sample_sizes,
            'preprocessing_times': [],
            'encryption_times': [],
            'computation_times': [],
            'decryption_times': [],
            'dp_times': [],
            'total_times': []
        }
        
        for n_samples in sample_sizes:
            print(f"\n{'='*70}")
            print(f" Testing with {n_samples} samples")
            print(f"{'='*70}")
            
            # Generate synthetic data
            np.random.seed(42)
            data = np.random.randn(n_samples, n_features)
            
            # Initialize pipeline
            pipeline = HEDPrivPipeline(
                poly_modulus_degree=8192,
                epsilon=1.0,
                delta=1e-5
            )
            
            # Setup
            pipeline.setup()
            
            # Preprocessing
            start = time.time()
            preprocessor = DataPreprocessor()
            preprocessor.scaler.fit(data)
            normalized_data = preprocessor.scaler.transform(data)
            prep_time = time.time() - start
            
            # Encryption
            start = time.time()
            encrypted_data = pipeline.encrypt_data(normalized_data)
            enc_time = time.time() - start
            
            # Computation
            start = time.time()
            decrypted_mean = pipeline.compute_encrypted_mean(encrypted_data)
            comp_time = pipeline.metrics['computation_time']
            dec_time = pipeline.metrics['decryption_time']
            
            # DP
            start = time.time()
            private_mean = pipeline.add_differential_privacy(
                decrypted_mean,
                n_samples=n_samples
            )
            dp_time = time.time() - start
            
            total_time = prep_time + enc_time + comp_time + dec_time + dp_time
            
            # Store results
            results['preprocessing_times'].append(prep_time)
            results['encryption_times'].append(enc_time)
            results['computation_times'].append(comp_time)
            results['decryption_times'].append(dec_time)
            results['dp_times'].append(dp_time)
            results['total_times'].append(total_time)
            
            print(f"\nResults for {n_samples} samples:")
            print(f"  Preprocessing: {prep_time:.3f}s")
            print(f"  Encryption: {enc_time:.3f}s")
            print(f"  Computation: {comp_time:.3f}s")
            print(f"  Decryption: {dec_time:.3f}s")
            print(f"  DP: {dp_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
        
        return results
    
    def benchmark_poly_modulus_degrees(
        self,
        degrees: List[int] = None,
        n_samples: int = 500
    ) -> Dict:
        """
        Benchmark different polynomial modulus degrees
        
        Args:
            degrees: List of polynomial modulus degrees
            n_samples: Number of samples
            
        Returns:
            Dictionary with benchmark results
        """
        if degrees is None:
            degrees = [4096, 8192, 16384]
        
        print("\n" + "="*70)
        print(" POLYNOMIAL MODULUS DEGREE BENCHMARK")
        print("="*70)
        print(f"\nDegrees: {degrees}")
        print(f"Samples: {n_samples}")
        print("\n" + "="*70 + "\n")
        
        results = {
            'degrees': degrees,
            'encryption_times': [],
            'computation_times': [],
            'total_times': [],
            'security_levels': []
        }
        
        # Generate data once
        np.random.seed(42)
        data = np.random.randn(n_samples, 4)
        preprocessor = DataPreprocessor()
        preprocessor.scaler.fit(data)
        normalized_data = preprocessor.scaler.transform(data)
        
        for degree in degrees:
            print(f"\n{'='*70}")
            print(f" Testing polynomial modulus degree: {degree}")
            print(f"{'='*70}")
            
            try:
                # Initialize pipeline
                pipeline = HEDPrivPipeline(
                    poly_modulus_degree=degree,
                    epsilon=1.0,
                    delta=1e-5
                )
                
                pipeline.setup()
                
                # Encryption
                start = time.time()
                encrypted_data = pipeline.encrypt_data(normalized_data)
                enc_time = time.time() - start
                
                # Computation
                start = time.time()
                decrypted_mean = pipeline.compute_encrypted_mean(encrypted_data)
                comp_time = pipeline.metrics['computation_time']
                
                total_time = enc_time + comp_time
                
                # Get security level
                security_level = pipeline.encryptor.context.security_level
                
                # Store results
                results['encryption_times'].append(enc_time)
                results['computation_times'].append(comp_time)
                results['total_times'].append(total_time)
                results['security_levels'].append(security_level)
                
                print(f"\nResults for degree {degree}:")
                print(f"  Security Level: {security_level} bits")
                print(f"  Encryption: {enc_time:.3f}s")
                print(f"  Computation: {comp_time:.3f}s")
                print(f"  Total: {total_time:.3f}s")
                
            except Exception as e:
                print(f"Error with degree {degree}: {e}")
                results['encryption_times'].append(None)
                results['computation_times'].append(None)
                results['total_times'].append(None)
                results['security_levels'].append(None)
        
        return results
    
    def plot_scalability(self, results: Dict):
        """Plot dataset size scalability results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sample_sizes = results['sample_sizes']
        
        # Plot 1: Individual component times
        ax1 = axes[0, 0]
        ax1.plot(sample_sizes, results['preprocessing_times'], 
                marker='o', label='Preprocessing', linewidth=2)
        ax1.plot(sample_sizes, results['encryption_times'], 
                marker='s', label='Encryption', linewidth=2)
        ax1.plot(sample_sizes, results['computation_times'], 
                marker='^', label='Computation', linewidth=2)
        ax1.plot(sample_sizes, results['decryption_times'], 
                marker='D', label='Decryption', linewidth=2)
        ax1.plot(sample_sizes, results['dp_times'], 
                marker='*', label='DP', linewidth=2)
        
        ax1.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Component Time Breakdown', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total time
        ax2 = axes[0, 1]
        ax2.plot(sample_sizes, results['total_times'], 
                marker='o', linewidth=3, markersize=10, color='red')
        ax2.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Total Execution Time Scalability', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(sample_sizes, results['total_times'], 1)
        p = np.poly1d(z)
        ax2.plot(sample_sizes, p(sample_sizes), "--", alpha=0.5, color='gray', 
                label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.2f}')
        ax2.legend()
        
        # Plot 3: Stacked bar chart
        ax3 = axes[1, 0]
        width = 0.6
        x = np.arange(len(sample_sizes))
        
        bottom = np.zeros(len(sample_sizes))
        colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']
        labels = ['Preprocessing', 'Encryption', 'Computation', 'Decryption', 'DP']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            values = results[f'{label.lower()}_times']
            ax3.bar(x, values, width, bottom=bottom, label=label, color=color)
            bottom += values
        
        ax3.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_title('Time Distribution by Component', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(sample_sizes)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Time per sample
        ax4 = axes[1, 1]
        time_per_sample = [t/n*1000 for t, n in zip(results['total_times'], sample_sizes)]
        ax4.plot(sample_sizes, time_per_sample, 
                marker='o', linewidth=3, markersize=10, color='green')
        ax4.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Time per Sample (ms)', fontsize=12, fontweight='bold')
        ax4.set_title('Efficiency: Time per Sample', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'scalability_benchmark.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        
        plt.show()
    
    def plot_security_performance_tradeoff(self, results: Dict):
        """Plot polynomial modulus degree results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        degrees = results['degrees']
        
        # Filter out None values
        valid_indices = [i for i, x in enumerate(results['total_times']) if x is not None]
        valid_degrees = [degrees[i] for i in valid_indices]
        valid_times = [results['total_times'][i] for i in valid_indices]
        valid_security = [results['security_levels'][i] for i in valid_indices]
        
        # Plot 1: Time vs Degree
        ax1 = axes[0]
        ax1.bar(range(len(valid_degrees)), valid_times, 
               color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Polynomial Modulus Degree', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance vs Security Parameter', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(valid_degrees)))
        ax1.set_xticklabels(valid_degrees)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Security level vs Time
        ax2 = axes[1]
        scatter = ax2.scatter(valid_times, valid_security, 
                             s=200, c=range(len(valid_degrees)), 
                             cmap='viridis', alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, (t, s, d) in enumerate(zip(valid_times, valid_security, valid_degrees)):
            ax2.annotate(f'{d}', (t, s), fontsize=10, fontweight='bold',
                        ha='center', va='center')
        
        ax2.set_xlabel('Total Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Security Level (bits)', fontsize=12, fontweight='bold')
        ax2.set_title('Security-Performance Tradeoff', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'security_performance_tradeoff.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, filename: str):
        """Save benchmark results to JSON"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    """Main execution function"""
    benchmark = PerformanceBenchmark(output_dir='results')
    
    # Benchmark 1: Dataset size scalability
    print("\n" + "="*70)
    print(" BENCHMARK 1: DATASET SIZE SCALABILITY")
    print("="*70)
    
    scalability_results = benchmark.benchmark_dataset_sizes(
        sample_sizes=[100, 250, 500, 1000, 1500],
        n_features=4
    )
    
    benchmark.plot_scalability(scalability_results)
    benchmark.save_results(scalability_results, 'scalability_benchmark.json')
    
    # Benchmark 2: Polynomial modulus degree
    print("\n" + "="*70)
    print(" BENCHMARK 2: SECURITY PARAMETER ANALYSIS")
    print("="*70)
    
    security_results = benchmark.benchmark_poly_modulus_degrees(
        degrees=[4096, 8192, 16384],
        n_samples=500
    )
    
    benchmark.plot_security_performance_tradeoff(security_results)
    benchmark.save_results(security_results, 'security_benchmark.json')
    
    print("\n" + "="*70)
    print(" BENCHMARKING COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {benchmark.output_dir}/")
    print("  - scalability_benchmark.png")
    print("  - scalability_benchmark.json")
    print("  - security_performance_tradeoff.png")
    print("  - security_benchmark.json")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
