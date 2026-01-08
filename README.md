# HEDPriv Framework

**Hybrid Encryption and Differential Privacy for Privacy-Preserving Data Analytics**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

HEDPriv is a comprehensive privacy-preserving analytics framework that combines two powerful cryptographic techniques:

- **Homomorphic Encryption (CKKS)**: Enables computation on encrypted data without decryption
- **Differential Privacy (DP)**: Provides mathematical guarantees against privacy breaches

This framework was developed as part of an MPhil thesis on privacy-preserving data analytics for sensitive medical data.

## Features

- **CKKS Homomorphic Encryption** using TenSEAL
- **Differential Privacy** with Gaussian mechanism
- **Privacy-Utility Tradeoff Analysis** with visualization
- **Performance Benchmarking** tools
- **Support for statistical queries** (mean, variance, sum)
- **Production-ready code** with comprehensive testing
- **Interactive Jupyter notebooks** for demonstrations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/samuelselasi/hedpriv.git
cd hedpriv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from src.hedpriv_pipeline import HEDPrivPipeline

# Initialize pipeline
pipeline = HEDPrivPipeline(
    poly_modulus_degree=8192,
    epsilon=1.0,
    delta=1e-5
)

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Print results
print(f"Private Mean: {results['private_mean']}")
print(f"Privacy Budget (ε): {pipeline.dp_mechanism.epsilon}")
```

## Dataset

This framework is designed for the **UCI Heart Disease Dataset**, which contains sensitive medical information:

- **Features**: Age, RestingBP, Cholesterol, MaxHR, etc.
- **Samples**: ~1000 patients
- **Type**: Numerical medical data requiring GDPR/HIPAA compliance

The framework can easily be adapted for other tabular datasets.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HEDPriv Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Data Preprocessing                                       │
│     └─ Load → Clean → Normalize → Split                     │
│                                                              │
│  2. CKKS Encryption (Client)                                 │
│     └─ Generate Keys → Encode → Encrypt                     │
│                                                              │
│  3. Homomorphic Computation (Cloud)                          │
│     └─ Encrypted Sum → Encrypted Mean → Encrypted Variance  │
│                                                              │
│  4. Differential Privacy (Trust Boundary)                    │
│     └─ Decrypt → Calculate Sensitivity → Add Noise          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
HEDPriv/
├── src/                      # Core framework
│   ├── preprocessing.py      # Data preprocessing
│   ├── ckks_encryption.py    # CKKS implementation
│   ├── differential_privacy.py # DP mechanisms
│   └── hedpriv_pipeline.py   # Main pipeline
├── experiments/              # Analysis scripts
│   ├── privacy_utility_tradeoff.py
│   └── performance_benchmark.py
├── tests/                    # Unit tests
├── notebooks/                # Jupyter demos
├── results/                  # Output directory
└── data/                     # Dataset directory
```

## Running Experiments

### Privacy-Utility Tradeoff Analysis

```bash
python experiments/privacy_utility_tradeoff.py
```

This generates:
- Privacy-utility tradeoff curves
- Performance comparison plots
- MSE analysis across different ε values

### Performance Benchmarking

```bash
python experiments/performance_benchmark.py
```

This evaluates:
- Dataset size scalability
- Security parameter analysis
- Computational overhead

## Results

The framework has been tested with:
- **Privacy budgets (ε)**: 0.1 to 10.0
- **Dataset sizes**: 100 to 2000 samples
- **Polynomial degrees**: 4096, 8192, 16384
- **Security levels**: 128-bit to 256-bit

Example results:
- **Encryption time**: ~0.1s per sample (poly_degree=8192)
- **HE computation error**: < 1e-4 MSE
- **DP noise overhead**: Controlled by ε parameter

## Use Cases

1. **Healthcare Analytics**: Compute statistics on patient data without exposing individual records
2. **Financial Services**: Perform risk analysis on encrypted transaction data
3. **Government Agencies**: Analyze census data with privacy guarantees
4. **Research**: Enable privacy-preserving machine learning

## Documentation

- **Setup Guide**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Reference**: See docstrings in source code
- **Jupyter Demo**: See `notebooks/demo.ipynb`
- **Thesis Chapter 3**: Implementation details mapped to methodology

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ckks.py -v

# Run with coverage
pytest --cov=src tests/
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{hedpriv2025,
  title={HEDPriv: A Hybrid Framework for Privacy-Preserving Data Analytics},
  authors={},
  year={2026},
  school={University of Ghana},
  type={MPhil Thesis}
}
```

## Contact

- **Authors**: 
- **Email**: 
- **GitHub**: [Samuel Selasi](https://github.com/samuelselasi)
- **LinkedIn**: [Your Profile]()

## Acknowledgments

- [OpenMined](https://www.openmined.org/) for TenSEAL library
- [Microsoft SEAL](https://github.com/microsoft/SEAL) for the underlying HE library
- UCI Machine Learning Repository for the Heart Disease dataset
- Your advisor and committee members

## References

1. Cheon, J. H., et al. (2017). "Homomorphic encryption for arithmetic of approximate numbers." *ASIACRYPT*.
2. Dwork, C., & Roth, A. (2014). "The algorithmic foundations of differential privacy." *Foundations and Trends in Theoretical Computer Science*.
3. Acar, A., et al. (2018). "A survey on homomorphic encryption schemes: Theory and implementation." *ACM Computing Surveys*.

---

**Note**: This is research software. While it implements standard cryptographic primitives, it has not undergone formal security audits. Use in production environments at your own risk.

**Star this repo if you find it helpful!**
