"""
HEDPriv: Hybrid Encryption and Differential Privacy Framework

A comprehensive framework for privacy-preserving data analytics combining:
- Homomorphic Encryption (CKKS scheme)
- Differential Privacy (Gaussian mechanism)

Author: Your Name
Date: 2026
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Samuel Selasi"
__email__ = "sskporvie001@st.ug.edu.gh"

from .preprocessing import DataPreprocessor
from .ckks_encryption import CKKSEncryptor
from .differential_privacy import GaussianMechanism
from .hedpriv_pipeline import HEDPrivPipeline

__all__ = [
    "DataPreprocessor",
    "CKKSEncryptor",
    "GaussianMechanism",
    "HEDPrivPipeline",
]

# Package metadata
PACKAGE_NAME = "hedpriv"
DESCRIPTION = "Hybrid Encryption and Differential Privacy Framework"
URL = "https://github.com/yourusername/HEDPriv"

# Framework configuration defaults
DEFAULT_CONFIG = {
    "poly_modulus_degree": 8192,
    "coeff_mod_bit_sizes": [60, 40, 40, 60],
    "global_scale": 2**40,
    "epsilon": 1.0,
    "delta": 1e-5,
    "random_state": 42,
}


def get_version():
    """Return the current version of HEDPriv."""
    return __version__


def get_config():
    """Return the default configuration."""
    return DEFAULT_CONFIG.copy()


# Print welcome message when package is imported
def _init_message():
    """Print initialization message."""
    print(f"HEDPriv Framework v{__version__} loaded successfully!")
    print(f"Author: {__author__}")
    print(f"For documentation, visit: {URL}")


# Uncomment the line below if you want to see the message on import
# _init_message()
