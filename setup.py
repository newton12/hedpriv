from setuptools import setup, find_packages

setup(
    name="hedpriv",
    version="0.1.0",
    description="HEDPriv: Hybrid Encryption and Differential Privacy Framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "tenseal>=0.3.14",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
