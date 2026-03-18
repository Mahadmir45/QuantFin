from setuptools import setup, find_packages

setup(
    name="mmm-bayesian",
    version="1.0.0",
    description="Bayesian Marketing Mix Modeling for Retail & Luxury Brands",
    author="QuantFin Analytics",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "pymc>=5.10",
        "arviz>=0.17",
        "pytensor>=2.18",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
        "seaborn>=0.13",
    ],
    entry_points={
        "console_scripts": [
            "mmm-bayesian=mmm_bayesian.cli:main",
        ],
    },
)
