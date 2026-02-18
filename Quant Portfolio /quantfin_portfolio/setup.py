"""Setup script for QuantFin Pro."""

from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "QuantFin Pro - Advanced Quantitative Finance Library"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'networkx>=2.6.0',
        'scikit-learn>=1.0.0',
        'yfinance>=0.2.0'
    ]

setup(
    name='quantfin-pro',
    version='1.0.0',
    author='Mahad Mir',
    author_email='your.email@example.com',
    description='Advanced Quantitative Finance Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Mahadmir45/QuantFin',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
        'ml': [
            'xgboost>=1.5.0',
            'lightgbm>=3.3.0',
            'tensorflow>=2.8.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'quantfin=quantfin.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)