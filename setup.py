from setuptools import setup, find_packages

setup(
    name="student_performance_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.13.0",
        "missingno>=0.5.2",
        "scipy>=1.10.0",
        "streamlit>=1.22.0",
        "joblib>=1.2.0",
    ],
)
