"""Setup configuration for MLETT package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlett",
    version="0.1.0",
    author="MLETT Team",
    description="ML Time Series forecasting on ETT dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.7.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
    ],
    entry_points={
        "console_scripts": [
            "mlett-train=scripts.train:main",
            "mlett-evaluate=scripts.evaluate:main",
            "mlett-predict=scripts.predict:main",
        ],
    },
)