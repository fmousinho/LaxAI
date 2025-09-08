from setuptools import find_packages, setup

setup(
    name="laxai-shared-libs",
    version="1.0.0",
    description="Shared libraries for LaxAI services",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.0",
        "toml>=0.10.0", 
        "python-dotenv>=1.0.0",
        "python-dateutil>=2.8.0",
        "structlog>=23.1.0",
        "python-json-logger>=2.0.0",
    ],
    python_requires=">=3.8",
)
