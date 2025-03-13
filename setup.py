from setuptools import setup, find_packages

setup(
    name="futuredynamic",
    version="0.1.0",
    description="Adaptive Memory Management for Efficient LLM Inference Across GPU Architectures",
    author="Your Name",
    author_email="email@example.com",
    url="https://github.com/yourusername/futuredynamic",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "psutil>=5.9.0",
        "pynvml>=11.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
