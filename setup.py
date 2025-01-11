from setuptools import setup, find_packages

setup(
    name="neurotune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.2",
        "tqdm>=4.50.0",
    ],
    author="Pawan Rama Mali",
    author_email="prm@outlook.in",
    description="A powerful library for neural network fine-tuning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PawanRamaMali/NeuroTune",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
