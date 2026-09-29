from setuptools import setup, find_packages

setup(
    name="imageToEquation",
    version="0.1.0",
    description="Differentiable raster-to-equation transcriber and continuous solver.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.2.0",
    ],
    entry_points={
        "console_scripts": [
            "transcribeImage=transcribeImage:main",
            "batchTranscribe=batchTranscribe:main",
            "runHpcTrainer=runHpcTrainer:main",
        ],
    },
)
