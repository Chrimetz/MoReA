from setuptools import setup, find_packages

setup(
    name="MoReA",
    version="0.1.0",
    packages=find_packages(include=["MoReA", "MoReA.*"]),
    install_requires=[
        "uvicorn",
        "numpy",
        "fastapi",
        "onnxruntime"
        "pydantic",
        "coremltools",
        "scikit-learn==1.1.0",
        "requests"
    ],
    author="Christopher A. Metz",
    entry_points={
        "console_scripts": [
            "MoReA = app.main:main"
        ]
    },
)