from setuptools import setup, find_packages



setup(
    name="morea",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "uvicorn",
        "numpy",
        "fastapi",
        "onnxruntime==1.21.0",
        "pydantic",
        "coremltools",
        "scikit-learn",
        "requests"
    ],
    author="Christopher A. Metz",
    entry_points={
        "console_scripts": [
            "morea = app.main:main"
        ]
    },
)