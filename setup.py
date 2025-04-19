from setuptools import setup, find_packages

setup(
    name="morea",  # Ensure this name is unique on PyPI
    version="0.1.0",  # Follow semantic versioning
    description="A RESTful API for deploying machine learning models quickly.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
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
    author="Khonshu",    
    url="https://github.com/Chrimetz/MoReA",  # Replace with your GitHub repo URL
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "morea = app.main:main"
        ]
    },
    include_package_data=True,  # Ensures files like README.md and LICENSE are included
)