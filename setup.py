from setuptools import setup, find_packages

excludes = [
    "tests",
    "tests.*",
    "examples",
    "examples.*",
    "Test/",
    "docs",
    "docs.*",
    "devtools",
    "devtools.*",
    "__pycache__",
    "__pycache__/",
    "__pycache__*",
]

setup(
    name="diffPLOG2TROE",
    version="1.0.0",
    author="Timoteo Dinelli",
    author_email="timoteo.dinelli@polimi.it",
    description="A differentiable PLOG to TROE refitter",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tdinelli/diffPLOG2TROE",
    packages=find_packages(exclude=excludes),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pytest",
        "matplotlib",
        "mkdocs",
        "jax",
        "chex",
        "optax",
        "nlopt",
    ],
)
