from setuptools import setup, find_packages

setup(
    name="dataprobe-automl",
    version="1.0.0",
    author="Sais-gallego",
    description="Automated Exploratory Data Analysis & Baseline AutoML Pipeline Generator",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sais-gallego/dataprobe-automl",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "dataprobe=dataprobe.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
