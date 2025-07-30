#!/usr/bin/env python3
"""
Setup script for the Document Extraction Evaluation Framework.

This script provides easy installation and development setup.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

# Read development requirements
dev_requirements = []
with open("requirements-dev.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            dev_requirements.append(line)

setup(
    name="doc-prompt-improver",
    version="1.0.0",
    author="Document Evaluation Team",
    author_email="team@example.com",
    description="Evaluation-Only Framework for Prompt-Driven Document Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chunkys0up7/DocPromptImprover",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": dev_requirements,
        "all": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "doc-evaluator=src.cli.main:main",
            "run-evaluation-demo=demos.comprehensive_demo:main",
            "run-tests=run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
    },
    keywords=[
        "document-processing",
        "evaluation",
        "machine-learning",
        "nlp",
        "dspy",
        "prompt-engineering",
        "ocr",
        "extraction",
        "quality-assessment",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Chunkys0up7/DocPromptImprover/issues",
        "Source": "https://github.com/Chunkys0up7/DocPromptImprover",
        "Documentation": "https://github.com/Chunkys0up7/DocPromptImprover#readme",
    },
) 