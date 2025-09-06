from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from __init__.py
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), "aigie", "__init__.py")
    with open(init_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="aigie",
    version=get_version(),
    author="Aigie Team",
    author_email="nirel@aigie.io",
    description="AI Agent Runtime Error Detection & Remediation with LLM-as-Judge Validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NirelNemirovsky/aigie-io",
    project_urls={
        "Bug Reports": "https://github.com/NirelNemirovsky/aigie-io/issues",
        "Source": "https://github.com/NirelNemirovsky/aigie-io",
        "Documentation": "https://aigie.readthedocs.io",
    },
    packages=find_packages(exclude=["tests*", "examples*", "venv*"]),
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
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Monitoring",
        "Topic :: System :: Logging",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "cloud": [
            "google-cloud-logging>=3.0.0",
            "google-cloud-monitoring>=2.0.0",
        ],
        "gemini": [
            "google-generativeai>=0.3.0",
            "langchain-google-genai>=0.0.5",
        ],
        "all": [
            "google-cloud-logging>=3.0.0",
            "google-cloud-monitoring>=2.0.0",
            "google-generativeai>=0.3.0",
            "langchain-google-genai>=0.0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "aigie=aigie.cli:main",
        ],
    },
    keywords=[
        "ai", "artificial-intelligence", "langchain", "langgraph", 
        "error-detection", "monitoring", "validation", "llm", "gemini",
        "agent", "runtime", "remediation", "auto-correction"
    ],
    include_package_data=True,
    zip_safe=False,
)
