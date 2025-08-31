from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aigie",
    version="0.1.0",
    author="Aigie Team",
    author_email="nirel@aigie.io",
    description="Real-time error detection and monitoring for LangChain and LangGraph applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/aigie",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cloud": [
            "google-cloud-logging>=3.0.0",
            "google-cloud-monitoring>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aigie=aigie.cli:main",
        ],
    },
)
