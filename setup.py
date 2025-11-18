"""
Setup configuration for RL Trading System
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "A modular reinforcement learning trading system"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="rl-trading-system",
    version="1.0.0",
    author="Claude Code",
    author_email="claude@anthropic.com",
    description="A modular reinforcement learning trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rl-trading-system",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rl-trading=rl_trading_system.cli:main",
        ],
    },
)
