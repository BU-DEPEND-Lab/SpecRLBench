# NOTE: Only tested with Python 3.10.
# Create a new venv and install the package with `pip install -e .` in the root directory of the project.
from setuptools import setup, find_packages


setup(
    name="specbench",
    version="0.1.0",
    description="SpecRLBench: A Benchmark for Generalization in Specification-Guided Reinforcement Learning",
    author="Zijian Guo, İlker Işık",
    packages=find_packages(include=["specbench", "specbench.*"]),
    install_requires=[
        "gymnasium>=0.29.1,<1.0", "pybullet", "numpy<2", "scipy"
    ],
    include_package_data=True,
    zip_safe=False,
)
