from setuptools import setup, find_packages

setup(
    name="morality_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy"],
)