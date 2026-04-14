from setuptools import setup, find_packages

setup(
    name="morality-gym-tabular",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "gymnasium",
    ],
    author="Simon Rosen",
    author_email="",
    description="Morality Gym Tabular Environment",
    keywords="reinforcement-learning, environment, agent, rl, gymnasium",
    url="https://github.com/SimonRosen173/morality-gym-tabular",
) 