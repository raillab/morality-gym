# Contributing to Morality Gym Tabular

Thank you for your interest in contributing to Morality Gym Tabular! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally: `git clone https://github.com/yourusername/morality-gym-tabular.git`
3. Create a new branch for your changes: `git checkout -b feature/your-feature-name`
4. Install the package in development mode: `pip install -e .`

## Types of Contributions

### Bug Reports

If you find a bug, please create an issue on GitHub with:

- A clear title and description
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, etc.)

### Feature Requests

If you have an idea for a new feature, please create an issue with:

- A clear title and description
- The problem the feature would solve
- Any relevant examples or use cases

### Documentation Improvements

Documentation is crucial for this project. Contributions to improve documentation are highly valued. This includes:

- Fixing typos or grammar errors
- Adding missing documentation
- Clarifying existing documentation
- Adding examples or tutorials

### Code Contributions

Code contributions can include:

- Bug fixes
- New features
- Optimizations
- Testing improvements

## Development Guidelines

### Code Style

We follow PEP 8 conventions for Python code. Please ensure your code adheres to these standards.

### Documentation

- All modules, classes, and methods should have docstrings
- Use NumPy or Google style docstrings
- Include examples where appropriate

### Testing

- All new features should include tests
- All bug fixes should include tests that reproduce the bug
- Run existing tests to ensure they still pass

### Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation as needed
3. Add tests for new features or bug fixes
4. Make sure all tests pass
5. Submit a pull request with a clear description of your changes

## Environment Setup

We recommend using a virtual environment for development:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
pip install -r requirements-dev.txt  # If available
```

## Adding New Environments

If you want to add a new moral dilemma environment:

1. Create a new directory in `morality_gym/environments/`
2. Implement the necessary components (entities, scenarios, dynamics)
3. Register the environment in `morality_gym/setup/setup.py`
4. Add relevant tests
5. Document the new environment

## Adding New Morality Trees

If you want to add a new ethical framework:

1. Implement a new subclass of `MoralityChain` in `morality_gym/morality_tree/morality_tree.py`
2. Register the new tree in `morality_gym/morality_tree/setup.py`
3. Add relevant tests
4. Document the new morality tree

## Questions

If you have questions about contributing, please create an issue with the label "question".

Thank you for contributing to Morality Gym Tabular! 