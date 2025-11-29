# Contributing to MedEval

Thank you for your interest in contributing to MedEval! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd medeval
```

2. Install in development mode with all dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting (line length 100)
- **Ruff**: Linting and import sorting
- **mypy**: Type checking (with relaxed settings for optional dependencies)

Run these before committing:
```bash
black .
ruff check --fix .
mypy medeval
```

## Testing

We use pytest for testing. Run tests with:
```bash
pytest
```

For coverage:
```bash
pytest --cov=medeval --cov-report=html
```

### Test Categories

- **Unit tests**: Fast, test individual functions
- **Acceptance tests**: Marked with `@pytest.mark.acceptance`, test end-to-end functionality
- **Slow tests**: Marked with `@pytest.mark.slow`, may take longer to run

Run only fast tests:
```bash
pytest -m "not slow"
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass and code is formatted
5. Update documentation if needed
6. Submit a pull request with a clear description

## Code Review

- All PRs require at least one approval
- Address review comments promptly
- Keep PRs focused and reasonably sized

## Adding New Metrics

When adding new metrics:

1. Place in appropriate module (`medeval/metrics/`)
2. Follow existing patterns for spacing handling
3. Support both 2D and 3D inputs
4. Add comprehensive tests
5. Document with docstrings including formulas

## Questions?

Open an issue for questions or discussions.

