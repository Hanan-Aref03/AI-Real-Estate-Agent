# Contributing Guide

## Development Setup

### 1. Fork and Clone

```bash
git clone <your-fork-url>
cd ai-real-estate-agent
```

### 2. Create Virtual Environment

```bash
uv sync
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Code Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints
- Docstrings for all functions/classes

### Formatting

```bash
# Format code
black app/ ui/

# Sort imports
ruff check app/ ui/ --fix
```

### Type Checking

```bash
mypy app/ ui/
```

### Testing

```bash
pytest tests/
pytest --cov=app tests/
```

## Project Structure Rules

- **Routers** - Add new endpoints in `app/routers/`
- **Models** - Put data models in `schemas.py`
- **Config** - Add configurations in `config.py`
- **Documentation** - New docs in `docs/` folder

## Adding a New Feature

### 1. Create a Router File

`app/routers/new_feature.py`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/feature", tags=["feature"])

@router.post("")
async def feature_endpoint():
    """Feature endpoint documentation."""
    pass
```

### 2. Register Router in Main

Update `app/main.py`:

```python
from app.routers import new_feature_router

app.include_router(new_feature_router)
```

### 3. Add Tests

Create `tests/test_new_feature.py`

### 4. Update Documentation

Add endpoint docs to `docs/API_DOCS.md`

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add new price prediction model
fix: Handle missing environment variables
docs: Update API documentation
refactor: Reorganize router structure
test: Add unit tests for model loader
```

## Pull Request Process

1. Update tests and documentation
2. Ensure all tests pass: `pytest`
3. Format code: `black app/ ui/`
4. Run type checker: `mypy app/ ui/`
5. Push to your fork
6. Create PR with detailed description

## Performance Guidelines

- Lazy load models when possible
- Cache LLM responses for repeated queries
- Use async/await for I/O operations
- Profile before optimizing

## Documentation

- Update relevant `.md` files in `docs/`
- Add docstrings to all functions
- Include examples for new features
- Keep README.md current

## Reporting Issues

Include:
- Python version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error traceback (if applicable)

## Questions?

- Check existing documentation in `docs/`
- Review similar existing code
- Open a discussion issue
