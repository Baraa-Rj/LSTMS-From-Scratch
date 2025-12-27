# LSTMS Python Environment

This is a minimal, ready-to-run Python environment with:

- Virtual environment managed by VS Code
- Modern packaging via `pyproject.toml`
- Sample module and CLI entry (`python -m lstms`)
- Basic tests with `pytest`

## Quick Start

```bash
# Activate venv implicitly via full Python path provided by VS Code
/home/dark/LSTMS/.venv/bin/python -m pip install -U pip
/home/dark/LSTMS/.venv/bin/python -m pip install -e .[test]

# Run the sample app
/home/dark/LSTMS/.venv/bin/python -m lstms

# Run tests
/home/dark/LSTMS/.venv/bin/python -m pytest
```

## Project Layout

- `src/lstms`: Package code
- `tests`: Pytest suite
- `pyproject.toml`: Project metadata and dependencies

## Notes

- This environment targets Python 3.10+
- You can add more dependencies under `[project.dependencies]` in `pyproject.toml`
# LSTMS-From-Scratch
