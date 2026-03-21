# Contributing to msfiddle

Thank you for your interest in contributing! This document explains how to set up your development environment, run tests, and submit changes.

> **Note:** The core model code (`dataset.py`, `model_tcn.py`, `utils/`) is maintained in the [FIDDLE](https://github.com/JosieHong/FIDDLE) repository and periodically synced here. If your change touches those files, please open an issue first to discuss the best approach.

## Development Setup

**1. Clone the repository**

```bash
git clone https://github.com/JosieHong/msfiddle.git
cd msfiddle
```

**2. Create a conda environment**

```bash
conda create -n msfiddle-dev python=3.10
conda activate msfiddle-dev
```

**3. Install the package in editable mode**

```bash
pip install -e .
pip install pytest black
```

**4. Install PyTorch** (required to run the full pipeline, not needed for utility tests)

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/#v1130) for your system. The package requires `torch>=1.13.0,<2.0.0`.

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers utility functions only (`mol_utils`, `msms_utils`, `refine_utils`). Tests do **not** require PyTorch or downloaded model weights.

## Code Style

This project uses [Black](https://black.readthedocs.io/) for code formatting. Before submitting a pull request, format your code with:

```bash
black .
```

To check formatting without making changes:

```bash
black --check .
```

> Do not reformat files that are synced from FIDDLE (`dataset.py`, `model_tcn.py`, `utils/`) — formatting changes there should be made upstream.

## What to Contribute

Good candidates for contributions:

- **Bug fixes** in utility functions (`utils/`)
- **New precursor type support** in `msms_utils.py`
- **Test coverage** for currently untested edge cases
- **Documentation** improvements

Please open an issue before starting work on larger changes.

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`
2. Make your changes and add tests if applicable
3. Format your code: `black .`
4. Ensure all tests pass: `pytest tests/ -v`
5. Open a pull request against `main` with a clear description of what was changed and why

## Reporting Issues

Please use [GitHub Issues](https://github.com/JosieHong/msfiddle/issues) to report bugs. Include:

- Your OS and Python version
- The full error traceback
- A minimal reproducible example if possible
