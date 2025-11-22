# Contributing to PyDTMDL

Thank you for your interest in contributing to PyDTMDL! This document provides guidelines for contributing to the project.

## Reporting Issues

If you encounter any issues while using PyDTMDL, please follow these steps to report them:

1. **Check Existing Issues**: Before creating a new issue, please check the [existing issues](https://github.com/iwatkot/pydtmdl/issues) to see if your issue has already been reported.

2. **Create a New Issue**: If your issue is not listed, you can create a new issue by clicking on the "New issue" button in the [Issues tab](https://github.com/iwatkot/pydtmdl/issues).

3. **Provide Detailed Information**: When creating a new issue, please provide as much detail as possible, including:
   - Coordinates of the center point
   - Size of the region
   - Provider being used
   - Error message if any
   - Stack trace if available
   - Screenshots if possible
   - Any additional information that you think might be useful

This will help understand the issue better and provide a quicker resolution.

## How to Contribute

ℹ️ You'll need to install [Git](https://git-scm.com/) and [Python](https://www.python.org/downloads/) (version 3.11 or higher) on your machine to contribute to the PyDTMDL project. You also must have a GitHub account to fork the repository and submit pull requests.

ℹ️ It's recommended to use [Visual Studio Code](https://code.visualstudio.com/) as your code editor, since the repository already contains a `.vscode` directory with the recommended settings and launch configurations for debugging.

1. **Fork the Repository**: Start by [forking the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). This creates a copy of the repository under your GitHub account.

2. **Clone Your Fork**: Clone your forked repository to your local machine using the command:
   ```bash
   git clone <your_forked_repository_url>
   ```

3. **Create a New Branch**: Before making any changes, create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Prepare a virtual environment**: It's recommended to use a virtual environment to manage dependencies. You can create a virtual environment and install dependencies using:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # For Windows
   source .venv/bin/activate  # For Linux/MacOS
   pip install -e ".[dev]"  # Install package in editable mode with dev dependencies
   ```

5. **Make Your Changes**: Now, you can make your changes in the codebase. Ensure that your code follows the project's coding standards and conventions.

6. **Use the demo.py script**: The `demo.py` script is provided to help you test your changes. If you're using VSCode, you can simply select the `demo.py` launch configuration and run it.
   
   If you're using the terminal, you can run the script with the following command:
   ```bash
   python demo.py
   ```

7. ⚠️ **Run MyPy**: The project relies on the static type checker [MyPy](https://mypy.readthedocs.io/en/stable/). Before submitting a pull request, ensure that your code passes MyPy checks. You can run MyPy with the following command:
   ```bash
   mypy pydtmdl
   ```
   
   ℹ️ The automatic checks will also be performed by the CI/CD pipeline, but it's a good practice to run them locally before submitting a pull request.

8. ⚠️ **Run Pylint**: The project uses [Pylint](https://pylint.pycqa.org/en/latest/) for code quality checks. Before submitting a pull request, ensure that your code passes Pylint checks. You can run Pylint with the following command:
   ```bash
   pylint pydtmdl
   ```
   
   ℹ️ The automatic checks will also be performed by the CI/CD pipeline, but it's a good practice to run them locally before submitting a pull request.


## Code Style

The PyDTMDL project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure that your code adheres to these guidelines and is properly formatted. Remember to run Pylint and MyPy to check for any style violations before submitting your pull request.

All methods, functions and classes must have type hints (including generic types) and docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

**If those requirements are not met, your pull request will not be accepted.**

## Adding a New DTM Provider

If you're interested in adding a new DTM provider, please refer to the [Contributing section in the README](README.md#contributing) for detailed instructions on how to implement a DTM provider.

### ⚠️ Critical Requirements for DTM Providers

When adding a new DTM provider, you **must** follow these requirements:

1. **Use Unified Download Methods**: All providers must use one of the three unified download methods provided by the base `DTMProvider` class:
   - `download_tif_files()` - For URL-based downloads
   - `download_file()` - For single file downloads (GET/POST)
   - `download_tiles_with_fetcher()` - For OGC services (WCS/WMS)

2. **Do Not Implement Custom Download Logic**: Do not use `requests` directly or implement your own download/retry logic. The unified methods provide:
   - Automatic retry logic with configurable attempts
   - Consistent error handling and logging
   - Progress tracking with tqdm
   - File caching
   - Timeout support
   - Authentication support

3. **Extend, Don't Replace**: If you need functionality not provided by the unified methods, **extend the base class methods** rather than implementing your own. This ensures all providers benefit from improvements.

4. **Follow Existing Patterns**: Review similar providers in [pydtmdl/providers/](https://github.com/iwatkot/pydtmdl/tree/main/pydtmdl/providers) to understand the correct implementation patterns.

**Pull requests that implement custom download logic will not be accepted.** This requirement ensures maintainability, consistency, and reliability across all providers.

## Submitting a Pull Request

Once you have made your changes and tested them, you can submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to the repository.

Please ensure that:
- Your code passes all MyPy and Pylint checks
- You have updated the README if necessary
- Your commit messages are clear and descriptive

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

If you have any questions about contributing, feel free to open an issue or start a discussion in the [Discussions tab](https://github.com/iwatkot/pydtmdl/discussions).
