# PyPI Publishing Guide for Aigie

This guide will help you publish Aigie to PyPI using GitHub Actions.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org)
2. **TestPyPI Account**: Create an account at [test.pypi.org](https://test.pypi.org) for testing
3. **GitHub Repository**: Your code should be in a GitHub repository

## Step 1: Create PyPI API Tokens

### For Production (PyPI)
1. Go to [pypi.org](https://pypi.org) and log in
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Give it a name like "aigie-publishing"
5. Set scope to "Entire account" (or specific project if you prefer)
6. Copy the token (starts with `pypi-`)

### For Testing (TestPyPI)
1. Go to [test.pypi.org](https://test.pypi.org) and log in
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Give it a name like "aigie-test-publishing"
5. Set scope to "Entire account"
6. Copy the token (starts with `pypi-`)

## Step 2: Configure GitHub Secrets

1. Go to your GitHub repository
2. Click on "Settings" tab
3. Click on "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Add these secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `PYPI_API_TOKEN` | Your PyPI API token | For production releases |
| `TEST_PYPI_API_TOKEN` | Your TestPyPI API token | For testing releases |

## Step 3: Update Repository URLs

Update the URLs in these files to match your actual repository:

### In `setup.py`:
```python
url="https://github.com/NirelNemirovsky/aigie-io",
project_urls={
    "Bug Reports": "https://github.com/NirelNemirovsky/aigie-io/issues",
    "Source": "https://github.com/NirelNemirovsky/aigie-io",
    "Documentation": "https://aigie.readthedocs.io",
},
```

### In `pyproject.toml`:
```toml
[project.urls]
Homepage = "https://github.com/NirelNemirovsky/aigie-io"
Documentation = "https://aigie.readthedocs.io"
Repository = "https://github.com/NirelNemirovsky/aigie-io"
"Bug Tracker" = "https://github.com/NirelNemirovsky/aigie-io/issues"
```

## Step 4: Test the Build Locally

Before publishing, test the build locally:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
twine check dist/*

# Test upload to TestPyPI (optional)
twine upload --repository testpypi dist/*
```

## Step 5: Publishing Process

### Method 1: GitHub Release (Recommended)
1. Update the version in `aigie/__init__.py`:
   ```python
   __version__ = "0.1.0"  # Update this
   ```
2. Commit and push your changes
3. Go to your GitHub repository
4. Click "Releases" → "Create a new release"
5. Create a new tag (e.g., `v0.1.0`)
6. Add release notes
7. Click "Publish release"
8. The GitHub Action will automatically build and publish to PyPI

### Method 2: Manual Trigger
1. Go to your GitHub repository
2. Click "Actions" tab
3. Select "Publish to PyPI" workflow
4. Click "Run workflow"
5. Select the branch and click "Run workflow"

## Step 6: Verify Publication

1. Check PyPI: https://pypi.org/project/aigie/
2. Test installation:
   ```bash
   pip install aigie
   ```
3. Test the CLI:
   ```bash
   aigie --help
   ```

## Version Management

### Semantic Versioning
Use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Examples:
- `0.1.0` - Initial release
- `0.1.1` - Bug fix
- `0.2.0` - New features
- `1.0.0` - First stable release

### Updating Version
1. Update `aigie/__init__.py`:
   ```python
   __version__ = "0.1.1"  # New version
   ```
2. Commit and push
3. Create a new release with the same tag

## Troubleshooting

### Common Issues

1. **"Package already exists"**
   - You can't overwrite existing versions
   - Increment the version number

2. **"Invalid credentials"**
   - Check your API token
   - Ensure the secret is set correctly in GitHub

3. **"Build failed"**
   - Check the GitHub Actions logs
   - Ensure all dependencies are in `requirements.txt`

4. **"Package not found after upload"**
   - PyPI indexing can take a few minutes
   - Check the PyPI project page directly

### Testing Before Production

Always test on TestPyPI first:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ aigie
```

## Security Notes

- Never commit API tokens to your repository
- Use GitHub Secrets for all sensitive information
- Rotate your API tokens regularly
- Use different tokens for TestPyPI and PyPI

## Next Steps

After successful publication:

1. **Documentation**: Set up documentation on Read the Docs
2. **CI/CD**: Add more GitHub Actions for testing, linting, etc.
3. **Monitoring**: Monitor download statistics on PyPI
4. **Community**: Respond to issues and feature requests

## Support

If you encounter issues:
1. Check the GitHub Actions logs
2. Verify your PyPI account and tokens
3. Test locally with `python -m build`
4. Check PyPI's status page for outages
