## Description

<!-- Provide a brief description of your changes -->



## Related Issue

<!-- Link to the issue this PR addresses (if applicable) -->
Closes #

## Type of Change

<!-- Mark the relevant option with an [x] -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Test improvements
- [ ] CI/CD or tooling changes

## Pre-submission Checklist

<!-- Complete these checks before submitting. Mark each with [x] when done. -->

### Code Quality
- [ ] I have run `uv run ruff check src tests` and there are no errors
- [ ] I have run `uv run ruff format src tests` to format my code
- [ ] I have run `uv run mypy src` and there are no type errors

### Testing
- [ ] I have run `uv run pytest` and all tests pass
- [ ] I have added tests that cover my changes (if applicable)
- [ ] My changes do not break any existing tests

### Documentation
- [ ] I have updated the documentation (if applicable)
- [ ] I have updated docstrings for any modified functions/classes
- [ ] I have updated the README if this changes user-facing behavior

### General
- [ ] My code follows the project's coding style
- [ ] I have performed a self-review of my code
- [ ] I have commented my code in hard-to-understand areas
- [ ] My changes generate no new warnings

## Testing Instructions

<!-- How can reviewers test your changes? -->

```bash
# Example commands to test
uv run pytest tests/test_specific.py -v
```

## Screenshots / Examples

<!-- If applicable, add screenshots or code examples showing the change -->

## Additional Notes

<!-- Any additional context, concerns, or discussion points -->


