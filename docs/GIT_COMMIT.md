# Commit message convention

## Standard format

The structure of a Conventional Commit is as follows:

```

<type>[(<scope>)]: <description>

[optional body]

[optional footer]
```

## Allowed Commit Types

| Type | Description | Example |
|------|-------------|---------|
| **feat** | New feature (MINOR in semantic versioning) | `feat(auth): add OAuth2 login` |
| **fix** | Bug fix (PATCH in semantic versioning) | `fix(button): fix crash on click` |
| **docs** | Documentation changes | `docs(readme): update installation` |
| **style** | Formatting, spaces, indentation (no code change) | `style: remove trailing whitespace` |
| **refactor** | Code refactoring without functional changes | `refactor(api): simplify request handler` |
| **perf** | Performance improvement | `perf: optimize image loading` |
| **test** | Add or modify tests | `test: add unit tests for auth` |
| **chore** | General tasks, dependencies | `chore(deps): update dependencies` |
| **ci** | CI/CD modifications | `ci: add GitHub Actions workflow` |
| **revert** | Revert a previous commit | `revert: remove deprecated API` |

## Best practices for messages [5][2][6]

### Header (first line)

1. **Limit to 50 characters** maximum
2. **No period at the end** of the line
3. **Imperative mood** (command, not past tense):
   - ✅ `fix: correct button alignment` (good)
   - ❌ `fixed: corrected button alignment` (bad)
   - ❌ `fixing: correcting button alignment` (bad)
4. **Lowercase** except for proper nouns
5. **Optional scope** in parentheses

### Body (message body)

- **Separate** with a blank line
- **Limit to 72 characters** per line
- **Explain the "what" and the "why"**, not the "how"
- Optional but recommended for complex commits

### Footer (page footer)

- **Issue references**: `Closes #123`, `Fixes #456`
- **Breaking changes**: `BREAKING CHANGE: description`
- Optional

## Complete examples

### Simple commit

```
fix(ui): correct button alignment on dashboard
```

### Commit with body

```
feat(auth): add JWT token validation

Implement JWT token validation for API endpoints.
This ensures that only authenticated users can access
protected resources.

Closes #1234
```

### Commit with breaking change

```
feat(api): redesign authentication flow

BREAKING CHANGE: Authentication endpoint URL changed
from /auth to /api/v2/auth. Update your clients accordingly.
```

### Commit with scope

```
refactor(payment): simplify stripe integration

Move payment logic to dedicated module for better
maintainability and testability.
```
