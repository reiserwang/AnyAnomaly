# PR Review Toolkit

Comprehensive PR review agents specializing in comments, tests, error handling, type design, code quality, and code simplification.

## Quick Start

```bash
/review-pr              # Full review
/review-pr tests errors # Specific aspects
/review-pr simplify     # Simplify code after review
```

## Agents

| Agent | Purpose |
|-------|---------|
| `code-reviewer` | Project guidelines compliance, bug detection |
| `code-simplifier` | Simplify code for clarity and maintainability |
| `comment-analyzer` | Verify comment accuracy and completeness |
| `pr-test-analyzer` | Analyze test coverage quality |
| `silent-failure-hunter` | Find hidden errors and silent failures |
| `type-design-analyzer` | Evaluate type encapsulation and invariants |

## Review Aspects

- **comments** - Analyze code comment accuracy
- **tests** - Review test coverage quality
- **errors** - Check error handling
- **types** - Analyze type design
- **code** - General code review
- **simplify** - Simplify code after review
- **all** - Run all reviews (default)

## Workflow

```
Before commit:    /review-pr code errors
Before PR:        /review-pr all
After PR fixes:   /review-pr [specific aspects]
```

## Output

Each agent provides:
- **Critical Issues** (must fix)
- **Important Issues** (should fix)
- **Suggestions** (nice to have)
- **Positive Observations**

## Integration

This plugin is located in `.shared/plugins/` for use by both Claude Code and Gemini CLI.
