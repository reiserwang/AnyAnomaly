# Feature Dev Plugin

Comprehensive feature development workflow with specialized agents for codebase exploration, architecture design, and quality review.

## Quick Start

```bash
/feature-dev Add user authentication with OAuth support
```

## Workflow Phases

1. **Discovery** - Understand what needs to be built
2. **Codebase Exploration** - Analyze existing patterns with `code-explorer` agents
3. **Clarifying Questions** - Fill in gaps before designing
4. **Architecture Design** - Design approaches with `code-architect` agents
5. **Implementation** - Build the feature
6. **Quality Review** - Review with `code-reviewer` agents
7. **Summary** - Document what was accomplished

## Agents

| Agent | Purpose |
|-------|---------|
| `code-architect` | Design feature architectures and blueprints |
| `code-explorer` | Trace and analyze existing code |
| `code-reviewer` | Review code quality and conventions |

## Usage

```bash
/feature-dev                    # Start guided workflow
/feature-dev Build REST API     # Start with feature description
```

## Key Principles

- Ask clarifying questions before implementing
- Understand codebase patterns first
- Present multiple architecture approaches
- Get user approval before implementation
