# Coding Standards

Environment and tooling preferences for all agents.

---

## Python

| Preference | Standard |
|------------|----------|
| **Package Manager** | `uv` |
| **Virtual Environment** | Always create `.venv` in project root |
| **Dependency File** | `pyproject.toml` (uv native) |

**Setup Commands:**
```bash
uv venv
uv pip install -r requirements.txt
# or
uv sync  # if using pyproject.toml
```

---

## JavaScript / TypeScript

| Preference | Standard |
|------------|----------|
| **Package Manager** | `bun` (preferred) or `npm` |
| **Lock File** | `bun.lockb` or `package-lock.json` |

**Setup Commands:**
```bash
bun install
bun run dev
```

---

## Environment Variables

- Never commit secrets to version control
- Use `.env.example` as a template (committed)
- Use `.env` for local overrides (gitignored)
- For production, use secret managers (Vault, AWS Secrets Manager)

---

## Git

| Preference | Standard |
|------------|----------|
| **Branch Naming** | `feature/<name>`, `fix/<name>`, `chore/<name>` |
| **Commit Style** | Conventional Commits (`feat:`, `fix:`, `docs:`) |

---

## Code Style

| Language | Formatter | Linter |
|----------|-----------|--------|
| Python | `ruff format` | `ruff check` |
| JS/TS | `prettier` | `eslint` |
| CSS | `prettier` | `stylelint` |

---

## Testing

| Language | Framework |
|----------|-----------|
| Python | `pytest` |
| JS/TS | `vitest` or `jest` |

---

## 🔒 Protected Protocols

> [!CAUTION]
> The following protocols are **MANDATORY** and must NEVER be removed during prompt revisions.

| Protocol | Location | Purpose |
|----------|----------|---------|
| **Autonomous Iteration Loop** | `orchestrator/AGENT.md` §71-95 | Enables "ship code while you sleep" sessions |
| **Iteration Loop Workflow** | `workflows/iteration-loop.md` | Step-by-step autonomous execution guide |

**Why Protected:**
- Core to autonomous agent operation
- Enables self-correcting behavior (up to 5 iterations)
- Provides clear escalation path when stuck
- Essential for unattended task completion

