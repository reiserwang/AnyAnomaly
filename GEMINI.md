# GEMINI Protocol
**Root Index & Team Shared Memory**

## 🤖 Agent Registry
| Role | Resp | Path |
| :--- | :--- | :--- |
| **Orchestrator** | **Orchestrator**. Start here. | [.agents/orchestrator/AGENT.md](.agents/orchestrator/AGENT.md) |
| **Planner** | **Thinking**. Specs, Arch, Plans. | [.agents/planner/AGENT.md](.agents/planner/AGENT.md) |
| **Coder** | **Build**. Reviewer, Dev, Test. | *(Standard)* |
| **Reviewer** | **Quality**. Sec, Perf, Refactor. | [.agents/code_reviewer/AGENT.md](.agents/code_reviewer/AGENT.md) |
| **Tester** | **Verify**. Plans, Auto-tests. | [.agents/tester/AGENT.md](.agents/tester/AGENT.md) |
| **DevOps** | **Ops**. Git, CI/CD, Docker. | [.agents/devops/AGENT.md](.agents/devops/AGENT.md) |
| **Security** | **Sec**. SBOM, Threat Model. | [.agents/security/AGENT.md](.agents/security/AGENT.md) |
| **UI/UX** | **Design**. Styles, Palettes. | [.agents/ui_ux/AGENT.md](.agents/ui_ux/AGENT.md) |
| **Writer** | **Docs**. API, Guides. | [.agents/tech_writer/AGENT.md](.agents/tech_writer/AGENT.md) |

## 📂 Artifact Standards
| Type | Path | Owner |
| :--- | :--- | :--- |
| **Specs** | `specs/` | Planner |
| **Design** | `design/` | Planner |
| **Docs** | `docs/` | Writer |
| **Tests** | `tests/` | Tester |
| **Sec** | `security/` | Security |
| **Standards** | [.agents/STANDARDS.md](.agents/STANDARDS.md) | **ALL** |
| **State** | `.agents/SCRATCHPAD.md` | **ALL** |

## 🧠 Shared Memory Protocol
1.  **State**: Always read/write `.agents/SCRATCHPAD.md` for active context.
2.  **Flow**: Orchestrator -> Planner (Think) -> Planner (Plan) -> Orchestrator -> Coder (Exec) -> Verify.
3.  **Git**: DevOps owns branches/PRs. Check `task.md` for top-level tracking.
