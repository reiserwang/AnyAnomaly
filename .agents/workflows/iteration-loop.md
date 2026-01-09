---
description: Run autonomous iteration loop until completion criteria met (Ralph Wiggum Technique)
---

# Autonomous Iteration Loop Workflow

Run this workflow to enable "ship code while you sleep" sessions.

## Prerequisites
- Task plan exists in `SCRATCHPAD.md` or `tasks/`
- Completion criteria defined in `SCRATCHPAD.md` → `## ✅ Completion Checklist`
- MAX_ITERATIONS set (default: 5)

## Steps

// turbo-all

1. **Read Current State**
   ```bash
   cat .agents/SCRATCHPAD.md
   ```

2. **Execute Current Task**
   - Run the atomic task from the plan
   - Follow existing agent protocols (Planner → Coder → etc.)

3. **Run Verification**
   ```bash
   # Run tests
   uv run pytest  # or: bun test
   
   # Run linter
   ruff check .   # or: eslint .
   
   # Run build
   uv run build   # or: bun run build
   ```

4. **Evaluate Results**
   - Check ALL items in `## ✅ Completion Checklist`
   - If ANY fail → proceed to step 5
   - If ALL pass → proceed to step 6

5. **Handle Failure**
   - Log to `SCRATCHPAD.md` → `## 📊 Failure Log`:
     ```markdown
     | N | <Error Type> | <Message> | <Lesson Learned> |
     ```
   - Increment `Iteration` counter
   - If Iteration < MAX_ITERATIONS: **GOTO step 2**
   - Else: **STOP** and use `notify_user` to escalate

6. **Handle Success**
   - Invoke DevOps agent for checkpoint commit:
     ```bash
     git add -A
     git commit -m "checkpoint(iter-N): <brief description>"
     ```
   - Update `SCRATCHPAD.md` → `## 📌 Checkpoint History`
   - Mark task complete
   - Proceed to next task or finish

## Abort Conditions
- MAX_ITERATIONS reached without success
- Security-impacting changes detected
- User explicitly requests stop

## Example Usage
```
/iteration-loop
"Run autonomous loop to implement the auth feature until all tests pass"
```
