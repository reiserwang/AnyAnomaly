# Ralph Wiggum Plugin

Implementation of the Ralph Wiggum technique for iterative, self-referential AI development loops.

## Quick Start

```bash
/ralph-loop "Build a REST API for todos. Requirements: CRUD operations, input validation, tests. Output <promise>COMPLETE</promise> when done." --completion-promise "COMPLETE" --max-iterations 50
```

## Commands

- `/ralph-loop` - Start an iterative development loop
- `/cancel-ralph` - Cancel an active Ralph loop
- `/ralph-help` - Get help on using the Ralph Wiggum technique

## How It Works

The same prompt is fed to the agent repeatedly. The "self-referential" aspect comes from the agent seeing its own previous work in the files and git history.

**Each iteration:**
1. Agent receives the SAME prompt
2. Works on the task, modifying files
3. Tries to exit
4. Stop hook intercepts and feeds the same prompt again
5. Iteratively improves until completion

## Options

- `--max-iterations <n>` - Stop after N iterations (default: unlimited)
- `--completion-promise <text>` - Phrase that signals completion

## Files

- `commands/` - Slash command definitions
- `hooks/stop-hook.sh` - Stop hook that implements the loop
- `scripts/setup-ralph-loop.sh` - Setup script for initializing loops

## Learn More

- [Original technique](https://ghuntley.com/ralph/)
- [Ralph Orchestrator](https://github.com/mikeyobrien/ralph-orchestrator)
