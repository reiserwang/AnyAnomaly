#!/usr/bin/env python3
"""
Shared Hook: Dangerous Command Blocker
=======================================
PreToolUse hook that blocks dangerous system commands before execution.
Works with both Claude Code and Gemini CLI.

Loads rules from shared blocklist at .shared/blocked_commands.json

Exit codes:
- 0: Command is safe to execute
- 2: Command is blocked (dangerous) - stderr is shown to the AI

Claude Code: PreToolUse hook with tool_name "Bash"
Gemini CLI: BeforeTool hook with tool_name "Shell", "run_command", etc.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Tuple, List, Dict

# ============================================================================
# SHARED RULES LOADER
# ============================================================================

def get_project_root() -> Path:
    """Find project root by looking for .shared directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".shared" / "blocked_commands.json").exists():
            return current
        current = current.parent
    # Fallback to script's parent (assumes .shared/ structure)
    return Path(__file__).resolve().parent.parent


def load_shared_rules() -> Tuple[List[str], List[str]]:
    """
    Load blocked command patterns from shared JSON file.
    
    Returns:
        Tuple of (blocked_commands, dangerous_patterns)
    """
    project_root = get_project_root()
    rules_path = project_root / ".shared" / "blocked_commands.json"
    
    if not rules_path.exists():
        print(f"Warning: Shared rules not found at {rules_path}", file=sys.stderr)
        return [], []
    
    try:
        with open(rules_path, "r") as f:
            data = json.load(f)
        
        blocked = [item["pattern"] for item in data.get("blocked_commands", [])]
        dangerous = [item["pattern"] for item in data.get("dangerous_patterns", [])]
        return blocked, dangerous
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading shared rules: {e}", file=sys.stderr)
        return [], []


# ============================================================================
# HOOK LOGIC
# ============================================================================

def is_dangerous_command(command: str, blocked_commands: List[str], dangerous_patterns: List[str]) -> Tuple[bool, str]:
    """
    Check if a command matches any dangerous patterns.
    
    Returns:
        Tuple of (is_dangerous, reason)
    """
    # Check blocked commands
    for pattern in blocked_commands:
        if re.search(pattern, command, re.IGNORECASE):
            return True, f"Blocked pattern: {pattern}"
    
    # Check dangerous patterns
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True, f"Dangerous pattern: {pattern}"
    
    return False, ""


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_name = input_data.get("tool_name", "")
    
    # Check for shell/bash tool invocations
    # Claude Code uses "Bash", Gemini CLI uses "Shell", "run_command", etc.
    shell_tools = ["Bash", "Shell", "run_command", "shell", "bash"]
    if tool_name not in shell_tools:
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    # Handle different input formats across tools
    command = tool_input.get("command", "") or tool_input.get("cmd", "") or tool_input.get("CommandLine", "")

    if not command:
        sys.exit(0)

    # Load shared rules
    blocked_commands, dangerous_patterns = load_shared_rules()
    
    if not blocked_commands and not dangerous_patterns:
        print("Warning: No blocking rules loaded, allowing command", file=sys.stderr)
        sys.exit(0)

    is_dangerous, reason = is_dangerous_command(command, blocked_commands, dangerous_patterns)
    
    if is_dangerous:
        print(f"⛔ BLOCKED: This command has been blocked by the security hook.", file=sys.stderr)
        print(f"   Command: {command[:100]}{'...' if len(command) > 100 else ''}", file=sys.stderr)
        print(f"   Reason: {reason}", file=sys.stderr)
        print(f"   To override, ask the user to run this command manually.", file=sys.stderr)
        # Exit code 2 blocks the tool call
        sys.exit(2)
    
    # Command is safe
    sys.exit(0)


if __name__ == "__main__":
    main()
