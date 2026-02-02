import subprocess
import json
import time
import requests
import sys
import os
import signal
import shutil

# Configuration
REPO_DIR = os.getcwd()
BACKEND_DIR = os.path.join(REPO_DIR, "backend")
BACKEND_APP = os.path.join(BACKEND_DIR, "app.py")
# Use 'uv run' to launch backend
VENV_PYTHON = "uv" 
TEST_SCRIPT = os.path.join(REPO_DIR, "test_system.py")
BASE_URL = "http://localhost:5001"

def run_command(command, cwd=None, capture_output=True):
    """Result of running a shell command."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            check=True,
            text=True,
            capture_output=capture_output
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Stderr: {e.stderr}")
        raise

def get_open_prs():
    """Get list of open PRs using gh CLI."""
    print("Fetching open PRs...")
    cmd = "gh pr list --json number,headRefName,title,author"
    output = run_command(cmd)
    return json.loads(output)

def wait_for_backend(timeout=60):
    """Wait for backend to be ready."""
    print("Waiting for backend to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Backend is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("Timeout waiting for backend.")
    return False

def start_backend():
    """Start backend server in background."""
    print("Starting backend server...")
    # Using Popen to start in background
    # We use the virtualenv python explicitly
    process = subprocess.Popen(
        [VENV_PYTHON, "app.py"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

def stop_backend(process):
    """Stop backend server."""
    if process:
        print("Stopping backend server...")
        try:
            os.kill(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except Exception as e:
            print(f"Error stopping backend: {e}")
            # Force kill if needed
            try:
                os.kill(process.pid, signal.SIGKILL)
            except:
                pass

def checkout_pr(pr_number):
    """Checkout a PR branch."""
    print(f"Checking out PR #{pr_number}...")
    run_command(f"gh pr checkout {pr_number}")

def run_tests():
    """Run the test suite."""
    print("Running system tests...")
    # Assuming test_system.py is the validation suite
    # It requires the backend to be running
    try:
        # Using sys.executable to ensure we use the same python env if needed, 
        # but test_system.py seems self-contained or relies on system python dependencies?
        # Let's check test_system.py again. It uses requests.
        # We should use the venv python ideally to ensure deps are there, or system python if setup.
        # user has 'test_system.py' in root.
        
        # Let's try running with the same python we are running this script with first, 
        # assuming the env has 'requests'. 
        # If this script is run with system python and it doesn't have requests, it might fail.
        # But test_system.py imports requests. 
        
        subprocess.run(
            [sys.executable, TEST_SCRIPT],
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        print("Tests failed!")
        return False

def merge_pr(pr_number):
    """Merge the PR."""
    print(f"Merging PR #{pr_number}...")
    # Adding --admin to force merge if needed? No, standard merge.
    # --delete-branch to clean up
    run_command(f"gh pr merge {pr_number} --merge --delete-branch")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Automate PR integration.")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually merge.")
    args = parser.parse_args()

    try:
        prs = get_open_prs()
    except Exception as e:
        print(f"Failed to list PRs: {e}")
        return

    if not prs:
        print("No open PRs found.")
        return

    print(f"Found {len(prs)} open PRs.")

    current_branch = run_command("git branch --show-current")
    print(f"Current branch: {current_branch}")

    results = {}

    for pr in prs:
        pr_number = pr['number']
        title = pr['title']
        print(f"\n--- Processing PR #{pr_number}: {title} ---")

        backend_process = None
        try:
            checkout_pr(pr_number)
            
            # Start Backend
            backend_process = start_backend()
            if not wait_for_backend():
                print("Backend failed to start. Skipping PR.")
                results[pr_number] = "Backend Failure"
                continue

            # Run Tests
            if run_tests():
                print("Tests passed!")
                if args.dry_run:
                    print("Dry run: Would merge PR.")
                    results[pr_number] = "Success (Dry Run)"
                else:
                    merge_pr(pr_number)
                    results[pr_number] = "Merged"
            else:
                print("Tests failed. Skipping PR.")
                results[pr_number] = "Test Failure"

        except Exception as e:
            print(f"Error processing PR #{pr_number}: {e}")
            results[pr_number] = f"Error: {e}"
        finally:
            stop_backend(backend_process)
            # Find and kill any lingering processes on port 5001 just in case
            subprocess.run("lsof -ti:5001 | xargs kill -9", shell=True, capture_output=True)

    # Return to original branch (usually main)
    print(f"\nReturning to {current_branch}...")
    run_command(f"git checkout {current_branch}")

    print("\n--- Summary ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
