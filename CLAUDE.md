# Safety Rules (STRICT — MUST FOLLOW)

## 🚫 Destructive Actions
- NEVER delete files or directories (no rm -rf)
- NEVER overwrite existing files without explicit permission
- NEVER run git reset --hard or similar destructive git commands

## 🖥️ Cluster Safety
- NEVER run jobs on login nodes
- ALWAYS use job scheduler (e.g., bsub, sbatch)
- NEVER submit excessive jobs (limit parallel jobs to reasonable number)
- NEVER modify shared modules or system environments

## 🔥 Job Protection
- NEVER kill running jobs without explicit permission
- NEVER interrupt long-running jobs
- ALWAYS ask before modifying job scripts that are currently running

## 📦 Environment Safety
- NEVER delete or overwrite conda/virtual environments
- NEVER modify base environment
- ALWAYS create new environments if needed

## 📂 File Safety
- ALWAYS write outputs to new files or new directories
- NEVER overwrite results/ or data/ folders
- ALWAYS confirm before modifying existing scripts

## ⚠️ Code Modification
- NEVER modify existing code without explicit approval
- Adding new code is allowed if it does not alter existing logic
- ALWAYS explain changes before applying them
- ALWAYS show diffs before major edits

## 🤖 Execution Rules

### Default Behavior
- Proceed autonomously for safe operations:
  - creating new files
  - adding new code (without modifying existing code)
  - reading files and analyzing data
  - running non-destructive commands

### Require Permission (MANDATORY)
- ALWAYS ask BEFORE:
  - modifying existing code
  - overwriting any file
  - deleting files
  - changing environments or configurations
  - modifying running jobs or job scripts

### Safety Rule
- When unsure, ASK before acting

## 🧠 General Principle
- When in doubt, ASK before acting


# Audit, Logging & Debug Rules (STRICT)

## 📝 Logging (MANDATORY)
- ALWAYS write a persistent markdown log file (e.g., logs/task_log_<timestamp>.md)
- NEVER overwrite logs

### Log Header (REQUIRED)
- Task objective (what is being done and why)
- Start time
- Working directory
- Key files involved

## 🔄 Step Logging
For each major step, record:
- What was done
- Why it was done
- Files/commands involved
- Outcome

## 📂 Change Tracking
- ALWAYS log all created/modified files
- NEVER make silent changes

## ⚙️ Command Trace
- Log important commands with purpose + summarized result

## 🔍 Self-Check & Debug (MANDATORY)
After ANY code change or execution:
- Check for errors (syntax/runtime)
- Verify outputs exist and are complete
- Check results for correctness and consistency
- Ask: “Does this make sense?”

IF anything is suspicious:
- CONTINUE debugging before proceeding
- Log:
  - issue
  - root cause
  - fix

## 📊 Result Validation (MANDATORY)
- ALWAYS verify results against expectations:
  - dimensions, formats, values, ranges
  - consistency with inputs/metadata
- Explicitly state:
  - whether results are correct
  - whether they make sense

## 🧠 Decision Transparency
- For non-trivial choices, briefly explain reasoning

## 📁 Final Summary (REQUIRED)
At task end, log:
- Steps performed
- Files changed
- Debugging performed
- Validation performed
- Final assessment:
  - objective achieved or not
  - correctness confidence
  - remaining risks/uncertainties

## ⚠️ General Principle
- Assume all work will be audited later
- Optimize for clarity, traceability, and reproducibility

# Permission & Execution Policy

## Default Behavior
- DO NOT ask for permission for routine, safe operations
- Proceed autonomously for:
  - creating new files
  - adding new code (without modifying existing code)
  - reading files and analyzing data
  - running non-destructive commands

## Require Permission (MANDATORY)
- ALWAYS ask for permission BEFORE:
  - modifying existing code 
  - overwriting any existing file
  - deleting files or directories
  - changing configurations or environments
  - modifying scripts that affect running jobs

## Code Modification Rule
- BEFORE modifying existing code:
  - explain what will change
  - explain why it is needed
  - WAIT for explicit approval

## Overwrite Protection
- NEVER overwrite files silently
- If overwrite is needed:
  - explain what will be replaced
  - propose alternative (e.g., new file)
  - WAIT for approval

## Safety Priority
- When unsure whether an action is safe:
  - ASK for permission
- Otherwise:
  - proceed without interruption

## Goal
- Minimize unnecessary interruptions
- Maintain safety for destructive or irreversible actions


## Bash Execution Policy

- Treat standard bash commands as safe ONLY within the current project directory
- NEVER run commands outside the current project directory unless explicitly approved

- DO NOT ask for permission for:
  - read-only commands (e.g., ls, cat, grep)
  - non-destructive commands within the project directory
  - creating new files or directories within the project directory

- ALWAYS ask for permission BEFORE running any command that may:
  - overwrite existing files
  - delete files or directories
  - modify existing code or data
  - affect running jobs, environments, or external systems
  - access or modify files outside the project directory

- When unsure whether a command is safe:
  - ASK before executing

## Scope Rule

- The "current project directory" means the directory where Claude Code is invoked and its subdirectories
- NEVER assume permission to operate outside this scope