# R-thinker

This repository is the downstream research workspace for reverse-thinking
agents. It depends on the sibling `tempobench` repository during local
development, so changes made in TempoBench are visible here without copying
benchmark code into this project.

The legacy R-thinker implementation was restored from the old `chess-llm`
history at commit `c36cbbb` (`2026-04-22`, `大改后完整版1.0`). See
`docs/RESTORED_FROM_HISTORY.md` for the exact source and restored paths.

## Layout

```text
R-thinker/
  src/reverse_thinking_agent/  # agent and research code
  Agent/                       # restored legacy R-thinker agents
  Env/                         # restored legacy runtime compatibility layer
  configs/                     # reverse-thinking configs
  experiments/                 # restored and new experiment entry points
  Tools/                       # restored legacy frontend tools
  docs/                        # research notes and design docs
  artifacts/                   # generated outputs, ignored by git
```

## Local Setup

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` installs `../tempobench` in editable mode. During normal
development, update Bench in the sibling `tempobench` repository and this
project will import the updated code from the same local checkout.

The restored April 2026 code has extra dependencies. Install them only when you
need to run the legacy scripts:

```powershell
python -m pip install -r requirements-legacy.txt
Copy-Item config.example.json config.json
```

For reproducible experiments, pin Bench to a tag or commit before running the
experiment and record that version in the experiment notes.

## Quick Check

```powershell
reverse-thinking
```
