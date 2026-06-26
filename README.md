# Reverse Thinking Agent

This repository is the downstream research workspace for reverse-thinking
agents. It depends on the sibling `tempobench` repository during local
development, so changes made in TempoBench are visible here without copying
benchmark code into this project.

## Layout

```text
reverse-thinking-agent/
  src/reverse_thinking_agent/  # agent and research code
  configs/                     # reverse-thinking configs
  experiments/                 # experiment entry points and notes
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

For reproducible experiments, pin Bench to a tag or commit before running the
experiment and record that version in the experiment notes.

## Quick Check

```powershell
reverse-thinking
```
