# Restored R-thinker Snapshot

This repository includes legacy R-thinker code restored from the old
`chess-llm` Git history.

- Source repository: `ChessLLM_Project/chess-llm`
- Source commit: `c36cbbb74f21de1ba83434178bbb861d11e4f273`
- Commit date: `2026-04-22 20:31:12 +0800`
- Commit message: `大改后完整版1.0`

Restored code and assets:

- `Agent/R_Thinker/`
- `Agent/R_thinker_variants/`
- `Agent/prompt/`
- `Agent/*_agents/`
- `Agent/utils/`
- `Env/`
- `experiments/`
- `Tools/`
- `main_realtime_chess.py`
- `main_realtime_common.py`
- `main_realtime_maze.py`

The historical `Agent/logs/`, `Experiments/logs/`, `Experiments/results/`, and
`Experiments/records/` runtime outputs were intentionally not restored.

The historical `config.json` contained real API credentials. It was not copied.
Use `config.example.json` as a safe template and keep your local `config.json`
untracked.
