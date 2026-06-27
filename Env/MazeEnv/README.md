# MazeEnv

Real-time Maze environment with a single blue pursuing a red, plus a web UI.

## Components

- `RealtimeMazeEnv.py`: Core environment implementing symmetric action APIs and red/blue logic.
- `webapp/`: Flask + Socket.IO web UI
	- `app.py`: Maze web server entry point
	- `templates/Realtimemaze.html`: UI template
	- `static/`: assets and client logic

## API (Environment)

- `get_legal_actions(role)`: legal moves for `'red'` or `'blue'`.
- `apply_action(role, move)`: apply a single move string; returns success boolean.
- `frame_state()`: current frame with keys like `red`, `blue`, `goal`.
- `check_game_status()`: returns `{ game_over, event }`, where `event ∈ { 'goal','caught','timeout' }`.

Notes:
- The environment enforces a single blue (no plural chasers).
- The controller pushes frames even on invalid/no actions to keep the UI responsive.

## Run the Web UI

Install dependencies from the project `requirements.txt`, then:

```bash
python Env/MazeEnv/webapp/app.py
```

Open http://127.0.0.1:5000

