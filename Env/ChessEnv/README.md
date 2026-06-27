# ChessEnv

Real-time Chess environment and web UI.

## Components

- `RealtimeChessEnv.py`: Core environment with cooldown, legal move generation, enhanced FEN, and termination checks.
- `webapp/`: Flask + Socket.IO web UI
	- `app.py`: Chess web server entry point
	- `templates/Realtimechess.html`: UI template
	- `static/`: assets and client logic

## API (Environment)

- `get_legal_moves_for_color(color)`: list of legal UCI moves for `'w'` or `'b'`.
- `process_move_action(move_uci, color)`: apply a UCI move for the specified color.
- `get_enhancedFENfull()`: returns enhanced FEN string with extra metadata for agents.
- `check_game_status()`: termination info including `game_over` and reason.

## Run the Web UI

Make sure dependencies are installed (see project `requirements.txt`). Then:

```bash
python Env/ChessEnv/webapp/app.py
```

Open http://127.0.0.1:5000

## Notes

- Agents for Chess implement `get_action(enhanced_FEN_full, color, legal_moves)`.
- For experiments and LLM configuration, see the project root `README.md` and `config.json`.
