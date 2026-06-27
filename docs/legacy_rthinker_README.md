# Real-time Chess & Maze Platform

A unified real-time platform for Chess and Maze with web UIs, multiple agent types (rule-based, random, dual-system ThinkAgent, LLM), and reproducible experiments.

## ✨ Key Features

- 🤖 **Multi-Agent**: Random, rule-based, Human, ThinkAgent (dual-system), and LLM agents
- 🌐 **Web Interfaces**: Flask + Socket.IO real-time UIs for Chess and Maze
- ⚡ **Real-time Loops**: Robust controller with heartbeat frame pushes to avoid UI stalls
- 🧠 **LLM Integration**: Pluggable provider and model via `config.json`
- 🔧 **Symmetric APIs**: Unified action APIs across environments to simplify agents
- ♟️ **Enhanced FEN**: Advanced Chess board state with metadata for agents

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Supported platforms: Windows, Linux, macOS

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.json` and set:
- `api_settings`: your API key, base URL, timeouts
- `llm_settings`: temperature, max_tokens, `default_model`
- `agent_types`: logical keys mapped to agent class names
- `baselines.use_adapters`: true to use adapter-wrapped baselines (Reflexion/CoT/MemoryLLM/Coding)
- `experiments.methods_override`: override methods list (e.g., `["Rule","Random"]`) for smoke tests

### Launching Games

#### Maze Web UI (Recommended)
```bash
python Env/MazeEnv/webapp/app.py
```
Visit http://127.0.0.1:5000

#### Chess Web UI
```bash
python Env/ChessEnv/webapp/app.py
```
Visit http://127.0.0.1:5000

## 🤖 Agent Types

### 1. Random Agent (`Random_Agent`)
- Selects random legal moves
- Configuration example: `Random_Agent`

### 2. Human Player (`Human_Agent`)
- Human player interaction through web interface
- Configuration example: `Human_Agent`

### 3. Rule-based Agent (`Rule_Agent`)
- Chess rule-based intelligent decision making
- Configuration example: `Rule_Agent`

### 4. Dual-system Thinking Agent (`Think_Agent`)
- Combines fast thinking and slow thinking systems
- Configuration example: `Think_Agent`

### 5. LLM Agent (`LLM_Agent`)
- Uses large language models for chess decision making
- Configuration example: `LLM_Agent:qwen/qwen3-235b-a22b:free`

## 🧪 Experiments

Run pairwise comparisons across methods in Chess and Maze (CSV outputs in `Experiments/results/`):

```bash
python Experiments/run_exp1_main_baselines.py
# or ablations
python Experiments/run_exp2_ablation.py
```

Tips:
- For quick local checks, set `"experiments": { "methods_override": ["Rule", "Random"] }` in `config.json`.
- Toggle adapter-wrapped baselines with `"baselines": { "use_adapters": true }`.

## 🛠️ Development Guide

### Adding New Agent Types

1. Create a new agent class in the `Agent/` directory
2. Implement the `get_action` method
3. Add support in the `AgentManager.load_agent` method

### Extending Web Interface

- Modify files in `Env/ChessEnv/webapp/`
- Static resources are located in `static/` directory
- Template files are located in `templates/` directory

### LLM Providers

Models and providers are configurable via `config.json`. Use any provider that supports Chat Completions compatible with your integration.

## 📜 License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests to improve this project.

## 📞 Contact

For questions or suggestions, please use GitHub Issues to contact us.

## 🙏 Acknowledgments

- Built with [python-chess](https://python-chess.readthedocs.io/) for chess logic
- Web interface powered by [Flask](https://flask.palletsprojects.com/) and [SocketIO](https://socket.io/)
- LLM integration through [OpenRouter](https://openrouter.ai/)