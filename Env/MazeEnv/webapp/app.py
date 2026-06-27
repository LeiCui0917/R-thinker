"""
MazeEnv.webapp.app
===================

Responsibilities:
- Provide Flask + SocketIO service to drive `RealtimeMazeEnv` and push frame data to frontend.
- Route interfaces aligned with ChessEnv: `/` template page, `/agent_options`, `/set_models`, `/start_game`, `/pause`, `/end`.
- Controller `MazeController` manages environment loop and push logic.
"""

# =============================
# 1. Initialization and Global State
# =============================
import os, sys, threading, time, json, random
from flask_socketio import SocketIO
from flask import Flask, render_template, jsonify, request
# Solve cross-directory import issue: Add project root directory to sys.path
# Project root path: C:\Users\CUILEI\Desktop\ChessLLM_Project\chess-llm
# Current file directory: C:\Users\CUILEI\Desktop\ChessLLM_Project\chess-llm\Env\MazeEnv\webapp
# Three levels up is the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Maze Environment
from Env.MazeEnv.RealtimeMazeEnv import RealtimeMazeEnv
# Import Main Controller
from main_realtime_maze import RealtimeMazeController
from Agent.available_agents import MAZE_AGENT_NAMES, build_agent_groups

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Read configuration file (Keep Agent & LLM options)
CONFIG = {}
try:
    cfg_path = os.path.join(project_root, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            CONFIG = json.load(f)
except Exception as e:
    print(f"Cannot read config.json: {e}")

# Global Runtime State
state = {
    'controller': None,
    'red_model': None,
    'blue_model': None
}


def _get_preview_payload():
    controller = state.get('controller')
    if controller and getattr(controller, 'env', None) is not None:
        env = controller.env
        return {
            'ascii': env.render_ascii(),
            'frame': env.frame_state(),
            'game_over': False,
        }

    gs = (CONFIG.get('game_settings', {}) or {})
    env = RealtimeMazeEnv(
        maze_size=int(gs.get('maze_logical_size', 15)),
        loop_probability=float(gs.get('maze_loop_probability', 0.1)),
        step_duration=float(gs.get('maze_step_duration', 0.5)),
        trail_release_seconds=float(gs.get('maze_trail_release_seconds', 0.0)),
        seed=None,
    )
    return {
        'ascii': env.render_ascii(),
        'frame': env.frame_state(),
        'game_over': False,
    }

@app.route('/')
def index():
    return render_template('Realtimemaze.html')

# =============================
# 3. Web Parameter Return Module
# =============================
# Return agent type and LLM model type
@app.route('/agent_options', methods=['GET'])
def agent_options():
    # Return available agent and model list
    llm_models_map = CONFIG.get('llm_available_models', {})
    default_llm = CONFIG.get('llm_settings', {}).get('default_model')
    llm_models = [v for _, v in llm_models_map.items()]
    groups = build_agent_groups(MAZE_AGENT_NAMES, llm_models, default_llm)
    wd = (CONFIG.get('web_defaults', {}) or {}).get('maze', {}) or {}
    default_selection = {
        'red': {
            'group': (wd.get('red', {}) or {}).get('group', 'llm_base'),
            'agent': (wd.get('red', {}) or {}).get('agent', f'LLM_Agent:{default_llm}' if default_llm else 'LLM_Agent'),
        },
        'blue': {
            'group': (wd.get('blue', {}) or {}).get('group', 'rule'),
            'agent': (wd.get('blue', {}) or {}).get('agent', 'Rule_Agent'),
        },
    }
    return jsonify({
        'success': True,
        'agents': MAZE_AGENT_NAMES,
        'default_llm': default_llm
        ,
        'agent_groups': groups,
        'default_selection': default_selection,
    })

@app.route('/set_models', methods=['POST'])
def set_models():
    data = {}
    if request.is_json:
        try:
            data = request.get_json(force=True) or {}
        except Exception:
            data = {}
    else:
        data = request.form.to_dict()
    
    red_model_name = data.get('red_model')
    blue_model_name = data.get('blue_model')

    if not red_model_name or not blue_model_name:
        return jsonify({'success': False, 'error': 'red_model and blue_model are required'}), 400

    state['red_model'] = red_model_name
    state['blue_model'] = blue_model_name
        
    return jsonify({'success': True})

@app.route('/start_game', methods=['POST'])
def start_game():
    # Stop old game
    if state.get('controller'):
        state['controller'].stop_game()
        state['controller'] = None
    
    # Create new controller
    controller = RealtimeMazeController(
        red_model=state.get('red_model') or 'Rule_Agent',
        blue_model=state.get('blue_model') or 'Rule_Agent',
        push_callback=push_state_to_frontend
    )
    
    ok = controller.start_game()
    if ok:
        state['controller'] = controller
        socketio.emit('pause_state', {'paused': False})
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Start failed'})


@app.route('/initial_state', methods=['GET'])
def initial_state():
    try:
        return jsonify({'success': True, **_get_preview_payload()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pause', methods=['POST'])
def pause():
    controller = state.get('controller')
    if not controller:
        return jsonify({'error': 'Controller not initialized', 'paused': False})
    
    # Toggle pause state
    is_paused = controller.events["pause"].is_set()
    if is_paused:
        controller.resume_game()
        new_state = False
    else:
        controller.pause_game()
        new_state = True

    # Broadcast pause state to frontend
    socketio.emit('pause_state', {'paused': new_state})

    return jsonify({'paused': new_state})

@app.route('/end', methods=['POST'])
def end():
    controller = state.get('controller')
    if controller:
        controller.stop_game()
    return jsonify({'ended': True})

# =============================
# IV. SocketIO Events
# =============================
# 🛠️ Adjust push_state_to_frontend function signature to accept a single state dictionary
def push_state_to_frontend(data):
    payload = {
        'ascii': data.get('ascii', ''),
        'frame': data.get('frame'),
        'game_over': bool(data.get('game_over'))
    }
    if 'termination_reason' in data:
        payload['termination_reason'] = data.get('termination_reason')
    socketio.emit('update_state', payload)

@socketio.on('connect')
def handle_connect(auth=None):
    # If controller exists, push current state
    controller = state.get('controller')
    if controller:
        try:
            push_state_to_frontend({'ascii': controller.env.render_ascii(), 'frame': controller.env.frame_state()})
        except Exception:
            pass

@socketio.on('human_move')
def handle_human_move_socket(data):
    move, role = data.get('uci'), data.get('color')
    ctrl = state.get('controller')
    if not ctrl:
        return {'success': False, 'error': 'no controller'}
    agent = ctrl.red_agent if role == 'red' else ctrl.blue_agent
    agent.set_pending_move(move)
    try:
        getattr(ctrl, 'move_event', None) and ctrl.move_event.set()
    except Exception:
        pass
    return {'success': True}


# =============================
# VIII. Main Program Entry
# =============================
if __name__ == '__main__':
    print("Maze server started successfully! Please visit: http://127.0.0.1:5001")
    # Start with socketio.run, supporting WebSocket/long-polling
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    # python MazeEnv\webapp\app.py
