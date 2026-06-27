"""
app.py — Chess Game Controller and Web Interface
Responsibilities:
 1. Receive web parameters → Update controller parameters
 2. Receive chess moves from controller → Update web display (including cooldown status)
"""

"""
🔄 Frontend-Backend Data Interaction:
Backend → Frontend (WebSocket Real-time Push via Flask-SocketIO)
 Data Format:
 {
   "fen": "<FEN string>",
   "cooldown_state": [
       {"square": "e2", "duration": 5, "total": 10}, # Full cooldown info
       {"square": "g1", "duration": 3, "total": 10}
   ]
 }

Frontend → Backend (HTTP Calls)
 /set_models  → Set models for both sides
 /start_game  → Start controller (background thread)
 /pause       → Pause/Resume
 /end         → End game
"""

# =============================
# 1. Initialization and Global State
# =============================
import os, json, sys
from flask_socketio import SocketIO 
from flask import Flask, render_template, jsonify, request
# Solve cross-directory import issue: Add project root directory to sys.path
# Project root path: C:\Users\CUILEI\Desktop\ChessLLM_Project\chess-llm
# Current file directory: C:\Users\CUILEI\Desktop\ChessLLM_Project\chess-llm\Env\ChessEnv\webapp
# Three levels up is the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now can import modules located in project root normally
from main_realtime_chess import RealtimeChessController  
from Agent.available_agents import CHESS_AGENT_NAMES, build_agent_groups
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global Runtime State
state = {
    'white_model': None,
    'black_model': None,
    'last_fen': "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Standard initial FEN
    'cooldown_state': [],
    'thread': None # Used to store controller thread instance
}

# Signal control migrated to RealtimeChessController internal event dictionary, removed module-level events to avoid redundancy

# Return Home HTML
@app.route('/')
def index():
    # Fix: Change filename from 'index.html' to 'Realtimechess.html'
    return render_template('Realtimechess.html')  # Assuming HTML is in templates/Realtimechess.html
# =============================
# 2. Web Parameter Initialization Module
# =============================
def load_config():
    """Read config.json and return dictionary"""
    # config.json is located at project root
    path = os.path.join(project_root, 'config.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)
    
@app.route('/get_model_list', methods=['GET'])
def get_model_list():
    """Return all available agent types and LLM model list"""
    config = load_config()
    llm_models = list(config.get('llm_available_models', {}).values())
    default_llm = config.get('llm_settings', {}).get('default_model')
    groups = build_agent_groups(CHESS_AGENT_NAMES, llm_models, default_llm)
    wd = (config.get('web_defaults', {}) or {}).get('chess', {}) or {}
    default_selection = {
        'white': {
            'group': (wd.get('white', {}) or {}).get('group', 'llm_base'),
            'agent': (wd.get('white', {}) or {}).get('agent', f'LLM_Agent:{default_llm}' if default_llm else 'LLM_Agent'),
        },
        'black': {
            'group': (wd.get('black', {}) or {}).get('group', 'rule'),
            'agent': (wd.get('black', {}) or {}).get('agent', 'Rule_Agent'),
        },
    }
    return jsonify({
        'success': True,
        'agents': CHESS_AGENT_NAMES,
        'default_llm': default_llm,
        'agent_groups': groups,
        'default_selection': default_selection,
    })

# =============================
# 3. Web Parameter Return Module
# =============================
# Return agent type and LLM model type
@app.route('/set_models', methods=['POST'])
def set_models():
    """Update memory after web selects model (Merged version)"""
    data = request.get_json() or {}
    
    # Basic Agent Type (Required)
    state['white_model'] = data.get('white_agent')
    state['black_model'] = data.get('black_agent')
    
    # LLM Model (Optional)
    state['white_llm_model'] = data.get('white_llm')
    state['black_llm_model'] = data.get('black_llm')

    # Return unified format
    return jsonify({
        'success': True,
        'white': {
            'type': state['white_model'],
            'llm_model': state['white_llm_model'] if str(state['white_model']).startswith('LLM_Agent') else None
        },
        'black': {
            'type': state['black_model'],
            'llm_model': state['black_llm_model'] if str(state['black_model']).startswith('LLM_Agent') else None
        }
    })

@app.route('/start_game', methods=['POST'])
def start_game():
    white, black = state['white_model'], state['black_model']
    
    if not white or not black:
        return jsonify({'success': False, 'error': 'Please select models for both sides first'})
    
    # Create and start the controller
    controller = RealtimeChessController(
        white_model=white,
        black_model=black,
        push_callback=push_state_to_frontend,  # Add callback function
    )
    if controller.start_game():
        state['controller'] = controller
        # New game: Authoritatively reset the frontend pause flag to ensure the cooldown animation does not maintain the last pause state
        try:
            socketio.emit('pause_state', {'paused': False})
        except Exception as e:
            print(f"[WARN] Failed to reset pause state: {e}")
        # Push the opening position to the frontend immediately after successful startup
        try:
            initial_fen = controller.env.get_enhancedFENfull()
            push_state_to_frontend({
                'fen': initial_fen,
                'cooldown_duration': getattr(controller.env, 'cooldown', None)
            })
        except Exception as e:
            print(f"[WARN] Initial FEN push failed: {e}")
        return jsonify({'success': True, 'info': 'Controller started'})
    return jsonify({'success': False, 'error': 'Start failed'})

@app.route('/pause', methods=['POST'])
def pause():
    """Toggle pause/resume (based on controller instance internal events)"""
    controller = state.get('controller')
    if not controller:
        return jsonify({'error': 'Controller not initialized', 'paused': False})

    # Use controller internal events as authoritative state
    pause_evt = controller.events.get('pause')
    if pause_evt and pause_evt.is_set():
        # currently paused -> resume
        controller.resume_game()
        new_state = False
    else:
        controller.pause_game()
        new_state = True

    # Broadcast pause state to frontend (used to freeze/resume cooldown animation)
    socketio.emit('pause_state', {'paused': new_state})

    return jsonify({'paused': new_state})

@app.route('/end', methods=['POST'])
def end():
    """End game"""
    controller = state.get('controller')
    if controller:
        controller.stop_game()  # Call controller method
    return jsonify({'ended': True})

# =============================
# IV. SocketIO Events
# =============================
# 🛠️ Adjust push_state_to_frontend function signature to accept a single state dictionary
def push_state_to_frontend(data):
    """
    Called by the main controller to push board and cooldown status in real-time
    """
    # Compatible receiver: prioritize 'fen' (Enhanced_FEN_Full), backward compatible with old key names
    fen_full = data.get('fen') or data.get('enhanced_FEN_full') or data.get('Enhancedfen_full') or data.get('Enhanced_FEN_Full') or state['last_fen']

    # Update internal state, only store fen (no longer construct cooldown_state, receiver parses as needed)
    state['last_fen'] = fen_full

    # Receive cooldown_duration provided by the caller (priority), otherwise use the existing value in the module
    incoming_cd = data.get('cooldown_duration')
    if incoming_cd is not None:
        incoming_cd = float(incoming_cd)
        state['cooldown_duration'] = incoming_cd

    # Send unified SocketIO event 'update_state': send core state and optional end information
    payload = {
        'fen': fen_full,
        'cooldown_duration': state.get('cooldown_duration'),
        'game_over': bool(data.get('game_over', data.get('event') == 'game_over'))
    }
    # Pass through optional fields (if they exist)
    if 'termination_reason' in data:
        payload['termination_reason'] = data.get('termination_reason')

    socketio.emit('update_state', payload)

@socketio.on('connect')
def handle_connect():
    # When the client connects, immediately push the current board state to help the frontend initialize quickly
    socketio.emit('update_state', {
        'fen': state['last_fen'],
        'cooldown_duration': state.get('cooldown_duration')
    })

@socketio.on('human_move')
def handle_human_move_socket(data):
    """
    Handle human move requests sent by the frontend via Socket.IO
    Same function as HTTP POST route, but responds via Socket.IO event
    """
    uci, color = data.get('uci'), data.get('color')
    ctrl = state.get('controller')
    if not ctrl:
        return {'success': False, 'error': 'no controller'}
    agent = ctrl.white_agent if color == 'w' else ctrl.black_agent
    agent.set_pending_move(uci)
    # Event wakeup: Notify the main control loop to check the action immediately to reduce waiting latency
    try:
        getattr(ctrl, 'move_event', None) and ctrl.move_event.set()
    except Exception:
        pass
    print(f"[DBG SOCKET] after set_pending_move pending={getattr(agent,'pending_move',None)}")
    return {'success': True}


# =============================
# VIII. Main Program Entry
# =============================
if __name__ == '__main__':
    print("Server started successfully! Please visit: http://127.0.0.1:5000")
    # Start with socketio.run, supporting WebSocket/long-polling
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    # python ChessEnv\webapp\app.py