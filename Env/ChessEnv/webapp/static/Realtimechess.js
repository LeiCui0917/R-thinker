/* ======================================================
 * 🔄 Frontend-Backend Interaction:
 * ------------------------------------------------------
 * Backend → Frontend (WebSocket Push)
 *   {
 *     "fen": "<Enhanced FEN>",               // Recommended: Use key 'fen' to carry enhanced_FEN_full
 *     "enhanced_FEN_full": "<Enhanced FEN>", // Optional alias (Backward compatibility/Internal consistency)
 *     "cooldown_state": [...],
 *     "game_over": false
 *   }
 * Frontend → Backend (HTTP Request)
 *   /set_models, /start_game, /pause, /end
 * ====================================================== */


/* ======================================================
 * 🎯 Real-time Chess Frontend Controller Script (Optimized Version)
 * ====================================================== */


/* ======================== Module 1: Piece Resource Detection ======================== */
let useLocalPieces = false;

async function testOnlinePieces() {
    const testPiece = 'wK';
    const url = `https://chessboardjs.com/img/chesspieces/wikipedia/${testPiece}.png`;
    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(true);
        img.onerror = () => { useLocalPieces = true; resolve(false); };
        img.src = url;
        setTimeout(() => { 
            if (!img.complete) { useLocalPieces = true; resolve(false); }
        }, 1000);
    });
}


/* ======================== Module 2: Board Initialization ======================== */
const board = Chessboard('board', {
    // Allow dragging, but we will decide whether to submit move in onDrop based on selected agent type
    draggable: true,
    position: 'start',
    pieceTheme: piece => `/static/img/chesspieces/wikipedia/${piece}.png`,
    onDrop: onPieceDrop
});
let currentFen = 'start';
// Click move state
let selectedSquare = null;
let selectedPieceColor = null;
// Pause state (used to freeze cooldown animation)
let isPaused = false;
// Prevent dragging moves when game over
let gameOverFlag = false;
// Cooldown alignment debounce threshold (seconds): Do not force snap update if difference between frontend animation remaining time and backend calculation is less than this value
const COOLDOWN_DEBOUNCE_SEC = 0.15;
let chessAgentGroups = [];


/* ======================== Module 3: Enhanced FEN Parsing ======================== */
// Parse Enhanced FEN (Full name: enhanced_FEN_full)
function parseEnhancedFen(enhanced_FEN_full) {
    if (!enhanced_FEN_full) return { fen: 'start', cooldownInfo: {} };
    try {
        const [fen, cooldownPart] = enhanced_FEN_full.split('|cooldown:');
        const cooldownInfo = {};
        if (cooldownPart) {
            cooldownPart.split(',').forEach(entry => {
                const [square, ts] = entry.split(':');
                const t = parseFloat(ts);
                if (square && !isNaN(t)) cooldownInfo[square] = t;
            });
        }
        return { fen: fen.trim(), cooldownInfo };
    } catch (e) {
        console.error('Failed to parse Enhanced FEN:', e);
        return { fen: enhanced_FEN_full, cooldownInfo: {} };
    }
}
/* ======================== Module 4: Board and Cooldown Rendering ======================== */
let cooldownPieces = [];

// Clear all cooldown overlays (used to completely clear residues after new game or game end)
function clearCooldownOverlays() {
    cooldownPieces.forEach(p => { try { p.overlay.remove(); } catch (e) {} });
    cooldownPieces = [];
}

function updateBoard(enhanced_FEN_full) {
    if (!enhanced_FEN_full) return {};
    const { fen: pureFen, cooldownInfo } = parseEnhancedFen(enhanced_FEN_full);
    currentFen = pureFen;
    board.position(pureFen);
    return cooldownInfo;
}

// Handle piece drop event (Minimal implementation: Submit to backend only when the side is set to Human)
function onPieceDrop(source, target, piece) {
    if (gameOverFlag) {
    return 'snapback';
    }
    const whiteSel = $('#whiteAgentModel').val();
    const blackSel = $('#blackAgentModel').val();
    const whiteIsHuman = whiteSel === 'Human_Agent';
    const blackIsHuman = blackSel === 'Human_Agent';

    if (!whiteIsHuman && !blackIsHuman) return 'snapback';
    if (!piece || piece.length < 1) return 'snapback';

    const movedPieceColor = piece[0] === 'w' ? 'w' : 'b';
    if (movedPieceColor === 'w' && !whiteIsHuman) return 'snapback';
    if (movedPieceColor === 'b' && !blackIsHuman) return 'snapback';

    const uci = source + target;
    // use Socket.IO with ack to reduce latency compared to HTTP
    try {
        socket.emit('human_move', { uci: uci, color: movedPieceColor }, function(res) {
            if (res && res.success) {
                if (res.fen) updateBoard(res.fen);
                // success: do nothing more here (server will also push updates)
            } else {
                // rejected by server: restore authoritative board
                board.position(currentFen);
            }
        });
    } catch (e) {
        // emit failed: restore
        board.position(currentFen);
    }

    return undefined;
}

function renderCooldowns(fen, cooldownState, gameOver, serverCooldown) {
    const cooldownInfo = updateBoard(fen);
    const now = Date.now() / 1000;
    const total = Number(serverCooldown);

    // Update cooldown layer (delay briefly to ensure board DOM update completes first, avoiding cooldown animation appearing before piece move)
    // Delay slightly before creating/updating overlay to ensure piece has completed visual move, avoiding the illusion of "cooldown appearing before piece moves"
    setTimeout(() => {

        Object.entries(cooldownInfo).forEach(([square, start]) => {
            const remaining = Math.max(0, Math.min(total, start + total - now));
            if (remaining <= 0) return;

            let item = cooldownPieces.find(p => p.square === square);
            if (!item) {
                const $sq = $(`.square-55d63[data-square="${square}"]`);
                if (!$sq.length) return;
                $sq.css('position', 'relative');
                const overlay = $('<div class="cooldown-overlay"></div>').appendTo($sq);
                overlay.css({
                    height: '100%',
                    top: 0,
                    position: 'absolute',
                    left: 0, width: '100%',
                    backgroundColor: 'rgba(0,150,255,0.3)',
                    pointerEvents: 'none',
                    'transform-origin': 'bottom',
                    'will-change': 'transform',
                    transform: `scaleY(${(remaining / total)})`
                });
                cooldownPieces.push({ square, remaining, total, startTime: start, overlay });
            } else {
                // Debounce: Do not update if difference with backend calculation is less than threshold, avoiding minor jitter
                const diff = Math.abs((item.remaining || 0) - remaining);
                if (diff >= COOLDOWN_DEBOUNCE_SEC) {
                    item.remaining = remaining;
                    item.total = total;
                }
            }
        });
    }, 60);

    // Clean up expired cooldown layers
    const t = Date.now() / 1000;
    cooldownPieces = cooldownPieces.filter(p => {
        if (p.startTime + p.total > t) return true;
        p.overlay.remove();
        return false;
    });

    if (gameOver) $('#status').text('Game Over');
}

function animateCooldowns() {
    const now = Date.now();
    const dt = (now - (animateCooldowns.lastTime || now)) / 1000;
    animateCooldowns.lastTime = now;

    // Freeze animation when paused (do not update overlay height)
    if (isPaused) {
        requestAnimationFrame(animateCooldowns);
        return;
    }

    cooldownPieces = cooldownPieces.filter(p => {
        p.remaining -= dt;
        if (p.remaining <= 0) { p.overlay.remove(); return false; }
        const pct = Math.max(0, Math.min(1, p.remaining / p.total));
        p.overlay.css('transform', `scaleY(${pct})`);
        return true;
    });

    requestAnimationFrame(animateCooldowns);
}
animateCooldowns();


/* ======================== Module 5: SocketIO Communication ======================== */
const socket = io();
socket.on('connect', () => console.log('✅ SocketIO Connected'));
socket.on('disconnect', () => console.warn('⚠️ SocketIO Disconnected'));
socket.on('update_state', data => {
    try {
        // Accept either the 'fen' key or the canonical 'enhanced_FEN_full' alias
        const fen_payload = data.fen;
        // game_over is provided by the backend, and the frontend renders the final state accordingly
        const gameOver = !!data.game_over;
        // Pass server-provided cooldown_duration into render function
        renderCooldowns(fen_payload, data.cooldown_state, gameOver, data.cooldown_duration);
        // Update global interception flag: disable dragging when game is over
        gameOverFlag = gameOver;
        // Status bar prompt: if game over and a reason is provided, display it
        if (data.termination_reason && gameOver) {
            $('#status').text(data.termination_reason);
        }
    } catch (e) { console.error('Socket data parsing error:', e); }
});
// Receive pause/resume state, control whether cooldown animation is frozen
socket.on('pause_state', data => {
    isPaused = !!(data && data.paused);
});


/* ======================== Module 6: Backend API Encapsulation ======================== */
const api = {
    getModelList: () => $.getJSON('/get_model_list'),
    setModels: data => $.ajax({
        url: '/set_models',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(data)
    }),
    startGame: () => $.post('/start_game'),
    pause: () => $.post('/pause'),
    end: () => $.post('/end')
};


/* ======================== Module 7: Agent Setup and Control ======================== */
// Load agents and models
function loadAgentTypes() {
    api.getModelList().done(data => {
        chessAgentGroups = data?.agent_groups || [];
        const defaults = data?.default_selection || {};

        const $whiteGroup = $('#whiteAgentGroup').empty();
        const $blackGroup = $('#blackAgentGroup').empty();
        $whiteGroup.append($('<option>').val('').text('Please select group'));
        $blackGroup.append($('<option>').val('').text('Please select group'));

        chessAgentGroups.forEach(g => {
            $whiteGroup.append($('<option>').val(g.key).text(g.label));
            $blackGroup.append($('<option>').val(g.key).text(g.label));
        });

        const defaultWhiteGroup = defaults?.white?.group || '';
        const defaultBlackGroup = defaults?.black?.group || '';
        $('#whiteAgentGroup').val(defaultWhiteGroup).trigger('change');
        $('#blackAgentGroup').val(defaultBlackGroup).trigger('change');

        const defaultWhiteAgent = defaults?.white?.agent;
        const defaultBlackAgent = defaults?.black?.agent;
        if (defaultWhiteAgent) $('#whiteAgentModel').val(defaultWhiteAgent);
        if (defaultBlackAgent) $('#blackAgentModel').val(defaultBlackAgent);
    });
}

function updateAgentSelector(side, groupKey) {
    const group = chessAgentGroups.find(g => g.key === groupKey);
    const $sel = $(`#${side}AgentModel`).empty();
    $sel.append($('<option>').val('').text('Please select agent'));
    if (!group) return;
    (group.options || []).forEach(opt => {
        const text = opt.label || opt.value;
        $sel.append($('<option>').val(opt.value).text(text));
    });

    const defaultOpt = (group.options || []).find(o => o.is_default) || group.options?.[0];
    if (defaultOpt) {
        $sel.val(defaultOpt.value);
    }
}

function setupAgentTypeHandlers() {
    $('#whiteAgentGroup').change(function() { updateAgentSelector('white', $(this).val()); });
    $('#blackAgentGroup').change(function() { updateAgentSelector('black', $(this).val()); });
}

function getSelectedAgent(side) {
    const selected = ($(`#${side}AgentModel`).val() || '').trim();
    return { model_name: selected || null };
}


/* ======================== Module 8: Game Control ======================== */
function startLLMGame() {
    const white = getSelectedAgent('white');
    const black = getSelectedAgent('black');
    if (!white.model_name || !black.model_name) {
        $('#status').text('Please select both agents');
        return;
    }
    $('#status').text('Game starting...');

    const data = {
        white_agent: white.model_name,
        black_agent: black.model_name
    };
    clearCooldownOverlays();
    api.setModels(data).done(res => {
        // Check if the backend confirms successful setup
        if (!res || !res.success) {
            const msg = res && res.error ? res.error : 'Setup failed';
            return $('#status').text(msg);
        }
        api.startGame().done(r => {
            if (!r || !r.success) {
                const msg = r && r.error ? r.error : 'Start failed';
                $('#status').text(msg);
            } else {
                $('#status').text('Game in progress...');
            }
        }).fail((xhr, status, err) => {
            $('#status').text(`Start request failed: ${status} ${err}`);
        });
    });
}

function togglePause() {
    api.pause().done(res => {
        // If the backend returns an error, display the error message to the user
        if (res && res.error) {
            $('#status').text(`Operation failed: ${res.error}`);
            return;
        }

        if (res) {
            const paused = !!res.paused;
            $('#pauseBtn').text(paused ? 'Resume Game' : 'Pause Game');
            $('#status').text(paused ? 'Game Paused' : 'Game in progress...');
        }
    }).fail((xhr, status, err) => {
        // Give feedback when request fails (network or server error)
        $('#status').text(`Request failed: ${status} ${err}`);
    });
}

function endGame() {
    if (!confirm('Are you sure you want to end the current game?')) return;
    api.end().done(() => {
        $('#status').text('Game Ended');
        $('#pauseBtn').text('Pause Game');
    });
}


/* ======================== Module 9: Initialization Entry ======================== */
$(document).ready(() => {
    loadAgentTypes();
    setupAgentTypeHandlers();
    board.position('start');
    $('#startBtn').click(startLLMGame);
    $('#pauseBtn').click(togglePause);
    $('#endBtn').click(endGame);
    console.log('✅ Frontend initialization complete');
});