/* ======================================================
 * 🎯 Maze Frontend Controller Script (Refactored Version, Consistent with ChessEnv)
 * Module Division:
 *  1) Resource Loading
 *  2) Frame Rendering
 *  3) SocketIO Communication
 *  4) Backend API Encapsulation
 *  5) Agent Settings and Control
 *  6) Game Control
 *  7) Initialization Entry
 * ====================================================== */

/* ======================== Module 1: Runtime State ======================== */
// Canvas Initialization
const canvas = document.getElementById('mazeCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;

// Blue Direction Tracking
let chaserAngle = 0;
let agentAngle = 0;
let lastFrame = null; // Store last frame data for repainting

let mazeAgentGroups = [];
let currentMazeSelection = { red: '', blue: '' };

const mazeAssets = window.MAZE_ASSETS || {};
const goalIcon = new Image();
let goalIconLoaded = false;

if (mazeAssets.goalIcon) {
    goalIcon.onload = () => {
        goalIconLoaded = true;
        if (lastFrame) drawFrame(lastFrame);
    };
    goalIcon.src = mazeAssets.goalIcon;
}

/* ======================== Module 2: Terrain Textures (Canvas Patterns) ======================== */
// 2) 墙用草坪纹理（基于原来的绿色，叠加随机斑点）
let grassPattern = null;

function buildGrassPattern() {
    if (!ctx) return null;
    const pc = document.createElement('canvas');
    pc.width = 48;
    pc.height = 48;
    const pctx = pc.getContext('2d');
    if (!pctx) return null;

    // Base grass color (keep original wall green)
    pctx.fillStyle = '#4CAF50';
    pctx.fillRect(0, 0, pc.width, pc.height);

    // Speckles
    pctx.fillStyle = 'rgba(30, 90, 30, 0.28)';
    for (let i = 0; i < 140; i++) {
        const x = Math.floor(Math.random() * pc.width);
        const y = Math.floor(Math.random() * pc.height);
        const r = 1 + Math.floor(Math.random() * 2);
        pctx.beginPath();
        pctx.arc(x, y, r, 0, Math.PI * 2);
        pctx.fill();
    }

    try {
        return ctx.createPattern(pc, 'repeat');
    } catch (e) {
        return null;
    }
}

function ensureGrassPattern() {
    if (!ctx) return;
    if (grassPattern) return;
    grassPattern = buildGrassPattern();
}

/* ======================== Module 3: Frame Rendering ======================== */
function drawFrame(frame) {
    if (!ctx || !frame) return;
    lastFrame = frame; // Update cache
    
    const layout = getCanvasLayout(frame);
    if (!layout) return;
    const { W, H, vw, vh, cellSize, labelSpace, ox, oy } = layout;
    
    // Enable high quality image smoothing
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const drawAxes = () => {
        // Adaptive label stride to avoid clutter on small cells.
        const stride = cellSize >= 18 ? 1 : (cellSize >= 12 ? 2 : 5);
        const fontPx = Math.max(10, Math.floor(cellSize * 0.42));
        ctx.save();
        ctx.fillStyle = '#2c3e50';
        ctx.font = `${fontPx}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Column indices (x) on top
        const topY = oy - Math.floor(labelSpace * 0.55);
        for (let x = 0; x < W; x += stride) {
            const cx = ox + x * cellSize + cellSize / 2;
            ctx.fillText(String(x), cx, topY);
        }

        // Row indices (y) on left
        ctx.textAlign = 'right';
        const leftX = ox - Math.floor(labelSpace * 0.20);
        for (let y = 0; y < H; y += stride) {
            const cy = oy + y * cellSize + cellSize / 2;
            ctx.fillText(String(y), leftX, cy);
        }

        ctx.restore();
    };

    // 1. Background
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, vw, vh);

    ensureGrassPattern();

    // Parse ASCII Map
    const mapRows = frame.map ? frame.map.split('\n') : [];
    
    // Helper function: Check if it is a wall
    const isWall = (x, y) => {
        if (y >= 0 && y < mapRows.length && mapRows[y] && x >= 0 && x < mapRows[y].length) {
            return mapRows[y][x] === '#';
        }
        return false;
    };

    // 2. Draw Map
    // 1) 格子不要有边框：不再画 strokeRect
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const cx = ox + x * cellSize;
            const cy = oy + y * cellSize;

            if (isWall(x, y)) {
                ctx.fillStyle = grassPattern || '#4CAF50';
                ctx.fillRect(cx, cy, cellSize, cellSize);

            } else {
                ctx.fillStyle = '#ECEFF1';
                ctx.fillRect(cx, cy, cellSize, cellSize);
            }
        }
    }

    // Axis labels (draw after base tiles for clarity)
    drawAxes();

    // 2.1 只给“草坪墙”画外轮廓 + 立体阴影：一片草坪一个框（不画内部格子边界）
    // - 上/左边缘：高光
    // - 下/右边缘：阴影
    const edgeW = Math.max(1, cellSize * 0.10);

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = edgeW;

    // Shadow pass (bottom/right)
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.22)';
    ctx.beginPath();
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (!isWall(x, y)) continue;
            const x0 = ox + x * cellSize;
            const y0 = oy + y * cellSize;
            const x1 = x0 + cellSize;
            const y1 = y0 + cellSize;
            if (!isWall(x + 1, y)) { ctx.moveTo(x1, y0); ctx.lineTo(x1, y1); }
            if (!isWall(x, y + 1)) { ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); }
        }
    }
    ctx.stroke();

    // Highlight pass (top/left)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
    ctx.beginPath();
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (!isWall(x, y)) continue;
            const x0 = ox + x * cellSize;
            const y0 = oy + y * cellSize;
            const x1 = x0 + cellSize;
            const y1 = y0 + cellSize;
            if (!isWall(x - 1, y)) { ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); }
            if (!isWall(x, y - 1)) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); }
        }
    }
    ctx.stroke();

    // Subtle outer outline to keep edges crisp
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
    ctx.lineWidth = Math.max(1, cellSize * 0.06);
    ctx.beginPath();
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (!isWall(x, y)) continue;
            const x0 = ox + x * cellSize;
            const y0 = oy + y * cellSize;
            const x1 = x0 + cellSize;
            const y1 = y0 + cellSize;
            if (!isWall(x - 1, y)) { ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); }
            if (!isWall(x + 1, y)) { ctx.moveTo(x1, y0); ctx.lineTo(x1, y1); }
            if (!isWall(x, y - 1)) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); }
            if (!isWall(x, y + 1)) { ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); }
        }
    }
    ctx.stroke();
    ctx.restore();

    // 2.5 Draw Visited Cells (Race Mode)
    // Backend format: visited: { red: [{x,y}, ...], blue: [{x,y}, ...] }
    const visited = frame.visited || null;
    if (visited && Array.isArray(visited.red)) {
        ctx.save();
        // 轨迹填充：内缩 + 半透明
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = '#F44336';
        const inset = Math.max(1, Math.floor(cellSize * 0.12));
        const w2 = Math.max(1, cellSize - inset * 2);
        visited.red.forEach(p => {
            if (!p) return;
            const x = Number(p.x), y = Number(p.y);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return;
            if (isWall(x, y)) return;
            const cx = ox + x * cellSize;
            const cy = oy + y * cellSize;
            ctx.fillRect(cx + inset, cy + inset, w2, w2);
        });
        ctx.restore();
    }
    if (visited && Array.isArray(visited.blue)) {
        ctx.save();
        // 轨迹填充：内缩 + 半透明
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = '#2196F3';
        const inset = Math.max(1, Math.floor(cellSize * 0.12));
        const w2 = Math.max(1, cellSize - inset * 2);
        visited.blue.forEach(p => {
            if (!p) return;
            const x = Number(p.x), y = Number(p.y);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return;
            if (isWall(x, y)) return;
            const cx = ox + x * cellSize;
            const cy = oy + y * cellSize;
            ctx.fillRect(cx + inset, cy + inset, w2, w2);
        });
        ctx.restore();
    }

    // 3. Draw Goals
    const drawGoalMarker = (g) => {
        if (!g) return;
        const x = Number(g.x), y = Number(g.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        const gx = ox + x * cellSize;
        const gy = oy + y * cellSize;
        const size = Math.max(16, Math.floor(cellSize * 0.95));
        const offset = Math.floor((cellSize - size) / 2);

        ctx.save();
        if (goalIconLoaded && goalIcon.naturalWidth > 0 && goalIcon.naturalHeight > 0) {
            ctx.drawImage(goalIcon, gx + offset, gy + offset, size, size);
        } else {
            const cx = gx + cellSize / 2;
            const cy = gy + cellSize / 2;
            ctx.fillStyle = '#FF7043';
            ctx.beginPath();
            ctx.arc(cx, cy, cellSize * 0.30, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    };

    const goalMarkers = frame.goal ? [frame.goal] : [];
    goalMarkers.forEach(drawGoalMarker);

    // 4) 小车不使用图片：用简单几何图形表示
    const drawCar = (centerX, centerY, angleDeg, color) => {
        const r = Math.max(7, cellSize * 0.46);

        ctx.save();
        ctx.translate(centerX, centerY);

        // 椭圆落地阴影（不随车旋转）
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.20)';
        ctx.beginPath();
        if (typeof ctx.ellipse === 'function') {
            ctx.ellipse(0, r * 0.72, r * 0.70, r * 0.28, 0, 0, Math.PI * 2);
        } else {
            // Fallback: scaled circle
            ctx.translate(0, r * 0.72);
            ctx.scale(1.0, 0.4);
            ctx.arc(0, 0, r * 0.70, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.restore();

        // halo
        const gradient = ctx.createRadialGradient(0, 0, r * 0.2, 0, 0, r * 1.6);
        gradient.addColorStop(0, `${color}99`);
        gradient.addColorStop(1, `${color}00`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, r * 1.6, 0, Math.PI * 2);
        ctx.fill();

        ctx.rotate((angleDeg || 0) * Math.PI / 180);

        // body (more obvious arrow: head + tail)
        ctx.beginPath();
        ctx.moveTo(0, -r * 1.05);               // head tip
        ctx.lineTo(r * 0.75, -r * 0.20);        // right shoulder
        ctx.lineTo(r * 0.42, r * 0.95);         // right tail
        ctx.lineTo(0, r * 0.60);                // tail notch
        ctx.lineTo(-r * 0.42, r * 0.95);        // left tail
        ctx.lineTo(-r * 0.75, -r * 0.20);       // left shoulder
        ctx.closePath();

        ctx.fillStyle = color;
        ctx.fill();

        // double outline for contrast on both grass and road
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.lineWidth = Math.max(2, r * 0.22);
        ctx.stroke();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = Math.max(2, r * 0.14);
        ctx.stroke();

        // direction line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.lineWidth = Math.max(2, r * 0.10);
        ctx.beginPath();
        ctx.moveTo(0, r * 0.35);
        ctx.lineTo(0, -r * 0.55);
        ctx.stroke();

        // cockpit dot
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(0, 0, Math.max(2, r * 0.20), 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    };

    // 4. Draw Blue (single)
    const blue = frame.blue;
    if (blue) {
        const cx = blue.x, cy = blue.y;
        
        // Calculate direction
        if (blue.dir === 'N') chaserAngle = 0;
        else if (blue.dir === 'E') chaserAngle = 90;
        else if (blue.dir === 'S') chaserAngle = 180;
        else if (blue.dir === 'W') chaserAngle = 270;

        const centerX = ox + cx * cellSize + cellSize / 2;
        const centerY = oy + cy * cellSize + cellSize / 2;

        drawCar(centerX, centerY, chaserAngle, '#2196F3');
    }

    // 5. Draw Red
    const red = frame.red;
    if (red) {
        const ax = red.x, ay = red.y;
        
        // Prefer backend-provided direction (accurate even for multi-cell moves).
        if (red.dir === 'N') agentAngle = 0;
        else if (red.dir === 'E') agentAngle = 90;
        else if (red.dir === 'S') agentAngle = 180;
        else if (red.dir === 'W') agentAngle = 270;

        const centerX = ox + ax * cellSize + cellSize / 2;
        const centerY = oy + ay * cellSize + cellSize / 2;
        
        drawCar(centerX, centerY, agentAngle, '#F44336');
    }
}

/* ======================== Module 4: SocketIO Communication ======================== */
const socket = io();
socket.on('connect', () => console.log('✅ SocketIO Connected'));
socket.on('disconnect', () => console.warn('⚠️ SocketIO Disconnected'));

socket.on('update_state', data => {
    const gameOver = !!data.game_over;
    
    if (data.frame) drawFrame(data.frame);
    
    if (data.termination_reason && gameOver) {
        $('#status').text(data.termination_reason);
    }
});

/* ======================== Module 5: Backend API Wrapper ======================== */
const api = {
    getModelList: () => $.getJSON('/agent_options'),
    getInitialState: () => $.getJSON('/initial_state'),
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

/* ======================== Module 6: Agent Setup and Control ======================== */
function loadAgentTypes() {
    api.getModelList().done(res => {
        if (!res || !res.success) return;

        mazeAgentGroups = res.agent_groups || [];
        const defaults = res.default_selection || {};

        const $red = $('#redAgentGroup').empty();
        const $blue = $('#blueAgentGroup').empty();

        $red.append($('<option>').val('').text('Select Group'));
        $blue.append($('<option>').val('').text('Select Group'));

        mazeAgentGroups.forEach(group => {
            $red.append($('<option>').val(group.key).text(group.label));
            $blue.append($('<option>').val(group.key).text(group.label));
        });

        const defaultRedGroup = defaults?.red?.group || '';
        const defaultBlueGroup = defaults?.blue?.group || '';
        $('#redAgentGroup').val(defaultRedGroup).trigger('change');
        $('#blueAgentGroup').val(defaultBlueGroup).trigger('change');

        const defaultRedAgent = defaults?.red?.agent;
        const defaultBlueAgent = defaults?.blue?.agent;
        if (defaultRedAgent) $('#redAgentModel').val(defaultRedAgent);
        if (defaultBlueAgent) $('#blueAgentModel').val(defaultBlueAgent);
    }).fail((xhr, status, err) => {
        $('#status').text(`Failed to get options: ${status} ${err}`);
    });
}

function updateAgentSelector(side, groupKey) {
    const group = mazeAgentGroups.find(g => g.key === groupKey);
    const $sel = $(`#${side}AgentModel`).empty();
    $sel.append($('<option>').val('').text('Select Agent...'));
    if (!group) return;
    (group.options || []).forEach(opt => {
        $sel.append($('<option>').val(opt.value).text(opt.label || opt.value));
    });
    const defaultOpt = (group.options || []).find(o => o.is_default) || group.options?.[0];
    if (defaultOpt) {
        $sel.val(defaultOpt.value);
    }
}

function setupAgentTypeHandlers() {
    $('#redAgentGroup').change(function() { updateAgentSelector('red', $(this).val()); });
    $('#blueAgentGroup').change(function() { updateAgentSelector('blue', $(this).val()); });
}

function getSelectedAgent(side) {
    const selected = ($(`#${side}AgentModel`).val() || '').trim();
    if (!selected) {
        return { model_name: null };
    }
    return { model_name: selected };
}

/* ======================== Module 7: Game Control ======================== */
function startMaze() {
    const red = getSelectedAgent('red');
    const blue = getSelectedAgent('blue');
    currentMazeSelection = {
        red: red.model_name || '',
        blue: blue.model_name || '',
    };

    if (!red.model_name || !blue.model_name) {
        $('#status').text('Please select both agents (and LLM models if needed)');
        return;
    }

    $('#status').text('Maze Starting...');
    
    // Build request data
    const payload = {
            red_model: red.model_name,
            blue_model: blue.model_name
    };

    // Set models and start game
    api.setModels(payload).done(res => {
        if (!res || !res.success) {
            const msg = res && res.error ? res.error : 'Setup failed';
            return $('#status').text(msg);
        }
        
        api.startGame().done(r => {
            if (!r || !r.success) {
                const msg = r && r.error ? r.error : 'Start failed';
                $('#status').text(msg);
            } else {
                $('#status').text('Maze Running...');
            }
        }).fail((xhr, status, err) => {
            $('#status').text(`Start request failed: ${status} ${err}`);
        });
    }).fail((xhr, status, err) => {
        $('#status').text(`Setup request failed: ${status} ${err}`);
    });
}

function togglePause() {
    api.pause().done(res => {
        if (res && res.error) {
            $('#status').text(`Operation failed: ${res.error}`);
            return;
        }

        if (res) {
            const paused = !!res.paused;
            $('#pauseBtn').text(paused ? 'Resume Game' : 'Pause Game');
            $('#status').text(paused ? 'Game Paused' : 'Maze Running...');
        }
    }).fail((xhr, status, err) => {
        $('#status').text(`Pause request failed: ${status} ${err}`);
    });
}

function endGame() {
    if (!confirm('Are you sure you want to end the current maze game?')) return;
    
    api.end().done(() => {
        $('#status').text('Game Ended');
        $('#pauseBtn').text('Pause Game');
    }).fail((xhr, status, err) => {
        $('#status').text(`End request failed: ${status} ${err}`);
    });
}

function loadInitialMaze() {
    api.getInitialState().done(res => {
        if (!res || !res.success) return;
        if (res.frame) drawFrame(res.frame);
        $('#status').text('Ready to start');
    }).fail(() => {
        $('#status').text('Waiting to start...');
    });
}

/* ======================== Module 7: Initialization Entry ======================== */
$(document).ready(() => {
    loadInitialMaze();
    loadAgentTypes();
    setupAgentTypeHandlers();
    $('#startBtn').click(startMaze);
    $('#pauseBtn').click(togglePause);
    $('#endBtn').click(endGame);
    
    // Listen for window resize, redraw last frame to maintain clarity
    window.addEventListener('resize', () => {
        if (lastFrame) requestAnimationFrame(() => drawFrame(lastFrame));
    });

    console.log('✅ Maze Frontend Initialized');
});

function getCanvasLayout(frame, mutateCanvas = true) {
    if (!ctx || !canvas || !frame) return null;
    const W = Number(frame.w) || 10;
    const H = Number(frame.h) || 10;
    const pad = 2;
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.floor(canvas.clientWidth);
    const cssH = Math.floor(canvas.clientHeight);
    if (cssW === 0 || cssH === 0) return null;

    const vw = Math.max(1, Math.floor(cssW * dpr));
    const vh = Math.max(1, Math.floor(cssH * dpr));
    let cellSize = Math.floor(Math.min((vw - pad * 2) / (W + 1), (vh - pad * 2) / (H + 1)));
    cellSize = Math.max(1, cellSize);
    const labelSpace = Math.max(10, Math.floor(cellSize * 0.95));
    const totalW = cellSize * W + labelSpace;
    const totalH = cellSize * H + labelSpace;
    const ox = Math.floor((vw - totalW) / 2) + labelSpace;
    const oy = Math.floor((vh - totalH) / 2) + labelSpace;
    if (mutateCanvas) {
        canvas.width = vw;
        canvas.height = vh;
    }
    return { W, H, vw, vh, cellSize, labelSpace, ox, oy, dpr };
}

function selectedHumanRole(mouseButton = 0) {
    const redIsHuman = (currentMazeSelection.red || '') === 'Human_Agent';
    const blueIsHuman = (currentMazeSelection.blue || '') === 'Human_Agent';
    if (redIsHuman && !blueIsHuman) return 'red';
    if (!redIsHuman && blueIsHuman) return 'blue';
    if (redIsHuman && blueIsHuman) return mouseButton === 2 ? 'blue' : 'red';
    return null;
}

function emitHumanMazeMove(evt) {
    if (!lastFrame || !canvas) return;
    const role = selectedHumanRole(evt.button || 0);
    if (!role) return;

    const layout = getCanvasLayout(lastFrame, false);
    if (!layout) return;
    const { W, H, cellSize, ox, oy } = layout;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / Math.max(1, rect.width);
    const scaleY = canvas.height / Math.max(1, rect.height);
    const px = (evt.clientX - rect.left) * scaleX;
    const py = (evt.clientY - rect.top) * scaleY;
    const x = Math.floor((px - ox) / cellSize);
    const y = Math.floor((py - oy) / cellSize);

    if (x < 0 || y < 0 || x >= W || y >= H) return;
    socket.emit('human_move', { uci: [x, y], color: role });
    $('#status').text(`Human ${role} target: (${x}, ${y})`);
}

if (canvas) {
    canvas.addEventListener('click', emitHumanMazeMove);
    canvas.addEventListener('contextmenu', evt => {
        evt.preventDefault();
        emitHumanMazeMove(evt);
    });
}
