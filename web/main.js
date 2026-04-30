const buildId = new URL(import.meta.url).search;
const wasmModulePath = `./pkg/steinbeisser_bg.wasm${buildId}`;
const workerModulePath = `./worker.js${buildId}`;

const engineModule = await import(`./pkg/steinbeisser.js${buildId}`);
const boardModule = await import(`./board.js${buildId}`);

const {
  default: initEngine,
  apply_move,
  legal_moves_for_selection,
  new_session,
  session_status,
  undo_full_turn,
} = engineModule;

const {
  CELLS,
  FRAME_POINTS,
  VIEWBOX,
  occupantAt,
  parsePositionString,
  renderBoard,
} = boardModule;

const DEFAULT_MAX_DEPTH = 17;
const DEFAULT_MAX_TIME_SECONDS = 3;
const MIN_MAX_DEPTH = 1;
const MAX_MAX_DEPTH = 64;
const MIN_MAX_TIME_SECONDS = 0.1;
const MAX_MAX_TIME_SECONDS = 600;
const MAX_TIME_STEP_SECONDS = 0.1;
const EVAL_EXPECTED_OUTCOME_SCORE_SCALE = 996;
const MAX_MARBLES_PER_SIDE = 14;

const state = {
  session: null,
  selected: [],
  candidates: [],
  evaluationWhiteScore: 0,
  humanColor: 'black',
  depthLimitEnabled: false,
  timeLimitEnabled: true,
  maxDepth: DEFAULT_MAX_DEPTH,
  maxTimeSeconds: DEFAULT_MAX_TIME_SECONDS,
  editMode: false,
  demoMode: false,
  thinking: false,
  thinkingSide: null,
  activeSearchKind: null,
  worker: null,
  workerReady: null,
  nextRequestId: 1,
  activeRequestId: null,
  lastEngineSearchInfo: null,
  debugEnabled: false,
  lastEngineDebugInfo: null,
  notice: '',
  noticeKind: null,
};
let initialSessionPosition = null;

const boardContainer = document.querySelector('[data-board]');
const evalRail = document.querySelector('.eval-rail');
const evalBar = document.querySelector('.eval-bar');
const evalFill = document.querySelector('[data-eval-fill]');
const evalLabel = document.querySelector('[data-eval-label]');
const demoButton = document.querySelector('[data-action="demo"]');
const toggleSideButton = document.querySelector('[data-action="toggle-side"]');
const takeBackButton = document.querySelector('[data-action="takeback"]');
const editPositionButton = document.querySelector('[data-action="edit-position"]');
const positionEditor = document.querySelector('[data-position-editor]');
const positionInput = document.querySelector('[data-position-input]');
const resetButton = document.querySelector('[data-action="reset"]');
const depthLimitToggle = document.querySelector('[data-toggle="depth-limit"]');
const timeLimitToggle = document.querySelector('[data-toggle="time-limit"]');
const debugToggle = document.querySelector('[data-toggle="debug"]');
const maxDepthInput = document.querySelector('[data-setting="max-depth"]');
const maxTimeInput = document.querySelector('[data-setting="max-time-seconds"]');
const thinkingDots = document.querySelector('[data-thinking-dots]');
const searchInfoText = document.querySelector('[data-search-info]');
const debugPanel = document.querySelector('[data-debug-panel]');
const debugSpeed = document.querySelector('[data-debug-speed]');
const debugTime = document.querySelector('[data-debug-time]');
const debugNodes = document.querySelector('[data-debug-nodes]');
const debugDepth = document.querySelector('[data-debug-depth]');
const debugPly = document.querySelector('[data-debug-ply]');
const noticeText = document.querySelector('[data-notice]');
const frameTop = Math.min(...FRAME_POINTS.map(([, y]) => y));
const frameLeftTip = FRAME_POINTS.reduce(
  (leftmost, point) => (point[0] < leftmost[0] ? point : leftmost),
  FRAME_POINTS[0],
);
const evalBarTopRatio = (frameTop - (VIEWBOX.minY ?? 0)) / VIEWBOX.height;
const evalBarHeightRatio = ((frameLeftTip[1] - frameTop) * 2) / VIEWBOX.height;

function clampInteger(value, fallback, minimum, maximum) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(maximum, Math.max(minimum, parsed));
}

function clampNumber(value, fallback, minimum, maximum) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(maximum, Math.max(minimum, parsed));
}

function clampSteppedNumber(value, fallback, minimum, maximum, step) {
  const clamped = clampNumber(value, fallback, minimum, maximum);
  const stepped = Math.round(clamped / step) * step;
  return Number(Math.min(maximum, Math.max(minimum, stepped)).toFixed(10));
}

function syncSearchInputs() {
  maxDepthInput.value = String(state.maxDepth);
  maxTimeInput.value = String(state.maxTimeSeconds);
}

function compressFenRow(values) {
  let row = '';
  let emptyCount = 0;
  for (const value of values) {
    if (!value) {
      emptyCount += 1;
      continue;
    }
    if (emptyCount) {
      row += String(emptyCount);
      emptyCount = 0;
    }
    row += value;
  }
  if (emptyCount) {
    row += String(emptyCount);
  }
  return row;
}

function serializePlayStrategyFen(positionState) {
  const rows = [];
  for (let rowIndex = 0; rowIndex < 9; rowIndex += 1) {
    const rowCells = CELLS.filter((cell) => cell.rowIndex === rowIndex);
    rows.push(
      compressFenRow(
        rowCells.map((cell) => {
          const occupant = occupantAt(positionState, cell.coord);
          if (occupant === 'black') {
            return 'S';
          }
          if (occupant === 'white') {
            return 's';
          }
          return '';
        }),
      ),
    );
  }

  const whiteEjected = Math.max(0, MAX_MARBLES_PER_SIDE - positionState.white.size);
  const blackEjected = Math.max(0, MAX_MARBLES_PER_SIDE - positionState.black.size);
  const side = positionState.sideToMove === 'white' ? 'w' : 'b';
  return `${rows.join('/')} 0 0 ${side} ${whiteEjected} ${blackEjected}`;
}

function editedResult(positionState, turnIndex) {
  if (positionState.black.size <= 8) {
    return {
      kind: 'win',
      winner: 'white',
      reason: 'black_marbles_reduced_to_eight',
    };
  }
  if (positionState.white.size <= 8) {
    return {
      kind: 'win',
      winner: 'black',
      reason: 'white_marbles_reduced_to_eight',
    };
  }
  if (turnIndex >= 350) {
    if (positionState.black.size > positionState.white.size) {
      return {
        kind: 'win',
        winner: 'black',
        reason: 'max_turns_material_advantage',
      };
    }
    if (positionState.white.size > positionState.black.size) {
      return {
        kind: 'win',
        winner: 'white',
        reason: 'max_turns_material_advantage',
      };
    }
    return {
      kind: 'draw',
      winner: null,
      reason: 'max_turns_even_material',
    };
  }
  return null;
}

function sessionFromPosition(position, turnIndex = 0) {
  const positionState = parsePositionString(position);
  const session = {
    position,
    sideToMove: positionState.sideToMove,
    historyPositions: [],
    noProgressPly: 0,
    turnIndex,
    lastEngineReverseMove: null,
    moveStack: [],
    blackCount: positionState.black.size,
    whiteCount: positionState.white.size,
    lastMove: null,
    result: editedResult(positionState, turnIndex),
  };
  session_status(session);
  return session;
}

function parsePlayStrategyFen(value) {
  const tokens = value.trim().split(/\s+/).filter(Boolean);
  const boardToken = tokens[0];
  if (!boardToken || !boardToken.includes('/')) {
    return null;
  }
  return serializePlayStrategyFen(parsePositionString(value));
}

function parsePastedPosition(value) {
  const fenPosition = parsePlayStrategyFen(value);
  if (fenPosition) {
    return fenPosition;
  }
  throw new Error('Paste PlayStrategy FEN');
}

function refreshPositionInput() {
  if (!state.session) {
    positionInput.value = '';
    return;
  }
  positionInput.value = serializePlayStrategyFen(parsePositionString(state.session.position));
  requestAnimationFrame(() => {
    positionInput.scrollLeft = 0;
  });
}

function engineColor() {
  return state.humanColor === 'black' ? 'white' : 'black';
}

function canHumanAct(status) {
  return (
    !state.editMode &&
    !state.demoMode &&
    !isBlockingSearch() &&
    !status.isGameOver &&
    status.sideToMove === state.humanColor
  );
}

function setSearchLimitEnabled(limit, enabled) {
  if (limit === 'depth') {
    state.depthLimitEnabled = enabled;
  } else {
    state.timeLimitEnabled = enabled;
  }

  if (!state.depthLimitEnabled && !state.timeLimitEnabled) {
    if (limit === 'depth') {
      state.timeLimitEnabled = true;
    } else {
      state.depthLimitEnabled = true;
    }
  }
}

function formatNps(nodes, elapsedMs) {
  const seconds = Math.max(elapsedMs, 1) / 1000;
  const nps = nodes / seconds;
  const knps = nps / 1_000;
  if (knps >= 100) {
    return `${knps.toFixed(0)} kNs`;
  }
  if (knps >= 10) {
    return `${knps.toFixed(1)} kNs`;
  }
  return `${knps.toFixed(2)} kNs`;
}

function formatElapsedTime(elapsedMs) {
  return `${(Math.max(elapsedMs, 1) / 1000).toFixed(2)} s`;
}

function formatElapsedMilliseconds(elapsedMs) {
  return `${Math.round(Math.max(elapsedMs, 1))} ms`;
}

function formatPlyCounter(turnIndex) {
  return `${turnIndex}/350`;
}

function clearSearchInfo() {
  state.lastEngineSearchInfo = null;
}

function clearDebugInfo() {
  state.lastEngineDebugInfo = null;
}

function buildLastEngineSearchInfo(depth, elapsedMs) {
  const depthLimited =
    state.depthLimitEnabled &&
    (!state.timeLimitEnabled || depth >= state.maxDepth);

  if (depthLimited) {
    return `Time ${formatElapsedTime(elapsedMs)}`;
  }

  return `Depth ${depth}`;
}

function setNotice(text, kind = 'status') {
  state.notice = text;
  state.noticeKind = kind;
}

function clearNotice() {
  state.notice = '';
  state.noticeKind = null;
}

function clearTransientNotice() {
  if (state.noticeKind === 'status') {
    clearNotice();
  }
}

function clearSelection() {
  state.selected = [];
  state.candidates = [];
}

function isBlockingSearch() {
  return state.activeRequestId !== null;
}

function effectiveWhiteScore(sessionStatus) {
  if (sessionStatus.result?.kind === 'win') {
    return sessionStatus.result.winner === 'white' ? 4000 : -4000;
  }
  if (sessionStatus.result?.kind === 'draw') {
    return 0;
  }
  return state.evaluationWhiteScore;
}

function whiteExpectedOutcomeProbability(score) {
  const scaled = Math.max(-12, Math.min(12, score / EVAL_EXPECTED_OUTCOME_SCORE_SCALE));
  return 1 / (1 + Math.exp(-scaled));
}

function formatWinProbability(probability) {
  return `${Math.round(probability * 100)}%`;
}

function renderEvaluationBar(sessionStatus) {
  let whiteFraction;
  if (sessionStatus.result?.kind === 'win') {
    whiteFraction = sessionStatus.result.winner === 'white' ? 1 : 0;
  } else if (sessionStatus.result?.kind === 'draw') {
    whiteFraction = 0.5;
  } else {
    whiteFraction = whiteExpectedOutcomeProbability(effectiveWhiteScore(sessionStatus));
  }
  evalFill.style.height = `${whiteFraction * 100}%`;

  if (Math.abs(whiteFraction - 0.5) < 0.005 || sessionStatus.result?.kind === 'draw') {
    evalBar.dataset.leading = 'even';
    evalLabel.textContent = formatWinProbability(0.5);
  } else if (whiteFraction > 0.5) {
    evalBar.dataset.leading = 'white';
    evalLabel.textContent = formatWinProbability(whiteFraction);
  } else {
    evalBar.dataset.leading = 'black';
    evalLabel.textContent = formatWinProbability(1 - whiteFraction);
  }
}

function syncEvaluationBarGeometry() {
  const boardSvg = boardContainer.querySelector('.board-svg');
  if (!boardSvg || !evalBar || !evalRail) {
    return;
  }

  const boardHeight = boardSvg.getBoundingClientRect().height;
  evalRail.style.marginTop = `${boardHeight * evalBarTopRatio}px`;
  evalBar.style.height = `${boardHeight * evalBarHeightRatio}px`;
}

function resultCopy(result) {
  if (!result) {
    return '';
  }
  if (result.kind === 'draw') {
    return 'Draw';
  }
  return result.winner === 'black' ? 'Black wins' : 'White wins';
}

function render() {
  const status = session_status(state.session);
  const searchInFlight = state.activeRequestId !== null;
  const blockingSearch = isBlockingSearch();
  const locked =
    state.editMode ||
    state.demoMode ||
    blockingSearch ||
    status.isGameOver ||
    status.sideToMove !== state.humanColor;
  const showThinking =
    searchInFlight &&
    !status.result &&
    !!state.thinkingSide;

  renderBoard({
    container: boardContainer,
    session: state.session,
    selected: state.selected,
    candidates: state.candidates,
    locked,
    onCellClick: handleCellClick,
    onCandidateClick: handleCandidateClick,
    onBackgroundClick: handleBoardBackgroundClick,
    editMode: state.editMode,
    onEditDrop: handleEditDrop,
  });

  renderEvaluationBar(status);
  syncEvaluationBarGeometry();
  evalRail.classList.toggle('is-hidden', !state.debugEnabled);
  demoButton.textContent = state.demoMode ? 'Pause self-play' : 'Self-play';
  demoButton.setAttribute('aria-pressed', state.demoMode ? 'true' : 'false');
  demoButton.disabled = state.editMode;
  toggleSideButton.textContent = state.humanColor === 'black' ? 'Play white' : 'Play black';
  toggleSideButton.disabled = state.editMode || state.demoMode;
  takeBackButton.disabled = state.editMode || state.demoMode || !status.canTakeBack;
  resetButton.disabled = state.demoMode || state.session.position === initialSessionPosition;
  editPositionButton.textContent = state.editMode ? 'Done' : 'Edit';
  editPositionButton.setAttribute('aria-pressed', state.editMode ? 'true' : 'false');
  editPositionButton.disabled = false;
  positionEditor.hidden = !state.editMode;
  depthLimitToggle.setAttribute('aria-pressed', state.depthLimitEnabled ? 'true' : 'false');
  timeLimitToggle.setAttribute('aria-pressed', state.timeLimitEnabled ? 'true' : 'false');
  depthLimitToggle.disabled = state.demoMode;
  timeLimitToggle.disabled = state.demoMode;
  maxDepthInput.disabled = state.demoMode;
  maxTimeInput.disabled = state.demoMode;
  debugToggle.setAttribute('aria-pressed', state.debugEnabled ? 'true' : 'false');
  maxDepthInput.classList.toggle('is-inactive', !state.depthLimitEnabled);
  maxTimeInput.classList.toggle('is-inactive', !state.timeLimitEnabled);
  syncSearchInputs();

  if (showThinking) {
    thinkingDots.dataset.side = state.thinkingSide;
  } else {
    delete thinkingDots.dataset.side;
  }
  thinkingDots.hidden = !showThinking;

  const noticeValue = state.notice || (status.result ? resultCopy(status.result) : '');
  const showDebugInfo = state.debugEnabled;
  const debugInfo = state.lastEngineDebugInfo;
  const showSearchInfo = false;
  debugPanel.hidden = !showDebugInfo;
  if (showDebugInfo) {
    debugSpeed.textContent = debugInfo?.speed ?? '';
    debugTime.textContent = debugInfo?.time ?? '';
    debugNodes.textContent = debugInfo?.nodes ?? '';
    debugDepth.textContent = debugInfo?.depth ?? '';
    debugPly.textContent = debugInfo?.ply ?? '';
  } else {
    debugSpeed.textContent = '';
    debugTime.textContent = '';
    debugNodes.textContent = '';
    debugDepth.textContent = '';
    debugPly.textContent = '';
  }

  searchInfoText.hidden = !showSearchInfo;
  searchInfoText.textContent = '';

  noticeText.hidden = !noticeValue;
  noticeText.textContent = noticeValue;
}

function refreshCandidates() {
  if (!state.selected.length) {
    state.candidates = [];
    return;
  }
  state.candidates = legal_moves_for_selection(state.session, state.selected);
}

function setEditedPosition(positionState) {
  const position = serializePlayStrategyFen(positionState);
  state.session = sessionFromPosition(position, state.session.turnIndex);
  clearSelection();
  clearSearchInfo();
  clearDebugInfo();
  state.evaluationWhiteScore = 0;
  clearNotice();
}

function handleEditDrop(payload, targetCoord) {
  if (!state.editMode) {
    return;
  }

  const positionState = parsePositionString(state.session.position);
  const sourceSet = payload.color === 'black' ? positionState.black : positionState.white;

  if (payload.source === 'board') {
    if (payload.coord === targetCoord) {
      return;
    }
    sourceSet.delete(payload.coord);
  } else if (!targetCoord) {
    return;
  }

  if (targetCoord) {
    positionState.black.delete(targetCoord);
    positionState.white.delete(targetCoord);
    sourceSet.add(targetCoord);
  }

  setEditedPosition(positionState);
  refreshPositionInput();
  render();
}

function applyPositionText() {
  const value = positionInput.value.trim();
  if (!value) {
    refreshPositionInput();
    return true;
  }

  try {
    const position = parsePastedPosition(value);
    state.session = sessionFromPosition(position, state.session.turnIndex);
    clearSelection();
    clearSearchInfo();
    clearDebugInfo();
    state.evaluationWhiteScore = 0;
    clearNotice();
    refreshPositionInput();
    render();
    return true;
  } catch (error) {
    setNotice(error.message || String(error), 'error');
    render();
    return false;
  }
}

function handleBoardBackgroundClick() {
  const status = session_status(state.session);
  if (!canHumanAct(status)) {
    return;
  }
  clearTransientNotice();
  clearSelection();
  render();
}

function handleCellClick(coord) {
  const status = session_status(state.session);
  if (!canHumanAct(status)) {
    return;
  }
  clearTransientNotice();

  const positionState = parsePositionString(state.session.position);
  const occupant = occupantAt(positionState, coord);
  if (occupant !== state.humanColor) {
    clearSelection();
    render();
    return;
  }

  if (state.selected.length === 1 && state.selected[0] === coord) {
    clearSelection();
    render();
    return;
  }
  state.selected = [coord];
  refreshCandidates();
  render();
}

async function handleCandidateClick(candidate) {
  await applyHumanMove(candidate.move);
}

async function applyHumanMove(moveText) {
  const status = session_status(state.session);
  if (!canHumanAct(status)) {
    return;
  }
  clearTransientNotice();

  try {
    clearSearchInfo();
    state.session = apply_move(state.session, moveText);
    clearSelection();
    clearNotice();
    render();

    const nextStatus = session_status(state.session);
    if (!nextStatus.isGameOver) {
      await ensureEngineTurnIfNeeded();
    }
  } catch (error) {
    setNotice(error.message || String(error), 'error');
    render();
  }
}

function handleWorkerMessage(event) {
  const message = event.data;
  if (message.type === 'ready') {
    if (state.workerReady?.resolve) {
      state.workerReady.resolve();
      state.workerReady = { promise: state.workerReady.promise, resolve: null };
    }
    return;
  }

  if (message.type === 'result') {
    if (message.requestId !== state.activeRequestId) {
      return;
    }
    state.activeRequestId = null;
    state.activeSearchKind = null;
    state.thinking = false;
    state.thinkingSide = null;
    const elapsedMs = message.elapsedMs ?? 0;
    state.lastEngineSearchInfo = buildLastEngineSearchInfo(
      message.result.depth,
      elapsedMs,
    );
    state.evaluationWhiteScore = message.result.whitePerspectiveScore;
    state.session = apply_move(state.session, message.result.bestMove);
    const statusAfterMove = session_status(state.session);
    state.lastEngineDebugInfo = {
      depth: `${message.result.depth} ply`,
      speed: formatNps(message.result.nodes, elapsedMs),
      time: formatElapsedMilliseconds(elapsedMs),
      nodes: message.result.nodes.toLocaleString(),
      ply: formatPlyCounter(statusAfterMove.turnIndex),
    };
    clearNotice();
    render();
    void maybeStartDemoTurn();
    return;
  }

  if (message.type === 'error') {
    if (message.requestId !== state.activeRequestId) {
      return;
    }
    state.activeRequestId = null;
    state.activeSearchKind = null;
    state.thinking = false;
    state.thinkingSide = null;
    state.demoMode = false;
    clearSearchInfo();
    setNotice(message.error || 'Search failed', 'error');
    render();
  }
}

function createSearchWorker() {
  if (state.worker) {
    state.worker.terminate();
  }

  const worker = new Worker(workerModulePath, { type: 'module' });
  worker.addEventListener('message', handleWorkerMessage);
  worker.addEventListener('error', (event) => {
    setNotice(event.message || 'Worker error', 'error');
    state.thinking = false;
    state.thinkingSide = null;
    state.activeRequestId = null;
    state.activeSearchKind = null;
    state.demoMode = false;
    clearSearchInfo();
    render();
  });

  let resolveReady;
  const promise = new Promise((resolve) => {
    resolveReady = resolve;
  });

  state.worker = worker;
  state.workerReady = { promise, resolve: resolveReady };
  worker.postMessage({ type: 'init' });
  return promise;
}

async function ensureWorkerReady() {
  if (!state.worker) {
    await createSearchWorker();
  }
  return state.workerReady?.promise;
}

async function cancelSearch() {
  if (!state.worker) {
    return;
  }
  state.activeRequestId = null;
  state.activeSearchKind = null;
  state.thinking = false;
  state.thinkingSide = null;
  await createSearchWorker();
}

async function beginEngineTurn() {
  const status = session_status(state.session);
  if (status.isGameOver) {
    return;
  }
  const maxDepth = state.depthLimitEnabled ? state.maxDepth : 0;
  const maxTimeMs = state.timeLimitEnabled ? Math.round(state.maxTimeSeconds * 1000) : 0;

  const requestId = state.nextRequestId;
  state.nextRequestId += 1;
  state.activeRequestId = requestId;
  state.activeSearchKind = 'move';
  state.thinking = true;
  state.thinkingSide = status.sideToMove;
  clearNotice();
  render();
  await ensureWorkerReady();
  if (state.activeRequestId !== requestId || state.activeSearchKind !== 'move') {
    return;
  }
  state.worker.postMessage({
    type: 'search',
    kind: 'move',
    requestId,
    maxDepth,
    maxTimeMs,
    session: state.session,
  });
}

async function ensureEngineTurnIfNeeded() {
  const status = session_status(state.session);
  if (state.demoMode || state.thinking || status.isGameOver || status.sideToMove === state.humanColor) {
    return;
  }
  await beginEngineTurn();
}

async function maybeStartDemoTurn() {
  const status = session_status(state.session);
  if (!state.demoMode || state.thinking || status.isGameOver) {
    return;
  }
  await beginEngineTurn();
}

async function resetBoardState() {
  state.session = new_session();
  clearSelection();
  clearSearchInfo();
  clearDebugInfo();
  state.evaluationWhiteScore = 0;
  clearNotice();
  render();
  await ensureEngineTurnIfNeeded();
}

editPositionButton.addEventListener('click', async () => {
  clearTransientNotice();
  if (!state.editMode) {
    state.demoMode = false;
    if (state.activeRequestId !== null) {
      await cancelSearch();
    }
    state.editMode = true;
    clearSelection();
    clearSearchInfo();
    refreshPositionInput();
    render();
    positionInput.focus();
    positionInput.select();
    return;
  }

  if (!applyPositionText()) {
    return;
  }
  state.editMode = false;
  clearSelection();
  render();
  await ensureEngineTurnIfNeeded();
});

demoButton.addEventListener('click', async () => {
  clearTransientNotice();
  if (state.demoMode) {
    state.demoMode = false;
    if (state.activeRequestId !== null) {
      await cancelSearch();
    }
    clearSearchInfo();
    render();
    return;
  }

  state.demoMode = true;
  clearSelection();
  render();
  await maybeStartDemoTurn();
});

debugToggle.addEventListener('click', () => {
  state.debugEnabled = !state.debugEnabled;
  render();
});

toggleSideButton.addEventListener('click', async () => {
  clearTransientNotice();
  state.demoMode = false;
  if (state.activeRequestId !== null) {
    await cancelSearch();
  }
  state.humanColor = engineColor();
  await resetBoardState();
});

depthLimitToggle.addEventListener('click', async () => {
  clearTransientNotice();
  if (isBlockingSearch()) {
    return;
  }
  setSearchLimitEnabled('depth', !state.depthLimitEnabled);
  render();
});

timeLimitToggle.addEventListener('click', async () => {
  clearTransientNotice();
  if (isBlockingSearch()) {
    return;
  }
  setSearchLimitEnabled('time', !state.timeLimitEnabled);
  render();
});

maxDepthInput.addEventListener('change', async () => {
  clearTransientNotice();
  state.maxDepth = clampInteger(
    maxDepthInput.value,
    DEFAULT_MAX_DEPTH,
    MIN_MAX_DEPTH,
    MAX_MAX_DEPTH,
  );
  syncSearchInputs();
  render();
});

maxTimeInput.addEventListener('change', async () => {
  clearTransientNotice();
  state.maxTimeSeconds = clampSteppedNumber(
    maxTimeInput.value,
    DEFAULT_MAX_TIME_SECONDS,
    MIN_MAX_TIME_SECONDS,
    MAX_MAX_TIME_SECONDS,
    MAX_TIME_STEP_SECONDS,
  );
  syncSearchInputs();
  render();
});

takeBackButton.addEventListener('click', async () => {
  clearTransientNotice();
  try {
    if (state.activeRequestId !== null) {
      await cancelSearch();
    }
    state.session = undo_full_turn(state.session);
    clearSelection();
    clearSearchInfo();
    clearDebugInfo();
    state.evaluationWhiteScore = 0;
    clearNotice();
    render();
    await ensureEngineTurnIfNeeded();
    await maybeStartDemoTurn();
  } catch (error) {
    setNotice(error.message || String(error), 'error');
    render();
  }
});

resetButton.addEventListener('click', async () => {
  clearTransientNotice();
  state.demoMode = false;
  state.editMode = false;
  if (state.activeRequestId !== null) {
    await cancelSearch();
  }
  await resetBoardState();
});

positionInput.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') {
    return;
  }
  event.preventDefault();
  applyPositionText();
});

positionInput.addEventListener('change', applyPositionText);

await initEngine({ module_or_path: wasmModulePath });
state.session = new_session();
initialSessionPosition = state.session.position;
await createSearchWorker();
render();

window.addEventListener('resize', syncEvaluationBarGeometry);
