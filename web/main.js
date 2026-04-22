const buildId = new URL(import.meta.url).search;
const wasmModulePath = `./pkg/steinbeisser_wasm_engine_bg.wasm${buildId}`;
const workerModulePath = `./worker.js${buildId}`;

const engineModule = await import(`./pkg/steinbeisser_wasm_engine.js${buildId}`);
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
const MIN_MAX_TIME_SECONDS = 1;
const MAX_MAX_TIME_SECONDS = 600;
const SPEED_TEST_DEPTH = 17;
const EVAL_EXPECTED_OUTCOME_SCORE_SCALE = 996;

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
  demoMode: false,
  thinking: false,
  thinkingSide: null,
  activeSearchKind: null,
  worker: null,
  workerReady: null,
  nextRequestId: 1,
  activeRequestId: null,
  lastEngineSearchInfo: null,
  speedTestResult: null,
  notice: '',
  noticeKind: null,
};
let initialSessionPosition = null;

const boardContainer = document.querySelector('[data-board]');
const evalRail = document.querySelector('.eval-rail');
const evalBar = document.querySelector('.eval-bar');
const evalFill = document.querySelector('[data-eval-fill]');
const plyCounterValue = document.querySelector('.ply-counter-value');
const speedTestButton = document.querySelector('[data-action="speed-test"]');
const demoButton = document.querySelector('[data-action="demo"]');
const toggleSideButton = document.querySelector('[data-action="toggle-side"]');
const takeBackButton = document.querySelector('[data-action="takeback"]');
const resetButton = document.querySelector('[data-action="reset"]');
const depthLimitToggle = document.querySelector('[data-toggle="depth-limit"]');
const timeLimitToggle = document.querySelector('[data-toggle="time-limit"]');
const maxDepthInput = document.querySelector('[data-setting="max-depth"]');
const maxTimeInput = document.querySelector('[data-setting="max-time-seconds"]');
const thinkingDots = document.querySelector('[data-thinking-dots]');
const searchInfoText = document.querySelector('[data-search-info]');
const speedTestPanel = document.querySelector('[data-speed-test-panel]');
const speedTestSpeed = document.querySelector('[data-speed-test-speed]');
const speedTestTime = document.querySelector('[data-speed-test-time]');
const speedTestNodes = document.querySelector('[data-speed-test-nodes]');
const speedTestDepth = document.querySelector('[data-speed-test-depth]');
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

function syncSearchInputs() {
  maxDepthInput.value = String(state.maxDepth);
  maxTimeInput.value = String(state.maxTimeSeconds);
}

function engineColor() {
  return state.humanColor === 'black' ? 'white' : 'black';
}

function canHumanAct(status) {
  return !state.demoMode && !state.thinking && !status.isGameOver && status.sideToMove === state.humanColor;
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

function clearSearchInfo() {
  state.lastEngineSearchInfo = null;
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
  state.speedTestResult = null;
  state.notice = text;
  state.noticeKind = kind;
}

function clearNotice() {
  state.notice = '';
  state.noticeKind = null;
}

function clearTransientNotice() {
  if (state.noticeKind === 'speed-test' || state.speedTestResult) {
    state.speedTestResult = null;
    clearNotice();
  }
}

function clearSelection() {
  state.selected = [];
  state.candidates = [];
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
  const speedTestInFlight = state.activeSearchKind === 'speed-test';
  const locked = state.demoMode || searchInFlight || status.isGameOver || status.sideToMove !== state.humanColor;
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
  });

  renderEvaluationBar(status);
  syncEvaluationBarGeometry();
  plyCounterValue.textContent = `${status.turnIndex} / 350`;
  speedTestButton.textContent = state.activeSearchKind === 'speed-test' ? 'Testing...' : 'Speed';
  speedTestButton.disabled = searchInFlight || state.demoMode;
  demoButton.textContent = state.demoMode ? 'Stop Demo' : 'Demo';
  demoButton.setAttribute('aria-pressed', state.demoMode ? 'true' : 'false');
  demoButton.disabled = searchInFlight && !state.demoMode;
  toggleSideButton.textContent = state.humanColor === 'black' ? 'Play white' : 'Play black';
  toggleSideButton.disabled = state.demoMode || state.activeSearchKind === 'speed-test';
  takeBackButton.disabled = state.demoMode || speedTestInFlight || !status.canTakeBack;
  resetButton.disabled =
    state.demoMode || speedTestInFlight || state.session.position === initialSessionPosition;
  depthLimitToggle.setAttribute('aria-pressed', state.depthLimitEnabled ? 'true' : 'false');
  timeLimitToggle.setAttribute('aria-pressed', state.timeLimitEnabled ? 'true' : 'false');
  depthLimitToggle.disabled = searchInFlight || state.demoMode;
  timeLimitToggle.disabled = searchInFlight || state.demoMode;
  maxDepthInput.disabled = searchInFlight || state.demoMode || !state.depthLimitEnabled;
  maxTimeInput.disabled = searchInFlight || state.demoMode || !state.timeLimitEnabled;
  syncSearchInputs();

  if (showThinking) {
    thinkingDots.dataset.side = state.thinkingSide;
  } else {
    delete thinkingDots.dataset.side;
  }
  thinkingDots.hidden = !showThinking;

  const noticeValue = state.notice || (status.result ? resultCopy(status.result) : '');
  const showSpeedTestResult = !noticeValue && !showThinking && !!state.speedTestResult;
  const showSearchInfo =
    !noticeValue &&
    !showSpeedTestResult &&
    (!showThinking || state.demoMode) &&
    !!state.lastEngineSearchInfo;
  speedTestPanel.hidden = !showSpeedTestResult;
  if (showSpeedTestResult) {
    speedTestSpeed.textContent = state.speedTestResult.speed;
    speedTestTime.textContent = state.speedTestResult.time;
    speedTestNodes.textContent = state.speedTestResult.nodes;
    speedTestDepth.textContent = state.speedTestResult.depth;
  } else {
    speedTestSpeed.textContent = '';
    speedTestTime.textContent = '';
    speedTestNodes.textContent = '';
    speedTestDepth.textContent = '';
  }

  searchInfoText.hidden = !showSearchInfo;
  searchInfoText.textContent = showSearchInfo ? state.lastEngineSearchInfo : '';

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
  const status = session_status(state.session);
  if (!canHumanAct(status)) {
    return;
  }
  clearTransientNotice();

  try {
    clearSearchInfo();
    state.session = apply_move(state.session, candidate.move);
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
    const requestKind = state.activeSearchKind;
    state.activeRequestId = null;
    state.activeSearchKind = null;
    state.thinking = false;
    state.thinkingSide = null;
    if (requestKind === 'speed-test') {
      state.speedTestResult = {
        speed: formatNps(message.result.nodes, message.elapsedMs ?? 0),
        time: formatElapsedTime(message.elapsedMs ?? 0),
        nodes: message.result.nodes.toLocaleString(),
        depth: String(message.result.depth),
      };
      clearNotice();
      render();
      return;
    }
    state.lastEngineSearchInfo = buildLastEngineSearchInfo(
      message.result.depth,
      message.elapsedMs ?? 0,
    );
    state.evaluationWhiteScore = message.result.whitePerspectiveScore;
    state.session = apply_move(state.session, message.result.bestMove);
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
    setNotice(
      message.kind === 'speed-test'
        ? message.error || 'Speed test failed'
        : message.error || 'Search failed',
      message.kind === 'speed-test' ? 'speed-test' : 'error',
    );
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
  const maxTimeMs = state.timeLimitEnabled ? state.maxTimeSeconds * 1000 : 0;

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

async function runSpeedTest() {
  if (state.thinking || state.demoMode) {
    return;
  }
  clearTransientNotice();

  const status = session_status(state.session);
  if (status.isGameOver) {
    setNotice('Speed test unavailable in a finished game', 'speed-test');
    render();
    return;
  }

  const requestId = state.nextRequestId;
  state.nextRequestId += 1;
  state.activeRequestId = requestId;
  state.activeSearchKind = 'speed-test';
  state.thinking = true;
  state.thinkingSide = status.sideToMove;
  clearNotice();
  render();
  await ensureWorkerReady();
  if (state.activeRequestId !== requestId || state.activeSearchKind !== 'speed-test') {
    return;
  }
  state.worker.postMessage({
    type: 'search',
    kind: 'speed-test',
    requestId,
    maxDepth: SPEED_TEST_DEPTH,
    maxTimeMs: 0,
    session: state.session,
  });
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
  state.evaluationWhiteScore = 0;
  clearNotice();
  render();
  await ensureEngineTurnIfNeeded();
}

demoButton.addEventListener('click', async () => {
  clearTransientNotice();
  if (state.demoMode) {
    state.demoMode = false;
    if (state.thinking) {
      await cancelSearch();
    }
    await resetBoardState();
    return;
  }

  state.demoMode = true;
  clearSelection();
  render();
  await maybeStartDemoTurn();
});

speedTestButton.addEventListener('click', async () => {
  clearTransientNotice();
  await runSpeedTest();
});

toggleSideButton.addEventListener('click', async () => {
  clearTransientNotice();
  state.demoMode = false;
  if (state.thinking) {
    await cancelSearch();
  }
  state.humanColor = engineColor();
  await resetBoardState();
});

depthLimitToggle.addEventListener('click', () => {
  clearTransientNotice();
  if (state.activeRequestId !== null) {
    return;
  }
  setSearchLimitEnabled('depth', !state.depthLimitEnabled);
  render();
});

timeLimitToggle.addEventListener('click', () => {
  clearTransientNotice();
  if (state.activeRequestId !== null) {
    return;
  }
  setSearchLimitEnabled('time', !state.timeLimitEnabled);
  render();
});

maxDepthInput.addEventListener('change', () => {
  clearTransientNotice();
  state.maxDepth = clampInteger(
    maxDepthInput.value,
    DEFAULT_MAX_DEPTH,
    MIN_MAX_DEPTH,
    MAX_MAX_DEPTH,
  );
  syncSearchInputs();
});

maxTimeInput.addEventListener('change', () => {
  clearTransientNotice();
  state.maxTimeSeconds = clampInteger(
    maxTimeInput.value,
    DEFAULT_MAX_TIME_SECONDS,
    MIN_MAX_TIME_SECONDS,
    MAX_MAX_TIME_SECONDS,
  );
  syncSearchInputs();
});

takeBackButton.addEventListener('click', async () => {
  clearTransientNotice();
  try {
    if (state.thinking) {
      await cancelSearch();
    }
    state.session = undo_full_turn(state.session);
    clearSelection();
    clearSearchInfo();
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
  if (state.thinking) {
    await cancelSearch();
  }
  await resetBoardState();
});

await initEngine({ module_or_path: wasmModulePath });
state.session = new_session();
initialSessionPosition = state.session.position;
await createSearchWorker();
render();

window.addEventListener('resize', syncEvaluationBarGeometry);
