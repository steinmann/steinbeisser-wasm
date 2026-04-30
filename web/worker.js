const buildId = new URL(self.location.href).search;
const wasmModulePath = `./pkg/steinbeisser_bg.wasm${buildId}`;

let ready = false;
let cancelledRequestId = null;
let engineModulePromise = null;
let searchBestMove = null;
let searchBestMoveWithLimits = null;
let initEngine = null;

async function ensureEngineReady() {
  if (!engineModulePromise) {
    engineModulePromise = import(`./pkg/steinbeisser.js${buildId}`);
  }

  if (ready) {
    return;
  }

  const engineModule = await engineModulePromise;
  initEngine ??= engineModule.default;
  searchBestMove ??= engineModule.search_best_move;
  searchBestMoveWithLimits ??= engineModule.search_best_move_with_limits;
  await initEngine({ module_or_path: wasmModulePath });
  ready = true;
}

self.addEventListener('message', async (event) => {
  const message = event.data;

  if (message.type === 'init') {
    await ensureEngineReady();
    self.postMessage({ type: 'ready' });
    return;
  }

  if (message.type === 'cancel') {
    cancelledRequestId = message.requestId;
    return;
  }

  if (message.type !== 'search') {
    return;
  }

  try {
    await ensureEngineReady();

    const startedAt = performance.now();
    const result = searchBestMoveWithLimits
      ? searchBestMoveWithLimits(message.session, message.maxDepth, message.maxTimeMs)
      : searchBestMove(message.session, message.maxDepth);
    const elapsedMs = performance.now() - startedAt;
    if (cancelledRequestId === message.requestId) {
      cancelledRequestId = null;
      return;
    }

    self.postMessage({
      type: 'result',
      kind: message.kind ?? 'move',
      requestId: message.requestId,
      elapsedMs,
      result,
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      kind: message.kind ?? 'move',
      requestId: message.requestId,
      error: error?.message || String(error),
    });
  }
});
