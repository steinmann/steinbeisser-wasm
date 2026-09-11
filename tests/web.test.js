import test from 'node:test';
import assert from 'node:assert/strict';
import { CELLS, parsePositionString, renderBoard } from '../web/board.js';
import { SearchWorkerClient } from '../web/worker-client.js';

class FakeWorker extends EventTarget {
  sent = [];
  postMessage(message) { this.sent.push(message); }
  terminate() { this.terminated = true; }
  message(data) { this.dispatchEvent(new MessageEvent('message', { data })); }
}
const client = (messages = [], failures = []) => new SearchWorkerClient(
  'worker.js', (e) => messages.push(e.data), (e) => failures.push(e.message), FakeWorker,
);

test('FEN checks row bounds before expanding numeric runs', () => {
  for (const row of ['999999999', '9'.repeat(400), '0', '6', 'SSSSSS']) {
    assert.throws(() => parsePositionString(`${row}/6/7/8/9/8/7/6/5 0 0 b 0 0`));
  }
  const p = parsePositionString('5/6/7/8/9/8/7/6/5 0 0 w 0 0');
  assert.equal(p.black.size, 0);
  assert.equal(p.sideToMove, 'white');
});
test('worker readiness and messages', async () => {
  const messages = []; const c = client(messages);
  assert.deepEqual(c.worker.sent, [{ type: 'init' }]);
  c.worker.message({ type: 'ready' });
  await c.ready;
  c.postMessage({ type: 'search', requestId: 1 });
  c.worker.message({ type: 'result', requestId: 1 });
  assert.equal(messages.length, 1);
});
test('initialization failure rejects readiness', async () => {
  const failures = []; const c = client([], failures);
  c.worker.message({ type: 'init-error', error: 'bad wasm' });
  await assert.rejects(c.ready, /bad wasm/);
  assert.equal(c.status, 'failed');
  assert.equal(c.worker.terminated, true);
  assert.deepEqual(failures, ['bad wasm']);
});
test('replacement rejects old readiness and ignores old results', async () => {
  const messages = []; const c = client(messages);
  c.terminate();
  await assert.rejects(c.ready, /disposed/);
  c.worker.message({ type: 'ready' });
  c.worker.message({ type: 'result' });
  assert.equal(c.status, 'disposed');
  assert.deepEqual(messages, []);
});
test('native worker failure settles initialization', async () => {
  const c = client();
  const e = new Event('error'); e.message = 'import failed';
  c.worker.dispatchEvent(e);
  await assert.rejects(c.ready, /import failed/);
  assert.throws(() => c.postMessage({ type: 'search' }), /failed/);
});

test('synchronous worker construction or initial post failure settles readiness', async () => {
  for (const WorkerType of [
    class { constructor() { throw new Error('construction failed'); } },
    class extends FakeWorker { postMessage() { throw new Error('posting failed'); } },
  ]) {
    const failures = [];
    const c = new SearchWorkerClient('worker.js', () => {}, (e) => failures.push(e.message), WorkerType);
    await assert.rejects(c.ready, /failed/);
    assert.equal(c.status, 'failed');
    assert.equal(failures.length, 1);
    c.terminate();
    c.terminate();
  }
});

test('decode failure after readiness reports once and ignores late events', async () => {
  const messages = []; const failures = []; const c = client(messages, failures);
  assert.throws(() => c.postMessage({ type: 'search' }), /initializing/);
  c.worker.message({ type: 'ready' });
  await c.ready;
  c.worker.dispatchEvent(new Event('messageerror'));
  c.worker.dispatchEvent(new Event('error'));
  c.worker.message({ type: 'result', requestId: 1 });
  assert.deepEqual(failures, ['Could not decode worker response']);
  assert.deepEqual(messages, []);
  assert.equal(c.worker.terminated, true);
});

// Minimal SVG surface: exercise real board event handlers without a DOM dependency.
class SvgElement extends EventTarget {
  children = [];
  attributes = {};
  style = {};
  captures = new Set();
  setAttribute(name, value) { this.attributes[name] = value; }
  append(...nodes) {
    for (const node of nodes) { node.parent = this; this.children.push(node); }
  }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  setPointerCapture(id) { this.captures.add(id); }
  hasPointerCapture(id) { return this.captures.has(id); }
  releasePointerCapture(id) { this.captures.delete(id); }
  getScreenCTM() { return { inverse() { return {}; } }; }
  createSVGPoint() { return { matrixTransform() { return { x: this.x, y: this.y }; } }; }
}

function pointer(type, cell = CELLS[1], pointerId = 1) {
  return Object.assign(new Event(type, { cancelable: true }), {
    button: 0, pointerId, clientX: cell.x, clientY: cell.y,
  });
}

test('cancelled or lost edit pointer never commits a drop and clears its ghost', (t) => {
  const previousDocument = globalThis.document;
  globalThis.document = { createElementNS: () => new SvgElement() };
  t.after(() => {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  });
  for (const ending of ['pointercancel', 'lostpointercapture']) {
    const container = new SvgElement(); const drops = [];
    renderBoard({ container, session: {
      position: 'S4/6/7/8/9/8/7/6/5 0 0 b 0 0', blackCount: 1, whiteCount: 0,
    }, selected: [], candidates: [], editMode: true, locked: true, maxMarblesPerSide: 14,
    onEditDrop: (...args) => drops.push(args) });
    const root = container.children[0];
    const marble = root.children[4].children[0];
    const ghosts = root.children.at(-1);
    marble.dispatchEvent(pointer('pointerdown', CELLS[0]));
    assert.equal(ghosts.children.length, 1);
    assert.equal(root.hasPointerCapture(1), true);
    // A second touch cannot replace the first gesture or cancel it.
    marble.dispatchEvent(pointer('pointerdown', CELLS[0], 2));
    root.dispatchEvent(pointer(ending, CELLS[1], 2));
    assert.equal(ghosts.children.length, 1);
    root.dispatchEvent(pointer('pointermove'));
    root.dispatchEvent(pointer(ending));
    root.dispatchEvent(pointer('pointerup'));
    assert.deepEqual(drops, []);
    assert.equal(ghosts.children.length, 0);
    assert.equal(root.hasPointerCapture(1), false);
    // Cancellation leaves the board ready for the next ordinary drop.
    marble.dispatchEvent(pointer('pointerdown', CELLS[0], 3));
    root.dispatchEvent(pointer('pointerup', CELLS[1], 3));
    assert.deepEqual(drops, [[{ source: 'board', color: 'black', coord: CELLS[0].coord }, CELLS[1].coord]]);
    assert.equal(ghosts.children.length, 0);
  }
});
