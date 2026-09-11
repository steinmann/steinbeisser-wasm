import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { CELLS, parsePositionString } from '../web/board.js';

const bundleDirectory = process.env.STEINBEISSER_WASM_DIR;

test('built WASM supports frontend session, search and compact/legacy undo contracts', {
  skip: !bundleDirectory && 'Set STEINBEISSER_WASM_DIR to the generated pkg directory',
}, async () => {
  const directory = resolve(bundleDirectory);
  const engine = await import(pathToFileURL(resolve(directory, 'steinbeisser.js')));
  await engine.default({ module_or_path: await readFile(resolve(directory, 'steinbeisser_bg.wasm')) });
  assert.equal(engine.max_marbles_per_side(), 14);
  const initial = engine.new_session();
  assert.equal(engine.session_status(initial).sideToMove, 'black');
  assert.equal(parsePositionString(initial.position).black.size, initial.blackCount);
  assert.deepEqual(engine.session_from_position(initial.position, 0), initial);

  const blackCells = parsePositionString(initial.position).black;
  const humanMove = CELLS.filter(({ coord }) => blackCells.has(coord))
    .flatMap(({ coord }) => engine.legal_moves_for_selection(initial, [coord]))[0];
  assert.ok(humanMove);
  const afterHuman = engine.apply_move(initial, humanMove.move);
  assert.equal(engine.session_status(afterHuman).canTakeBack, true);
  const search = engine.search_best_move_with_limits(structuredClone(afterHuman), 1, 100);
  assert.equal(typeof search.bestMove, 'string');
  assert.equal(search.depth, 1);
  const afterEngine = engine.apply_move(afterHuman, search.bestMove);
  assert.equal(afterEngine.turnIndex, 2);
  assert.equal(afterEngine.moveStack.length, 2);
  for (const entry of afterEngine.moveStack) {
    assert.equal(typeof entry.historyLen, 'number');
    assert.equal(Object.hasOwn(entry, 'historyPositions'), false);
  }
  assert.deepEqual(engine.undo_full_turn(structuredClone(afterEngine)), initial);
  assert.deepEqual(engine.undo_full_turn(afterHuman), initial);

  const legacy = structuredClone(afterEngine);
  for (const entry of legacy.moveStack) {
    entry.historyPositions = legacy.historyPositions.slice(0, entry.historyLen);
    delete entry.historyLen;
  }
  assert.deepEqual(engine.undo_full_turn(legacy), initial);
  assert.equal(engine.session_from_position(initial.position, 350).result.kind, 'draw');
  assert.throws(() => engine.session_from_position('999999999/6/7/8/9/8/7/6/5 0 0 b 0 0', 0));
});
