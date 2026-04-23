const SVG_NS = 'http://www.w3.org/2000/svg';

export const VIEWBOX = { minX: 0, minY: -120, width: 1884, height: 1680 };
export const SLOT_RADIUS = 74;
const BLACK_PIECE_OUTER_RADIUS = 65;
const WHITE_PIECE_OUTER_RADIUS = 64.75;
const BLACK_PIECE_INNER_RADIUS = 60.5;
const WHITE_PIECE_INNER_RADIUS = 60.25;
const BLACK_CAPTURE_OUTER_RADIUS = BLACK_PIECE_OUTER_RADIUS;
const WHITE_CAPTURE_OUTER_RADIUS = WHITE_PIECE_OUTER_RADIUS;
const BLACK_CAPTURE_INNER_RADIUS = BLACK_PIECE_INNER_RADIUS;
const WHITE_CAPTURE_INNER_RADIUS = WHITE_PIECE_INNER_RADIUS;
export const FRAME_POINTS = [
  [533, 60],
  [1314, 60],
  [1700, 734],
  [1319, 1400],
  [528, 1400],
  [160, 734],
];

export const COLORS = {
  pageBackground: '#1A1D23',
  frameStroke: '#505B73',
  slotFill: '#333B4A',
  slotStroke: '#4E5971',
  blackOuter: '#1E2430',
  blackInner: '#0B0F17',
  whiteOuter: '#4E5971',
  whiteInner: '#E6EBF2',
  accent: '#FF8A47',
};

const ROWS = 'ABCDEFGHI';
const ROW_LENGTHS = [5, 6, 7, 8, 9, 8, 7, 6, 5];
const ROW_START_X = [616, 537.5, 459, 380.5, 302, 380.5, 459, 537.5, 616];
const ROW_Y = [191, 326, 461, 596, 731, 866, 1001, 1136, 1271];
const COLUMN_STEP = 157;
const CAPTURE_XS = [537.5, 694.5, 851.5, 1008.5, 1165.5, 1322.5];
const BLACK_CAPTURE_Y = -20;
const WHITE_CAPTURE_Y = 1490;
const MAX_MARBLES_PER_SIDE = 14;

const DIRECTION_AXIAL = {
  E: [1, 0],
  SE: [0, 1],
  SW: [-1, 1],
  W: [-1, 0],
  NW: [0, -1],
  NE: [1, -1],
};

const DIRECTION_SCREEN = {
  E: [157, 0],
  SE: [78.5, 135],
  SW: [-78.5, 135],
  W: [-157, 0],
  NW: [-78.5, -135],
  NE: [78.5, -135],
};

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function buildCells() {
  const cells = [];
  for (let rowIndex = 0; rowIndex < ROWS.length; rowIndex += 1) {
    const rowLength = ROW_LENGTHS[rowIndex];
    const startX = ROW_START_X[rowIndex];
    const y = ROW_Y[rowIndex];
    const r = rowIndex - 4;
    const qMin = Math.max(-4, -r - 4);

    for (let columnIndex = 0; columnIndex < rowLength; columnIndex += 1) {
      const q = qMin + columnIndex;
      const coord = `${ROWS[rowIndex]}${columnIndex + 1}`;
      cells.push({
        coord,
        rowIndex,
        columnIndex: columnIndex + 1,
        x: startX + (COLUMN_STEP * columnIndex),
        y,
        axial: {
          q,
          r,
          s: -q - r,
        },
      });
    }
  }
  return cells;
}

export const CELLS = buildCells();
export const CELL_MAP = new Map(CELLS.map((cell) => [cell.coord, cell]));
const AXIAL_KEY_TO_COORD = new Map(
  CELLS.map((cell) => [`${cell.axial.q},${cell.axial.r}`, cell.coord]),
);

function coordList(rawValue) {
  if (!rawValue || rawValue === '-') {
    return [];
  }
  return rawValue.split(',');
}

export function parsePositionString(position) {
  const parts = position.trim().split(';');
  const sideToMove = parts[1]?.slice(4) === 'w' ? 'white' : 'black';
  const black = new Set(coordList(parts[2]?.slice(6)));
  const white = new Set(coordList(parts[3]?.slice(6)));
  return { sideToMove, black, white };
}

export function occupantAt(positionState, coord) {
  if (positionState.black.has(coord)) {
    return 'black';
  }
  if (positionState.white.has(coord)) {
    return 'white';
  }
  return null;
}

function lineSignature(cell, axis) {
  if (axis === 'q') {
    return cell.axial.q;
  }
  if (axis === 'r') {
    return cell.axial.r;
  }
  return cell.axial.s;
}

function lineOrder(cell, axis) {
  if (axis === 'q') {
    return cell.axial.r;
  }
  if (axis === 'r') {
    return cell.axial.q;
  }
  return cell.axial.q;
}

export function isContiguousSelection(selection) {
  if (selection.length <= 1) {
    return true;
  }

  const cells = selection.map((coord) => CELL_MAP.get(coord)).filter(Boolean);
  if (cells.length !== selection.length) {
    return false;
  }

  const axes = ['q', 'r', 's'].filter((axis) =>
    cells.every((cell) => lineSignature(cell, axis) === lineSignature(cells[0], axis)),
  );
  if (!axes.length) {
    return false;
  }

  return axes.some((axis) => {
    const ordered = [...cells].sort((left, right) => lineOrder(left, axis) - lineOrder(right, axis));
    for (let index = 1; index < ordered.length; index += 1) {
      if (lineOrder(ordered[index], axis) !== lineOrder(ordered[index - 1], axis) + 1) {
        return false;
      }
    }
    return true;
  });
}

export function translateCoord(coord, direction) {
  const cell = CELL_MAP.get(coord);
  const delta = DIRECTION_AXIAL[direction];
  if (!cell || !delta) {
    return null;
  }
  const key = `${cell.axial.q + delta[0]},${cell.axial.r + delta[1]}`;
  return AXIAL_KEY_TO_COORD.get(key) ?? null;
}

function normalizedDirectionVector(direction) {
  const [dx, dy] = DIRECTION_SCREEN[direction] ?? [0, 0];
  const length = Math.hypot(dx, dy) || 1;
  return { x: dx / length, y: dy / length };
}

function hexDistance(left, right) {
  return Math.max(
    Math.abs(left.axial.q - right.axial.q),
    Math.abs(left.axial.r - right.axial.r),
    Math.abs(left.axial.s - right.axial.s),
  );
}

function broadsideDestinationPoint(candidate, selectedCoord) {
  const selectedCell = selectedCoord ? CELL_MAP.get(selectedCoord) : null;
  if (!selectedCell) {
    return null;
  }

  const primarySource = candidate.sourceCells
    .map((coord) => CELL_MAP.get(coord))
    .filter(Boolean)
    .sort((left, right) => {
      const distanceDiff = hexDistance(right, selectedCell) - hexDistance(left, selectedCell);
      if (distanceDiff !== 0) {
        return distanceDiff;
      }
      return right.coord.localeCompare(left.coord);
    })[0];

  if (!primarySource) {
    return null;
  }

  const destinationCoord = translateCoord(primarySource.coord, candidate.direction);
  if (!destinationCoord) {
    return null;
  }

  const destinationCell = CELL_MAP.get(destinationCoord);
  return destinationCell ? { x: destinationCell.x, y: destinationCell.y } : null;
}

export function candidateDotPoint(candidate, selected = []) {
  if (candidate.isEjection) {
    const anchor = CELL_MAP.get(candidate.anchorCell);
    if (!anchor) {
      return { x: 0, y: 0 };
    }
    const unit = normalizedDirectionVector(candidate.direction);
    return {
      x: anchor.x + (unit.x * SLOT_RADIUS),
      y: anchor.y + (unit.y * SLOT_RADIUS),
    };
  }

  const selectedCoord = selected.length === 1 ? selected[0] : null;
  if (selectedCoord && candidate.sourceCells.length === 1) {
    const selectedDestination = translateCoord(selectedCoord, candidate.direction);
    if (selectedDestination) {
      const destinationCell = CELL_MAP.get(selectedDestination);
      if (destinationCell) {
        return { x: destinationCell.x, y: destinationCell.y };
      }
    }
  }

  if (candidate.isBroadside) {
    const broadsidePoint = broadsideDestinationPoint(candidate, selectedCoord);
    if (broadsidePoint) {
      return broadsidePoint;
    }
  }

  const anchor = CELL_MAP.get(candidate.anchorCell);
  return anchor ? { x: anchor.x, y: anchor.y } : { x: 0, y: 0 };
}

function pointKey(point) {
  return `${Math.round(point.x)},${Math.round(point.y)}`;
}

function preferredCandidate(current, next) {
  if (!current) {
    return next;
  }
  const currentIsSingle = current.sourceCells.length === 1;
  const nextIsSingle = next.sourceCells.length === 1;
  if (currentIsSingle !== nextIsSingle) {
    return nextIsSingle ? next : current;
  }
  if (next.isBroadside !== current.isBroadside) {
    return next.isBroadside ? next : current;
  }
  if (next.sourceCells.length !== current.sourceCells.length) {
    return next.sourceCells.length < current.sourceCells.length ? next : current;
  }
  if (next.isPush !== current.isPush) {
    return next.isPush ? current : next;
  }
  return next.move < current.move ? next : current;
}

function appendSlotLayer(group, positionState) {
  for (const cell of CELLS) {
    if (occupantAt(positionState, cell.coord)) {
      continue;
    }
    const slot = svg('polygon', {
      points: '-64,-37 0,-74 64,-37 64,37 0,74 -64,37',
      transform: `translate(${cell.x} ${cell.y})`,
      fill: COLORS.slotFill,
      stroke: COLORS.slotStroke,
      'stroke-width': 4,
      'stroke-linejoin': 'round',
    });
    group.append(slot);
  }
}

function appendPieces(group, positionState) {
  for (const cell of CELLS) {
    const occupant = occupantAt(positionState, cell.coord);
    if (!occupant) {
      continue;
    }
    const outerFill = occupant === 'black' ? COLORS.blackOuter : COLORS.whiteOuter;
    const innerFill = occupant === 'black' ? COLORS.blackInner : COLORS.whiteInner;
    const outerRadius =
      occupant === 'black' ? BLACK_PIECE_OUTER_RADIUS : WHITE_PIECE_OUTER_RADIUS;
    const innerRadius =
      occupant === 'black' ? BLACK_PIECE_INNER_RADIUS : WHITE_PIECE_INNER_RADIUS;

    const marbleGroup = svg('g');
    marbleGroup.append(
      svg('circle', {
        cx: cell.x,
        cy: cell.y,
        r: outerRadius,
        fill: outerFill,
      }),
    );
    marbleGroup.append(
      svg('circle', {
        cx: cell.x,
        cy: cell.y,
        r: innerRadius,
        fill: innerFill,
      }),
    );
    group.append(marbleGroup);
  }
}

function appendCapturedPieces(group, session) {
  const missingBlack = Math.max(0, MAX_MARBLES_PER_SIDE - (session.blackCount ?? MAX_MARBLES_PER_SIDE));
  const missingWhite = Math.max(0, MAX_MARBLES_PER_SIDE - (session.whiteCount ?? MAX_MARBLES_PER_SIDE));

  const whiteCaptureXs = CAPTURE_XS.slice(0, missingWhite);
  const blackCaptureXs = CAPTURE_XS.slice(CAPTURE_XS.length - missingBlack);

  for (const x of whiteCaptureXs) {
    const marbleGroup = svg('g', { 'pointer-events': 'none' });
    marbleGroup.append(
      svg('circle', {
        cx: x,
        cy: WHITE_CAPTURE_Y,
        r: WHITE_CAPTURE_OUTER_RADIUS,
        fill: COLORS.whiteOuter,
      }),
    );
    marbleGroup.append(
      svg('circle', {
        cx: x,
        cy: WHITE_CAPTURE_Y,
        r: WHITE_CAPTURE_INNER_RADIUS,
        fill: COLORS.whiteInner,
      }),
    );
    group.append(marbleGroup);
  }

  for (const x of blackCaptureXs) {
    const marbleGroup = svg('g', { 'pointer-events': 'none' });
    marbleGroup.append(
      svg('circle', {
        cx: x,
        cy: BLACK_CAPTURE_Y,
        r: BLACK_CAPTURE_OUTER_RADIUS,
        fill: COLORS.blackOuter,
      }),
    );
    marbleGroup.append(
      svg('circle', {
        cx: x,
        cy: BLACK_CAPTURE_Y,
        r: BLACK_CAPTURE_INNER_RADIUS,
        fill: COLORS.blackInner,
      }),
    );
    group.append(marbleGroup);
  }
}

function appendSelectionMarkers(group, selected) {
  for (const coord of selected) {
    const cell = CELL_MAP.get(coord);
    if (!cell) {
      continue;
    }
    group.append(
      svg('circle', {
        cx: cell.x,
        cy: cell.y,
        r: 23,
        fill: COLORS.accent,
        'pointer-events': 'none',
      }),
    );
  }
}

function appendCandidateDots(group, candidates, selected, onCandidateClick) {
  const uniqueCandidates = new Map();
  for (const candidate of candidates) {
    const point = candidateDotPoint(candidate, selected);
    const key = pointKey(point);
    const existing = uniqueCandidates.get(key);
    uniqueCandidates.set(key, {
      point,
      candidate: preferredCandidate(existing?.candidate, candidate),
    });
  }

  for (const { point, candidate } of uniqueCandidates.values()) {
    const hit = svg('circle', {
      cx: point.x,
      cy: point.y,
      r: 32,
      fill: 'transparent',
      'pointer-events': 'all',
      tabindex: '0',
      role: 'button',
      'aria-label': `Play ${candidate.move}`,
    });
    hit.addEventListener('click', (event) => {
      event.stopPropagation();
      onCandidateClick(candidate);
    });
    hit.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
        onCandidateClick(candidate);
      }
    });

    const dot = svg('circle', {
      cx: point.x,
      cy: point.y,
      r: 18,
      fill: COLORS.accent,
      'pointer-events': 'none',
    });

    group.append(hit, dot);
  }
}

function appendHitTargets(group, onCellClick, locked) {
  for (const cell of CELLS) {
    const hit = svg('circle', {
      cx: cell.x,
      cy: cell.y,
      r: 68,
      fill: 'transparent',
      'pointer-events': locked ? 'none' : 'all',
      tabindex: locked ? '-1' : '0',
      role: 'button',
      'aria-label': cell.coord,
    });
    if (!locked) {
      hit.addEventListener('click', (event) => {
        event.stopPropagation();
        onCellClick(cell.coord);
      });
      hit.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          event.stopPropagation();
          onCellClick(cell.coord);
        }
      });
    }
    group.append(hit);
  }
}

export function renderBoard({
  container,
  session,
  selected,
  candidates,
  onCellClick,
  onCandidateClick,
  onBackgroundClick,
  locked,
}) {
  container.textContent = '';
  const positionState = parsePositionString(session.position);

  const svgRoot = svg('svg', {
    viewBox: `${VIEWBOX.minX} ${VIEWBOX.minY} ${VIEWBOX.width} ${VIEWBOX.height}`,
    class: 'board-svg',
    'aria-label': 'Abalone board',
  });
  svgRoot.addEventListener('click', () => onBackgroundClick());

  svgRoot.append(
    svg('rect', {
      x: VIEWBOX.minX,
      y: VIEWBOX.minY,
      width: VIEWBOX.width,
      height: VIEWBOX.height,
      fill: COLORS.pageBackground,
    }),
  );

  const capturedPieces = svg('g');
  appendCapturedPieces(capturedPieces, session);
  svgRoot.append(capturedPieces);

  svgRoot.append(
    svg('polygon', {
      points: FRAME_POINTS.map(([x, y]) => `${x},${y}`).join(' '),
      fill: 'none',
      stroke: COLORS.frameStroke,
      'stroke-width': 6,
      'stroke-linejoin': 'round',
    }),
  );

  const slots = svg('g');
  appendSlotLayer(slots, positionState);
  svgRoot.append(slots);

  const pieces = svg('g');
  appendPieces(pieces, positionState);
  svgRoot.append(pieces);

  const hits = svg('g');
  appendHitTargets(hits, onCellClick, locked);
  svgRoot.append(hits);

  const overlay = svg('g');
  appendSelectionMarkers(overlay, selected);
  appendCandidateDots(overlay, candidates, selected, onCandidateClick);
  svgRoot.append(overlay);

  container.append(svgRoot);
}
