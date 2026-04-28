## Design

### Board

The board is a 61-cell hex grid represented with axial coordinates internally. Each cell also has a compact numeric id, so geometry lookups, neighbor masks, and row/column coordinates are cheap to move between. Positions keep sorted marble lists for canonical state and compact bitsets for fast occupancy tests. The move generator and search work mostly from those bitsets and precomputed geometry tables.

### Movegen

Move generation considers every legal group of one, two, or three friendly marbles. Each group is tested in the six hex directions as either an inline move or a broadside move. Inline moves handle pushes and ejections by checking the ray in front of the group, while broadside moves require all translated destination cells to be empty. The search path uses precomputed source-group and direction tables so legality checks avoid rebuilding the same geometry over and over.

### Eval

Evaluation is an embedded NNUE-style network stored in `net.mlp`. The sparse features are the occupied cell/color pairs, maintained incrementally in a 58-wide accumulator for both black and white perspectives. Eight dense features are mixed into the same 58-wide layer, including shape features such as edge pressure, contact pairs, liberties, singletons, remaining turns, and no-ejection ply. Inference pads that 58-wide layer to 64 values, applies an 8-bit quantized hidden layer with 32 units, then an 8-bit quantized output layer. The sparse input weights are 16-bit, while the hidden/output weights and activations are 8-bit.

### Search

Search is iterative deepening alpha-beta with aspiration windows around the previous iteration score. It keeps a transposition table, eval cache, history scores, countermoves, and correction history across searches. Move ordering prioritizes hash moves, ejections, countermoves, and history-guided candidates, with partial sorting near the front of the move list. The tree uses null-move pruning, futility pruning, late move reductions, and late move pruning to cut low-value branches. Timed searches poll the clock periodically and use a deadline margin so the engine can stop cleanly before the budget is exhausted.

## Usage

The standalone engine speaks a minimal line protocol over stdin/stdout. Each request is one space-delimited line. Each response is one space-delimited line. The engine returns the updated FEN after applying its chosen move, not the move itself.

Input:

```text
<board> <black_score> <white_score> <side> <no_ejection_ply> <move_number> <time_ms> <depth>
```

| # | Field | Example | Meaning |
|---:|---|---|---|
| 1 | `board` | `ss1SS/sssSSS/1ss1SS1/8/9/8/1SS1ss1/SSSsss/SS1ss` | 9-row PlayStrategy board, `S` black, `s` white, `1` empty |
| 2 | `black_score` | `0` | White marbles ejected |
| 3 | `white_score` | `0` | Black marbles ejected |
| 4 | `side` | `b` | Side to move: `b` or `w` |
| 5 | `no_ejection_ply` | `0` | Half-moves since last ejection |
| 6 | `move_number` | `1` | Full move number |
| 7 | `time_ms` | `3000` | Search time budget in ms |
| 8 | `depth` | `0` | Max depth, `0` means no depth cap |

Output:

```text
<board> <black_score> <white_score> <side> <no_ejection_ply> <move_number> <score> <depth> <nodes> <elapsed_ms>
```

| # | Field | Example | Meaning |
|---:|---|---|---|
| 1 | `board` | `ss1SS/sssSSS/1ss1SS1/8/9/3S4/1SS1ss1/SSSsss/1S1ss` | Updated board after engine move |
| 2 | `black_score` | `0` | Updated black score |
| 3 | `white_score` | `0` | Updated white score |
| 4 | `side` | `w` | Side to move after engine move |
| 5 | `no_ejection_ply` | `1` | Updated half-moves since ejection |
| 6 | `move_number` | `1` | Updated full move number |
| 7 | `score` | `703` | Engine evaluation/search score |
| 8 | `depth` | `17` | Completed search depth |
| 9 | `nodes` | `11845632` | Nodes searched |
| 10 | `elapsed_ms` | `2986` | Engine-measured elapsed time |

Use `depth = 0` for no depth cap. Use `time_ms = 0` with `depth > 0` for fixed-depth search. `time_ms = 0` and `depth = 0` is invalid.

Timed Belgian Daisy search, 3000 ms, no depth cap:

```text
ss1SS/sssSSS/1ss1SS1/8/9/8/1SS1ss1/SSSsss/SS1ss 0 0 b 0 1 3000 0
```

Example response:

```text
ss1SS/sssSSS/1ss1SS1/8/9/3S4/1SS1ss1/SSSsss/1S1ss 0 0 w 1 1 703 17 11845632 2986
```
