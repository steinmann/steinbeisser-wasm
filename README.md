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

The standalone engine speaks a minimal line protocol over stdin/stdout. Each request is one space-delimited line. Each response is one space-delimited line. The engine returns the updated FEN after applying its chosen move.

Input:

```text
SS1ss/SSSsss/1SS1ss1/8/9/8/1ss1SS1/sssSSS/ss1SS 0 0 b 0 1 0 1
```

| # | Field | Example | Meaning |
|---:|---|---|---|
| 1 | `board` | `SS1ss/SSSsss/1SS1ss1/8/9/8/1ss1SS1/sssSSS/ss1SS` | `S` black, `s` white, `n` empty |
| 2 | `black_score` | `0` | White marbles ejected |
| 3 | `white_score` | `0` | Black marbles ejected |
| 4 | `side` | `b` | Side to move |
| 5 | `no_ejection_ply` | `0` | Half-moves since last ejection |
| 6 | `move_number` | `1` | Full move number |
| 7 | `time_ms` | `0` | Search time budget in ms |
| 8 | `depth` | `1` | Max depth; `0` means no cap |

Output:

```text
S2ss/SSSsss/1SSSss1/8/9/8/1ss1SS1/sssSSS/ss1SS 0 0 w 1 1 846 1 56 0
```

| # | Field | Example | Meaning |
|---:|---|---|---|
| 1 | `board` | `S2ss/SSSsss/1SSSss1/8/9/8/1ss1SS1/sssSSS/ss1SS` | Board after move |
| 2 | `black_score` | `0` | Updated black score |
| 3 | `white_score` | `0` | Updated white score |
| 4 | `side` | `w` | Side to move |
| 5 | `no_ejection_ply` | `1` | Half-moves since ejection |
| 6 | `move_number` | `1` | Full move number |
| 7 | `score` | `846` | Search score |
| 8 | `depth` | `1` | Completed depth |
| 9 | `nodes` | `56` | Nodes searched |
| 10 | `elapsed_ms` | `0` | Engine elapsed time |

Use `depth = 0` for no depth cap. Use `time_ms = 0` with `depth > 0` for fixed-depth search. `time_ms = 0` and `depth = 0` is invalid.

## GitHub Tournament

### Rules

- Engines: 8 playable GitHub Abalone engines.
- Format: round robin.
- Openings: all 60 Wall of Variations FEN positions from [`data/positions`](data/positions).
- Games: each matchup played every opening once from each side, for 120 games per matchup.
- Time control: 100 ms per move.
- Steinbeisser build: native compiled Rust from this repository, not WebAssembly.
- Output: one clean JSON file per matchup in [`data/games`](data/games).

### Results

Elo is calculated from score percentage against the field.

| # | Author | Repo | Score | Score % | Elo | W-D-L |
|---:|---|---|---:|---:|---:|---:|
| 1 | steinmann | [steinbeisser-wasm][steinbeisser-wasm] | 840/840 | 100.0% | +inf | 840-0-0 |
| 2 | elchairoy | [Gnizabalone][Gnizabalone] | 720/840 | 85.7% | +311 | 720-0-120 |
| 3 | Retam1 | [abalone-agent][abalone-agent] | 521.5/840 | 62.1% | +86 | 497-49-294 |
| 4 | ilagko | [AbaloneWeb][AbaloneWeb] | 403.5/840 | 48.0% | -14 | 364-79-397 |
| 5 | altin | [abalone-ai][abalone-ai] | 403/840 | 48.0% | -14 | 355-96-389 |
| 6 | MichielVerloop | [AbaloneAI][AbaloneAI] | 229/840 | 27.3% | -170 | 170-118-552 |
| 7 | negjafari | [AI-abalone][AI-abalone] | 124/840 | 14.8% | -305 | 2-244-594 |
| 8 | AlirezaNR1 | [Abalone-AI][Abalone-AI] | 119/840 | 14.2% | -313 | 2-234-604 |

[steinbeisser-wasm]: https://github.com/steinmann/steinbeisser-wasm
[Gnizabalone]: https://github.com/elchairoy/Gnizabalone
[abalone-agent]: https://github.com/Retam1/abalone-agent
[abalone-ai]: https://github.com/altin/abalone-ai
[AbaloneWeb]: https://github.com/ilagko/AbaloneWeb
[AbaloneAI]: https://github.com/MichielVerloop/AbaloneAI
[AI-abalone]: https://github.com/negjafari/AI-abalone
[Abalone-AI]: https://github.com/AlirezaNR1/Abalone-AI
