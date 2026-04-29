## Play

https://steinmann.github.io/steinbeisser-wasm/

## Standalone Engine

The standalone engine speaks a minimal line protocol over stdin/stdout. Each request is one space-delimited line. Each response is one space-delimited line. The engine returns the updated FEN after applying its chosen move.

Example input, from the Belgian Daisy start with black to move:

```text
ss1SS/sssSSS/1ss1SS1/8/9/8/1SS1ss1/SSSsss/SS1ss 0 0 B 0 1 10000 0
```

| # | Field | Meaning |
|---:|---|---|
| 1 | `board` | `S` black, `s` white, `1-9` empty |
| 2 | `black_score` | White marbles ejected |
| 3 | `white_score` | Black marbles ejected |
| 4 | `side` | Side to move |
| 5 | `no_ejection_ply` | Plies since ejection |
| 6 | `move_number` | Full move |
| 7 | `time_ms` | Budget in ms |
| 8 | `depth` | Max depth; `0` no cap |

Example output:

```text
ss1SS/sssSSS/1ss1SS1/8/9/3S4/1SS1ss1/SSSsss/1S1ss 0 0 w 1 1 698 18 41377792 9985
```

| # | Field | Meaning |
|---:|---|---|
| 1 | `board` | Board after move |
| 2 | `black_score` | Updated black score |
| 3 | `white_score` | Updated white score |
| 4 | `side` | Side to move |
| 5 | `no_ejection_ply` | Plies since ejection |
| 6 | `move_number` | Full move |
| 7 | `score` | Search score |
| 8 | `depth` | Completed depth |
| 9 | `nodes` | Nodes searched |
| 10 | `elapsed_ms` | Engine elapsed time |

Use `depth = 0` for no depth cap. Use `time_ms = 0` with `depth > 0` for fixed-depth search. `time_ms = 0` and `depth = 0` is invalid.

## GitHub Tournament

### Rules

- Engines: All 8 playable GitHub Abalone engines.
- Format: Round robin.
- Openings: All 60 Wall of Variations FEN positions from [`data/positions`](data/positions).
- Games: Each matchup played every opening once from each side, for 120 games per matchup.
- Time control: 100 ms per move.
- Winning: Eject 6 enemy marbles, or lead on ejections after the 350-ply cap. Tied ejections are a draw.
- Output: One clean JSON file per matchup in [`data/games`](data/games).

### Results

| # | Author | Repo | Score | Score % | W-D-L |
|---:|---|---|---:|---:|---:|
| 1 | steinmann | [steinbeisser-wasm][steinbeisser-wasm] | 840/840 | 100.0% | 840-0-0 |
| 2 | elchairoy | [Gnizabalone][Gnizabalone] | 720/840 | 85.7% | 720-0-120 |
| 3 | Retam1 | [abalone-agent][abalone-agent] | 521.5/840 | 62.1% | 497-49-294 |
| 4 | ilagko | [AbaloneWeb][AbaloneWeb] | 403.5/840 | 48.0% | 364-79-397 |
| 5 | altin | [abalone-ai][abalone-ai] | 403/840 | 48.0% | 355-96-389 |
| 6 | MichielVerloop | [AbaloneAI][AbaloneAI] | 229/840 | 27.3% | 170-118-552 |
| 7 | negjafari | [AI-abalone][AI-abalone] | 124/840 | 14.8% | 2-244-594 |
| 8 | AlirezaNR1 | [Abalone-AI][Abalone-AI] | 119/840 | 14.2% | 2-234-604 |

[steinbeisser-wasm]: https://github.com/steinmann/steinbeisser-wasm
[Gnizabalone]: https://github.com/elchairoy/Gnizabalone
[abalone-agent]: https://github.com/Retam1/abalone-agent
[abalone-ai]: https://github.com/altin/abalone-ai
[AbaloneWeb]: https://github.com/ilagko/AbaloneWeb
[AbaloneAI]: https://github.com/MichielVerloop/AbaloneAI
[AI-abalone]: https://github.com/negjafari/AI-abalone
[Abalone-AI]: https://github.com/AlirezaNR1/Abalone-AI

## CodinGame Abalone League
https://www.codingame.com/multiplayer/bot-programming/abalone/leaderboard
