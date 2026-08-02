#![allow(dead_code, hidden_glob_reexports, private_interfaces)]

#[path = "../engine/src/board.rs"]
mod board;
#[path = "../engine/src/eval.rs"]
mod eval;
#[path = "../engine/src/movegen.rs"]
mod movegen;
#[path = "../engine/src/search.rs"]
mod search;

use std::collections::HashSet;
use std::env;
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use board::{geometry, CellId, Color, Coord, Position, Symmetry};
use search::{search_timed_with_turn, MAX_GAME_TURNS};

const MAX_EJECTIONS_PER_SIDE: usize = 3;
const DEFAULT_MAX_ABS_SCORE: i32 = 3500;

struct Args {
    n: usize,
    s: i32,
    t: u64,
    p: (usize, usize),
}

impl Default for Args {
    fn default() -> Self {
        Self {
            n: 1,
            s: DEFAULT_MAX_ABS_SCORE,
            t: 5,
            p: (50, 300),
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = args()?;
    let mut rng = Rng::new();
    let mut seen = HashSet::with_capacity(args.n.saturating_mul(2));
    let mut out = io::BufWriter::new(io::stdout().lock());
    let mut written = 0;

    while written < args.n {
        let candidate = random_position(&args, &mut rng)?;
        if !seen.insert(key(&candidate.pos)) {
            continue;
        }

        let turn = candidate.ply.min(MAX_GAME_TURNS as usize) as u16;
        if i64::from(static_score(&candidate.pos, candidate.quiet, turn)).abs() > i64::from(args.s)
        {
            continue;
        }

        let result =
            search_timed_with_turn(&candidate.pos, &[], candidate.quiet, turn, args.t, None)
                .map_err(|error| {
                    format!(
                        "search failed at {}: {error}",
                        fen(
                            &candidate.pos,
                            candidate.black_score,
                            candidate.white_score,
                            candidate.quiet,
                            candidate.ply
                        )
                    )
                })?;
        if result.best_move.is_none() || i64::from(result.score).abs() > i64::from(args.s) {
            continue;
        }

        writeln!(
            out,
            "{}",
            fen(
                &candidate.pos,
                candidate.black_score,
                candidate.white_score,
                candidate.quiet,
                candidate.ply
            )
        )
        .map_err(|error| format!("{error}"))?;
        written += 1;
    }

    Ok(())
}

fn args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "-n" => args.n = value.parse().map_err(|_| "bad -n".to_owned())?,
            "-s" => args.s = value.parse().map_err(|_| "bad -s".to_owned())?,
            "-t" => args.t = value.parse().map_err(|_| "bad -t".to_owned())?,
            "-p" => args.p = range(&value)?,
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if args.s < 0 || args.t == 0 || args.p.0 > args.p.1 {
        return Err("bad arguments".to_owned());
    }
    Ok(args)
}

fn range(value: &str) -> Result<(usize, usize), String> {
    let (a, b) = value.split_once(',').ok_or_else(|| "bad -p".to_owned())?;
    Ok((
        a.parse().map_err(|_| "bad -p".to_owned())?,
        b.parse().map_err(|_| "bad -p".to_owned())?,
    ))
}

struct Candidate {
    pos: Position,
    black_score: usize,
    white_score: usize,
    quiet: u16,
    ply: usize,
}

fn random_position(args: &Args, rng: &mut Rng) -> Result<Candidate, String> {
    let black_score = rng.usize(MAX_EJECTIONS_PER_SIDE + 1);
    let white_score = rng.usize(MAX_EJECTIONS_PER_SIDE + 1);
    let black_count = Position::MAX_PIECES_PER_SIDE - white_score;
    let white_count = Position::MAX_PIECES_PER_SIDE - black_score;
    let total = black_count + white_count;
    let side = if rng.usize(2) == 0 {
        Color::Black
    } else {
        Color::White
    };
    let ply = args.p.0 + rng.usize(args.p.1 - args.p.0 + 1);
    let quiet = 0;

    let mut cells: Vec<CellId> = (0..board::CELL_COUNT)
        .map(|cell| CellId::new(cell as u8).unwrap())
        .collect();
    shuffle(&mut cells, rng);
    let black = cells[..black_count].to_vec();
    let white = cells[black_count..total].to_vec();
    let pos = Position::new(side, black, white).map_err(|error| format!("{error}"))?;

    Ok(Candidate {
        pos,
        black_score,
        white_score,
        quiet,
        ply,
    })
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        items.swap(i, rng.usize(i + 1));
    }
}

fn static_score(pos: &Position, quiet: u16, turn: u16) -> i32 {
    let model = eval::nnue();
    let (black, white) = bits(pos, Symmetry::identity());
    let shape = eval::build_feature_shape(black, white);
    let acc = model.root_accumulator(pos);
    model.evaluate_with_accumulator_bits(
        pos.side_to_move() == Color::Black,
        &shape,
        black,
        white,
        turn as f32,
        quiet as f32,
        &acc,
    )
}

fn key(pos: &Position) -> u128 {
    Symmetry::all()
        .into_iter()
        .map(|sym| bit_key(pos, sym))
        .min()
        .unwrap()
}

fn bit_key(pos: &Position, sym: Symmetry) -> u128 {
    let (black, white) = bits(pos, sym);
    let side = match pos.side_to_move() {
        Color::Black => 0,
        Color::White => 1,
    };
    u128::from(black) | (u128::from(white) << board::CELL_COUNT) | (side << (board::CELL_COUNT * 2))
}

fn bits(pos: &Position, sym: Symmetry) -> (u64, u64) {
    let mut black = 0u64;
    let mut white = 0u64;
    for cell in pos.black() {
        black |= 1u64 << geometry().transform(cell, sym).as_u8();
    }
    for cell in pos.white() {
        white |= 1u64 << geometry().transform(cell, sym).as_u8();
    }
    (black, white)
}

fn fen(pos: &Position, black_score: usize, white_score: usize, quiet: u16, ply: usize) -> String {
    format!(
        "{} {} {} {} {} {}",
        board(pos),
        black_score,
        white_score,
        side(pos.side_to_move()),
        quiet,
        ply
    )
}

fn board(pos: &Position) -> String {
    let mut out = String::with_capacity(70);
    for row in Coord::MIN_ROW..=Coord::MAX_ROW {
        if row > Coord::MIN_ROW {
            out.push('/');
        }
        let mut empty = 0u8;
        for col in 1..=Coord::row_length(row).unwrap() {
            let cell = geometry()
                .index_of_coord(Coord::new(row, col).unwrap())
                .unwrap();
            match pos.occupant(cell) {
                Some(color) => {
                    if empty > 0 {
                        out.push_str(&empty.to_string());
                        empty = 0;
                    }
                    out.push(match color {
                        Color::Black => 'S',
                        Color::White => 's',
                    });
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
    }
    out
}

fn side(color: Color) -> &'static str {
    match color {
        Color::Black => "b",
        Color::White => "w",
    }
}

struct Rng(u64);

impl Rng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn usize(&mut self, n: usize) -> usize {
        (self.u64() as usize) % n
    }
}
