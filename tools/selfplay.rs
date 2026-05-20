#![allow(dead_code, hidden_glob_reexports, private_interfaces)]

#[path = "../engine/src/board.rs"]
mod board;
#[path = "../engine/src/eval.rs"]
mod eval;
#[path = "../engine/src/movegen.rs"]
mod movegen;
#[path = "../engine/src/search.rs"]
mod search;

use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::str::FromStr;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use board::{Color, Move, Position};
use movegen::PositionState;
use search::MAX_GAME_TURNS;

const MAX_PIECES: usize = Position::MAX_PIECES_PER_SIDE;
const WIN_SCORE: usize = 6;
const BOOTSTRAPS: usize = 2000;
const DEFAULT_MAX_ABS_SCORE: i32 = 3500;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Match,
    Generate,
    BuildLocal,
    BuildRef,
    BuildSource,
}

#[derive(Clone, Debug)]
struct Args {
    mode: Mode,
    openings: PathBuf,
    pairs: usize,
    games: usize,
    parallel_games: usize,
    target_samples: usize,
    time_ms: u64,
    time_set: bool,
    depth: u8,
    seed: u64,
    max_abs_score: Option<i32>,
    data_out: Option<PathBuf>,
    games_out: Option<PathBuf>,
    engine_source: PathBuf,
    reference_source: Option<PathBuf>,
    repo: PathBuf,
    github_ref: String,
    fetch: bool,
    keep_temp: bool,
    github_bin: Option<PathBuf>,
    local_bin: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::Match,
            openings: PathBuf::from("data/random100K.fen"),
            pairs: 100,
            games: usize::MAX,
            parallel_games: 1,
            target_samples: 20_000,
            time_ms: 5,
            time_set: false,
            depth: 0,
            seed: 1,
            max_abs_score: Some(DEFAULT_MAX_ABS_SCORE),
            data_out: None,
            games_out: None,
            engine_source: PathBuf::from("CodinGame/steinbeisser-rc5.rs"),
            reference_source: None,
            repo: PathBuf::new(),
            github_ref: "origin/main".to_owned(),
            fetch: true,
            keep_temp: false,
            github_bin: None,
            local_bin: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Limits {
    time_ms: u64,
    depth: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GameState {
    position: Position,
    black_score: usize,
    white_score: usize,
    no_progress: u16,
    ply: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EngineId {
    Github,
    Local,
    SelfplayA,
    SelfplayB,
}

#[derive(Clone, Debug)]
struct EngineStats {
    moves: usize,
    nodes: u64,
    engine_ms: u64,
    response_ms: u128,
    depth_sum: u64,
    illegal: usize,
    errors: usize,
}

impl EngineStats {
    fn new() -> Self {
        Self {
            moves: 0,
            nodes: 0,
            engine_ms: 0,
            response_ms: 0,
            depth_sum: 0,
            illegal: 0,
            errors: 0,
        }
    }

    fn record(&mut self, reply: &EngineReply) {
        self.moves += 1;
        self.nodes = self.nodes.saturating_add(reply.nodes);
        self.engine_ms = self.engine_ms.saturating_add(reply.elapsed_ms);
        self.response_ms = self.response_ms.saturating_add(reply.response_ms);
        self.depth_sum = self.depth_sum.saturating_add(u64::from(reply.depth));
    }

    fn add(&mut self, other: &Self) {
        self.moves += other.moves;
        self.nodes = self.nodes.saturating_add(other.nodes);
        self.engine_ms = self.engine_ms.saturating_add(other.engine_ms);
        self.response_ms = self.response_ms.saturating_add(other.response_ms);
        self.depth_sum = self.depth_sum.saturating_add(other.depth_sum);
        self.illegal += other.illegal;
        self.errors += other.errors;
    }

    fn avg_depth(&self) -> f64 {
        div(self.depth_sum as f64, self.moves as f64)
    }

    fn avg_response_ms(&self) -> f64 {
        div(self.response_ms as f64, self.moves as f64)
    }

    fn avg_nps(&self) -> f64 {
        div(self.nodes as f64 * 1000.0, self.engine_ms.max(1) as f64)
    }
}

#[derive(Clone, Debug)]
struct EngineReply {
    state: GameState,
    score: i32,
    depth: u8,
    nodes: u64,
    elapsed_ms: u64,
    response_ms: u128,
    best_move: Option<String>,
    fast_move_raw: Option<u16>,
}

struct EngineProcess {
    name: &'static str,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stats: EngineStats,
}

#[derive(Clone, Debug)]
struct TrainingRow {
    ply: u16,
    engine: &'static str,
    side: Color,
    fen: String,
    best_move: String,
    fast_move_raw: Option<u16>,
    score: i32,
    depth: u8,
    nodes: u64,
    elapsed_ms: u64,
    response_ms: u128,
    no_progress: u16,
    caused_ejection: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Outcome {
    Win(Color, &'static str),
    Draw(&'static str),
}

#[derive(Clone, Debug)]
struct GameRecord {
    local_points: f64,
    outcome: Outcome,
    plies: u16,
}

#[derive(Clone, Debug)]
struct PlayedGame {
    record: GameRecord,
    rows: Vec<TrainingRow>,
}

#[derive(Clone, Debug)]
struct BuiltEngines {
    github: PathBuf,
    local: PathBuf,
    temp_root: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct BuiltLocal {
    local: PathBuf,
    temp_root: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = args()?;
    if args.repo.as_os_str().is_empty() {
        args.repo = git_root(&env::current_dir().map_err(|error| format!("{error}"))?)?;
    }
    if args.mode == Mode::Generate && !args.time_set {
        args.time_ms = 100;
    }

    match args.mode {
        Mode::Match => run_match(&args),
        Mode::Generate => run_generate(&args),
        Mode::BuildLocal => run_build_local(&args),
        Mode::BuildRef => run_build_ref(&args),
        Mode::BuildSource => run_build_source(&args),
    }
}

fn run_build_local(args: &Args) -> Result<(), String> {
    let out = args
        .local_bin
        .as_ref()
        .ok_or_else(|| "build-local needs --local-bin <file>".to_owned())?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }

    let mut build_args = args.clone();
    build_args.local_bin = None;
    let built = build_local_engine(&build_args)?;
    fs::copy(&built.local, out).map_err(|error| format!("{}: {error}", out.display()))?;
    println!("local_bin,{}", out.display());
    cleanup(args.keep_temp, built.temp_root);
    Ok(())
}

fn run_build_ref(args: &Args) -> Result<(), String> {
    let out = args
        .github_bin
        .as_ref()
        .ok_or_else(|| "build-ref needs --github-bin <file>".to_owned())?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }

    let root = temp_root("steinbeisser-selfplay-ref");
    let source_root = root.join("source");
    fs::create_dir_all(&source_root).map_err(|error| format!("{error}"))?;
    archive_ref(&args.repo, &args.github_ref, &source_root, &root)?;
    let built = build_native_engine(&source_root, &root, "ref")?;
    fs::copy(&built, out).map_err(|error| format!("{}: {error}", out.display()))?;
    println!("github_bin,{}", out.display());
    cleanup(args.keep_temp, Some(root));
    Ok(())
}

fn run_build_source(args: &Args) -> Result<(), String> {
    let out = args
        .local_bin
        .as_ref()
        .ok_or_else(|| "build-source needs --local-bin <file>".to_owned())?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }

    let source = args
        .engine_source
        .canonicalize()
        .map_err(|error| format!("{}: {error}", args.engine_source.display()))?;
    let root = temp_root("steinbeisser-selfplay-source");
    fs::create_dir_all(&root).map_err(|error| format!("{error}"))?;
    let wrapper = root.join("source_engine.rs");
    fs::write(&wrapper, render_source_engine(&source))
        .map_err(|error| format!("{}: {error}", wrapper.display()))?;
    run_command(
        Command::new("rustc")
            .arg("--edition=2021")
            .arg("-Awarnings")
            .arg("-O")
            .arg(&wrapper)
            .arg("-o")
            .arg(out),
    )?;
    println!("local_bin,{}", out.display());
    cleanup(args.keep_temp, Some(root));
    Ok(())
}

fn run_match(args: &Args) -> Result<(), String> {
    let limits = Limits {
        time_ms: args.time_ms,
        depth: args.depth,
    };
    let built = build_match_engines(args)?;
    let openings = load_openings(&args.openings)?;
    if openings.is_empty() {
        return Err("opening file is empty".to_owned());
    }

    let mut order = (0..openings.len()).collect::<Vec<_>>();
    shuffle(&mut order, &mut Rng::new(args.seed));

    let mut github = EngineProcess::spawn("github", &built.github)?;
    let mut local = EngineProcess::spawn("local", &built.local)?;
    let mut training = open_training(args.data_out.as_ref())?;
    let mut records = Vec::with_capacity(args.pairs * 2);
    let mut pair_points = Vec::with_capacity(args.pairs);
    let started = Instant::now();

    for pair in 0..args.pairs {
        let opening = openings[order[pair % order.len()]].clone();
        let first = play_game(
            &opening,
            EngineId::Github,
            EngineId::Local,
            &mut github,
            &mut local,
            limits,
            args.max_abs_score,
        )?;
        let second = play_game(
            &opening,
            EngineId::Local,
            EngineId::Github,
            &mut github,
            &mut local,
            limits,
            args.max_abs_score,
        )?;
        write_training(
            training.as_mut(),
            pair * 2,
            &first.rows,
            first.record.outcome,
        )?;
        write_training(
            training.as_mut(),
            pair * 2 + 1,
            &second.rows,
            second.record.outcome,
        )?;
        pair_points.push(first.record.local_points + second.record.local_points);
        records.push(first.record);
        records.push(second.record);
    }

    if let Some(writer) = training.as_mut() {
        writer.flush().map_err(|error| format!("{error}"))?;
    }

    let summary = summarize(&records, &pair_points);
    print_summary(
        args,
        &built,
        &github.stats,
        &local.stats,
        &summary,
        started.elapsed().as_secs_f64(),
    );

    cleanup(args.keep_temp, built.temp_root);
    Ok(())
}

fn run_generate(args: &Args) -> Result<(), String> {
    let out = args
        .games_out
        .as_ref()
        .ok_or_else(|| "generate needs --games-out <file>".to_owned())?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }

    let limits = Limits {
        time_ms: args.time_ms,
        depth: args.depth,
    };
    let built = build_local_engine(args)?;
    let openings = load_openings(&args.openings)?;
    if openings.is_empty() {
        return Err("opening file is empty".to_owned());
    }
    let mut order = (0..openings.len()).collect::<Vec<_>>();
    shuffle(&mut order, &mut Rng::new(args.seed));

    let started = Instant::now();
    let (games, sample_count, stats) = if args.parallel_games == 1 {
        generate_serial(&built.local, &openings, &order, args, limits)?
    } else {
        generate_parallel(&built.local, openings, order, args, limits)?
    };

    write_fen_samples_jsonl(out, &games)?;
    println!("mode,generate");
    println!("samples_out,{}", out.display());
    println!("games,{}", games.len());
    println!("samples,{sample_count}");
    println!(
        "max_abs_score,{}",
        args.max_abs_score
            .map(|limit| limit.to_string())
            .unwrap_or_else(|| "none".to_owned())
    );
    println!("time_ms,{}", args.time_ms);
    println!("elapsed_s,{:.3}", started.elapsed().as_secs_f64());
    print_engine("selfplay", &stats);

    cleanup(args.keep_temp, built.temp_root);
    Ok(())
}

fn generate_serial(
    engine: &Path,
    openings: &[GameState],
    order: &[usize],
    args: &Args,
    limits: Limits,
) -> Result<(Vec<(String, PlayedGame)>, usize, EngineStats), String> {
    let mut black = EngineProcess::spawn("selfplay-a", engine)?;
    let mut white = EngineProcess::spawn("selfplay-b", engine)?;
    let mut games = Vec::<(String, PlayedGame)>::new();
    let mut sample_count = 0usize;

    while sample_count < args.target_samples && games.len() < args.games {
        let opening_index = order[games.len() % order.len()];
        let opening = openings[opening_index].clone();
        let name = format!("fen-{opening_index:06}");
        let game = play_game(
            &opening,
            EngineId::SelfplayA,
            EngineId::SelfplayB,
            &mut black,
            &mut white,
            limits,
            args.max_abs_score,
        )?;
        sample_count += game.rows.len();
        games.push((name, game));
    }

    let mut stats = EngineStats::new();
    stats.add(&black.stats);
    stats.add(&white.stats);
    Ok((games, sample_count, stats))
}

fn generate_parallel(
    engine: &Path,
    openings: Vec<GameState>,
    order: Vec<usize>,
    args: &Args,
    limits: Limits,
) -> Result<(Vec<(String, PlayedGame)>, usize, EngineStats), String> {
    let engine = Arc::new(engine.to_path_buf());
    let openings = Arc::new(openings);
    let order = Arc::new(order);
    let next_game = Arc::new(AtomicUsize::new(0));
    let sample_count = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let error = Arc::new(Mutex::new(None::<String>));
    let mut handles = Vec::with_capacity(args.parallel_games);

    for worker in 0..args.parallel_games {
        let engine = Arc::clone(&engine);
        let openings = Arc::clone(&openings);
        let order = Arc::clone(&order);
        let next_game = Arc::clone(&next_game);
        let sample_count = Arc::clone(&sample_count);
        let stop = Arc::clone(&stop);
        let error = Arc::clone(&error);
        let target_samples = args.target_samples;
        let max_games = args.games;
        let max_abs_score = args.max_abs_score;

        handles.push(thread::spawn(
            move || -> (Vec<(usize, String, PlayedGame)>, EngineStats) {
                let mut local_games = Vec::new();
                let mut stats = EngineStats::new();
                let result = (|| -> Result<(), String> {
                    let mut black = EngineProcess::spawn("selfplay-a", &engine)?;
                    let mut white = EngineProcess::spawn("selfplay-b", &engine)?;
                    loop {
                        if stop.load(Ordering::Relaxed)
                            || sample_count.load(Ordering::Relaxed) >= target_samples
                        {
                            break;
                        }
                        let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                        if game_index >= max_games {
                            break;
                        }
                        if sample_count.load(Ordering::Relaxed) >= target_samples {
                            break;
                        }
                        let opening_index = order[game_index % order.len()];
                        let opening = openings[opening_index].clone();
                        let name = format!("fen-{opening_index:06}-w{worker:02}-{game_index:08}");
                        let game = play_game(
                            &opening,
                            EngineId::SelfplayA,
                            EngineId::SelfplayB,
                            &mut black,
                            &mut white,
                            limits,
                            max_abs_score,
                        )?;
                        sample_count.fetch_add(game.rows.len(), Ordering::Relaxed);
                        local_games.push((game_index, name, game));
                    }
                    stats.add(&black.stats);
                    stats.add(&white.stats);
                    Ok(())
                })();
                if let Err(message) = result {
                    stop.store(true, Ordering::Relaxed);
                    if let Ok(mut slot) = error.lock() {
                        if slot.is_none() {
                            *slot = Some(message);
                        }
                    }
                }
                (local_games, stats)
            },
        ));
    }

    let mut indexed_games = Vec::new();
    let mut stats = EngineStats::new();
    for handle in handles {
        let (mut worker_games, worker_stats) = handle
            .join()
            .map_err(|_| "selfplay worker panicked".to_owned())?;
        indexed_games.append(&mut worker_games);
        stats.add(&worker_stats);
    }
    let message = {
        let mut guard = error.lock().map_err(|error| format!("{error}"))?;
        guard.take()
    };
    if let Some(message) = message {
        return Err(message);
    }

    indexed_games.sort_by_key(|(index, _, _)| *index);
    let games = indexed_games
        .into_iter()
        .map(|(_, name, game)| (name, game))
        .collect::<Vec<_>>();
    let samples = games.iter().map(|(_, game)| game.rows.len()).sum();
    Ok((games, samples, stats))
}

fn args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = env::args().skip(1).peekable();
    if let Some(mode) = it.peek().cloned() {
        match mode.as_str() {
            "match" => {
                args.mode = Mode::Match;
                it.next();
            }
            "generate" => {
                args.mode = Mode::Generate;
                it.next();
            }
            "build-local" => {
                args.mode = Mode::BuildLocal;
                it.next();
            }
            "build-ref" => {
                args.mode = Mode::BuildRef;
                it.next();
            }
            "build-source" => {
                args.mode = Mode::BuildSource;
                it.next();
            }
            _ => {}
        }
    }

    while let Some(flag) = it.next() {
        match flag.as_str() {
            "-o" | "--openings" => args.openings = next_path(&mut it, &flag)?,
            "-n" | "--pairs" => args.pairs = next_value(&mut it, &flag)?,
            "--games" => args.games = next_value(&mut it, &flag)?,
            "--parallel-games" => args.parallel_games = next_value(&mut it, &flag)?,
            "--target-samples" => args.target_samples = next_value(&mut it, &flag)?,
            "-t" | "--time" => {
                args.time_ms = next_value(&mut it, &flag)?;
                args.time_set = true;
            }
            "-d" | "--depth" => args.depth = next_value(&mut it, &flag)?,
            "--seed" => args.seed = next_value(&mut it, &flag)?,
            "--max-abs-score" => args.max_abs_score = Some(next_value(&mut it, &flag)?),
            "--no-score-filter" => args.max_abs_score = None,
            "--data-out" => args.data_out = Some(next_path(&mut it, &flag)?),
            "--games-out" => args.games_out = Some(next_path(&mut it, &flag)?),
            "--engine" => args.engine_source = next_path(&mut it, &flag)?,
            "--reference-engine" | "--reference-source" => {
                args.reference_source = Some(next_path(&mut it, &flag)?)
            }
            "--repo" => args.repo = next_path(&mut it, &flag)?,
            "--github-ref" => args.github_ref = next_string(&mut it, &flag)?,
            "--github-bin" => args.github_bin = Some(next_path(&mut it, &flag)?),
            "--local-bin" => args.local_bin = Some(next_path(&mut it, &flag)?),
            "--no-fetch" => args.fetch = false,
            "--keep-temp" => args.keep_temp = true,
            "-h" | "--help" => return Err(usage()),
            _ => return Err(format!("unknown option {flag}\n\n{}", usage())),
        }
    }

    if args.pairs == 0 || args.target_samples == 0 || args.parallel_games == 0 {
        return Err("counts must be greater than zero".to_owned());
    }
    if args.time_ms == 0 && args.depth == 0 {
        return Err("at least one of --time or --depth must be non-zero".to_owned());
    }
    if matches!(args.max_abs_score, Some(limit) if limit < 0) {
        return Err("--max-abs-score must be non-negative".to_owned());
    }
    if args.mode == Mode::Match && args.github_bin.is_some() != args.local_bin.is_some() {
        return Err("--github-bin and --local-bin must be used together in match mode".to_owned());
    }
    Ok(args)
}

fn usage() -> String {
    "usage:\n\
       selfplay match [-o data/random100K.fen] [-n pairs] [-t ms] [--github-ref origin/main]\n\
       selfplay generate [-o data/random100K.fen] --games-out samples.jsonl [--target-samples 20000] [--parallel-games 15] [--max-abs-score 3500] [-t 100]\n\
       selfplay build-local --repo . --local-bin /tmp/steinbeisser-native\n\
       selfplay build-ref --repo . --github-ref v1.0 --github-bin /tmp/steinbeisser-v1\n\
       selfplay build-source --engine CodinGame/steinbeisser-rc8.rs --local-bin /tmp/steinbeisser-source\n"
        .to_owned()
}

fn next_string<I: Iterator<Item = String>>(it: &mut I, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn next_path<I: Iterator<Item = String>>(it: &mut I, flag: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(next_string(it, flag)?))
}

fn next_value<T: FromStr, I: Iterator<Item = String>>(it: &mut I, flag: &str) -> Result<T, String> {
    next_string(it, flag)?
        .parse::<T>()
        .map_err(|_| format!("bad value for {flag}"))
}

fn build_match_engines(args: &Args) -> Result<BuiltEngines, String> {
    if let (Some(github), Some(local)) = (&args.github_bin, &args.local_bin) {
        return Ok(BuiltEngines {
            github: github.clone(),
            local: local.clone(),
            temp_root: None,
        });
    }

    if args.fetch {
        run_command(
            Command::new("git")
                .arg("-C")
                .arg(&args.repo)
                .arg("fetch")
                .arg("origin")
                .arg("main"),
        )?;
    }

    let root = temp_root("steinbeisser-selfplay-match");
    let github_src = root.join("github");
    let local_src = root.join("local");
    fs::create_dir_all(&github_src).map_err(|error| format!("{error}"))?;
    fs::create_dir_all(&local_src).map_err(|error| format!("{error}"))?;

    archive_ref(&args.repo, &args.github_ref, &github_src, &root)?;
    copy_tree(&args.repo, &local_src)?;

    let github = build_native_engine(&github_src, &root, "github")?;
    let local = build_native_engine(&local_src, &root, "local")?;
    Ok(BuiltEngines {
        github,
        local,
        temp_root: Some(root),
    })
}

fn build_local_engine(args: &Args) -> Result<BuiltLocal, String> {
    if let Some(local) = &args.local_bin {
        return Ok(BuiltLocal {
            local: local.clone(),
            temp_root: None,
        });
    }

    let root = temp_root("steinbeisser-selfplay-local");
    let local_src = root.join("local");
    fs::create_dir_all(&local_src).map_err(|error| format!("{error}"))?;
    copy_tree(&args.repo, &local_src)?;
    let local = build_native_engine(&local_src, &root, "local")?;
    Ok(BuiltLocal {
        local,
        temp_root: Some(root),
    })
}

fn materialize_reference_source(
    args: &Args,
    workspace: &Path,
) -> Result<(PathBuf, Option<PathBuf>), String> {
    let source = args
        .reference_source
        .as_ref()
        .unwrap_or(&args.engine_source);
    match materialize_github_reference_source(args, source) {
        Ok(reference) => Ok(reference),
        Err(git_error) => {
            let local = resolve_reference_file(source, workspace)?;
            eprintln!(
                "warning: using local reference source {} ({git_error})",
                local.display()
            );
            Ok((local, None))
        }
    }
}

fn resolve_reference_file(source: &Path, workspace: &Path) -> Result<PathBuf, String> {
    for candidate in [
        source.to_path_buf(),
        workspace.join(source),
        workspace.join("CodinGame").join(source),
    ] {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "could not resolve reference source {}",
        source.display()
    ))
}

fn materialize_github_reference_source(
    args: &Args,
    source: &Path,
) -> Result<(PathBuf, Option<PathBuf>), String> {
    if args.fetch {
        run_command(
            Command::new("git")
                .arg("-C")
                .arg(&args.repo)
                .arg("fetch")
                .arg("origin")
                .arg("main"),
        )?;
    }
    let root = temp_root("steinbeisser-selfplay-reference");
    fs::create_dir_all(&root).map_err(|error| format!("{}: {error}", root.display()))?;
    let source_in_repo = args
        .reference_source
        .as_deref()
        .unwrap_or(source)
        .strip_prefix(&args.repo)
        .unwrap_or(source);
    let spec = format!("{}:{}", args.github_ref, source_in_repo.display());
    let output = Command::new("git")
        .arg("-C")
        .arg(&args.repo)
        .arg("show")
        .arg(&spec)
        .output()
        .map_err(|error| format!("{error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "failed to read GitHub reference source {spec}: {stderr}"
        ));
    }
    let file_name = args
        .reference_source
        .as_deref()
        .unwrap_or(source)
        .file_name()
        .ok_or_else(|| "engine source has no file name".to_owned())?;
    let path = root.join(file_name);
    fs::write(&path, output.stdout).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok((path, Some(root)))
}

fn archive_ref(repo: &Path, git_ref: &str, dst: &Path, temp: &Path) -> Result<(), String> {
    let archive = temp.join("github.tar");
    run_command(
        Command::new("git")
            .arg("-C")
            .arg(repo)
            .arg("archive")
            .arg("--format=tar")
            .arg("--output")
            .arg(&archive)
            .arg(git_ref),
    )?;
    run_command(
        Command::new("tar")
            .arg("-xf")
            .arg(&archive)
            .arg("-C")
            .arg(dst),
    )?;
    prune_skipped_roots(dst)?;
    Ok(())
}

fn prune_skipped_roots(root: &Path) -> Result<(), String> {
    for name in [".git", ".build", "target", "dist", "data", ".DS_Store"] {
        let path = root.join(name);
        if path.is_dir() {
            fs::remove_dir_all(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        } else if path.is_file() {
            fs::remove_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    Ok(())
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), String> {
    for entry in fs::read_dir(src).map_err(|error| format!("{}: {error}", src.display()))? {
        let entry = entry.map_err(|error| format!("{error}"))?;
        let name = entry.file_name();
        if skip_copy_name(&name) {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        let file_type = entry.file_type().map_err(|error| format!("{error}"))?;
        if file_type.is_dir() {
            fs::create_dir_all(&dst_path).map_err(|error| format!("{error}"))?;
            copy_tree(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            fs::copy(&src_path, &dst_path).map_err(|error| format!("{error}"))?;
        }
    }
    Ok(())
}

fn skip_copy_name(name: &OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(".git" | ".build" | "target" | "dist" | "data" | ".DS_Store")
    )
}

fn build_native_engine(src: &Path, root: &Path, name: &str) -> Result<PathBuf, String> {
    let bin_dir = src.join("engine/src/bin");
    fs::create_dir_all(&bin_dir).map_err(|error| format!("{error}"))?;
    fs::write(bin_dir.join("steinbeisser_native.rs"), NATIVE_ENGINE)
        .map_err(|error| format!("{error}"))?;

    let target_dir = root.join(format!("target-{name}"));
    run_command(
        Command::new("cargo")
            .arg("build")
            .arg("--manifest-path")
            .arg(src.join("engine/Cargo.toml"))
            .arg("--target-dir")
            .arg(&target_dir)
            .arg("--release")
            .arg("--bin")
            .arg("steinbeisser_native"),
    )?;

    let exe = if cfg!(windows) {
        "steinbeisser_native.exe"
    } else {
        "steinbeisser_native"
    };
    Ok(target_dir.join("release").join(exe))
}

fn run_command(command: &mut Command) -> Result<(), String> {
    let status = command.status().map_err(|error| format!("{error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed with status {status:?}"))
    }
}

fn run_capture(command: &mut Command) -> Result<String, String> {
    let output = command.output().map_err(|error| format!("{error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");
    if output.status.success() {
        Ok(stdout)
    } else {
        Err(format!("command failed with status {:?}", output.status))
    }
}

fn ensure_cargo_bin(workspace: &Path, bin: &str, manifest: &Path) -> Result<PathBuf, String> {
    let target_dir = cargo_target_directory(manifest).unwrap_or_else(|| workspace.join("target"));
    let exe = target_dir.join("release").join(if cfg!(windows) {
        format!("{bin}.exe")
    } else {
        bin.to_owned()
    });
    if exe.is_file() {
        return Ok(exe);
    }
    run_command(
        Command::new("cargo")
            .arg("build")
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--release"),
    )?;
    if exe.is_file() {
        Ok(exe)
    } else {
        Err(format!("built binary was not found at {}", exe.display()))
    }
}

fn cargo_target_directory(manifest: &Path) -> Option<PathBuf> {
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--format-version")
        .arg("1")
        .arg("--no-deps")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    json_string_field(&text, "target_directory").map(PathBuf::from)
}

fn json_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut rest = text.split_once(&needle)?.1;
    rest = rest.split_once(':')?.1.trim_start();
    let raw = rest.strip_prefix('"')?;
    let mut out = String::new();
    let mut escaped = false;
    for ch in raw.chars() {
        if escaped {
            out.push(match ch {
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                'b' => '\u{0008}',
                'f' => '\u{000c}',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            });
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn git_root(cwd: &Path) -> Result<PathBuf, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .arg("rev-parse")
        .arg("--show-toplevel")
        .output()
        .map_err(|error| format!("{error}"))?;
    if !output.status.success() {
        return Err("not inside a git repository".to_owned());
    }
    let text = String::from_utf8(output.stdout).map_err(|error| format!("{error}"))?;
    Ok(PathBuf::from(text.trim()))
}

fn load_openings(path: &Path) -> Result<Vec<GameState>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut out = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("{error}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let state = GameState::parse(trimmed)
            .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1))?;
        state.validate()?;
        out.push(state);
    }
    Ok(out)
}

impl GameState {
    fn parse(line: &str) -> Result<Self, String> {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 6 {
            return Err("FEN needs at least 6 fields".to_owned());
        }
        Ok(Self {
            position: Position::from_str(line).map_err(|error| format!("{error}"))?,
            black_score: fields[1]
                .parse()
                .map_err(|_| "bad black score".to_owned())?,
            white_score: fields[2]
                .parse()
                .map_err(|_| "bad white score".to_owned())?,
            no_progress: fields[4]
                .parse()
                .map_err(|_| "bad no-progress ply".to_owned())?,
            ply: fields[5].parse().map_err(|_| "bad ply".to_owned())?,
        })
    }

    fn validate(&self) -> Result<(), String> {
        let expected_black = MAX_PIECES.saturating_sub(self.position.white().len());
        let expected_white = MAX_PIECES.saturating_sub(self.position.black().len());
        if self.black_score != expected_black || self.white_score != expected_white {
            return Err(format!(
                "score/material mismatch: got {} {}, expected {} {}",
                self.black_score, self.white_score, expected_black, expected_white
            ));
        }
        Ok(())
    }

    fn side(&self) -> Color {
        self.position.side_to_move()
    }

    fn total_score(&self) -> usize {
        self.black_score + self.white_score
    }

    fn fen(&self) -> String {
        format!(
            "{} {} {} {} {} {}",
            board_text(&self.position),
            self.black_score,
            self.white_score,
            side_text(self.side()),
            self.no_progress,
            self.ply
        )
    }

    fn input(&self, limits: Limits) -> String {
        format!("{} {} {}\n", self.fen(), limits.time_ms, limits.depth)
    }
}

impl EngineProcess {
    fn spawn(name: &'static str, path: &Path) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "missing engine stdin".to_owned())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "missing engine stdout".to_owned())?;
        Ok(Self {
            name,
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stats: EngineStats::new(),
        })
    }

    fn search(&mut self, state: &GameState, limits: Limits) -> Result<EngineReply, String> {
        let start = Instant::now();
        self.stdin
            .write_all(state.input(limits).as_bytes())
            .and_then(|_| self.stdin.flush())
            .map_err(|error| {
                self.stats.errors += 1;
                format!("{} write failed: {error}", self.name)
            })?;

        let mut line = String::new();
        let read = self.stdout.read_line(&mut line).map_err(|error| {
            self.stats.errors += 1;
            format!("{} read failed: {error}", self.name)
        })?;
        if read == 0 {
            self.stats.errors += 1;
            return Err(format!("{} exited", self.name));
        }
        let reply = parse_reply(&line, start.elapsed().as_millis())?;
        self.stats.record(&reply);
        Ok(reply)
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn parse_reply(line: &str, response_ms: u128) -> Result<EngineReply, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 12 {
        return Err(format!(
            "engine reply needs 12 fields including move and fast_move_raw, got {}",
            fields.len()
        ));
    }
    Ok(EngineReply {
        state: GameState::parse(line)?,
        score: fields[6]
            .parse()
            .map_err(|_| "bad reply score".to_owned())?,
        depth: fields[7]
            .parse()
            .map_err(|_| "bad reply depth".to_owned())?,
        nodes: fields[8]
            .parse()
            .map_err(|_| "bad reply nodes".to_owned())?,
        elapsed_ms: fields[9]
            .parse()
            .map_err(|_| "bad reply elapsed_ms".to_owned())?,
        response_ms,
        best_move: Some(fields[10].to_owned()),
        fast_move_raw: Some(
            fields[11]
                .parse()
                .map_err(|_| "bad reply fast_move_raw".to_owned())?,
        ),
    })
}

fn play_game(
    opening: &GameState,
    black_engine: EngineId,
    white_engine: EngineId,
    first: &mut EngineProcess,
    second: &mut EngineProcess,
    limits: Limits,
    max_abs_score: Option<i32>,
) -> Result<PlayedGame, String> {
    let mut state = opening.clone();
    let mut rows = Vec::new();

    loop {
        if let Some(outcome) = detect_result(&state) {
            let plies = rows.len().min(usize::from(u16::MAX)) as u16;
            return Ok(PlayedGame {
                record: GameRecord {
                    local_points: local_points(outcome, black_engine, white_engine),
                    outcome,
                    plies,
                },
                rows,
            });
        }

        let side = state.side();
        let engine_id = if side == Color::Black {
            black_engine
        } else {
            white_engine
        };
        let before = state.clone();
        let reply = engine_for(first, second, engine_id).search(&before, limits)?;
        let best_move = adjudicate(&before, &reply).map_err(|error| {
            engine_for(first, second, engine_id).stats.illegal += 1;
            format!(
                "illegal engine reply from {}: {error}",
                engine_name(engine_id)
            )
        })?;

        if keep_training_row(&reply, max_abs_score) {
            rows.push(TrainingRow {
                ply: before.ply,
                engine: engine_name(engine_id),
                side,
                fen: before.fen(),
                best_move: best_move.to_string(),
                fast_move_raw: reply.fast_move_raw,
                score: reply.score,
                depth: reply.depth,
                nodes: reply.nodes,
                elapsed_ms: reply.elapsed_ms,
                response_ms: reply.response_ms,
                no_progress: before.no_progress,
                caused_ejection: reply.state.total_score() > before.total_score(),
            });
        }
        state = reply.state;
    }
}

fn keep_training_row(reply: &EngineReply, max_abs_score: Option<i32>) -> bool {
    if detect_result(&reply.state).is_some() {
        return false;
    }
    if let Some(limit) = max_abs_score {
        if i64::from(reply.score).abs() > i64::from(limit) {
            return false;
        }
    }
    true
}

fn engine_for<'a>(
    first: &'a mut EngineProcess,
    second: &'a mut EngineProcess,
    id: EngineId,
) -> &'a mut EngineProcess {
    match id {
        EngineId::Github | EngineId::SelfplayA => first,
        EngineId::Local | EngineId::SelfplayB => second,
    }
}

fn adjudicate(before: &GameState, reply: &EngineReply) -> Result<Move, String> {
    let after = &reply.state;
    after.validate()?;
    if after.ply != before.ply.saturating_add(1) {
        return Err("ply did not increment by one".to_owned());
    }
    let expected_no_progress = if after.total_score() > before.total_score() {
        0
    } else {
        before.no_progress.saturating_add(1)
    };
    if after.no_progress != expected_no_progress {
        return Err("bad no-progress ply".to_owned());
    }

    if let Some(best_move) = &reply.best_move {
        let candidate_move = Move::from_str(best_move).map_err(|error| format!("{error}"))?;
        let mut state =
            PositionState::new(before.position.clone()).map_err(|error| format!("{error:?}"))?;
        state
            .apply_move(&candidate_move)
            .map_err(|error| format!("{error:?}"))?;
        if canonical_position(state.position())? == after.position {
            return Ok(candidate_move);
        }
        return Err("reported move does not produce reported position".to_owned());
    }

    let state =
        PositionState::new(before.position.clone()).map_err(|error| format!("{error:?}"))?;
    for candidate_move in state.generate_legal_moves() {
        let mut next = state.clone();
        next.apply_move(&candidate_move)
            .map_err(|error| format!("{error:?}"))?;
        if canonical_position(next.position())? == after.position {
            return Ok(candidate_move);
        }
    }
    Err("next position is not legal".to_owned())
}

fn canonical_position(position: &Position) -> Result<Position, String> {
    Position::new(
        position.side_to_move(),
        position.black().to_vec(),
        position.white().to_vec(),
    )
    .map_err(|error| format!("{error}"))
}

fn detect_result(state: &GameState) -> Option<Outcome> {
    if state.black_score >= WIN_SCORE {
        return Some(Outcome::Win(Color::Black, "six_ejections"));
    }
    if state.white_score >= WIN_SCORE {
        return Some(Outcome::Win(Color::White, "six_ejections"));
    }
    if state.ply >= MAX_GAME_TURNS {
        return Some(match state.black_score.cmp(&state.white_score) {
            std::cmp::Ordering::Greater => Outcome::Win(Color::Black, "max_turns_score"),
            std::cmp::Ordering::Less => Outcome::Win(Color::White, "max_turns_score"),
            std::cmp::Ordering::Equal => Outcome::Draw("max_turns_even_score"),
        });
    }
    None
}

fn local_points(outcome: Outcome, black_engine: EngineId, white_engine: EngineId) -> f64 {
    match outcome {
        Outcome::Draw(_) => 0.5,
        Outcome::Win(Color::Black, _) if black_engine == EngineId::Local => 1.0,
        Outcome::Win(Color::White, _) if white_engine == EngineId::Local => 1.0,
        Outcome::Win(_, _) => 0.0,
    }
}

fn open_training(path: Option<&PathBuf>) -> Result<Option<BufWriter<File>>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let mut writer =
        BufWriter::new(File::create(path).map_err(|error| format!("{}: {error}", path.display()))?);
    writeln!(
        writer,
        "game_id\tply\tengine\tside\tfen\tmove\tscore\tdepth\tnodes\telapsed_ms\tresponse_ms\tresult\twinner\treason"
    )
    .map_err(|error| format!("{error}"))?;
    Ok(Some(writer))
}

fn write_training(
    writer: Option<&mut BufWriter<File>>,
    game_id: usize,
    rows: &[TrainingRow],
    outcome: Outcome,
) -> Result<(), String> {
    let Some(writer) = writer else {
        return Ok(());
    };
    let winner = match outcome {
        Outcome::Win(color, _) => side_text(color),
        Outcome::Draw(_) => "draw",
    };
    let reason = match outcome {
        Outcome::Win(_, reason) | Outcome::Draw(reason) => reason,
    };
    for row in rows {
        let result = match outcome {
            Outcome::Draw(_) => 0,
            Outcome::Win(color, _) if color == row.side => 1,
            Outcome::Win(_, _) => -1,
        };
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            game_id,
            row.ply,
            row.engine,
            side_text(row.side),
            row.fen,
            row.best_move,
            row.score,
            row.depth,
            row.nodes,
            row.elapsed_ms,
            row.response_ms,
            result,
            winner,
            reason
        )
        .map_err(|error| format!("{error}"))?;
    }
    Ok(())
}

fn write_fen_samples_jsonl(path: &Path, games: &[(String, PlayedGame)]) -> Result<(), String> {
    let mut writer =
        BufWriter::new(File::create(path).map_err(|error| format!("{}: {error}", path.display()))?);
    let run_file = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("selfplay.jsonl");
    for (game_idx, (name, game)) in games.iter().enumerate() {
        for row in &game.rows {
            let result = match game.record.outcome {
                Outcome::Draw(_) => 0,
                Outcome::Win(color, _) if color == row.side => 1,
                Outcome::Win(_, _) => -1,
            };
            let clipped = row.score.clamp(-10_000, 10_000);
            write!(
                writer,
                "{{\"format\":\"steinbeisser-fen-sample-v1\",\"run_file\":\"{}\",\"game_id\":{},\"opening_name\":\"{}\",\"fen\":\"{}\",\"side_to_move\":\"{}\",\"ply\":{},\"game_ply\":{},\"effective_game_turns_played\":{},\"no_progress_plies\":{},\"move\":\"{}\",",
                escape_json(run_file),
                game_idx,
                escape_json(name),
                escape_json(&row.fen),
                side_text(row.side),
                row.ply,
                row.ply,
                row.ply,
                row.no_progress,
                escape_json(&row.best_move)
            )
            .map_err(|error| format!("{error}"))?;
            if let Some(raw) = row.fast_move_raw {
                write!(writer, "\"fast_move_raw\":{},", raw).map_err(|error| format!("{error}"))?;
            }
            writeln!(
                writer,
                "\"score\":{},\"mean_score\":{},\"mean_clipped_score\":{},\"result\":{},\"mean_result\":{},\"result_bucket\":{},\"completed_depth\":{},\"mean_completed_depth\":{},\"nodes\":{},\"elapsed_ms\":{},\"caused_ejection\":{},\"ejection_rate\":{},\"occurrence_count\":1,\"sample_weight\":1.0}}",
                row.score,
                row.score,
                clipped,
                result,
                result,
                result,
                row.depth,
                row.depth,
                row.nodes,
                row.elapsed_ms,
                if row.caused_ejection { "true" } else { "false" },
                if row.caused_ejection { "1.0" } else { "0.0" },
            )
            .map_err(|error| format!("{error}"))?;
        }
    }
    writer.flush().map_err(|error| format!("{error}"))?;
    Ok(())
}

#[derive(Clone, Debug)]
struct Summary {
    games: usize,
    local_wins: usize,
    draws: usize,
    github_wins: usize,
    local_points: f64,
    elo: f64,
    ci_low: f64,
    ci_high: f64,
    avg_plies: f64,
}

fn summarize(records: &[GameRecord], pair_points: &[f64]) -> Summary {
    let mut local_wins = 0;
    let mut draws = 0;
    let mut github_wins = 0;
    let mut local_points_sum = 0.0;
    let mut plies = 0u64;

    for record in records {
        local_points_sum += record.local_points;
        plies += u64::from(record.plies);
        match record.local_points {
            1.0 => local_wins += 1,
            0.5 => draws += 1,
            _ => github_wins += 1,
        }
    }

    let games = records.len();
    let elo = elo(local_points_sum, games as f64);
    let (ci_low, ci_high) = elo_ci(pair_points);
    Summary {
        games,
        local_wins,
        draws,
        github_wins,
        local_points: local_points_sum,
        elo,
        ci_low,
        ci_high,
        avg_plies: div(plies as f64, games as f64),
    }
}

fn elo_ci(pair_points: &[f64]) -> (f64, f64) {
    if pair_points.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = Rng::new(0x51e1_f00d);
    let mut samples = Vec::with_capacity(BOOTSTRAPS);
    for _ in 0..BOOTSTRAPS {
        let mut points = 0.0;
        for _ in 0..pair_points.len() {
            points += pair_points[rng.usize(pair_points.len())];
        }
        samples.push(elo(points, (pair_points.len() * 2) as f64));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (
        samples[((BOOTSTRAPS as f64) * 0.025) as usize],
        samples[((BOOTSTRAPS as f64) * 0.975) as usize],
    )
}

fn elo(points: f64, games: f64) -> f64 {
    let p = ((points + 0.5) / (games + 1.0)).clamp(0.000001, 0.999999);
    400.0 * (p / (1.0 - p)).log10()
}

fn print_summary(
    args: &Args,
    built: &BuiltEngines,
    github: &EngineStats,
    local: &EngineStats,
    summary: &Summary,
    elapsed_s: f64,
) {
    println!("github_ref,{}", args.github_ref);
    println!("github_bin,{}", built.github.display());
    println!("local_bin,{}", built.local.display());
    if let Some(root) = &built.temp_root {
        println!("temp,{}", root.display());
    }
    println!("openings,{}", args.openings.display());
    println!("pairs,{}", args.pairs);
    println!("games,{}", summary.games);
    println!("time_ms,{}", args.time_ms);
    println!("depth,{}", args.depth);
    println!("elapsed_s,{elapsed_s:.3}");
    println!();
    println!("result,local_vs_github");
    println!(
        "w_d_l,{}-{}-{}",
        summary.local_wins, summary.draws, summary.github_wins
    );
    println!(
        "score,{:.1}/{:.0}",
        summary.local_points, summary.games as f64
    );
    println!(
        "score_pct,{:.3}",
        100.0 * div(summary.local_points, summary.games as f64)
    );
    println!("elo,{:.1}", summary.elo);
    println!("elo_95_ci,{:.1},{:.1}", summary.ci_low, summary.ci_high);
    println!("avg_plies,{:.2}", summary.avg_plies);
    println!();
    print_engine("github", github);
    print_engine("local", local);
}

fn print_engine(name: &str, stats: &EngineStats) {
    println!("{name}_moves,{}", stats.moves);
    println!("{name}_avg_depth,{:.2}", stats.avg_depth());
    println!("{name}_avg_nps,{:.0}", stats.avg_nps());
    println!("{name}_avg_response_ms,{:.2}", stats.avg_response_ms());
    println!("{name}_illegal,{}", stats.illegal);
    println!("{name}_errors,{}", stats.errors);
}

fn render_source_engine(source: &Path) -> String {
    SOURCE_ENGINE.replace("__SOURCE__", &rust_string_literal(source))
}

fn rust_string_literal(path: &Path) -> String {
    let text = path.to_string_lossy();
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn outcome_json(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Win(Color::Black, _) => "black_win",
        Outcome::Win(Color::White, _) => "white_win",
        Outcome::Draw(_) => "draw_equal_captures",
    }
}

fn board_text(position: &Position) -> String {
    position
        .to_string()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_owned()
}

fn side_text(color: Color) -> &'static str {
    match color {
        Color::Black => "b",
        Color::White => "w",
    }
}

fn engine_name(id: EngineId) -> &'static str {
    match id {
        EngineId::Github => "github",
        EngineId::Local => "local",
        EngineId::SelfplayA | EngineId::SelfplayB => "selfplay",
    }
}

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}

fn temp_root(prefix: &str) -> PathBuf {
    env::temp_dir().join(format!("{prefix}-{}-{}", std::process::id(), unix_ms()))
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn cleanup(keep_temp: bool, root: Option<PathBuf>) {
    if !keep_temp {
        if let Some(root) = root {
            let _ = fs::remove_dir_all(root);
        }
    }
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        items.swap(i, rng.usize(i + 1));
    }
}

#[derive(Clone, Debug)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
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

const NATIVE_ENGINE: &str = r#"
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::time::Instant;

use steinbeisser::search::{
    search_fixed_depth_with_turn, search_timed_depth_with_turn, search_timed_with_turn,
};
use steinbeisser::{Move, Position, PositionState};

const MAX_PIECES: usize = Position::MAX_PIECES_PER_SIDE;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.map_err(|error| format!("{error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = search_line(&line)?;
        writeln!(out, "{reply}").map_err(|error| format!("{error}"))?;
        out.flush().map_err(|error| format!("{error}"))?;
    }
    Ok(())
}

fn search_line(line: &str) -> Result<String, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 8 {
        return Err(format!("input needs 8 fields, got {}", fields.len()));
    }
    let position = Position::from_str(line).map_err(|error| format!("{error}"))?;
    let black_score: usize = fields[1].parse().map_err(|_| "bad black score".to_owned())?;
    let white_score: usize = fields[2].parse().map_err(|_| "bad white score".to_owned())?;
    let no_progress: u16 = fields[4].parse().map_err(|_| "bad no-progress".to_owned())?;
    let ply: u16 = fields[5].parse().map_err(|_| "bad ply".to_owned())?;
    let time_ms: u64 = fields[6].parse().map_err(|_| "bad time".to_owned())?;
    let depth: u8 = fields[7].parse().map_err(|_| "bad depth".to_owned())?;
    let start = Instant::now();
    let result = match (time_ms, depth) {
        (0, 0) => return Err("time and depth cannot both be zero".to_owned()),
        (0, depth) => search_fixed_depth_with_turn(&position, &[], no_progress, ply, depth, None),
        (time_ms, 0) => search_timed_with_turn(&position, &[], no_progress, ply, time_ms, None),
        (time_ms, depth) => {
            search_timed_depth_with_turn(&position, &[], no_progress, ply, time_ms, depth, None)
        }
    }?;
    let best_move = result.best_move.ok_or_else(|| "no legal move".to_owned())?;
    let fast_move_raw = compact_fast_move_raw(&position, &best_move)?;
    let mut state = PositionState::new(position).map_err(|error| format!("{error:?}"))?;
    state.apply_move(&best_move).map_err(|error| format!("{error:?}"))?;
    let next = canonical_position(state.position())?;
    let next_black_score = MAX_PIECES.saturating_sub(next.white().len());
    let next_white_score = MAX_PIECES.saturating_sub(next.black().len());
    let next_no_progress = if next_black_score + next_white_score > black_score + white_score {
        0
    } else {
        no_progress.saturating_add(1)
    };
    Ok(format!(
        "{} {} {} {} {} {} {} {} {} {} {} {}",
        board_text(&next),
        next_black_score,
        next_white_score,
        side_text(next.side_to_move()),
        next_no_progress,
        ply.saturating_add(1),
        result.score,
        result.depth,
        result.nodes,
        start.elapsed().as_millis(),
        best_move,
        fast_move_raw
    ))
}

fn compact_fast_move_raw(position: &Position, mv: &Move) -> Result<u16, String> {
    let state = PositionState::new(position.clone()).map_err(|error| format!("{error:?}"))?;
    for (index, candidate) in state.generate_legal_moves().iter().enumerate() {
        if candidate == mv {
            return Ok(index as u16);
        }
    }
    Err(format!("best move {mv} is not legal"))
}

fn canonical_position(position: &Position) -> Result<Position, String> {
    Position::new(
        position.side_to_move(),
        position.black().to_vec(),
        position.white().to_vec(),
    )
    .map_err(|error| format!("{error}"))
}

fn board_text(position: &Position) -> String {
    position
        .to_string()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_owned()
}

fn side_text(color: steinbeisser::Color) -> &'static str {
    match color {
        steinbeisser::Color::Black => "b",
        steinbeisser::Color::White => "w",
    }
}
"#;

const SOURCE_ENGINE: &str = r#"
#[path = __SOURCE__]
mod bot;

pub mod ac {
    pub use crate::bot::ac::*;
}

pub mod ar {
    pub use crate::bot::ar::*;
}

use std::io::{self, BufRead, Write};
use std::time::Instant;

use bot::ac::{gm, Co, Coord, Mv, Po};
use bot::ar::Rq;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.map_err(|error| format!("{error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = search_line(&line)?;
        writeln!(out, "{reply}").map_err(|error| format!("{error}"))?;
        out.flush().map_err(|error| format!("{error}"))?;
    }
    Ok(())
}

fn search_line(line: &str) -> Result<String, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 8 {
        return Err(format!("input needs 8 fields, got {}", fields.len()));
    }
    let position = parse_position(fields[0], fields[3])?;
    let black_score: usize = fields[1].parse().map_err(|_| "bad black score".to_owned())?;
    let white_score: usize = fields[2].parse().map_err(|_| "bad white score".to_owned())?;
    let no_progress: u16 = fields[4].parse().map_err(|_| "bad no-progress".to_owned())?;
    let ply: u16 = fields[5].parse().map_err(|_| "bad ply".to_owned())?;
    let time_ms: u64 = fields[6].parse().map_err(|_| "bad time".to_owned())?;
    let depth: u8 = fields[7].parse().map_err(|_| "bad depth".to_owned())?;
    let start = Instant::now();
    let result = match (time_ms, depth) {
        (0, 0) => return Err("time and depth cannot both be zero".to_owned()),
        (0, depth) => bot::sp_depth_with_gt(&position, &[], no_progress, ply, depth),
        (time_ms, 0) => bot::sp_with_gt(&position, &[], no_progress, ply, time_ms),
        (time_ms, depth) => {
            bot::sp_depth_with_gt(&position, &[], no_progress, ply, depth)
                .or_else(|_| bot::sp_with_gt(&position, &[], no_progress, ply, time_ms))
        }
    }?;
    let best_move = result.bm.ok_or_else(|| "no legal move".to_owned())?;
    let fast_move_raw = compact_fast_move_raw(&position, &best_move)?;
    let mut state = Rq::new(position).map_err(|error| format!("{error}"))?;
    state.apply_move(&best_move).map_err(|error| format!("{error:?}"))?;
    let next = canonical_position(state.position())?;
    let next_black_score = Po::MAX_PIECES_PER_SIDE.saturating_sub(next.white().len());
    let next_white_score = Po::MAX_PIECES_PER_SIDE.saturating_sub(next.black().len());
    let next_no_progress = if next_black_score + next_white_score > black_score + white_score {
        0
    } else {
        no_progress.saturating_add(1)
    };
    Ok(format!(
        "{} {} {} {} {} {} {} {} {} {} {} {}",
        board_text(&next),
        next_black_score,
        next_white_score,
        side_text(next.side_to_move()),
        next_no_progress,
        ply.saturating_add(1),
        result.score,
        result.dp,
        result.nodes,
        start.elapsed().as_millis(),
        best_move,
        fast_move_raw
    ))
}

fn compact_fast_move_raw(position: &Po, mv: &Mv) -> Result<u16, String> {
    let state = Rq::new(position.clone()).map_err(|error| format!("{error}"))?;
    for (index, candidate) in state.generate_legal_moves().iter().enumerate() {
        if candidate == mv {
            return Ok(index as u16);
        }
    }
    Err(format!("best move {mv} is not legal"))
}

fn canonical_position(position: &Po) -> Result<Po, String> {
    Po::new(
        position.side_to_move(),
        position.black().to_vec(),
        position.white().to_vec(),
    )
    .map_err(|error| format!("{error:?}"))
}

fn parse_position(board: &str, side: &str) -> Result<Po, String> {
    let side_to_move = Co::parse(side).ok_or_else(|| "bad side to move".to_owned())?;
    let mut black = Vec::new();
    let mut white = Vec::new();
    let rows = board.split('/').collect::<Vec<_>>();
    if rows.len() != 9 {
        return Err(format!("board has {} rows", rows.len()));
    }
    for (row_index, row_text) in rows.iter().enumerate() {
        let row = row_index as u8;
        let expected = Coord::row_length(row).ok_or_else(|| "bad row".to_owned())?;
        let mut column = 1_u8;
        let mut chars = row_text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch.is_ascii_digit() {
                let mut count = ch.to_digit(10).unwrap() as u8;
                while let Some(next) = chars.peek().copied() {
                    if !next.is_ascii_digit() {
                        break;
                    }
                    chars.next();
                    count = count
                        .saturating_mul(10)
                        .saturating_add(next.to_digit(10).unwrap() as u8);
                }
                if count == 0 {
                    return Err("zero empty run".to_owned());
                }
                column = column.saturating_add(count);
                continue;
            }
            let coord = Coord::new(row, column).ok_or_else(|| "bad board column".to_owned())?;
            let cell = gm().index_of_coord(coord).ok_or_else(|| "bad board cell".to_owned())?;
            match ch {
                'S' => black.push(cell),
                's' => white.push(cell),
                _ => return Err(format!("bad board cell {ch}")),
            }
            column = column.saturating_add(1);
        }
        if column.saturating_sub(1) != expected {
            return Err("bad row length".to_owned());
        }
    }
    Po::new(side_to_move, black, white).map_err(|error| format!("{error:?}"))
}

fn board_text(position: &Po) -> String {
    let mut rows = Vec::with_capacity(9);
    for row in Coord::MIN_ROW..=Coord::MAX_ROW {
        let mut row_text = String::new();
        let mut empty_count = 0_u8;
        for column in 1..=Coord::row_length(row).unwrap() {
            let coord = Coord::new(row, column).unwrap();
            let cell = gm().index_of_coord(coord).unwrap();
            let marble = match position.occupant(cell) {
                Some(Co::Black) => Some('S'),
                Some(Co::White) => Some('s'),
                None => None,
            };
            if let Some(marble) = marble {
                if empty_count > 0 {
                    row_text.push_str(&empty_count.to_string());
                    empty_count = 0;
                }
                row_text.push(marble);
            } else {
                empty_count += 1;
            }
        }
        if empty_count > 0 {
            row_text.push_str(&empty_count.to_string());
        }
        rows.push(row_text);
    }
    rows.join("/")
}

fn side_text(color: Co) -> &'static str {
    match color {
        Co::Black => "b",
        Co::White => "w",
    }
}
"#;
