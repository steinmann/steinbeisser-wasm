#[allow(dead_code, hidden_glob_reexports, private_interfaces)]
#[path = "../../../engine/src/board.rs"]
mod board;
#[allow(dead_code, hidden_glob_reexports, private_interfaces)]
#[path = "../../../engine/src/eval.rs"]
mod eval;
#[allow(dead_code)]
#[path = "../materialize.rs"]
mod materialize;
#[allow(dead_code, hidden_glob_reexports, private_interfaces)]
#[path = "../../../engine/src/movegen.rs"]
mod movegen;
#[allow(dead_code)]
#[path = "../sample.rs"]
mod sample;
#[allow(dead_code, hidden_glob_reexports, private_interfaces)]
#[path = "../../../engine/src/search.rs"]
mod search;

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::str::FromStr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use board::{Color, Move, Position};
use movegen::PositionState;
use sample::BinarySample;
use search::MAX_GAME_TURNS;

const MAX_PIECES: usize = Position::MAX_PIECES_PER_SIDE;
const WIN_SCORE: usize = 6;
const BOOTSTRAPS: usize = 2000;
pub(crate) const DEFAULT_MAX_ABS_SCORE: i32 = 3500;

type NamedGames = Vec<(String, PlayedGame)>;
type GenerationResult = Result<(NamedGames, usize, EngineStats), String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Match,
    Generate,
    BuildRef,
}

#[derive(Clone, Debug)]
struct Args {
    mode: Mode,
    openings: PathBuf,
    pairs: usize,
    parallel_games: usize,
    target_samples: usize,
    time_ms: u64,
    time_set: bool,
    depth: u8,
    seed: u64,
    max_abs_score: Option<i32>,
    games_out: Option<PathBuf>,
    repo: PathBuf,
    github_ref: String,
    github_bin: Option<PathBuf>,
    local_bin: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::Match,
            openings: PathBuf::from("data/random100K.fen"),
            pairs: 100,
            parallel_games: 1,
            target_samples: 20_000,
            time_ms: 5,
            time_set: false,
            depth: 0,
            seed: 1,
            max_abs_score: Some(DEFAULT_MAX_ABS_SCORE),
            games_out: None,
            repo: PathBuf::new(),
            github_ref: "origin/main".to_owned(),
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
}

#[derive(Clone, Debug)]
pub(crate) struct EngineStats {
    pub(crate) moves: usize,
    pub(crate) nodes: u64,
    pub(crate) engine_ms: u64,
    pub(crate) response_ms: u128,
    pub(crate) depth_sum: u64,
    pub(crate) illegal: usize,
    pub(crate) errors: usize,
}

impl EngineStats {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn add(&mut self, other: &Self) {
        self.moves += other.moves;
        self.nodes = self.nodes.saturating_add(other.nodes);
        self.engine_ms = self.engine_ms.saturating_add(other.engine_ms);
        self.response_ms = self.response_ms.saturating_add(other.response_ms);
        self.depth_sum = self.depth_sum.saturating_add(other.depth_sum);
        self.illegal += other.illegal;
        self.errors += other.errors;
    }

    pub(crate) fn avg_depth(&self) -> f64 {
        div(self.depth_sum as f64, self.moves as f64)
    }

    pub(crate) fn avg_response_ms(&self) -> f64 {
        div(self.response_ms as f64, self.moves as f64)
    }

    pub(crate) fn avg_nps(&self) -> f64 {
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
    side: Color,
    fen: String,
    score: i32,
    depth: u8,
    nodes: u64,
    elapsed_ms: u64,
    no_progress: u16,
    caused_ejection: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Outcome {
    Win(Color, &'static str),
    Draw(&'static str),
}

#[derive(Clone, Debug)]
pub(crate) struct GameRecord {
    pub(crate) local_points: f64,
    outcome: Outcome,
    pub(crate) plies: u16,
}

#[derive(Clone, Debug)]
struct PlayedGame {
    record: GameRecord,
    rows: Vec<TrainingRow>,
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

#[derive(Clone, Debug)]
struct Rng(u64);

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
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
            "build-ref" => {
                args.mode = Mode::BuildRef;
                it.next();
            }
            _ => {}
        }
    }

    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--openings" => args.openings = next_path(&mut it, &flag)?,
            "--pairs" => args.pairs = next_value(&mut it, &flag)?,
            "--parallel-games" => args.parallel_games = next_value(&mut it, &flag)?,
            "--target-samples" => args.target_samples = next_value(&mut it, &flag)?,
            "--time" => {
                args.time_ms = next_value(&mut it, &flag)?;
                args.time_set = true;
            }
            "--seed" => args.seed = next_value(&mut it, &flag)?,
            "--max-abs-score" => args.max_abs_score = Some(next_value(&mut it, &flag)?),
            "--games-out" => args.games_out = Some(next_path(&mut it, &flag)?),
            "--repo" => args.repo = next_path(&mut it, &flag)?,
            "--github-ref" => args.github_ref = next_string(&mut it, &flag)?,
            "--github-bin" => args.github_bin = Some(next_path(&mut it, &flag)?),
            "--local-bin" => args.local_bin = Some(next_path(&mut it, &flag)?),
            "--help" => return Err(usage()),
            _ => return Err(format!("unknown option {flag}\n\n{}", usage())),
        }
    }

    if args.pairs == 0 || args.target_samples == 0 || args.parallel_games == 0 {
        return Err("counts must be greater than zero".to_owned());
    }
    if args.time_ms == 0 && args.depth == 0 {
        return Err("--time must be greater than zero".to_owned());
    }
    if matches!(args.max_abs_score, Some(limit) if limit < 0) {
        return Err("--max-abs-score must be non-negative".to_owned());
    }
    Ok(args)
}

fn usage() -> String {
    "usage:\n\
       selfplay match --repo . --github-bin <ref-bin> --local-bin <candidate-bin> --openings <fen> --pairs <n> --time <ms>\n\
       selfplay generate --repo . --local-bin <engine-bin> --openings <fen> --games-out <samples.sbin> --target-samples <n> --parallel-games <n> --max-abs-score 3500 --time <ms>\n\
       selfplay build-ref --repo . --github-ref <release-tag> --github-bin <ref-bin>\n"
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
    cleanup(Some(root));
    Ok(())
}

fn build_match_engines(args: &Args) -> Result<BuiltEngines, String> {
    let github = args
        .github_bin
        .clone()
        .ok_or_else(|| "match needs --github-bin <file>".to_owned())?;
    let local = args
        .local_bin
        .clone()
        .ok_or_else(|| "match needs --local-bin <file>".to_owned())?;
    Ok(BuiltEngines {
        github,
        local,
        temp_root: None,
    })
}

fn build_local_engine(args: &Args) -> Result<BuiltLocal, String> {
    let local = args
        .local_bin
        .clone()
        .ok_or_else(|| "generate needs --local-bin <file>".to_owned())?;
    Ok(BuiltLocal {
        local,
        temp_root: None,
    })
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

fn cleanup(root: Option<PathBuf>) {
    if let Some(root) = root {
        let _ = fs::remove_dir_all(root);
    }
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

fn build_native_engine(src: &Path, root: &Path, name: &str) -> Result<PathBuf, String> {
    materialize::build_native_engine(src, &root.join(format!("target-{name}")))
        .map_err(|error| format!("{error:#}"))
}

fn run_command(command: &mut Command) -> Result<(), String> {
    let status = command.status().map_err(|error| format!("{error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed with status {status:?}"))
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
        Mode::BuildRef => run_build_ref(&args),
    }
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
        pair_points.push(first.record.local_points + second.record.local_points);
        records.push(first.record);
        records.push(second.record);
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

    cleanup(built.temp_root);
    Ok(())
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

    write_fen_samples(out, &games)?;
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

    cleanup(built.temp_root);
    Ok(())
}

fn generate_serial(
    engine: &Path,
    openings: &[GameState],
    order: &[usize],
    args: &Args,
    limits: Limits,
) -> GenerationResult {
    let mut engine = EngineProcess::spawn("selfplay", engine)?;
    let mut games = Vec::<(String, PlayedGame)>::new();
    let mut sample_count = 0usize;

    while sample_count < args.target_samples {
        let opening_index = order[games.len() % order.len()];
        let opening = openings[opening_index].clone();
        let name = format!("fen-{opening_index:06}");
        let game = play_selfplay_game(&opening, &mut engine, limits, args.max_abs_score)?;
        sample_count += game.rows.len();
        games.push((name, game));
    }

    let mut stats = EngineStats::new();
    stats.add(&engine.stats);
    Ok((games, sample_count, stats))
}

fn generate_parallel(
    engine: &Path,
    openings: Vec<GameState>,
    order: Vec<usize>,
    args: &Args,
    limits: Limits,
) -> GenerationResult {
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
        let max_abs_score = args.max_abs_score;

        handles.push(thread::spawn(
            move || -> (Vec<(usize, String, PlayedGame)>, EngineStats) {
                let mut local_games = Vec::new();
                let mut stats = EngineStats::new();
                let result = (|| -> Result<(), String> {
                    let mut engine = EngineProcess::spawn("selfplay", &engine)?;
                    loop {
                        if stop.load(Ordering::Relaxed)
                            || sample_count.load(Ordering::Relaxed) >= target_samples
                        {
                            break;
                        }
                        let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                        if sample_count.load(Ordering::Relaxed) >= target_samples {
                            break;
                        }
                        let opening_index = order[game_index % order.len()];
                        let opening = openings[opening_index].clone();
                        let name = format!("fen-{opening_index:06}-w{worker:02}-{game_index:08}");
                        let game =
                            play_selfplay_game(&opening, &mut engine, limits, max_abs_score)?;
                        sample_count.fetch_add(game.rows.len(), Ordering::Relaxed);
                        local_games.push((game_index, name, game));
                    }
                    stats.add(&engine.stats);
                    Ok(())
                })();
                if let Err(message) = result {
                    stop.store(true, Ordering::Relaxed);
                    if let Ok(mut slot) = error.lock()
                        && slot.is_none()
                    {
                        *slot = Some(message);
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
        let reply = parse_reply(&line, start.elapsed().as_millis()).map_err(|error| {
            self.stats.errors += 1;
            format!("{} bad reply: {error}", self.name)
        })?;
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
    let _: u16 = fields[11]
        .parse()
        .map_err(|_| "bad reply fast_move_raw".to_owned())?;
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
        adjudicate(&before, &reply).map_err(|error| {
            engine_for(first, second, engine_id).stats.illegal += 1;
            format!(
                "illegal engine reply from {}: {error}",
                engine_name(engine_id)
            )
        })?;

        if keep_training_row(&reply, max_abs_score) {
            rows.push(TrainingRow {
                ply: before.ply,
                side,
                fen: before.fen(),
                score: reply.score,
                depth: reply.depth,
                nodes: reply.nodes,
                elapsed_ms: reply.elapsed_ms,
                no_progress: before.no_progress,
                caused_ejection: reply.state.total_score() > before.total_score(),
            });
        }
        state = reply.state;
    }
}

fn play_selfplay_game(
    opening: &GameState,
    engine: &mut EngineProcess,
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
                    local_points: 0.5,
                    outcome,
                    plies,
                },
                rows,
            });
        }

        let side = state.side();
        let before = state.clone();
        let reply = engine.search(&before, limits)?;
        adjudicate(&before, &reply).map_err(|error| {
            engine.stats.illegal += 1;
            format!("illegal engine reply from selfplay: {error}")
        })?;

        if keep_training_row(&reply, max_abs_score) {
            rows.push(TrainingRow {
                ply: before.ply,
                side,
                fen: before.fen(),
                score: reply.score,
                depth: reply.depth,
                nodes: reply.nodes,
                elapsed_ms: reply.elapsed_ms,
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
    if let Some(limit) = max_abs_score
        && i64::from(reply.score).abs() > i64::from(limit)
    {
        return false;
    }
    true
}

fn engine_for<'a>(
    first: &'a mut EngineProcess,
    second: &'a mut EngineProcess,
    id: EngineId,
) -> &'a mut EngineProcess {
    match id {
        EngineId::Github => first,
        EngineId::Local => second,
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

fn write_fen_samples(path: &Path, games: &[(String, PlayedGame)]) -> Result<(), String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !name.ends_with(&format!(".{}", sample::BINARY_SAMPLE_EXTENSION))
        && !name.ends_with(&format!(".{}.tmp", sample::BINARY_SAMPLE_EXTENSION))
    {
        return Err(format!(
            "selfplay generate writes only .{} sample shards",
            sample::BINARY_SAMPLE_EXTENSION
        ));
    }
    write_fen_samples_binary(path, games)
}

fn write_fen_samples_binary(path: &Path, games: &[(String, PlayedGame)]) -> Result<(), String> {
    let mut samples = Vec::<BinarySample>::new();
    for (_name, game) in games {
        for row in &game.rows {
            let state = GameState::parse(&row.fen)?;
            let result = match game.record.outcome {
                Outcome::Draw(_) => 0.0,
                Outcome::Win(color, _) if color == row.side => 1.0,
                Outcome::Win(_, _) => -1.0,
            };
            samples.push(BinarySample {
                black_bits: state.position.black_bits(),
                white_bits: state.position.white_bits(),
                side_to_move_is_black: row.side == Color::Black,
                ply: row.ply as f32,
                no_progress_plies: row.no_progress as f32,
                score: row.score as f32,
                clipped_score: row.score.clamp(-10_000, 10_000) as f32,
                result,
                result_bucket: result as i32,
                completed_depth: row.depth as f32,
                nodes: row.nodes,
                elapsed_ms: row.elapsed_ms,
                caused_ejection: row.caused_ejection,
                occurrence_count: 1,
                sample_weight: 1.0,
            });
        }
    }
    sample::write_samples(path, &samples).map_err(|error| format!("{}: {error}", path.display()))
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
    }
}

fn div(a: f64, b: f64) -> f64 {
    if b == 0.0 { 0.0 } else { a / b }
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        items.swap(i, rng.usize(i + 1));
    }
}

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
