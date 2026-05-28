use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;

#[derive(Debug)]
struct ScreenMatchArgs {
    selfplay_bin: PathBuf,
    repo: PathBuf,
    candidate: PathBuf,
    baseline: PathBuf,
    openings: PathBuf,
    games: usize,
    time_ms: u64,
    seed: u64,
    github_ref: String,
    allow_local_failure: bool,
    allow_baseline_failure: bool,
}

#[derive(Serialize)]
struct MatchSummary {
    wins: usize,
    draws: usize,
    losses: usize,
    elo: f64,
    elo_lower: f64,
    elo_upper: f64,
    forfeit: bool,
}

pub(super) fn run_screen_match_command<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_screen_match_args(args)?;
    let summary = run_screen_match(&args)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn parse_screen_match_args<I>(args: I) -> Result<ScreenMatchArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut selfplay_bin = None::<PathBuf>;
    let mut repo = None::<PathBuf>;
    let mut candidate = None::<PathBuf>;
    let mut baseline = None::<PathBuf>;
    let mut openings = None::<PathBuf>;
    let mut games = None::<usize>;
    let mut time_ms = None::<u64>;
    let mut seed = None::<u64>;
    let mut github_ref = "baseline".to_owned();
    let mut allow_local_failure = false;
    let mut allow_baseline_failure = false;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--selfplay-bin" => {
                selfplay_bin = Some(PathBuf::from(required_value(&mut args, "--selfplay-bin")?))
            }
            "--repo" => repo = Some(PathBuf::from(required_value(&mut args, "--repo")?)),
            "--candidate" => candidate = Some(PathBuf::from(required_value(&mut args, &flag)?)),
            "--baseline" => baseline = Some(PathBuf::from(required_value(&mut args, &flag)?)),
            "--openings" => {
                openings = Some(PathBuf::from(required_value(&mut args, "--openings")?))
            }
            "--games" => games = Some(parse_value(&required_value(&mut args, "--games")?, &flag)?),
            "--time-ms" => time_ms = Some(parse_value(&required_value(&mut args, &flag)?, &flag)?),
            "--seed" => seed = Some(parse_value(&required_value(&mut args, "--seed")?, &flag)?),
            "--github-ref" => github_ref = required_value(&mut args, "--github-ref")?,
            "--allow-local-failure" => allow_local_failure = true,
            "--allow-baseline-failure" => allow_baseline_failure = true,
            _ => bail!(
                "unknown argument {flag}; usage: nnue screen-match --selfplay-bin <nnue-selfplay> --repo <repo> --candidate <bin> --baseline <bin> --openings <fen> --games <n> --time-ms <ms>"
            ),
        }
    }
    Ok(ScreenMatchArgs {
        selfplay_bin: selfplay_bin
            .ok_or_else(|| anyhow::anyhow!("missing required --selfplay-bin"))?,
        repo: repo.ok_or_else(|| anyhow::anyhow!("missing required --repo"))?,
        candidate: candidate.ok_or_else(|| anyhow::anyhow!("missing required --candidate"))?,
        baseline: baseline.ok_or_else(|| anyhow::anyhow!("missing required --baseline"))?,
        openings: openings.ok_or_else(|| anyhow::anyhow!("missing required --openings"))?,
        games: games.ok_or_else(|| anyhow::anyhow!("missing required --games"))?,
        time_ms: time_ms.ok_or_else(|| anyhow::anyhow!("missing required --time-ms"))?,
        seed: seed.unwrap_or(1),
        github_ref,
        allow_local_failure,
        allow_baseline_failure,
    })
}

fn required_value<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn parse_value<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("bad value for {flag}: {value}"))
}

fn run_screen_match(args: &ScreenMatchArgs) -> Result<MatchSummary> {
    let output = Command::new(&args.selfplay_bin)
        .arg("match")
        .arg("--repo")
        .arg(&args.repo)
        .arg("--github-bin")
        .arg(&args.baseline)
        .arg("--local-bin")
        .arg(&args.candidate)
        .arg("--openings")
        .arg(&args.openings)
        .arg("--pairs")
        .arg((args.games / 2).to_string())
        .arg("--time")
        .arg(args.time_ms.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--github-ref")
        .arg(&args.github_ref)
        .output()
        .with_context(|| format!("failed to run {}", args.selfplay_bin.display()))?;
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        return parse_match_summary(&text);
    }
    if let Some(summary) = failed_match_result(&text, args) {
        return Ok(summary);
    }
    bail!("selfplay match failed with status {}", output.status)
}

fn parse_match_summary(text: &str) -> Result<MatchSummary> {
    let mut fields = BTreeMap::<&str, &str>::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(',') {
            fields.insert(key.trim(), value.trim());
        }
    }
    let wdl = fields
        .get("w_d_l")
        .ok_or_else(|| anyhow::anyhow!("match output is missing w_d_l"))?;
    let mut parts = wdl.split('-');
    let wins = parse_usize(parts.next(), "wins")?;
    let draws = parse_usize(parts.next(), "draws")?;
    let losses = parse_usize(parts.next(), "losses")?;
    if parts.next().is_some() {
        bail!("bad w_d_l field {wdl}");
    }
    let elo = parse_f64(fields.get("elo").copied(), "elo")?;
    let ci = fields
        .get("elo_95_ci")
        .ok_or_else(|| anyhow::anyhow!("match output is missing elo_95_ci"))?;
    let (lower, upper) = ci
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("bad elo_95_ci field {ci}"))?;
    Ok(MatchSummary {
        wins,
        draws,
        losses,
        elo,
        elo_lower: lower.parse().context("bad elo_95_ci lower bound")?,
        elo_upper: upper.parse().context("bad elo_95_ci upper bound")?,
        forfeit: false,
    })
}

fn failed_match_result(text: &str, args: &ScreenMatchArgs) -> Option<MatchSummary> {
    let lower = text.to_lowercase();
    let local_path = args.candidate.to_string_lossy().to_lowercase();
    let baseline_path = args.baseline.to_string_lossy().to_lowercase();
    let local_failed = lower.contains("illegal engine reply from local")
        || lower.contains("local exited")
        || lower.contains("local broken pipe")
        || lower.contains("local write failed")
        || lower.contains("local read failed")
        || lower.contains("local bad reply")
        || match_output_count(&lower, "local_illegal") > 0
        || match_output_count(&lower, "local_errors") > 0
        || (lower.contains(&local_path) && path_failure_marker(&lower));
    let baseline_failed = lower.contains("illegal engine reply from github")
        || lower.contains("github exited")
        || lower.contains("github broken pipe")
        || lower.contains("github write failed")
        || lower.contains("github read failed")
        || lower.contains("github bad reply")
        || match_output_count(&lower, "github_illegal") > 0
        || match_output_count(&lower, "github_errors") > 0
        || (lower.contains(&baseline_path) && path_failure_marker(&lower));
    if local_failed {
        return args
            .allow_local_failure
            .then(|| forfeit_match_result(0, 0, args.games));
    }
    if baseline_failed {
        return args
            .allow_baseline_failure
            .then(|| forfeit_match_result(args.games, 0, 0));
    }
    if unknown_engine_failure(&lower) {
        if args.allow_local_failure && !args.allow_baseline_failure {
            return Some(forfeit_match_result(0, 0, args.games));
        }
        if args.allow_baseline_failure && !args.allow_local_failure {
            return Some(forfeit_match_result(args.games, 0, 0));
        }
        if args.allow_local_failure && args.allow_baseline_failure {
            return Some(forfeit_match_result(0, 0, args.games));
        }
    }
    None
}

fn forfeit_match_result(wins: usize, draws: usize, losses: usize) -> MatchSummary {
    let games = wins + draws + losses;
    let points = wins as f64 + 0.5 * draws as f64;
    let elo = elo_from_points(points, games);
    MatchSummary {
        wins,
        draws,
        losses,
        elo,
        elo_lower: elo,
        elo_upper: elo,
        forfeit: true,
    }
}

fn elo_from_points(points: f64, games: usize) -> f64 {
    if games == 0 {
        return 0.0;
    }
    let score = ((points + 0.5) / (games as f64 + 1.0)).clamp(0.001, 0.999);
    -400.0 * ((1.0 / score) - 1.0).log10()
}

fn match_output_count(text: &str, key: &str) -> usize {
    let prefix = format!("{key},");
    text.lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn path_failure_marker(text: &str) -> bool {
    [
        "no such file",
        "permission denied",
        "exec format error",
        "text file busy",
        "bad cpu type",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn unknown_engine_failure(text: &str) -> bool {
    [
        "bad reply",
        "write failed",
        "read failed",
        "broken pipe",
        "engine reply needs",
        "bad reply score",
        "bad reply depth",
        "bad reply nodes",
        "bad reply elapsed_ms",
        "bad reply fast_move_raw",
        "reported move does not produce reported position",
        "ply did not increment by one",
        "bad no-progress ply",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn parse_usize(value: Option<&str>, label: &str) -> Result<usize> {
    value
        .ok_or_else(|| anyhow::anyhow!("missing {label} in w_d_l"))?
        .parse()
        .with_context(|| format!("bad {label} in w_d_l"))
}

fn parse_f64(value: Option<&str>, label: &str) -> Result<f64> {
    value
        .ok_or_else(|| anyhow::anyhow!("match output is missing {label}"))?
        .parse()
        .with_context(|| format!("bad {label}"))
}
