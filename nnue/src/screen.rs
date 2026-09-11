use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;
#[path = "match_protocol.rs"]
#[allow(dead_code)] // Emission is used by the selfplay binary.
mod protocol;
use protocol::{Actor, MatchEvent, PREFIX};

#[derive(Debug)]
struct ScreenMatchArgs {
    selfplay_bin: PathBuf,
    repo: PathBuf,
    candidate: PathBuf,
    baseline: PathBuf,
    openings: PathBuf,
    games: usize,
    parallel_games: usize,
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
    let mut parallel_games = 1usize;
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
            "--parallel-games" => {
                parallel_games = parse_value(&required_value(&mut args, &flag)?, &flag)?
            }
            "--seed" => seed = Some(parse_value(&required_value(&mut args, "--seed")?, &flag)?),
            "--github-ref" => github_ref = required_value(&mut args, "--github-ref")?,
            "--allow-local-failure" => allow_local_failure = true,
            "--allow-baseline-failure" => allow_baseline_failure = true,
            _ => bail!(
                "unknown argument {flag}; usage: nnue screen-match --selfplay-bin <nnue-selfplay> --repo <repo> --candidate <bin> --baseline <bin> --openings <fen> --games <n> --time-ms <ms> [--parallel-games <n>]"
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
        parallel_games,
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
    if args.games == 0 || !args.games.is_multiple_of(2) {
        bail!("--games must be a positive even number");
    }
    if args.parallel_games == 0 {
        bail!("--parallel-games must be positive");
    }
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
        .arg("--parallel-games")
        .arg(args.parallel_games.to_string())
        .arg("--time")
        .arg(args.time_ms.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--github-ref")
        .arg(&args.github_ref)
        .output()
        .with_context(|| format!("failed to run {}", args.selfplay_bin.display()))?;
    let text = String::from_utf8_lossy(&output.stdout);
    parse_events(&text, args, output.status.success()).with_context(|| {
        format!(
            "selfplay status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn parse_events(text: &str, args: &ScreenMatchArgs, success: bool) -> Result<MatchSummary> {
    let events: Vec<MatchEvent> = text
        .lines()
        .filter_map(|line| line.strip_prefix(PREFIX))
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    if success {
        let [
            MatchEvent::Complete {
                games,
                wins,
                draws,
                losses,
                elo,
                elo_lower,
                elo_upper,
                local_illegal,
                local_errors,
                github_illegal,
                github_errors,
            },
        ] = events.as_slice()
        else {
            bail!("expected exactly one complete structured match result");
        };
        if *games != args.games
            || wins
                .checked_add(*draws)
                .and_then(|n| n.checked_add(*losses))
                != Some(*games)
            || [
                *local_illegal,
                *local_errors,
                *github_illegal,
                *github_errors,
            ]
            .iter()
            .any(|&n| n != 0)
            || !elo.is_finite()
            || !elo_lower.is_finite()
            || !elo_upper.is_finite()
            || elo_lower > elo_upper
        {
            bail!("invalid or incomplete match result");
        }
        return Ok(MatchSummary {
            wins: *wins,
            draws: *draws,
            losses: *losses,
            elo: *elo,
            elo_lower: *elo_lower,
            elo_upper: *elo_upper,
            forfeit: false,
        });
    }
    // A forfeit requires explicit attribution at the failure site. Unknown or
    // conflicting failures must never be guessed from filenames or stderr text.
    let mut failed = None;
    for event in &events {
        match event {
            MatchEvent::Failure {
                actor: Some(actor), ..
            } if failed.is_none() || failed == Some(*actor) => failed = Some(*actor),
            _ => bail!("unattributed, conflicting or malformed failed match"),
        }
    }
    match failed {
        Some(Actor::Local) if args.allow_local_failure => {
            Ok(forfeit_match_result(0, 0, args.games))
        }
        Some(Actor::Github) if args.allow_baseline_failure => {
            Ok(forfeit_match_result(args.games, 0, 0))
        }
        _ => bail!("match failed; no authorized, unambiguous engine forfeit"),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    fn args() -> ScreenMatchArgs {
        ScreenMatchArgs {
            selfplay_bin: "selfplay".into(),
            repo: ".".into(),
            candidate: "candidate".into(),
            baseline: "baseline".into(),
            openings: "openings".into(),
            games: 2,
            parallel_games: 1,
            time_ms: 50,
            seed: 1,
            github_ref: "baseline".into(),
            allow_local_failure: true,
            allow_baseline_failure: true,
        }
    }
    fn line(event: MatchEvent) -> String {
        format!("{PREFIX}{}", serde_json::to_string(&event).unwrap())
    }
    #[test]
    fn accepts_only_complete_zero_error_counts() {
        let event = MatchEvent::Complete {
            games: 2,
            wins: 0,
            draws: 2,
            losses: 0,
            elo: 0.,
            elo_lower: 0.,
            elo_upper: 0.,
            local_illegal: 0,
            local_errors: 0,
            github_illegal: 0,
            github_errors: 0,
        };
        let text = line(event);
        assert_eq!(parse_events(&text, &args(), true).unwrap().draws, 2);
        assert!(parse_events(&(text.clone() + "\n" + &text), &args(), true).is_err());
        assert!(parse_events(&text.replace("\"games\":2", "\"games\":4"), &args(), true).is_err());
        assert!(
            parse_events(
                &text.replace("\"local_errors\":0", "\"local_errors\":1"),
                &args(),
                true
            )
            .is_err()
        );
    }
    #[test]
    fn never_guesses_anonymous_or_conflicting_failures() {
        assert!(parse_events("local bad reply", &args(), false).is_err());
        let local = line(MatchEvent::Failure {
            actor: Some(Actor::Local),
            kind: "reply".into(),
            message: "bad".into(),
        });
        assert_eq!(parse_events(&local, &args(), false).unwrap().losses, 2);
        let other = line(MatchEvent::Failure {
            actor: Some(Actor::Github),
            kind: "reply".into(),
            message: "bad".into(),
        });
        assert!(parse_events(&(local + "\n" + &other), &args(), false).is_err());
    }

    #[test]
    fn refuses_unknown_malformed_or_unauthorized_failures() {
        let local = line(MatchEvent::Failure {
            actor: Some(Actor::Local),
            kind: "reply".into(),
            message: "bad".into(),
        });
        let unknown = line(MatchEvent::Failure {
            actor: None,
            kind: "worker_panic".into(),
            message: "failed".into(),
        });
        assert!(parse_events(&unknown, &args(), false).is_err());
        assert!(parse_events(&(local.clone() + "\n" + &unknown), &args(), false).is_err());
        assert!(parse_events(&(local.clone() + "\n" + PREFIX + "{"), &args(), false).is_err());
        assert!(parse_events(&local, &args(), true).is_err());
        let mut unauthorized = args();
        unauthorized.allow_local_failure = false;
        assert!(parse_events(&local, &unauthorized, false).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn failed_subprocess_attribution_comes_only_from_structured_stdout() -> Result<()> {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!(
            "steinbeisser-protocol-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let script = root.join("synthetic-selfplay");
        let local = line(MatchEvent::Failure {
            actor: Some(Actor::Local),
            kind: "spawn".into(),
            message: "missing candidate".into(),
        });
        // This fixture is a protocol-only process: no engine or game is run.
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' '{local}'\nprintf '%s\\n' 'github exited; local healthy' >&2\nexit 1\n"
            ),
        )?;
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700))?;
        let mut args = args();
        args.selfplay_bin = script.clone();
        let result = run_screen_match(&args)?;
        assert!(result.forfeit);
        assert_eq!((result.wins, result.draws, result.losses), (0, 0, 2));
        fs::write(
            &script,
            "#!/bin/sh\nprintf '%s\\n' 'local bad reply' >&2\nexit 1\n",
        )?;
        assert!(run_screen_match(&args).is_err());
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
