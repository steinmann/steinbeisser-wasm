use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;

#[derive(Debug)]
struct MaterializeCandidateArgs {
    repo: PathBuf,
    reference_ref: String,
    model: PathBuf,
    source_dir: PathBuf,
    target: PathBuf,
    target_dir: PathBuf,
    candidate_id: String,
}

#[derive(Serialize)]
struct MaterializeCandidateReport {
    candidate_id: String,
    reference_ref: String,
    model: String,
    source_dir: String,
    target_dir: String,
    binary: String,
}

pub(super) fn run_materialize_candidate_command<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_materialize_candidate_args(args)?;
    let report = materialize_candidate(&args)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub fn build_native_engine(source: &Path, target_dir: &Path) -> Result<PathBuf> {
    let bin_dir = source.join("engine/src/bin");
    fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed to create {}", bin_dir.display()))?;
    fs::write(bin_dir.join("steinbeisser_native.rs"), NATIVE_ENGINE)
        .with_context(|| format!("failed to write native wrapper in {}", bin_dir.display()))?;

    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(source.join("engine/Cargo.toml"))
        .arg("--target-dir")
        .arg(target_dir)
        .arg("--release")
        .arg("--bin")
        .arg("steinbeisser_native");
    run_command(&mut command)?;

    let exe = if cfg!(windows) {
        "steinbeisser_native.exe"
    } else {
        "steinbeisser_native"
    };
    let built = target_dir.join("release").join(exe);
    if !built.is_file() {
        bail!("candidate build did not produce {}", built.display());
    }
    Ok(built)
}

fn parse_materialize_candidate_args<I>(args: I) -> Result<MaterializeCandidateArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut repo = None::<PathBuf>;
    let mut reference_ref = None::<String>;
    let mut model = None::<PathBuf>;
    let mut source_dir = None::<PathBuf>;
    let mut target = None::<PathBuf>;
    let mut target_dir = None::<PathBuf>;
    let mut candidate_id = None::<String>;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--repo" => repo = Some(PathBuf::from(required_value(&mut args, "--repo")?)),
            "--reference-ref" => reference_ref = Some(required_value(&mut args, &flag)?),
            "--model" => model = Some(PathBuf::from(required_value(&mut args, "--model")?)),
            "--source-dir" => {
                source_dir = Some(PathBuf::from(required_value(&mut args, "--source-dir")?))
            }
            "--target" => target = Some(PathBuf::from(required_value(&mut args, "--target")?)),
            "--target-dir" => {
                target_dir = Some(PathBuf::from(required_value(&mut args, "--target-dir")?))
            }
            "--candidate-id" => candidate_id = Some(required_value(&mut args, "--candidate-id")?),
            _ => bail!(
                "unknown argument {flag}; usage: nnue materialize-candidate --repo <repo> --reference-ref <ref> --model <model.nnq> --source-dir <dir> --target <bin> --target-dir <cargo-target>"
            ),
        }
    }
    let target = target.ok_or_else(|| anyhow::anyhow!("missing required --target"))?;
    let candidate_id = candidate_id.unwrap_or_else(|| {
        target
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("candidate")
            .to_owned()
    });
    Ok(MaterializeCandidateArgs {
        repo: repo.ok_or_else(|| anyhow::anyhow!("missing required --repo"))?,
        reference_ref: reference_ref
            .ok_or_else(|| anyhow::anyhow!("missing required --reference-ref"))?,
        model: model.ok_or_else(|| anyhow::anyhow!("missing required --model"))?,
        source_dir: source_dir.ok_or_else(|| anyhow::anyhow!("missing required --source-dir"))?,
        target,
        target_dir: target_dir.ok_or_else(|| anyhow::anyhow!("missing required --target-dir"))?,
        candidate_id,
    })
}

fn required_value<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn materialize_candidate(args: &MaterializeCandidateArgs) -> Result<MaterializeCandidateReport> {
    materialize_reference_source(&args.repo, &args.reference_ref, &args.source_dir)?;
    write_reference_net(&args.source_dir, &args.model)?;
    let built = build_native_engine(&args.source_dir, &args.target_dir)?;
    if let Some(parent) = args.target.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::copy(&built, &args.target).with_context(|| {
        format!(
            "failed to copy built binary {} to {}",
            built.display(),
            args.target.display()
        )
    })?;
    make_executable(&args.target)?;
    Ok(MaterializeCandidateReport {
        candidate_id: args.candidate_id.clone(),
        reference_ref: args.reference_ref.clone(),
        model: args.model.display().to_string(),
        source_dir: args.source_dir.display().to_string(),
        target_dir: args.target_dir.display().to_string(),
        binary: args.target.display().to_string(),
    })
}

fn materialize_reference_source(
    repo: &Path,
    reference_ref: &str,
    destination: &Path,
) -> Result<()> {
    remove_path(destination)?;
    fs::create_dir_all(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let archive = destination.with_extension("tar");
    remove_path(&archive)?;

    let mut git = Command::new("git");
    git.arg("-C")
        .arg(repo)
        .arg("archive")
        .arg("--format=tar")
        .arg("--output")
        .arg(&archive)
        .arg(reference_ref);
    run_command(&mut git)?;

    let mut tar = Command::new("tar");
    tar.arg("-xf").arg(&archive).arg("-C").arg(destination);
    run_command(&mut tar)?;
    remove_path(&archive)?;
    Ok(())
}

fn write_reference_net(source: &Path, model: &Path) -> Result<()> {
    let net = source.join("engine/src/net.mlp");
    if !net.is_file() {
        bail!("reference source is missing {}", net.display());
    }
    let payload =
        fs::read(model).with_context(|| format!("failed to read model {}", model.display()))?;
    fs::write(&net, encode_ascii85(&payload))
        .with_context(|| format!("failed to write {}", net.display()))?;
    Ok(())
}

fn encode_ascii85(payload: &[u8]) -> String {
    let mut encoded = String::new();
    let full_length = payload.len() / 4 * 4;
    for chunk in payload[..full_length].chunks_exact(4) {
        let mut value = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if value == 0 {
            encoded.push('z');
            continue;
        }
        let mut digits = [0_u32; 5];
        for digit in digits.iter_mut().rev() {
            *digit = value % 85;
            value /= 85;
        }
        for digit in digits {
            encoded.push(char::from_u32(digit + 33).expect("ascii85 digit"));
        }
    }
    let remainder = &payload[full_length..];
    if !remainder.is_empty() {
        let mut padded = [0_u8; 4];
        padded[..remainder.len()].copy_from_slice(remainder);
        let mut value = u32::from_be_bytes(padded);
        let mut digits = [0_u32; 5];
        for digit in digits.iter_mut().rev() {
            *digit = value % 85;
            value /= 85;
        }
        for digit in digits.into_iter().take(remainder.len() + 1) {
            encoded.push(char::from_u32(digit + 33).expect("ascii85 digit"));
        }
    }
    encoded
}

fn remove_path(path: &Path) -> Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else if path.is_file() {
        fs::remove_file(path).with_context(|| format!("failed to remove {}", path.display()))?;
    }
    Ok(())
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)
            .with_context(|| format!("failed to stat {}", path.display()))?
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("failed to chmod {}", path.display()))?;
    }
    Ok(())
}

fn run_command(command: &mut Command) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("failed to run {command:?}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    bail!(
        "command {command:?} failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout.trim_end(),
        stderr.trim_end()
    )
}

pub const NATIVE_ENGINE: &str = r#"
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
