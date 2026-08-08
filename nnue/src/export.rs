use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use crate::corpus::CorpusManifest;
use crate::sample;
use crate::tournament;

#[derive(Debug)]
struct ExportResultsArgs {
    summary: PathBuf,
    out_dir: PathBuf,
}

#[derive(Debug)]
struct ExportTrainingArgs {
    summary: PathBuf,
    out_dir: PathBuf,
    work_dir: PathBuf,
    corpus_dir: Option<PathBuf>,
    reference_ref: String,
}

pub(super) fn run_export_results_command<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_export_results_args(args)?;
    let summary = read_json(&args.summary)?;
    let result = export_results(&summary, &args.out_dir)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

pub(super) fn run_export_training_data_command<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_export_training_args(args)?;
    let summary = read_json(&args.summary)?;
    let result = export_training_data(&summary, &args)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn parse_export_results_args<I>(args: I) -> Result<ExportResultsArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut summary = None::<PathBuf>;
    let mut out_dir = None::<PathBuf>;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--summary" => summary = Some(PathBuf::from(required_value(&mut args, "--summary")?)),
            "--out-dir" => out_dir = Some(PathBuf::from(required_value(&mut args, "--out-dir")?)),
            _ => bail!(
                "unknown argument {flag}; usage: nnue export-results --summary <summary.json> --out-dir <dir>"
            ),
        }
    }
    Ok(ExportResultsArgs {
        summary: summary.ok_or_else(|| anyhow::anyhow!("missing required --summary"))?,
        out_dir: out_dir.ok_or_else(|| anyhow::anyhow!("missing required --out-dir"))?,
    })
}

fn parse_export_training_args<I>(args: I) -> Result<ExportTrainingArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut summary = None::<PathBuf>;
    let mut out_dir = None::<PathBuf>;
    let mut work_dir = None::<PathBuf>;
    let mut corpus_dir = None::<PathBuf>;
    let mut reference_ref = None::<String>;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--summary" => summary = Some(PathBuf::from(required_value(&mut args, "--summary")?)),
            "--out-dir" => out_dir = Some(PathBuf::from(required_value(&mut args, "--out-dir")?)),
            "--work-dir" => {
                work_dir = Some(PathBuf::from(required_value(&mut args, "--work-dir")?))
            }
            "--corpus-dir" => {
                corpus_dir = Some(PathBuf::from(required_value(&mut args, "--corpus-dir")?))
            }
            "--reference-ref" => {
                reference_ref = Some(required_value(&mut args, "--reference-ref")?)
            }
            _ => bail!(
                "unknown argument {flag}; usage: nnue export-positive-training-data --summary <summary.json> --out-dir <dir> --work-dir <dir> [--corpus-dir <dir>] --reference-ref <ref>"
            ),
        }
    }
    Ok(ExportTrainingArgs {
        summary: summary.ok_or_else(|| anyhow::anyhow!("missing required --summary"))?,
        out_dir: out_dir.ok_or_else(|| anyhow::anyhow!("missing required --out-dir"))?,
        work_dir: work_dir.ok_or_else(|| anyhow::anyhow!("missing required --work-dir"))?,
        corpus_dir,
        reference_ref: reference_ref
            .ok_or_else(|| anyhow::anyhow!("missing required --reference-ref"))?,
    })
}

fn required_value<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn export_results(summary: &Value, out_dir: &Path) -> Result<Value> {
    if summary.get("status").and_then(Value::as_str) != Some("completed") {
        return Ok(Value::Null);
    }
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;
    let json_path = out_dir.join("results.json");
    write_json(&json_path, summary)?;

    let mut result = Map::<String, Value>::new();
    result.insert("json".to_owned(), json!(json_path.display().to_string()));
    let standings = tournament::standings_from_summary(summary)?;
    if !standings.is_empty() {
        let markdown_path = out_dir.join("standings.md");
        fs::write(
            &markdown_path,
            tournament::tournament_table_lines(&standings).join("\n") + "\n",
        )
        .with_context(|| format!("failed to write {}", markdown_path.display()))?;
        result.insert(
            "standings".to_owned(),
            json!(markdown_path.display().to_string()),
        );
    }
    Ok(Value::Object(result))
}

fn export_training_data(summary: &Value, args: &ExportTrainingArgs) -> Result<Value> {
    if summary.get("status").and_then(Value::as_str) != Some("completed") {
        return Ok(Value::Null);
    }
    let standings = summary
        .get("standings")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("completed tournament summary is missing standings"))?;
    let reference_rows = standings
        .iter()
        .filter(|row| {
            row.get("is_reference")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let [reference_row] = reference_rows.as_slice() else {
        bail!(
            "completed tournament must contain exactly one reference row, found {}",
            reference_rows.len()
        );
    };
    let candidate_rows = standings
        .iter()
        .filter(|row| {
            !row.get("is_reference")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if candidate_rows.is_empty() {
        return Ok(json!({
            "status": "skipped",
            "reason": "no_candidate_rows",
            "reference_ref": args.reference_ref,
            "tournament": summary,
        }));
    }

    let source_row = tournament_winner(&candidate_rows)?;
    let release_candidate =
        f64_field(source_row, "elo_vs_latest") > f64_field(reference_row, "elo_vs_latest");
    let source_corpus = match &args.corpus_dir {
        Some(path) => path.clone(),
        None => candidate_corpus_dir(source_row, &args.work_dir)?,
    };
    let corpus_manifest = CorpusManifest::read(&source_corpus.join("manifest.json"))?;
    let winner_train_prefix_samples = int_field(source_row, "train_samples");
    if corpus_manifest.train.samples < winner_train_prefix_samples.max(0) as usize {
        bail!(
            "full corpus has {} training samples, fewer than winner prefix {}",
            corpus_manifest.train.samples,
            winner_train_prefix_samples
        );
    }
    copy_training_corpus(&source_corpus, &args.out_dir)?;
    let export = json!({
        "status": if release_candidate { "completed" } else { "retained" },
        "release_candidate": release_candidate,
        "training_dir": args.out_dir.display().to_string(),
        "reference_ref": args.reference_ref,
        "winner_player": source_row.get("player").cloned().unwrap_or(Value::Null),
        "winner_model": source_row.get("model").cloned().unwrap_or(Value::Null),
        "source_corpus_dir": source_corpus.display().to_string(),
        "train_prefix_samples": winner_train_prefix_samples,
        "corpus_train_samples": corpus_manifest.train.samples,
        "validation_samples": corpus_manifest.val.samples,
        "tournament_elo_vs_latest": f64_field(source_row, "elo_vs_latest"),
        "qval_loss": f64_field(source_row, "qval_loss"),
        "files": ["train.sbin", "val.sbin", "manifest.json"],
    });
    Ok(export)
}

fn tournament_winner<'a>(rows: &[&'a Value]) -> Result<&'a Value> {
    rows.iter()
        .copied()
        .max_by(|left, right| {
            f64_field(left, "elo_vs_latest")
                .total_cmp(&f64_field(right, "elo_vs_latest"))
                .then_with(|| {
                    f64_field(left, "score_pct").total_cmp(&f64_field(right, "score_pct"))
                })
                .then_with(|| int_field(left, "games").cmp(&int_field(right, "games")))
        })
        .ok_or_else(|| anyhow::anyhow!("completed tournament has no candidate rows"))
}

fn candidate_corpus_dir(row: &Value, work_dir: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::<PathBuf>::new();
    if let Some(raw) = row.get("corpus_dir").and_then(Value::as_str)
        && !raw.is_empty()
    {
        candidates.push(PathBuf::from(raw));
    }
    let cycle = int_field(row, "cycle");
    if cycle > 0 {
        candidates.push(work_dir.join(format!("cycle{cycle}_fen")));
    }
    for path in &candidates {
        if path.join("manifest.json").is_file()
            && path.join("train.sbin").is_file()
            && path.join("val.sbin").is_file()
        {
            return Ok(path.clone());
        }
    }
    let attempted = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    bail!(
        "missing positive net training corpus; tried {}",
        if attempted.is_empty() {
            "no candidate paths"
        } else {
            attempted.as_str()
        }
    )
}

fn copy_training_corpus(source: &Path, destination: &Path) -> Result<()> {
    let tmp = destination.with_file_name(format!(
        ".{}.tmp",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("training-data")
    ));
    if tmp.exists() {
        fs::remove_dir_all(&tmp).with_context(|| format!("failed to remove {}", tmp.display()))?;
    }
    fs::create_dir_all(&tmp).with_context(|| format!("failed to create {}", tmp.display()))?;
    let manifest_path = source.join("manifest.json");
    let mut manifest = CorpusManifest::read(&manifest_path)?;
    for split in ["train", "val"] {
        let split_manifest = manifest.split(split);
        let source_file = source.join(&split_manifest.file);
        if !source_file.is_file() {
            bail!(
                "training corpus manifest references missing {split} file {}",
                source_file.display()
            );
        }
        let actual_samples = sample::sample_count(&source_file)
            .with_context(|| format!("failed to validate {}", source_file.display()))?;
        if actual_samples != split_manifest.samples {
            bail!(
                "training corpus manifest claims {split} has {} samples, found {}",
                split_manifest.samples,
                actual_samples
            );
        }
        fs::copy(&source_file, tmp.join(&split_manifest.file))
            .with_context(|| format!("failed to copy {}", source_file.display()))?;
    }
    manifest.source_corpus_dir = Some(source.display().to_string());
    manifest.corpus_dir = destination.display().to_string();
    manifest.canonical_corpus_dir = destination.display().to_string();
    manifest.write(&tmp.join("manifest.json"))?;
    if destination.exists() {
        if destination.is_dir() {
            fs::remove_dir_all(destination)
                .with_context(|| format!("failed to remove {}", destination.display()))?;
        } else {
            fs::remove_file(destination)
                .with_context(|| format!("failed to remove {}", destination.display()))?;
        }
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::rename(&tmp, destination).with_context(|| {
        format!(
            "failed to move {} to {}",
            tmp.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")
        .with_context(|| format!("failed to write {}", path.display()))
}

fn int_field(row: &Value, key: &str) -> i64 {
    row.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn f64_field(row: &Value, key: &str) -> f64 {
    row.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}
