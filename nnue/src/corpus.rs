use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::*;
use crate::sample::{self, BinarySample};

const SPLIT_SHARD_SAMPLES: usize = 50_000;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CorpusManifest {
    pub format: String,
    pub feature_set: String,
    pub input_count: usize,
    pub max_active_features: usize,
    pub score_clip: i32,
    pub max_abs_score: i32,
    pub score_bucket_size: i32,
    pub ply_bucket_size: i32,
    pub train: CorpusSplit,
    pub val: CorpusSplit,
    pub train_class_counts: BTreeMap<String, usize>,
    pub corpus_dir: String,
    pub canonical_corpus_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_corpus_dir: Option<String>,
}

impl CorpusManifest {
    pub(crate) fn read(path: &Path) -> Result<Self> {
        serde_json::from_str(
            &fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))
    }

    pub(crate) fn write(&self, path: &Path) -> Result<()> {
        write_json(path, self)
    }

    pub(crate) fn split(&self, name: &str) -> &CorpusSplit {
        match name {
            "train" => &self.train,
            "val" => &self.val,
            _ => &self.train,
        }
    }

    pub(crate) fn class_count(&self, result_bucket: i32) -> usize {
        self.train_class_counts
            .get(&result_bucket.to_string())
            .copied()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CorpusSplit {
    pub file: String,
    pub samples: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shards: Vec<CorpusShard>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CorpusShard {
    pub file: String,
    pub samples: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CorpusBuildReport {
    pub samples: usize,
    pub corpus_dir: String,
    pub ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_samples: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_samples: Option<usize>,
}

#[derive(Debug)]
struct CorpusBuildArgs {
    shards_dir: PathBuf,
    work_dir: PathBuf,
    cycle: u32,
    max_samples: Option<usize>,
    validation_samples: usize,
    max_abs_score: i32,
    feature_set: String,
    input_count: usize,
    max_active_features: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ShardSignature {
    size: u64,
    mtime_ns: u128,
}

#[derive(Clone, Debug)]
struct AcceptedSample {
    key: String,
    record: BinarySample,
    result_bucket: i32,
}

#[derive(Default)]
struct SplitWriteReport {
    rows: usize,
    class_counts: BTreeMap<String, usize>,
}

pub(super) fn run_corpus_build_command<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_corpus_build_args(args)?;
    let report = build_corpus(&args)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_corpus_build_args<I>(args: I) -> Result<CorpusBuildArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut shards_dir = None::<PathBuf>;
    let mut work_dir = None::<PathBuf>;
    let mut cycle = None::<u32>;
    let mut max_samples = None::<usize>;
    let mut validation_samples = 10_000_usize;
    let mut max_abs_score = 3_500_i32;
    let mut feature_set = FEATURE_SET_NAME.to_owned();
    let mut input_count = INPUT_COUNT;
    let mut max_active_features = MAX_ACTIVE_FEATURES;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--shards" => shards_dir = Some(PathBuf::from(required_value(&mut args, "--shards")?)),
            "--work-dir" => {
                work_dir = Some(PathBuf::from(required_value(&mut args, "--work-dir")?))
            }
            "--cycle" => cycle = Some(parse_value(&required_value(&mut args, "--cycle")?, &flag)?),
            "--max-samples" => {
                max_samples = Some(parse_value(
                    &required_value(&mut args, "--max-samples")?,
                    &flag,
                )?)
            }
            "--validation-samples" => {
                validation_samples =
                    parse_value(&required_value(&mut args, "--validation-samples")?, &flag)?
            }
            "--max-abs-score" => {
                max_abs_score = parse_value(&required_value(&mut args, "--max-abs-score")?, &flag)?
            }
            "--feature-set" => feature_set = required_value(&mut args, "--feature-set")?,
            "--input-count" => {
                input_count = parse_value(&required_value(&mut args, "--input-count")?, &flag)?
            }
            "--max-active-features" => {
                max_active_features =
                    parse_value(&required_value(&mut args, "--max-active-features")?, &flag)?
            }
            _ => bail!(
                "unknown argument {flag}; usage: nnue corpus-build --shards <dir> --work-dir <dir> --cycle <n>"
            ),
        }
    }
    Ok(CorpusBuildArgs {
        shards_dir: shards_dir.ok_or_else(|| anyhow::anyhow!("missing required --shards"))?,
        work_dir: work_dir.ok_or_else(|| anyhow::anyhow!("missing required --work-dir"))?,
        cycle: cycle.ok_or_else(|| anyhow::anyhow!("missing required --cycle"))?,
        max_samples,
        validation_samples,
        max_abs_score,
        feature_set,
        input_count,
        max_active_features,
    })
}

fn build_corpus(args: &CorpusBuildArgs) -> Result<CorpusBuildReport> {
    if args.feature_set != FEATURE_SET_NAME {
        bail!("unsupported --feature-set {}", args.feature_set);
    }
    let data_dir = args.work_dir.join("corpus-data");
    fs::create_dir_all(&data_dir)?;
    fs::create_dir_all(&args.work_dir)?;

    let accepted_path = data_dir.join("accepted.sbin");
    let seen_path = data_dir.join("seen_keys.txt");
    let shard_state_path = data_dir.join("shards-v2.json");
    let mut signatures = load_shard_signatures(&shard_state_path)?;
    let mut seen = load_seen_keys(&seen_path)?;
    if signatures_invalidated(&args.shards_dir, &signatures)? {
        signatures.clear();
        seen.clear();
        remove_if_exists(&accepted_path)?;
        remove_if_exists(&seen_path)?;
        remove_if_exists(&shard_state_path)?;
    }

    let accepted_exists = accepted_path.is_file();
    let mut accepted_writer = BufWriter::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&accepted_path)?,
    );
    if !accepted_exists {
        sample::write_header(&mut accepted_writer)?;
    }
    let mut seen_writer = BufWriter::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&seen_path)?,
    );
    scan_shards(
        args,
        &mut signatures,
        &mut seen,
        &mut accepted_writer,
        &mut seen_writer,
    )?;
    accepted_writer.flush()?;
    seen_writer.flush()?;
    write_json(&shard_state_path, &signatures)?;

    let samples = load_accepted_prefix(&accepted_path, args.max_samples)?;
    let total_samples = samples.len();
    let corpus_dir = args.work_dir.join(format!("cycle{}_fen", args.cycle));
    if total_samples <= args.validation_samples {
        return Ok(CorpusBuildReport {
            samples: total_samples,
            corpus_dir: corpus_dir.display().to_string(),
            ready: false,
            train_samples: None,
            val_samples: None,
        });
    }

    let validation_keys = load_validation_keys(
        &args.work_dir.join("validation_keys.json"),
        &samples,
        args.validation_samples,
    )?;
    fs::create_dir_all(&corpus_dir)?;
    let train_source = data_dir.join("train.sbin");
    let val_source = data_dir.join("val.sbin");
    let train_report = write_split_prefix(&train_source, &samples, |sample| {
        !validation_keys.contains(&sample.key)
    })?;
    let val_report = write_split_prefix(&val_source, &samples, |sample| {
        validation_keys.contains(&sample.key)
    })?;
    let train_shards = write_split_shards(
        &data_dir.join("train-shards"),
        "train",
        &samples,
        |sample| !validation_keys.contains(&sample.key),
    )?;
    let val_shards = write_split_shards(&data_dir.join("val-shards"), "val", &samples, |sample| {
        validation_keys.contains(&sample.key)
    })?;
    link_cycle_file(&train_source, &corpus_dir.join("train.sbin"))?;
    link_cycle_file(&val_source, &corpus_dir.join("val.sbin"))?;
    link_cycle_dir(
        &data_dir.join("train-shards"),
        &corpus_dir.join("train-shards"),
    )?;
    link_cycle_dir(&data_dir.join("val-shards"), &corpus_dir.join("val-shards"))?;

    let manifest = CorpusManifest {
        format: "steinbeisser-fen-corpus-v1".to_owned(),
        feature_set: args.feature_set.clone(),
        input_count: args.input_count,
        max_active_features: args.max_active_features,
        score_clip: 10_000,
        max_abs_score: args.max_abs_score,
        score_bucket_size: 500,
        ply_bucket_size: 10,
        train: CorpusSplit {
            file: "train.sbin".to_owned(),
            samples: train_report.rows,
            source_file: Some(train_source.display().to_string()),
            shards: train_shards,
        },
        val: CorpusSplit {
            file: "val.sbin".to_owned(),
            samples: val_report.rows,
            source_file: Some(val_source.display().to_string()),
            shards: val_shards,
        },
        train_class_counts: train_report.class_counts,
        corpus_dir: corpus_dir.display().to_string(),
        canonical_corpus_dir: data_dir.display().to_string(),
        source_corpus_dir: None,
    };
    manifest.write(&corpus_dir.join("manifest.json"))?;
    Ok(CorpusBuildReport {
        samples: total_samples,
        corpus_dir: corpus_dir.display().to_string(),
        ready: true,
        train_samples: Some(train_report.rows),
        val_samples: Some(val_report.rows),
    })
}

fn scan_shards(
    args: &CorpusBuildArgs,
    signatures: &mut BTreeMap<String, ShardSignature>,
    seen: &mut HashSet<String>,
    accepted_writer: &mut BufWriter<fs::File>,
    seen_writer: &mut BufWriter<fs::File>,
) -> Result<()> {
    let paths = sorted_shard_files(&args.shards_dir)?;
    for path in paths {
        if args.max_samples.is_some_and(|limit| seen.len() >= limit) {
            break;
        }
        let signature = shard_signature(&path)?;
        let key = path.display().to_string();
        if signatures.get(&key).copied() == Some(signature) {
            continue;
        }
        let completed = scan_binary_shard(args, &path, seen, accepted_writer, seen_writer)?;
        if completed {
            signatures.insert(path.display().to_string(), signature);
        }
    }
    Ok(())
}

fn sorted_shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if !dir.is_dir() {
        return Ok(paths);
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str())
            == Some(sample::BINARY_SAMPLE_EXTENSION)
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn accept_record(
    args: &CorpusBuildArgs,
    record: BinarySample,
    seen: &mut HashSet<String>,
    accepted_writer: &mut BufWriter<fs::File>,
    seen_writer: &mut BufWriter<fs::File>,
) -> Result<bool> {
    let key = record.key();
    if seen.contains(&key)
        || terminal_search_record(&record)
        || high_abs_score_record(&record, args.max_abs_score)
    {
        return Ok(true);
    }
    seen.insert(key.clone());
    sample::write_record(accepted_writer, &record)?;
    seen_writer.write_all(key.as_bytes())?;
    seen_writer.write_all(b"\n")?;
    Ok(args.max_samples.is_none_or(|limit| seen.len() < limit))
}

fn scan_binary_shard(
    args: &CorpusBuildArgs,
    path: &Path,
    seen: &mut HashSet<String>,
    accepted_writer: &mut BufWriter<fs::File>,
    seen_writer: &mut BufWriter<fs::File>,
) -> Result<bool> {
    for record in sample::read_samples(path)
        .with_context(|| format!("failed to read binary shard {}", path.display()))?
    {
        if !accept_record(args, record, seen, accepted_writer, seen_writer)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn shard_signature(path: &Path) -> Result<ShardSignature> {
    let metadata = fs::metadata(path)?;
    let modified = metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Ok(ShardSignature {
        size: metadata.len(),
        mtime_ns: modified.as_nanos(),
    })
}

fn signatures_invalidated(
    shards_dir: &Path,
    signatures: &BTreeMap<String, ShardSignature>,
) -> Result<bool> {
    for (path, previous) in signatures {
        let path = Path::new(path);
        if !path.starts_with(shards_dir) || !path.exists() || shard_signature(path)? != *previous {
            return Ok(true);
        }
    }
    Ok(false)
}

fn load_shard_signatures(path: &Path) -> Result<BTreeMap<String, ShardSignature>> {
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn load_seen_keys(path: &Path) -> Result<HashSet<String>> {
    if !path.is_file() {
        return Ok(HashSet::new());
    }
    let file = fs::File::open(path)?;
    let mut seen = HashSet::new();
    for line in BufReader::new(file).lines() {
        let key = line?;
        if !key.is_empty() {
            seen.insert(key);
        }
    }
    Ok(seen)
}

fn load_accepted_prefix(path: &Path, max_samples: Option<usize>) -> Result<Vec<AcceptedSample>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let mut records = sample::read_samples(path)
        .with_context(|| format!("failed to read accepted samples {}", path.display()))?;
    if let Some(limit) = max_samples {
        records.truncate(limit);
    }
    let samples = records
        .into_iter()
        .map(|record| AcceptedSample {
            key: record.key(),
            result_bucket: record.result_bucket,
            record,
        })
        .collect();
    Ok(samples)
}

fn load_validation_keys(
    path: &Path,
    samples: &[AcceptedSample],
    validation_samples: usize,
) -> Result<HashSet<String>> {
    let available = samples
        .iter()
        .map(|sample| sample.key.as_str())
        .collect::<HashSet<_>>();
    let mut keys = Vec::<String>::new();
    if path.is_file() {
        let raw: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        if let Some(values) = raw.as_array() {
            for value in values {
                if let Some(key) = value.as_str()
                    && available.contains(key)
                {
                    keys.push(key.to_owned());
                }
            }
        }
    }
    let mut present = keys.iter().cloned().collect::<HashSet<_>>();
    if keys.len() < validation_samples {
        for sample in samples {
            if present.insert(sample.key.clone()) {
                keys.push(sample.key.clone());
                if keys.len() >= validation_samples {
                    break;
                }
            }
        }
    }
    keys.truncate(validation_samples);
    write_json(path, &keys)?;
    Ok(keys.into_iter().collect())
}

fn write_split_prefix<F>(
    path: &Path,
    samples: &[AcceptedSample],
    keep: F,
) -> Result<SplitWriteReport>
where
    F: Fn(&AcceptedSample) -> bool,
{
    let meta_path = path.with_file_name(format!(
        "{}.meta.json",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("split")
    ));
    let previous = read_split_meta(&meta_path)?;
    let selected_count = samples.iter().filter(|sample| keep(sample)).count();
    let can_append = path.is_file()
        && previous.0 <= selected_count
        && (previous.0 == 0
            || samples
                .iter()
                .filter(|sample| keep(sample))
                .nth(previous.0 - 1)
                .map(|sample| sample.key.as_str())
                == Some(previous.1.as_str()));

    let mut class_counts = BTreeMap::from([
        ("-1".to_owned(), 0_usize),
        ("0".to_owned(), 0_usize),
        ("1".to_owned(), 0_usize),
    ]);
    let mut rows = 0_usize;
    let mut last_key = String::new();
    let selected = samples
        .iter()
        .filter(|sample| keep(sample))
        .collect::<Vec<_>>();
    if can_append && path.is_file() {
        let mut writer = BufWriter::new(fs::OpenOptions::new().append(true).open(path)?);
        for sample in &selected {
            rows += 1;
            last_key = sample.key.clone();
            *class_counts
                .entry(sample.result_bucket.to_string())
                .or_default() += 1;
            if rows > previous.0 {
                sample::write_record(&mut writer, &sample.record)?;
            }
        }
        writer.flush()?;
    } else {
        let records = selected
            .iter()
            .map(|sample| {
                rows += 1;
                last_key = sample.key.clone();
                *class_counts
                    .entry(sample.result_bucket.to_string())
                    .or_default() += 1;
                sample.record.clone()
            })
            .collect::<Vec<_>>();
        sample::write_samples(path, &records)?;
    }
    write_json(
        &meta_path,
        &json!({
            "rows": rows,
            "last_key": last_key,
        }),
    )?;
    Ok(SplitWriteReport { rows, class_counts })
}

fn read_split_meta(path: &Path) -> Result<(usize, String)> {
    if !path.is_file() {
        return Ok((0, String::new()));
    }
    let raw: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok((
        raw.get("rows").and_then(Value::as_u64).unwrap_or(0) as usize,
        raw.get("last_key")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
    ))
}

fn write_split_shards<F>(
    dir: &Path,
    prefix: &str,
    samples: &[AcceptedSample],
    keep: F,
) -> Result<Vec<CorpusShard>>
where
    F: Fn(&AcceptedSample) -> bool,
{
    fs::create_dir_all(dir)?;
    let selected = samples
        .iter()
        .filter(|sample| keep(sample))
        .collect::<Vec<_>>();
    let mut shards = Vec::<CorpusShard>::new();
    for (index, chunk) in selected.chunks(SPLIT_SHARD_SAMPLES).enumerate() {
        let path = dir.join(format!("{prefix}-{index:06}.sbin"));
        let meta_path = path.with_extension("sbin.meta.json");
        let first_key = chunk
            .first()
            .map(|sample| sample.key.as_str())
            .unwrap_or("");
        let last_key = chunk.last().map(|sample| sample.key.as_str()).unwrap_or("");
        let expected_meta = json!({
            "rows": chunk.len(),
            "first_key": first_key,
            "last_key": last_key,
        });
        if read_json_value(&meta_path)? != Some(expected_meta.clone()) {
            let records = chunk
                .iter()
                .map(|sample| sample.record.clone())
                .collect::<Vec<_>>();
            sample::write_samples(&path, &records)?;
            write_json(&meta_path, &expected_meta)?;
        }
        shards.push(CorpusShard {
            file: path.display().to_string(),
            samples: chunk.len(),
            start: index * SPLIT_SHARD_SAMPLES,
            end: index * SPLIT_SHARD_SAMPLES + chunk.len(),
        });
    }

    let expected_count = selected.len().div_ceil(SPLIT_SHARD_SAMPLES);
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("sbin") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(raw_index) = stem.strip_prefix(&format!("{prefix}-")) else {
            continue;
        };
        if raw_index
            .parse::<usize>()
            .is_ok_and(|index| index >= expected_count)
        {
            remove_if_exists(&path)?;
            remove_if_exists(&path.with_extension("sbin.meta.json"))?;
        }
    }
    Ok(shards)
}

fn link_cycle_file(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() || destination.is_symlink() {
        fs::remove_file(destination)?;
    }
    #[cfg(unix)]
    {
        if let Ok(()) = std::os::unix::fs::symlink(source, destination) {
            return Ok(());
        }
    }
    fs::copy(source, destination)?;
    Ok(())
}

fn link_cycle_dir(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() || destination.is_symlink() {
        fs::remove_file(destination)
            .or_else(|error| {
                if error.kind() == std::io::ErrorKind::IsADirectory {
                    fs::remove_dir_all(destination)
                } else {
                    Err(error)
                }
            })
            .with_context(|| format!("failed to remove {}", destination.display()))?;
    }
    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(source, destination).is_ok() {
            return Ok(());
        }
    }
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let source_path = entry?.path();
        let Some(name) = source_path.file_name() else {
            continue;
        };
        fs::copy(&source_path, destination.join(name))?;
    }
    Ok(())
}

fn read_json_value(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&fs::read_to_string(path)?)?))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("json")
    ));
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn terminal_search_record(sample: &BinarySample) -> bool {
    let score = sample.score as i32;
    let ply = sample.ply as i32;
    let depth = sample.completed_depth as i32;
    score.abs() >= 99_000 || (score == 0 && ply + depth >= 350)
}

fn high_abs_score_record(sample: &BinarySample, max_abs_score: i32) -> bool {
    if max_abs_score <= 0 {
        return false;
    }
    (sample.clipped_score as i32).abs() > max_abs_score
}
