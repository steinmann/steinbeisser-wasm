use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

use super::*;
use crate::corpus::CorpusManifest;
use crate::sample::{self, BinarySample};

#[derive(Debug)]
struct PrepareDatasetArgs {
    dataset_path: PathBuf,
    out_dir: PathBuf,
    manifest_path: PathBuf,
    feature_set: String,
    normalization: DenseFeatureNormalization,
    sample_limit: Option<usize>,
    lambda_mix: f64,
}

#[derive(Default, Serialize)]
struct RunCounter {
    samples_seen: u64,
    samples_kept: u64,
    samples_dropped: u64,
    raw_occurrences_seen: u64,
    raw_occurrences_kept: u64,
    raw_occurrences_dropped: u64,
    effective_weight_mass: f64,
}

pub(super) fn run_prepare_dataset_command<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let parsed = parse_prepare_dataset_args(args)?;
    let report = prepare_dataset(&parsed)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_prepare_dataset_args<I>(args: I) -> Result<PrepareDatasetArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut dataset_path = None::<PathBuf>;
    let mut out_dir = None::<PathBuf>;
    let mut manifest_path = None::<PathBuf>;
    let mut feature_set = FEATURE_SET_NAME.to_owned();
    let mut normalization = DenseFeatureNormalization::identity();
    let mut sample_limit = None::<usize>;
    let mut lambda_mix = 1.0_f64;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--dataset" => {
                dataset_path = Some(PathBuf::from(required_value(&mut args, "--dataset")?))
            }
            "--out-dir" => out_dir = Some(PathBuf::from(required_value(&mut args, "--out-dir")?)),
            "--manifest" => {
                manifest_path = Some(PathBuf::from(required_value(&mut args, "--manifest")?))
            }
            "--feature-set" => feature_set = required_value(&mut args, "--feature-set")?,
            "--normalization-json" => {
                let raw = required_value(&mut args, "--normalization-json")?;
                normalization = serde_json::from_str(&raw)
                    .with_context(|| "failed to parse --normalization-json")?;
            }
            "--sample-limit" => {
                sample_limit = Some(parse_value(
                    &required_value(&mut args, "--sample-limit")?,
                    &flag,
                )?);
            }
            "--lambda" => {
                lambda_mix = parse_value(&required_value(&mut args, "--lambda")?, &flag)?;
            }
            _ => bail!(
                "unknown argument {flag}; usage: nnue prepare-dataset --dataset <train.sbin> --out-dir <cache-dir> --manifest <manifest.json>"
            ),
        }
    }
    Ok(PrepareDatasetArgs {
        dataset_path: dataset_path.ok_or_else(|| anyhow::anyhow!("missing required --dataset"))?,
        out_dir: out_dir.ok_or_else(|| anyhow::anyhow!("missing required --out-dir"))?,
        manifest_path: manifest_path
            .ok_or_else(|| anyhow::anyhow!("missing required --manifest"))?,
        feature_set,
        normalization,
        sample_limit,
        lambda_mix,
    })
}

fn prepare_dataset(args: &PrepareDatasetArgs) -> Result<Value> {
    let started = Instant::now();
    validate_prepare_dataset_args(args)?;
    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;
    let manifest = CorpusManifest::read(&args.manifest_path)?;
    let feature_set = NnueFeatureSet::current();
    let input_count = feature_set.input_count();
    let sparse_count = feature_set.sparse_input_count();

    let mut inputs = BufWriter::new(fs::File::create(args.out_dir.join("inputs.f32"))?);
    let mut targets = BufWriter::new(fs::File::create(args.out_dir.join("targets.f32"))?);
    let mut train_weights =
        BufWriter::new(fs::File::create(args.out_dir.join("train_weights.f32"))?);
    let mut eval_weights = BufWriter::new(fs::File::create(args.out_dir.join("eval_weights.f32"))?);
    let mut keep_probabilities = BufWriter::new(fs::File::create(
        args.out_dir.join("keep_probabilities.f32"),
    )?);
    let mut input_row = vec![0.0_f32; input_count];
    let mut input_bytes = Vec::<u8>::with_capacity(input_count * 4);
    let mut samples_seen = 0_u64;
    let mut samples_kept = 0_u64;
    let samples_dropped = 0_u64;
    let mut raw_occurrences_seen = 0_u64;
    let mut raw_occurrences_kept = 0_u64;
    let raw_occurrences_dropped = 0_u64;
    let mut total_multiplier = 0.0_f64;
    let mut min_multiplier = None::<f64>;
    let mut max_multiplier = None::<f64>;
    let mut effective_weight_mass = 0.0_f64;
    let mut repeat_weighted_samples = 0_u64;
    let drop_reasons = BTreeMap::<String, u64>::new();
    let mut phase_before = BTreeMap::<String, u64>::new();
    let mut phase_after = BTreeMap::<String, u64>::new();
    let mut result_buckets = BTreeMap::<String, u64>::new();
    let mut score_buckets = BTreeMap::<String, u64>::new();
    let mut occurrence_buckets = BTreeMap::<String, u64>::new();
    let mut run_counters = BTreeMap::<String, RunCounter>::new();
    let run_file_name = args
        .dataset_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<dataset>")
        .to_owned();

    {
        let mut handle_sample = |sample: BinarySample| -> Result<bool> {
            if let Some(limit) = args.sample_limit
                && samples_seen as usize >= limit
            {
                return Ok(false);
            }
            let occurrence_count = sample.occurrence_count.max(1);
            let occurrence_count_u64 = u64::from(occurrence_count);
            let run_entry = run_counters.entry(run_file_name.clone()).or_default();
            samples_seen += 1;
            raw_occurrences_seen += occurrence_count_u64;
            run_entry.samples_seen += 1;
            run_entry.raw_occurrences_seen += occurrence_count_u64;

            let ply = sample.ply;
            if ply >= 8.0 {
                increment_counter(
                    &mut phase_before,
                    phase_bucket_label(ply),
                    occurrence_count_u64,
                );
            }
            let black_bits = sample.black_bits;
            let white_bits = sample.white_bits;
            let side_to_move_is_black = sample.side_to_move_is_black;
            let no_progress_plies = sample.no_progress_plies.max(0.0);
            let score = sample.score;
            let clipped_score = sample.clipped_score;
            let result = sample.result;
            let result_bucket = sample.result_bucket;

            input_row.fill(0.0);
            let features = feature_set.feature_vector_from_bitboards_with_context(
                side_to_move_is_black,
                black_bits,
                white_bits,
                ply,
                no_progress_plies,
                args.normalization.clone(),
            );
            for &index in features.active_indices() {
                input_row[index as usize] = 1.0;
            }
            for (dense_index, value) in features.dense_values().iter().copied().enumerate() {
                input_row[sparse_count + dense_index] = value;
            }
            write_f32_slice(&mut inputs, &input_row, &mut input_bytes)?;

            let target = combined_target(TargetBlend {
                score,
                result,
                lambda_mix: args.lambda_mix,
            }) as f32;
            let base_weight = f64::from(sample.sample_weight);
            let class_weight = class_weight_for_bucket(&manifest, result_bucket);
            let repeat_multiplier = repeat_occurrence_multiplier(occurrence_count);
            if repeat_multiplier > 1.0 {
                repeat_weighted_samples += 1;
            }
            let multiplier = repeat_multiplier;
            let train_weight = base_weight * class_weight * multiplier;
            let eval_weight = 1.0;

            write_f32(&mut targets, target)?;
            write_f32(&mut train_weights, train_weight as f32)?;
            write_f32(&mut eval_weights, eval_weight)?;
            write_f32(&mut keep_probabilities, 1.0)?;

            samples_kept += 1;
            raw_occurrences_kept += occurrence_count_u64;
            run_entry.samples_kept += 1;
            run_entry.raw_occurrences_kept += occurrence_count_u64;
            run_entry.effective_weight_mass += train_weight;
            effective_weight_mass += train_weight;
            total_multiplier += multiplier;
            min_multiplier = Some(min_multiplier.map_or(multiplier, |value| value.min(multiplier)));
            max_multiplier = Some(max_multiplier.map_or(multiplier, |value| value.max(multiplier)));
            if ply >= 8.0 {
                increment_counter(
                    &mut phase_after,
                    phase_bucket_label(ply),
                    occurrence_count_u64,
                );
            }
            increment_counter(
                &mut result_buckets,
                result_bucket.to_string(),
                occurrence_count_u64,
            );
            increment_counter(
                &mut score_buckets,
                score_bucket_label_for_report(clipped_score),
                occurrence_count_u64,
            );
            increment_counter(
                &mut occurrence_buckets,
                occurrence_bucket_label(occurrence_count),
                occurrence_count_u64,
            );
            Ok(true)
        };

        for record in sample::read_samples(&args.dataset_path)
            .with_context(|| format!("failed to read {}", args.dataset_path.display()))?
        {
            if !handle_sample(record)? {
                break;
            }
        }
    }

    inputs.flush()?;
    targets.flush()?;
    train_weights.flush()?;
    eval_weights.flush()?;
    keep_probabilities.flush()?;

    let arrays = json!({
        "inputs": {"file": "inputs.f32", "dtype": "float32", "shape": [samples_kept, input_count as u64]},
        "targets": {"file": "targets.f32", "dtype": "float32", "shape": [samples_kept]},
        "train_weights": {"file": "train_weights.f32", "dtype": "float32", "shape": [samples_kept]},
        "eval_weights": {"file": "eval_weights.f32", "dtype": "float32", "shape": [samples_kept]},
        "keep_probabilities": {"file": "keep_probabilities.f32", "dtype": "float32", "shape": [samples_kept]},
    });
    Ok(json!({
        "profile": "full",
        "dataset_path": args.dataset_path.display().to_string(),
        "active_filters": [
            "profile:full",
            "class_weighting:inverse_sqrt",
            "repeat_occurrence_weight:sqrt_cap2",
        ],
        "class_weighting": "inverse_sqrt",
        "repeat_occurrence_weight": "sqrt_cap2",
        "samples_seen": samples_seen,
        "samples_kept": samples_kept,
        "samples_dropped": samples_dropped,
        "raw_occurrences_seen": raw_occurrences_seen,
        "raw_occurrences_kept": raw_occurrences_kept,
        "raw_occurrences_dropped": raw_occurrences_dropped,
        "drop_reasons": drop_reasons,
        "profile_stats": {
            "full": {
                "kept_samples": samples_kept,
                "dropped_samples": samples_dropped,
                "kept_raw_occurrences": raw_occurrences_kept,
                "dropped_raw_occurrences": raw_occurrences_dropped,
            }
        },
        "phase_bucket_raw_occurrences_before": phase_before,
        "phase_bucket_raw_occurrences_after": phase_after,
        "result_bucket_raw_occurrences_after": result_buckets,
        "score_bucket_raw_occurrences_after": score_buckets,
        "completed_depth_bucket_raw_occurrences_after": {},
        "occurrence_bucket_raw_occurrences_after": occurrence_buckets,
        "weight_multiplier": {
            "count": samples_kept,
            "sum": total_multiplier,
            "mean": if samples_kept > 0 { total_multiplier / samples_kept as f64 } else { 0.0 },
            "min": min_multiplier,
            "max": max_multiplier,
        },
        "repeat_occurrence_weighted_samples": repeat_weighted_samples,
        "per_run": run_counters,
        "effective_weight_mass": effective_weight_mass,
        "cache_hit": false,
        "cache_key": Value::Null,
        "cache_build_seconds": started.elapsed().as_secs_f64(),
        "cache_file": Value::Null,
        "cache_format": "raw-f32-v1",
        "array_format": "raw-f32-v1",
        "arrays": arrays,
    }))
}

fn validate_prepare_dataset_args(args: &PrepareDatasetArgs) -> Result<()> {
    if args.feature_set != FEATURE_SET_NAME {
        bail!("unsupported --feature-set {}", args.feature_set);
    }
    if !matches!(
        args.dataset_path
            .extension()
            .and_then(|value| value.to_str()),
        Some(sample::BINARY_SAMPLE_EXTENSION)
    ) {
        bail!(
            "prepare-dataset expects .{} sample input",
            sample::BINARY_SAMPLE_EXTENSION
        );
    }
    Ok(())
}

fn class_weight_for_bucket(manifest: &CorpusManifest, result_bucket: i32) -> f64 {
    let count = manifest.class_count(result_bucket);
    if count == 0 {
        return 1.0;
    }
    1.0 / (count as f64).sqrt()
}

fn repeat_occurrence_multiplier(occurrence_count: u32) -> f64 {
    (occurrence_count.max(1) as f64).sqrt().min(2.0)
}

fn write_f32(writer: &mut BufWriter<fs::File>, value: f32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f32_slice(
    writer: &mut BufWriter<fs::File>,
    values: &[f32],
    scratch: &mut Vec<u8>,
) -> Result<()> {
    scratch.clear();
    for value in values {
        scratch.extend_from_slice(&value.to_le_bytes());
    }
    writer.write_all(scratch)?;
    Ok(())
}

fn increment_counter(mapping: &mut BTreeMap<String, u64>, key: impl Into<String>, amount: u64) {
    *mapping.entry(key.into()).or_insert(0) += amount;
}

fn phase_bucket_label(mean_ply: f32) -> &'static str {
    if mean_ply < 16.0 {
        "8-15"
    } else if mean_ply < 32.0 {
        "16-31"
    } else if mean_ply < 64.0 {
        "32-63"
    } else {
        "64-127"
    }
}

fn score_bucket_label_for_report(score: f32) -> String {
    let start = (f64::from(score) / 500.0).floor() as i32 * 500;
    format!("{}..{}", start, start + 499)
}

fn occurrence_bucket_label(occurrence_count: u32) -> &'static str {
    match occurrence_count {
        0 | 1 => "1",
        2 => "2",
        3 | 4 => "3-4",
        5..=8 => "5-8",
        9..=16 => "9-16",
        _ => "17+",
    }
}
