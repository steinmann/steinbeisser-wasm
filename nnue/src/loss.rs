use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::core::{SparseMlpModel, TargetBlend, combined_target, huber_loss, linear_clip_score};
use crate::sample;
use crate::{parse_value, required_value};

#[derive(Debug)]
struct RuntimeLossArgs {
    model_path: PathBuf,
    dataset_path: PathBuf,
    lambda_mix: f64,
}

#[derive(Debug, Serialize)]
struct RuntimeLossReport {
    samples: u64,
    loss: f64,
    search_abs_error: f64,
    result_abs_error: f64,
}

pub(super) fn run_runtime_loss_command<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let parsed = parse_runtime_loss_args(args)?;
    let report = evaluate_runtime_loss(&parsed)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_runtime_loss_args<I>(args: I) -> Result<RuntimeLossArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut model_path = None::<PathBuf>;
    let mut dataset_path = None::<PathBuf>;
    let mut lambda_mix = 1.0_f64;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = Some(PathBuf::from(required_value(&mut args, "--model")?)),
            "--dataset" => {
                dataset_path = Some(PathBuf::from(required_value(&mut args, "--dataset")?))
            }
            "--lambda" => {
                lambda_mix = parse_value(&required_value(&mut args, "--lambda")?, &flag)?;
            }
            _ => bail!(
                "unknown argument {flag}; usage: nnue runtime-loss --model <model.json|model.nnq> --dataset <val.sbin> [--lambda <f64>]"
            ),
        }
    }
    Ok(RuntimeLossArgs {
        model_path: model_path.ok_or_else(|| anyhow::anyhow!("missing required --model"))?,
        dataset_path: dataset_path.ok_or_else(|| anyhow::anyhow!("missing required --dataset"))?,
        lambda_mix,
    })
}

fn evaluate_runtime_loss(args: &RuntimeLossArgs) -> Result<RuntimeLossReport> {
    validate_runtime_loss_args(args)?;
    let model = SparseMlpModel::load(&args.model_path)?;
    let feature_set = model.feature_set();
    let normalization = model.dense_normalization();
    let mut samples = 0_u64;
    let mut weighted_loss = 0.0_f64;
    let mut weighted_search_abs_error = 0.0_f64;
    let mut weighted_result_abs_error = 0.0_f64;
    let mut weight_sum = 0.0_f64;

    for sample in sample::read_samples(&args.dataset_path)
        .with_context(|| format!("failed to read {}", args.dataset_path.display()))?
    {
        let black_bits = sample.black_bits;
        let white_bits = sample.white_bits;
        let side_to_move_is_black = sample.side_to_move_is_black;
        let ply = sample.ply;
        let no_progress_plies = sample.no_progress_plies.max(0.0);
        let score = sample.score;
        let result = sample.result;
        let eval_weight = 1.0;
        let features = feature_set.feature_vector_from_bitboards_with_context(
            side_to_move_is_black,
            black_bits,
            white_bits,
            ply,
            no_progress_plies,
            normalization.clone(),
        );
        let prediction = f64::from(model.raw_output(&features));
        let predicted_score = f64::from(linear_clip_score(prediction as f32));
        let target = combined_target(TargetBlend {
            score,
            result,
            lambda_mix: args.lambda_mix,
        });
        weighted_loss += eval_weight * huber_loss(prediction - target);
        let search_error = predicted_score - f64::from(score);
        weighted_search_abs_error += eval_weight * search_error.abs();
        let result_error = prediction.clamp(-1.0, 1.0) - f64::from(result);
        weighted_result_abs_error += eval_weight * result_error.abs();
        weight_sum += eval_weight;
        samples += 1;
    }
    Ok(RuntimeLossReport {
        samples,
        loss: weighted_loss / weight_sum.max(1.0),
        search_abs_error: weighted_search_abs_error / weight_sum.max(1.0),
        result_abs_error: weighted_result_abs_error / weight_sum.max(1.0),
    })
}

fn validate_runtime_loss_args(args: &RuntimeLossArgs) -> Result<()> {
    if !args.model_path.is_file() {
        bail!("missing model {}", args.model_path.display());
    }
    if !args.dataset_path.is_file() {
        bail!("missing dataset {}", args.dataset_path.display());
    }
    if !matches!(
        args.dataset_path
            .extension()
            .and_then(|value| value.to_str()),
        Some(sample::BINARY_SAMPLE_EXTENSION)
    ) {
        bail!(
            "runtime-loss expects .{} sample input",
            sample::BINARY_SAMPLE_EXTENSION
        );
    }
    Ok(())
}
