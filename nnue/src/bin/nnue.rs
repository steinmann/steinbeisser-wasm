#![recursion_limit = "256"]

use anyhow::{Result, bail};

#[path = "../core.rs"]
mod core;
#[path = "../corpus.rs"]
mod corpus;
#[path = "../export.rs"]
mod export;
#[path = "../loss.rs"]
mod loss;
#[path = "../materialize.rs"]
mod materialize;
#[path = "../prepare.rs"]
mod prepare;
#[path = "../sample.rs"]
mod sample;
#[path = "../screen.rs"]
mod screen;
#[path = "../tournament.rs"]
mod tournament;

pub(crate) use core::*;

pub(crate) fn required_value<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}

pub(crate) fn parse_value<T>(raw: &str, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    raw.parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid {flag} value {raw}: {error}"))
}

fn main() {
    if let Err(error) = run_cli(std::env::args().skip(1)) {
        eprintln!("nnue: {error:#}");
        std::process::exit(1);
    }
}

fn run_cli<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("runtime-loss") => loss::run_runtime_loss_command(args),
        Some("feature-schema") => print_feature_schema(),
        Some("prepare-dataset") => prepare::run_prepare_dataset_command(args),
        Some("corpus-build") => corpus::run_corpus_build_command(args),
        Some("materialize-candidate") => materialize::run_materialize_candidate_command(args),
        Some("screen-match") => screen::run_screen_match_command(args),
        Some("export-results") => export::run_export_results_command(args),
        Some("export-positive-training-data") => export::run_export_training_data_command(args),
        Some("help") | Some("--help") | Some("-h") | None => {
            print_usage();
            Ok(())
        }
        Some(other) => bail!("unknown command {other}; run nnue help"),
    }
}

fn print_feature_schema() -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&current_feature_schema())?
    );
    Ok(())
}

fn print_usage() {
    println!(
        "usage: nnue feature-schema\n       nnue runtime-loss --model <model.json|model.nnq> --dataset <val.sbin>\n       nnue prepare-dataset --dataset <train.sbin> --out-dir <cache-dir> --manifest <manifest.json>\n       nnue corpus-build --shards <dir> --work-dir <dir> --cycle <n>"
    );
    println!(
        "       nnue materialize-candidate --repo <repo> --reference-ref <ref> --model <model.nnq> --source-dir <dir> --target <bin> --target-dir <cargo-target>\n       nnue screen-match --selfplay-bin <nnue-selfplay> --repo <repo> --candidate <bin> --baseline <bin> --openings <fen> --games <n> --time-ms <ms> [--parallel-games <n>]\n       nnue export-results --summary <summary.json> --out-dir <dir>\n       nnue export-positive-training-data --summary <summary.json> --out-dir <dir> --work-dir <dir> [--corpus-dir <dir>] --reference-ref <ref>"
    );
}
