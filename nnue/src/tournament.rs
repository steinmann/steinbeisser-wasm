use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct TournamentStanding {
    pub(crate) player: String,
    pub(crate) elo_vs_latest: f64,
    pub(crate) games: i64,
    pub(crate) score_pct: f64,
    pub(crate) wins: i64,
    pub(crate) draws: i64,
    pub(crate) losses: i64,
    pub(crate) screen_elo: f64,
    pub(crate) qval_loss: Option<f64>,
}

pub(crate) fn standings_from_summary(summary: &Value) -> Result<Vec<TournamentStanding>> {
    let raw = summary
        .get("standings")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    serde_json::from_value(raw).context("failed to parse tournament standings")
}

pub(super) fn tournament_table_lines(standings: &[TournamentStanding]) -> Vec<String> {
    let headers = [
        "Rank",
        "Player",
        "Elo vs latest",
        "Games",
        "Score",
        "WDL",
        "Screen Elo",
        "QVal",
    ];
    let mut rows = Vec::with_capacity(standings.len());
    for (index, row) in standings.iter().enumerate() {
        let qval_text = row
            .qval_loss
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "ref".to_owned());
        rows.push(vec![
            (index + 1).to_string(),
            markdown_cell(&row.player),
            format!("{:+.1}", row.elo_vs_latest),
            row.games.to_string(),
            format!("{:.2}%", row.score_pct),
            format!("{}-{}-{}", row.wins, row.draws, row.losses),
            format!("{:+.1}", row.screen_elo),
            qval_text,
        ]);
    }
    let mut lines = vec!["## Final positive-hit round robin".to_owned()];
    lines.extend(compact_markdown_table(&headers, &rows));
    lines
}

fn compact_markdown_table(headers: &[&str], rows: &[Vec<String>]) -> Vec<String> {
    let header_cells = headers
        .iter()
        .map(|header| (*header).to_owned())
        .collect::<Vec<_>>();
    let separator_cells = vec!["---".to_owned(); headers.len()];
    let mut lines = vec![
        format_markdown_row(&header_cells),
        format_markdown_row(&separator_cells),
    ];
    lines.extend(rows.iter().map(|row| format_markdown_row(row)));
    lines
}

fn format_markdown_row(cells: &[String]) -> String {
    let cells = cells.join(" | ");
    format!("| {cells} |")
}

fn markdown_cell(text: &str) -> String {
    text.replace('|', "\\|")
}
