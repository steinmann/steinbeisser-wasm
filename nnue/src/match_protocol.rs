//! Versioned machine interface. Human-readable match logs are not a protocol.
use serde::{Deserialize, Serialize};

pub const PREFIX: &str = "steinbeisser.match.v1\t";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Actor {
    Local,
    Github,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum MatchEvent {
    Complete {
        games: usize,
        wins: usize,
        draws: usize,
        losses: usize,
        elo: f64,
        elo_lower: f64,
        elo_upper: f64,
        local_illegal: usize,
        local_errors: usize,
        github_illegal: usize,
        github_errors: usize,
    },
    Failure {
        actor: Option<Actor>,
        kind: String,
        message: String,
    },
}

pub fn emit(event: &MatchEvent) {
    // One println locks stdout for the entire record, including concurrent failures.
    println!(
        "{PREFIX}{}",
        serde_json::to_string(event).expect("serializable match event")
    );
}

pub fn engine_error(name: &str, kind: &str, message: impl ToString) -> String {
    let message = message.to_string();
    emit(&MatchEvent::Failure {
        actor: match name {
            "local" => Some(Actor::Local),
            "github" => Some(Actor::Github),
            _ => None,
        },
        kind: kind.to_owned(),
        message: message.clone(),
    });
    format!("{name} {kind}: {message}")
}
