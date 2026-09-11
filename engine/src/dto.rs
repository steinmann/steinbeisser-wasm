//! Serializable boundary types shared by adapters and the game layer.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GameResultDto {
    pub kind: String,
    pub winner: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MoveStackEntryDto {
    pub position: String,
    /// Legacy input only. New snapshots refer to the single session history.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history_positions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_len: Option<usize>,
    pub no_progress_ply: u16,
    pub turn_index: u16,
    pub last_engine_reverse_move: Option<String>,
    pub last_move: Option<String>,
    pub result: Option<GameResultDto>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionDto {
    pub position: String,
    pub side_to_move: String,
    pub history_positions: Vec<String>,
    pub no_progress_ply: u16,
    pub turn_index: u16,
    pub last_engine_reverse_move: Option<String>,
    pub move_stack: Vec<MoveStackEntryDto>,
    pub black_count: usize,
    pub white_count: usize,
    pub last_move: Option<String>,
    pub result: Option<GameResultDto>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MoveCandidateDto {
    pub r#move: String,
    pub source_cells: Vec<String>,
    pub direction: String,
    pub anchor_cell: String,
    pub is_inline: bool,
    pub is_broadside: bool,
    pub is_push: bool,
    pub is_ejection: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultDto {
    pub best_move: String,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub white_perspective_score: i32,
    pub black_perspective_score: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StatusDto {
    pub side_to_move: String,
    pub black_count: usize,
    pub white_count: usize,
    pub turn_index: u16,
    pub no_progress_ply: u16,
    pub result: Option<GameResultDto>,
    pub is_game_over: bool,
    pub can_take_back: bool,
}
