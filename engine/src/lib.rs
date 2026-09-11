#![warn(unsafe_op_in_unsafe_fn)]

pub mod api;
mod board;
pub mod dto;
mod eval;
pub mod game;
mod movegen;
pub mod search;

pub use api::{
    GameResultDto, MoveCandidateDto, MoveStackEntryDto, SearchResultDto, SessionDto, StatusDto,
    apply_move, legal_moves_for_selection, new_session, search_best_move,
    search_best_move_with_limits, session_status, undo_full_turn,
};
pub use board::*;
pub use movegen::{MoveApplicationError, PositionState, UndoSnapshot};
pub use search::SearchResult;

#[cfg(feature = "diagnostics")]
pub use eval::diagnostics;
