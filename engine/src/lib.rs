#![allow(dead_code, hidden_glob_reexports, private_interfaces)]

pub mod api;
mod board;
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
pub use search::{MAX_GAME_TURNS, SearchResult};
