use crate::board::{ALL_DIRECTIONS, Color, Coord, Direction, Move, Position, geometry};
use crate::eval::nnue;
use crate::game::{
    SessionState, apply_move_state, legal_moves_for_selection_state, search_best_move_state,
    search_best_move_with_limits_state, session_status_state, undo_full_turn_state,
};
use crate::movegen::PositionState;
use crate::search::{MAX_GAME_TURNS, search_raw_with_turn};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use wasm_bindgen::prelude::*;
use web_time::Instant;

fn from_js<T>(value: JsValue) -> Result<T, JsValue>
where
    T: for<'de> Deserialize<'de>,
{
    serde_wasm_bindgen::from_value(value).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn to_js<T>(value: &T) -> Result<JsValue, JsValue>
where
    T: Serialize,
{
    serde_wasm_bindgen::to_value(value).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen]
pub fn new_session() -> Result<JsValue, JsValue> {
    to_js(&SessionState::new().to_dto())
}

#[wasm_bindgen]
pub fn legal_moves_for_selection(session: JsValue, selected: JsValue) -> Result<JsValue, JsValue> {
    let session = SessionState::try_from(from_js::<SessionDto>(session)?)
        .map_err(|error| JsValue::from_str(&error))?;
    let selected = from_js::<Vec<String>>(selected)?;
    let moves = legal_moves_for_selection_state(&session, &selected)
        .map_err(|error| JsValue::from_str(&error))?;
    to_js(&moves)
}

#[wasm_bindgen]
pub fn apply_move(session: JsValue, move_text: String) -> Result<JsValue, JsValue> {
    let session = SessionState::try_from(from_js::<SessionDto>(session)?)
        .map_err(|error| JsValue::from_str(&error))?;
    let next = apply_move_state(session, &move_text).map_err(|error| JsValue::from_str(&error))?;
    to_js(&next.to_dto())
}

#[wasm_bindgen]
pub fn undo_full_turn(session: JsValue) -> Result<JsValue, JsValue> {
    let session = SessionState::try_from(from_js::<SessionDto>(session)?)
        .map_err(|error| JsValue::from_str(&error))?;
    let next = undo_full_turn_state(session).map_err(|error| JsValue::from_str(&error))?;
    to_js(&next.to_dto())
}

#[wasm_bindgen]
pub fn search_best_move(session: JsValue, depth: u8) -> Result<JsValue, JsValue> {
    let session = SessionState::try_from(from_js::<SessionDto>(session)?)
        .map_err(|error| JsValue::from_str(&error))?;
    let result =
        search_best_move_state(&session, depth).map_err(|error| JsValue::from_str(&error))?;
    to_js(&result)
}

#[wasm_bindgen]
pub fn search_best_move_with_limits(
    session: JsValue,
    depth: u8,
    max_time_ms: u32,
) -> Result<JsValue, JsValue> {
    let session = SessionState::try_from(from_js::<SessionDto>(session)?)
        .map_err(|error| JsValue::from_str(&error))?;
    let result = search_best_move_with_limits_state(&session, depth, Some(u64::from(max_time_ms)))
        .map_err(|error| JsValue::from_str(&error))?;
    to_js(&result)
}

#[wasm_bindgen]
pub fn session_status(session: JsValue) -> Result<JsValue, JsValue> {
    let session = SessionState::try_from(from_js::<SessionDto>(session)?)
        .map_err(|error| JsValue::from_str(&error))?;
    to_js(&session_status_state(&session))
}

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
    pub history_positions: Vec<String>,
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

const CODINGAME_FIRST_TURN_MS: u64 = 500;
const CODINGAME_TURN_MS: u64 = 45;

#[derive(Clone, Debug)]
struct CodingameAction {
    raw: String,
    values: [i32; 5],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CodingameActionCodec;

const CODINGAME_DIRECTION_MAP: [Direction; 6] = [
    Direction::East,
    Direction::Ne,
    Direction::Nw,
    Direction::West,
    Direction::Sw,
    Direction::Se,
];

impl CodingameActionCodec {
    fn parse_action(&self, action: &CodingameAction) -> Option<Move> {
        let row0 = u8::try_from(action.values[1]).ok()?;
        let col0 = u8::try_from(action.values[0] + 1).ok()?;
        let start = geometry().index_of_coord(Coord::new(row0, col0)?)?;
        let row1 = u8::try_from(action.values[3]).ok()?;
        let col1 = u8::try_from(action.values[2] + 1).ok()?;
        let end = geometry().index_of_coord(Coord::new(row1, col1)?)?;
        let direction = *CODINGAME_DIRECTION_MAP.get(action.values[4] as usize)?;
        let cells = cells_between(start, end)?;
        Move::from_cells(&cells, direction).ok()
    }
}

#[derive(Debug)]
struct CodingameState {
    own_color: Color,
    prior_positions: Vec<Position>,
    last_position: Option<Position>,
    last_position_after_own_move: Option<Position>,
    last_total_score: Option<i32>,
    last_reverse_move: Option<Move>,
    no_progress_ply: u16,
    own_turn_count: usize,
}

impl CodingameState {
    fn new(own_color: Color) -> Self {
        Self {
            own_color,
            prior_positions: Vec::new(),
            last_position: None,
            last_position_after_own_move: None,
            last_total_score: None,
            last_reverse_move: None,
            no_progress_ply: 0,
            own_turn_count: 0,
        }
    }

    fn observe_position(&mut self, position: &Position, total_score: i32) {
        if let Some(previous) = self.last_position.replace(position.clone()) {
            self.prior_positions.push(previous);
            self.no_progress_ply = if self.last_total_score == Some(total_score) {
                self.no_progress_ply.saturating_add(2)
            } else {
                0
            };
        }
        self.last_total_score = Some(total_score);
    }

    fn time_budget_ms(&self) -> u64 {
        if self.own_turn_count == 0 {
            CODINGAME_FIRST_TURN_MS
        } else {
            CODINGAME_TURN_MS
        }
    }

    fn turns_played(&self) -> u16 {
        let own_turns_played = self.own_turn_count.saturating_mul(2);
        let side_offset = if self.own_color == Color::White { 1 } else { 0 };
        own_turns_played
            .saturating_add(side_offset)
            .min(MAX_GAME_TURNS as usize) as u16
    }

    fn finish_turn(&mut self) {
        self.own_turn_count = self.own_turn_count.saturating_add(1);
    }

    fn record_own_move(&mut self, position: &Position, candidate_move: Option<Move>) {
        self.last_position_after_own_move =
            candidate_move.and_then(|candidate_move| position_after_move(position, candidate_move));
        self.last_reverse_move = candidate_move.and_then(reverse_move);
    }
}

struct ChosenAction {
    raw: String,
    candidate_move: Option<Move>,
}

pub fn run_codingame_adapter() {
    let _ = run_codingame_from_stdin();
}

fn run_codingame_from_stdin() -> Result<(), String> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let my_id = read_required_line(&mut lines)?
        .trim()
        .parse::<u8>()
        .map_err(|_| String::new())?;
    let own_color = match my_id {
        1 => Color::White,
        2 => Color::Black,
        _ => return Err(String::new()),
    };
    let mut state = CodingameState::new(own_color);
    let _ = geometry();
    let _ = nnue();

    while let Some(score_line) = next_optional_line(&mut lines)? {
        let turn_started = Instant::now();
        let scores = parse_i32s(&score_line)?;
        if scores.len() != 2 {
            return Err(String::new());
        }

        let position = read_position(&mut lines, state.own_color)?;
        let _opponent_action_line = read_required_line(&mut lines)?;
        let legal_actions_count = read_required_line(&mut lines)?
            .trim()
            .parse::<usize>()
            .map_err(|_| String::new())?;
        let mut legal_actions = Vec::with_capacity(legal_actions_count);
        for _ in 0..legal_actions_count {
            legal_actions.push(parse_action(&read_required_line(&mut lines)?)?);
        }

        let turn_budget_ms = state.time_budget_ms();
        let pre_search_ms = turn_started.elapsed().as_millis() as u64;
        let search_time_ms = turn_budget_ms
            .saturating_sub(pre_search_ms)
            .saturating_sub(4)
            .max(1);
        state.observe_position(&position, scores[0].saturating_add(scores[1]));
        let turn_index = state.turns_played();
        let chosen = choose_action(
            &position,
            &state.prior_positions,
            state.no_progress_ply,
            turn_index,
            search_time_ms,
            state.last_reverse_move,
            &legal_actions,
        )?;

        println!("{}", chosen.raw);
        io::stdout().flush().map_err(|_| String::new())?;
        state.record_own_move(&position, chosen.candidate_move);
        state.finish_turn();
    }

    Ok(())
}

fn choose_action(
    position: &Position,
    history_positions: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    time_ms: u64,
    root_reverse_move: Option<Move>,
    legal_actions: &[CodingameAction],
) -> Result<ChosenAction, String> {
    if legal_actions.is_empty() {
        return Ok(ChosenAction {
            raw: "0 0 0 0 0".to_owned(),
            candidate_move: None,
        });
    }

    let codec = CodingameActionCodec;
    let result = search_raw_with_turn(
        position,
        history_positions,
        no_progress_ply,
        turn_index,
        time_ms,
        root_reverse_move,
    )?;

    if let Some(best_move) = result.best_move {
        for action in legal_actions {
            if codec.parse_action(action) == Some(best_move) {
                return Ok(ChosenAction {
                    raw: action.raw.clone(),
                    candidate_move: Some(best_move),
                });
            }
        }
    }

    Ok(ChosenAction {
        raw: legal_actions[0].raw.clone(),
        candidate_move: codec.parse_action(&legal_actions[0]),
    })
}

fn parse_action(line: &str) -> Result<CodingameAction, String> {
    let values = parse_i32s(line)?;
    let values: [i32; 5] = values.try_into().map_err(|_| String::new())?;
    Ok(CodingameAction {
        raw: line.trim().to_owned(),
        values,
    })
}

fn read_position(
    lines: &mut impl Iterator<Item = io::Result<String>>,
    side_to_move: Color,
) -> Result<Position, String> {
    let mut black = Vec::new();
    let mut white = Vec::new();
    let geometry = geometry();

    for row in 0..=8u8 {
        let raw = read_required_line(lines)?;
        let digits = raw
            .bytes()
            .filter(|value| matches!(value, b'0' | b'1' | b'2'))
            .collect::<Vec<_>>();
        let expected = Coord::row_length(row).unwrap() as usize;
        if digits.len() != expected {
            return Err(String::new());
        }

        for (offset, value) in digits.into_iter().enumerate() {
            let coord = Coord::new(row, offset as u8 + 1).unwrap();
            let cell = geometry.index_of_coord(coord).unwrap();
            match value {
                b'1' => white.push(cell),
                b'2' => black.push(cell),
                _ => {}
            }
        }
    }

    Position::new(side_to_move, black, white).map_err(|_| String::new())
}

pub(crate) fn reverse_move(candidate_move: Move) -> Option<Move> {
    let geometry = geometry();
    let mut destination_cells = [candidate_move.source_cells()[0]; 3];
    for (index, cell) in candidate_move.source_cells().iter().copied().enumerate() {
        destination_cells[index] =
            geometry.cell(cell).neighbors[candidate_move.direction().index()]?;
    }
    Move::from_cells(
        &destination_cells[..candidate_move.len()],
        candidate_move.direction().opposite(),
    )
    .ok()
}

fn position_after_move(position: &Position, candidate_move: Move) -> Option<Position> {
    let mut state = PositionState::new(position.clone()).ok()?;
    state.apply_move(&candidate_move).ok()?;
    Some(state.position().clone())
}

fn cells_between(
    start: crate::board::CellId,
    end: crate::board::CellId,
) -> Option<Vec<crate::board::CellId>> {
    if start == end {
        return Some(vec![start]);
    }

    let geometry = geometry();
    for direction in ALL_DIRECTIONS {
        let mut cells = vec![start];
        let mut current = start;
        while cells.len() < 3 {
            let Some(next) = geometry.cell(current).neighbors[direction.index()] else {
                break;
            };
            current = next;
            cells.push(current);
            if current == end {
                return Some(cells);
            }
        }
    }

    None
}

fn parse_i32s(line: &str) -> Result<Vec<i32>, String> {
    line.split_whitespace()
        .map(|value| value.parse::<i32>().map_err(|_| String::new()))
        .collect()
}

fn read_required_line(
    lines: &mut impl Iterator<Item = io::Result<String>>,
) -> Result<String, String> {
    next_optional_line(lines)?.ok_or_else(String::new)
}

fn next_optional_line(
    lines: &mut impl Iterator<Item = io::Result<String>>,
) -> Result<Option<String>, String> {
    match lines.next() {
        Some(result) => result.map(Some).map_err(|_| String::new()),
        None => Ok(None),
    }
}
