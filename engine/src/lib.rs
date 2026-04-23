#![allow(
    dead_code,
    hidden_glob_reexports,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    private_interfaces
)]

use serde::{Deserialize, Serialize};
use std::str::FromStr;
use wasm_bindgen::prelude::*;

include!("rc8_core.rs");

const START_POSITION: &str = "aba-v1;stm=b;black=A4,A5,B4,B5,B6,C5,C6,G2,G3,H1,H2,H3,I1,I2;white=A1,A2,B1,B2,B3,C2,C3,G5,G6,H4,H5,H6,I4,I5";

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

#[derive(Clone, Debug)]
struct MoveStackEntry {
    position: Po,
    history_positions: Vec<Po>,
    no_progress_ply: u16,
    turn_index: u16,
    last_engine_reverse_move: Option<Mv>,
    last_move: Option<Mv>,
    result: Option<GameResultDto>,
}

#[derive(Clone, Debug)]
struct SessionState {
    position: Po,
    history_positions: Vec<Po>,
    no_progress_ply: u16,
    turn_index: u16,
    last_engine_reverse_move: Option<Mv>,
    move_stack: Vec<MoveStackEntry>,
    last_move: Option<Mv>,
    result: Option<GameResultDto>,
}

impl SessionState {
    fn new() -> Self {
        Self {
            position: Po::from_str(START_POSITION).expect("valid Belgian Daisy start"),
            history_positions: Vec::new(),
            no_progress_ply: 0,
            turn_index: 0,
            last_engine_reverse_move: None,
            move_stack: Vec::new(),
            last_move: None,
            result: None,
        }
    }

    fn side_to_move(&self) -> Co {
        self.position._sT()
    }

    fn black_count(&self) -> usize {
        self.position.black().len()
    }

    fn white_count(&self) -> usize {
        self.position.white().len()
    }

    fn to_dto(&self) -> SessionDto {
        SessionDto {
            position: self.position.canonical_string(),
            side_to_move: color_name(self.side_to_move()).to_owned(),
            history_positions: self
                .history_positions
                .iter()
                .map(Po::canonical_string)
                .collect(),
            no_progress_ply: self.no_progress_ply,
            turn_index: self.turn_index,
            last_engine_reverse_move: self.last_engine_reverse_move.map(|mv| mv.to_string()),
            move_stack: self.move_stack.iter().map(MoveStackEntry::to_dto).collect(),
            black_count: self.black_count(),
            white_count: self.white_count(),
            last_move: self.last_move.map(|mv| mv.to_string()),
            result: self.result.clone(),
        }
    }
}

impl MoveStackEntry {
    fn from_state(state: &SessionState) -> Self {
        Self {
            position: state.position.clone(),
            history_positions: state.history_positions.clone(),
            no_progress_ply: state.no_progress_ply,
            turn_index: state.turn_index,
            last_engine_reverse_move: state.last_engine_reverse_move,
            last_move: state.last_move,
            result: state.result.clone(),
        }
    }

    fn to_dto(&self) -> MoveStackEntryDto {
        MoveStackEntryDto {
            position: self.position.canonical_string(),
            history_positions: self
                .history_positions
                .iter()
                .map(Po::canonical_string)
                .collect(),
            no_progress_ply: self.no_progress_ply,
            turn_index: self.turn_index,
            last_engine_reverse_move: self.last_engine_reverse_move.map(|mv| mv.to_string()),
            last_move: self.last_move.map(|mv| mv.to_string()),
            result: self.result.clone(),
        }
    }
}

impl TryFrom<MoveStackEntryDto> for MoveStackEntry {
    type Error = String;

    fn try_from(value: MoveStackEntryDto) -> Result<Self, Self::Error> {
        Ok(Self {
            position: parse_position(&value.position)?,
            history_positions: value
                .history_positions
                .into_iter()
                .map(|position| parse_position(&position))
                .collect::<Result<Vec<_>, _>>()?,
            no_progress_ply: value.no_progress_ply,
            turn_index: value.turn_index,
            last_engine_reverse_move: value
                .last_engine_reverse_move
                .as_deref()
                .map(parse_move)
                .transpose()?,
            last_move: value.last_move.as_deref().map(parse_move).transpose()?,
            result: value.result,
        })
    }
}

impl TryFrom<SessionDto> for SessionState {
    type Error = String;

    fn try_from(value: SessionDto) -> Result<Self, Self::Error> {
        Ok(Self {
            position: parse_position(&value.position)?,
            history_positions: value
                .history_positions
                .into_iter()
                .map(|position| parse_position(&position))
                .collect::<Result<Vec<_>, _>>()?,
            no_progress_ply: value.no_progress_ply,
            turn_index: value.turn_index,
            last_engine_reverse_move: value
                .last_engine_reverse_move
                .as_deref()
                .map(parse_move)
                .transpose()?,
            move_stack: value
                .move_stack
                .into_iter()
                .map(MoveStackEntry::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            last_move: value.last_move.as_deref().map(parse_move).transpose()?,
            result: value.result,
        })
    }
}

fn color_name(color: Co) -> &'static str {
    match color {
        Co::Black => "black",
        Co::White => "white",
    }
}

fn parse_position(value: &str) -> Result<Po, String> {
    Po::from_str(value).map_err(|_| format!("invalid position: {value}"))
}

fn parse_move(value: &str) -> Result<Mv, String> {
    Mv::from_str(value).map_err(|_| format!("invalid move: {value}"))
}

fn parse_cell(value: &str) -> Result<Ci, String> {
    let coord = Coord::parse(value).ok_or_else(|| format!("invalid cell id: {value}"))?;
    gm().index_of_coord(coord)
        .ok_or_else(|| format!("unknown cell id: {value}"))
}

fn parse_cells(values: &[String]) -> Result<Vec<Ci>, String> {
    let mut cells = Vec::with_capacity(values.len());
    for value in values {
        let cell = parse_cell(value)?;
        if cells.contains(&cell) {
            return Err(format!("duplicate cell in selection: {value}"));
        }
        cells.push(cell);
    }
    Ok(cells)
}

fn canonicalize_position(position: &Po) -> Result<Po, String> {
    Po::new(
        position._sT(),
        position.black().to_vec(),
        position.white().to_vec(),
    )
    .map_err(|_| "failed to canonicalize position".to_owned())
}

fn result_win(winner: Co, reason: &str) -> GameResultDto {
    GameResultDto {
        kind: "win".to_owned(),
        winner: Some(color_name(winner).to_owned()),
        reason: reason.to_owned(),
    }
}

fn result_draw(reason: &str) -> GameResultDto {
    GameResultDto {
        kind: "draw".to_owned(),
        winner: None,
        reason: reason.to_owned(),
    }
}

fn detect_result(
    position: &Po,
    _history_positions: &[Po],
    turn_index: u16,
) -> Option<GameResultDto> {
    if position.black().len() <= 8 {
        return Some(result_win(Co::White, "black_marbles_reduced_to_eight"));
    }
    if position.white().len() <= 8 {
        return Some(result_win(Co::Black, "white_marbles_reduced_to_eight"));
    }

    if turn_index >= MAX_GAME_TURNS {
        let black = position.black().len();
        let white = position.white().len();
        if black > white {
            return Some(result_win(Co::Black, "max_turns_material_advantage"));
        }
        if white > black {
            return Some(result_win(Co::White, "max_turns_material_advantage"));
        }
        return Some(result_draw("max_turns_even_material"));
    }

    None
}

fn validate_selection(position: &Po, selected: &[Ci]) -> Result<Vec<Ci>, String> {
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    if selected.len() > 3 {
        return Err("selection must contain at most three marbles".to_owned());
    }

    let side = position._sT();
    for cell in selected {
        if position.occupant(*cell) != Some(side) {
            return Err("selection contains a cell not owned by the side to move".to_owned());
        }
    }

    let mut sorted = selected.to_vec();
    sorted.sort_unstable();
    let _ = Mv::from_cells(&sorted, Di::East)
        .map_err(|_| "selection must form a contiguous line of one to three marbles".to_owned())?;
    Ok(sorted)
}

fn translated_destinations(mv: &Mv) -> Vec<Ci> {
    mv._ad()
        .iter()
        .filter_map(|cell| _cJ(*cell, mv.direction()))
        .collect()
}

fn broadside_anchor(mv: &Mv) -> Ci {
    let destinations = translated_destinations(mv);
    if destinations.is_empty() {
        return mv._ad()[0];
    }

    let gm = gm();
    let mut sum_q = 0.0f64;
    let mut sum_r = 0.0f64;
    for cell in &destinations {
        let axial = gm.cell(*cell).axial;
        sum_q += f64::from(axial.q);
        sum_r += f64::from(axial.r);
    }
    let len = destinations.len() as f64;
    let centroid_q = sum_q / len;
    let centroid_r = sum_r / len;

    destinations
        .into_iter()
        .min_by(|left, right| {
            let left_axial = gm.cell(*left).axial;
            let right_axial = gm.cell(*right).axial;
            let left_dist = (f64::from(left_axial.q) - centroid_q).powi(2)
                + (f64::from(left_axial.r) - centroid_r).powi(2);
            let right_dist = (f64::from(right_axial.q) - centroid_q).powi(2)
                + (f64::from(right_axial.r) - centroid_r).powi(2);
            left_dist
                .partial_cmp(&right_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        })
        .unwrap_or(mv._ad()[0])
}

fn ejection_anchor(position: &Po, mv: &Mv) -> Option<Ci> {
    let front = move_front_cell(mv._ad(), mv.direction())?;
    let enemy = position._sT().other();
    let mut current = _cJ(front, mv.direction())?;
    let mut last_enemy = current;

    while position.occupant(current) == Some(enemy) {
        last_enemy = current;
        match _cJ(current, mv.direction()) {
            Some(next) => current = next,
            None => return Some(last_enemy),
        }
    }

    Some(last_enemy)
}

fn move_candidate_from_move(position: &Po, rq: &Rq, mv: Mv) -> MoveCandidateDto {
    let axis = move_group_axis(mv._ad());
    let is_inline = axis.is_some_and(|ax| move_is_inline(ax, mv.direction()));
    let side = position._sT();
    let front = move_front_cell(mv._ad(), mv.direction());
    let first_ahead = front.and_then(|cell| _cJ(cell, mv.direction()));
    let is_push =
        is_inline && first_ahead.is_some_and(|cell| position.occupant(cell) == Some(side.other()));
    let is_ejection = rq._bY(&mv).map(|entry| entry.ej).unwrap_or(false);

    let anchor = if is_inline {
        if is_push {
            if is_ejection {
                ejection_anchor(position, &mv).unwrap_or_else(|| mv._ad()[0])
            } else {
                first_ahead.unwrap_or_else(|| mv._ad()[0])
            }
        } else {
            front
                .and_then(|cell| _cJ(cell, mv.direction()))
                .unwrap_or_else(|| mv._ad()[0])
        }
    } else {
        broadside_anchor(&mv)
    };

    MoveCandidateDto {
        r#move: mv.to_string(),
        source_cells: mv._ad().iter().map(ToString::to_string).collect(),
        direction: mv.direction().as_str().to_owned(),
        anchor_cell: anchor.to_string(),
        is_inline,
        is_broadside: !is_inline,
        is_push,
        is_ejection,
    }
}

fn legal_moves_for_selection_state(
    session: &SessionState,
    selected_ids: &[String],
) -> Result<Vec<MoveCandidateDto>, String> {
    if session.result.is_some() || selected_ids.is_empty() {
        return Ok(Vec::new());
    }

    let selected = parse_cells(selected_ids)?;
    let sorted = validate_selection(&session.position, &selected)?;
    let rq =
        Rq::new(session.position.clone()).map_err(|_| "invalid session position".to_owned())?;
    let show_moves_for_any_group_with_cell = sorted.len() == 1;
    let selected_cell = sorted[0];
    let mut moves = rq
        .generate_legal_moves()
        .into_iter()
        .filter(|mv| {
            if show_moves_for_any_group_with_cell {
                mv._ad().contains(&selected_cell)
            } else {
                mv._ad() == sorted.as_slice()
            }
        })
        .map(|mv| move_candidate_from_move(&session.position, &rq, mv))
        .collect::<Vec<_>>();
    moves.sort_by(|left, right| left.r#move.cmp(&right.r#move));
    Ok(moves)
}

fn apply_move_state(mut session: SessionState, move_text: &str) -> Result<SessionState, String> {
    if session.result.is_some() {
        return Err("cannot apply a move in a finished game".to_owned());
    }

    let mv = parse_move(move_text)?;
    let side = session.position._sT();
    let mut rq =
        Rq::new(session.position.clone()).map_err(|_| "invalid session position".to_owned())?;
    let move_entry = rq
        ._bY(&mv)
        .ok_or_else(|| "illegal move for current position".to_owned())?;
    let snapshot = MoveStackEntry::from_state(&session);

    rq.apply_move(&mv)
        .map_err(|_| "failed to apply move".to_owned())?;

    session.move_stack.push(snapshot);
    session.history_positions.push(session.position.clone());
    session.position = canonicalize_position(rq.position())?;
    session.no_progress_ply = if move_entry.ej {
        0
    } else {
        session.no_progress_ply.saturating_add(1)
    };
    session.turn_index = session.turn_index.saturating_add(1);
    session.last_move = Some(mv);
    if side == Co::White {
        session.last_engine_reverse_move = reverse_move(mv);
    }
    session.result = detect_result(
        &session.position,
        &session.history_positions,
        session.turn_index,
    );

    Ok(session)
}

fn restore_snapshot(session: &mut SessionState, snapshot: MoveStackEntry) {
    session.position = snapshot.position;
    session.history_positions = snapshot.history_positions;
    session.no_progress_ply = snapshot.no_progress_ply;
    session.turn_index = snapshot.turn_index;
    session.last_engine_reverse_move = snapshot.last_engine_reverse_move;
    session.last_move = snapshot.last_move;
    session.result = snapshot.result;
}

fn undo_full_turn_state(mut session: SessionState) -> Result<SessionState, String> {
    if session.move_stack.is_empty() {
        return Ok(session);
    }

    let pops = if session.position._sT() == Co::White {
        1
    } else {
        2
    };
    let mut restored = 0usize;
    for _ in 0..pops {
        let Some(snapshot) = session.move_stack.pop() else {
            break;
        };
        restore_snapshot(&mut session, snapshot);
        restored += 1;
    }

    if restored == 0 {
        return Err("no moves available to undo".to_owned());
    }

    Ok(session)
}

fn search_best_move_state(session: &SessionState, depth: u8) -> Result<SearchResultDto, String> {
    search_best_move_with_limits_state(session, depth, None)
}

fn search_best_move_with_limits_state(
    session: &SessionState,
    depth: u8,
    max_time_ms: Option<u64>,
) -> Result<SearchResultDto, String> {
    if session.result.is_some() {
        return Err("cannot search a finished game".to_owned());
    }

    let score_side = session.position._sT();
    let result = match (depth, max_time_ms.filter(|time_ms| *time_ms > 0)) {
        (0, None) => {
            return Err("at least one search limit must be enabled".to_owned());
        }
        (0, Some(time_ms)) => search_timed_with_gt(
            &session.position,
            &session.history_positions,
            session.no_progress_ply,
            session.turn_index,
            time_ms,
            session.last_engine_reverse_move,
        )?,
        (depth, Some(time_ms)) => {
            let hy = session
                .history_positions
                .iter()
                .map(_X::fp)
                .collect::<Vec<_>>();
            let mut searcher = Searcher::new_timed_depth(
                time_ms,
                depth,
                session.last_engine_reverse_move,
                hy.len() <= 1 && session.no_progress_ply == 0,
            );
            let result = searcher
                .search(
                    &session.position,
                    &hy,
                    session.no_progress_ply,
                    session.turn_index.min(MAX_GAME_TURNS),
                )
                .map(|(result, _)| result);
            searcher.persist();
            result?
        }
        (depth, None) => search_depth_with_gt(
            &session.position,
            &session.history_positions,
            session.no_progress_ply,
            session.turn_index,
            depth,
            session.last_engine_reverse_move,
        )?,
    };
    let best_move = result
        .bm
        .ok_or_else(|| "engine search returned no legal move".to_owned())?;
    let white_perspective_score = if score_side == Co::White {
        result.score
    } else {
        -result.score
    };

    Ok(SearchResultDto {
        best_move: best_move.to_string(),
        score: result.score,
        depth: result.dp,
        nodes: result.nodes,
        white_perspective_score,
        black_perspective_score: -white_perspective_score,
    })
}

fn session_status_state(session: &SessionState) -> StatusDto {
    StatusDto {
        side_to_move: color_name(session.position._sT()).to_owned(),
        black_count: session.black_count(),
        white_count: session.white_count(),
        turn_index: session.turn_index,
        no_progress_ply: session.no_progress_ply,
        result: session.result.clone(),
        is_game_over: session.result.is_some(),
        can_take_back: !session.move_stack.is_empty(),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn session_from_position(position: &str, side_result: Option<GameResultDto>) -> SessionState {
        SessionState {
            position: Po::from_str(position).unwrap(),
            history_positions: Vec::new(),
            no_progress_ply: 0,
            turn_index: 0,
            last_engine_reverse_move: None,
            move_stack: Vec::new(),
            last_move: None,
            result: side_result,
        }
    }

    #[test]
    fn new_session_uses_belgian_daisy_with_black_to_move() {
        let session = SessionState::new();
        assert_eq!(session.position.canonical_string(), START_POSITION);
        assert_eq!(session.position._sT(), Co::Black);
        assert_eq!(session.black_count(), 14);
        assert_eq!(session.white_count(), 14);
        assert!(session.result.is_none());
    }

    #[test]
    fn selection_returns_moves_for_a_black_marble_in_start_position() {
        let session = SessionState::new();
        let moves = legal_moves_for_selection_state(&session, &[String::from("C5")]).unwrap();
        assert!(!moves.is_empty());
        assert!(
            moves
                .iter()
                .all(|candidate| candidate.source_cells.iter().any(|cell| cell == "C5"))
        );
        assert!(
            moves
                .iter()
                .any(|candidate| candidate.source_cells.len() > 1)
        );
    }

    #[test]
    fn apply_move_and_undo_full_turn_restore_the_previous_session() {
        let session = SessionState::new();
        let rq = Rq::new(session.position.clone()).unwrap();
        let first_move = rq.generate_legal_moves()[0].to_string();
        let after_move = apply_move_state(session.clone(), &first_move).unwrap();
        assert_eq!(after_move.position._sT(), Co::White);
        let restored = undo_full_turn_state(after_move).unwrap();
        assert_eq!(restored.to_dto(), session.to_dto());
    }

    #[test]
    fn search_wrapper_returns_a_legal_white_move_at_fixed_depth() {
        let session = session_from_position(
            "aba-v1;stm=w;black=A4,A5,B4,B5,B6,C5,C6,H1,H2;white=A1,A2,B1,B2,B3,C2,C3,H4,H5",
            None,
        );
        let search = search_best_move_state(&session, 17).unwrap();
        assert_eq!(search.depth, 17);

        let rq = Rq::new(session.position.clone()).unwrap();
        let legal_moves = rq
            .generate_legal_moves()
            .into_iter()
            .map(|mv| mv.to_string())
            .collect::<Vec<_>>();
        assert!(legal_moves.contains(&search.best_move));
    }

    #[test]
    fn search_replies_after_a_black_opening_move() {
        let session = SessionState::new();
        let after_black = apply_move_state(session, "A5,B5,C5>SW").unwrap();
        let search = search_best_move_state(&after_black, 3).unwrap();

        let rq = Rq::new(after_black.position.clone()).unwrap();
        let legal_moves = rq
            .generate_legal_moves()
            .into_iter()
            .map(|mv| mv.to_string())
            .collect::<Vec<_>>();
        assert!(legal_moves.contains(&search.best_move));
    }

    #[test]
    fn search_with_limits_returns_a_legal_white_move() {
        let session = SessionState::new();
        let after_black = apply_move_state(session, "A5,B5,C5>SW").unwrap();
        let search = search_best_move_with_limits_state(&after_black, 3, Some(1_000)).unwrap();

        let rq = Rq::new(after_black.position.clone()).unwrap();
        let legal_moves = rq
            .generate_legal_moves()
            .into_iter()
            .map(|mv| mv.to_string())
            .collect::<Vec<_>>();
        assert!(legal_moves.contains(&search.best_move));
        assert!(search.depth <= 3);
    }

    #[test]
    fn zero_time_limit_uses_fixed_depth_behavior() {
        let session = SessionState::new();
        let after_black = apply_move_state(session, "A5,B5,C5>SW").unwrap();
        let fixed = search_best_move_state(&after_black, 3).unwrap();
        let zero_time = search_best_move_with_limits_state(&after_black, 3, Some(0)).unwrap();

        assert_eq!(zero_time.depth, fixed.depth);
        assert_eq!(zero_time.best_move, fixed.best_move);
        assert_eq!(zero_time.score, fixed.score);
    }

    #[test]
    fn time_only_limit_returns_a_legal_white_move() {
        let session = SessionState::new();
        let after_black = apply_move_state(session, "A5,B5,C5>SW").unwrap();
        let search = search_best_move_with_limits_state(&after_black, 0, Some(1_000)).unwrap();

        let rq = Rq::new(after_black.position.clone()).unwrap();
        let legal_moves = rq
            .generate_legal_moves()
            .into_iter()
            .map(|mv| mv.to_string())
            .collect::<Vec<_>>();
        assert!(legal_moves.contains(&search.best_move));
        assert!(search.depth >= 1);
    }

    #[test]
    fn search_depth_is_capped_by_remaining_max_turns() {
        let mut session = SessionState::new();
        session.turn_index = MAX_GAME_TURNS - 2;

        let fixed = search_best_move_state(&session, 17).unwrap();
        assert!(fixed.depth <= 2);

        let timed = search_best_move_with_limits_state(&session, 0, Some(20)).unwrap();
        assert!(timed.depth <= 2);
    }

    #[test]
    fn core_search_stops_at_max_turn_horizon() {
        let session = SessionState::new();
        let gt = MAX_GAME_TURNS - 2;

        let fixed = search_depth_with_gt(
            &session.position,
            &session.history_positions,
            session.no_progress_ply,
            gt,
            17,
            None,
        )
        .unwrap();
        assert!(fixed.dp <= 2);

        let timed = search_timed_with_gt(
            &session.position,
            &session.history_positions,
            session.no_progress_ply,
            gt,
            20,
            None,
        )
        .unwrap();
        assert!(timed.dp <= 2);
    }

    #[test]
    fn final_ply_search_returns_terminal_score() {
        let position = Po::from_str("aba-v1;stm=b;black=A4,A5,B4,B5,B6,C5,C6,G2,G3;white=A1,A2,B1,B2,B3,C2,C3,G5,G6,H4,H5,H6,I4,I5").unwrap();

        let result = search_depth_with_gt(&position, &[], 0, MAX_GAME_TURNS - 1, 17, None).unwrap();

        assert_eq!(result.dp, 1);
        assert!(
            result.score.abs() >= _D - 1 || result.score == 0,
            "expected terminal-class score at max-turn horizon, got {}",
            result.score
        );
    }

    #[test]
    fn five_ply_horizon_search_returns_terminal_score() {
        let position = Po::from_str("aba-v1;stm=b;black=A4,A5,B4,B5,B6,C5,C6,G2,G3;white=A1,A2,B1,B2,B3,C2,C3,G5,G6,H4,H5,H6,I4,I5").unwrap();

        let result = search_depth_with_gt(&position, &[], 0, MAX_GAME_TURNS - 5, 17, None).unwrap();

        assert_eq!(result.dp, 5);
        assert!(
            result.score.abs() >= _D - 5 || result.score == 0,
            "expected terminal-class score at max-turn horizon, got {}",
            result.score
        );
    }

    #[test]
    fn disabling_all_search_limits_is_rejected() {
        let session = SessionState::new();
        let after_black = apply_move_state(session, "A5,B5,C5>SW").unwrap();
        let error = search_best_move_with_limits_state(&after_black, 0, Some(0)).unwrap_err();
        assert!(error.contains("at least one search limit"));
    }

    #[test]
    fn game_end_detection_does_not_mark_threefold_repetition_as_draw() {
        let position = Po::from_str(START_POSITION).unwrap();
        let result = detect_result(&position, &[position.clone(), position.clone()], 0);
        assert!(result.is_none());
    }

    #[test]
    fn game_end_detection_marks_even_material_at_max_turns_as_draw() {
        let position = Po::from_str(START_POSITION).unwrap();
        let result = detect_result(&position, &[], MAX_GAME_TURNS).unwrap();
        assert_eq!(result.kind, "draw");
        assert_eq!(result.reason, "max_turns_even_material");
    }

    #[test]
    fn game_end_detection_marks_material_leader_as_winner_at_max_turns() {
        let position = Po::from_str(
            "aba-v1;stm=b;black=A4,A5,B4,B5,B6,C5,C6,G2,G3,H1,H2,H3,I1,I2;white=A1,A2,B1,B2,B3,C2,C3,G5,G6,H4,H5,H6,I4",
        )
        .unwrap();
        let result = detect_result(&position, &[], MAX_GAME_TURNS).unwrap();
        assert_eq!(result.kind, "win");
        assert_eq!(result.winner.as_deref(), Some("black"));
        assert_eq!(result.reason, "max_turns_material_advantage");
    }
}
