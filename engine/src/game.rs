use crate::MAX_GAME_TURNS;
use crate::api::{
    GameResultDto, MoveCandidateDto, MoveStackEntryDto, SearchResultDto, SessionDto, StatusDto,
    reverse_move,
};
use crate::board::{CellId, Color, Coord, Direction, Move, Position, geometry};
use crate::movegen::PositionState;
use crate::search::{
    SearchResult, move_front_cell, move_group_axis, move_is_inline, neighbor_cell,
    search_fixed_depth_with_turn, search_timed_depth_with_turn, search_timed_with_turn,
};
use std::str::FromStr;

pub const START_POSITION: &str = "SS1ss/SSSsss/1SS1ss1/8/9/8/1ss1SS1/sssSSS/ss1SS 0 0 B 0 0";

#[derive(Clone, Debug)]
struct MoveStackEntry {
    position: Position,
    history_positions: Vec<Position>,
    no_progress_ply: u16,
    turn_index: u16,
    last_engine_reverse_move: Option<Move>,
    last_move: Option<Move>,
    result: Option<GameResultDto>,
}

#[derive(Clone, Debug)]
pub struct SessionState {
    position: Position,
    history_positions: Vec<Position>,
    no_progress_ply: u16,
    turn_index: u16,
    last_engine_reverse_move: Option<Move>,
    move_stack: Vec<MoveStackEntry>,
    last_move: Option<Move>,
    result: Option<GameResultDto>,
}

impl SessionState {
    pub fn new() -> Self {
        Self {
            position: Position::from_str(START_POSITION).expect("valid Belgian Daisy start"),
            history_positions: Vec::new(),
            no_progress_ply: 0,
            turn_index: 0,
            last_engine_reverse_move: None,
            move_stack: Vec::new(),
            last_move: None,
            result: None,
        }
    }

    pub fn from_position(position: Position, result: Option<GameResultDto>) -> Self {
        Self {
            position,
            history_positions: Vec::new(),
            no_progress_ply: 0,
            turn_index: 0,
            last_engine_reverse_move: None,
            move_stack: Vec::new(),
            last_move: None,
            result,
        }
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn history_positions(&self) -> &[Position] {
        &self.history_positions
    }

    pub fn no_progress_ply(&self) -> u16 {
        self.no_progress_ply
    }

    pub fn set_turn_index(&mut self, turn_index: u16) {
        self.turn_index = turn_index;
    }

    pub fn result(&self) -> Option<&GameResultDto> {
        self.result.as_ref()
    }

    fn side_to_move(&self) -> Color {
        self.position.side_to_move()
    }

    pub fn black_count(&self) -> usize {
        self.position.black().len()
    }

    pub fn white_count(&self) -> usize {
        self.position.white().len()
    }

    pub fn to_dto(&self) -> SessionDto {
        SessionDto {
            position: self.position.canonical_string(),
            side_to_move: color_name(self.side_to_move()).to_owned(),
            history_positions: self
                .history_positions
                .iter()
                .map(Position::canonical_string)
                .collect(),
            no_progress_ply: self.no_progress_ply,
            turn_index: self.turn_index,
            last_engine_reverse_move: self
                .last_engine_reverse_move
                .map(|candidate_move| candidate_move.to_string()),
            move_stack: self.move_stack.iter().map(MoveStackEntry::to_dto).collect(),
            black_count: self.black_count(),
            white_count: self.white_count(),
            last_move: self
                .last_move
                .map(|candidate_move| candidate_move.to_string()),
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
                .map(Position::canonical_string)
                .collect(),
            no_progress_ply: self.no_progress_ply,
            turn_index: self.turn_index,
            last_engine_reverse_move: self
                .last_engine_reverse_move
                .map(|candidate_move| candidate_move.to_string()),
            last_move: self
                .last_move
                .map(|candidate_move| candidate_move.to_string()),
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

fn color_name(color: Color) -> &'static str {
    match color {
        Color::Black => "black",
        Color::White => "white",
    }
}

fn parse_position(value: &str) -> Result<Position, String> {
    Position::from_str(value).map_err(|_| format!("invalid position: {value}"))
}

fn parse_move(value: &str) -> Result<Move, String> {
    Move::from_str(value).map_err(|_| format!("invalid move: {value}"))
}

fn parse_cell(value: &str) -> Result<CellId, String> {
    let coord = Coord::parse(value).ok_or_else(|| format!("invalid cell id: {value}"))?;
    geometry()
        .index_of_coord(coord)
        .ok_or_else(|| format!("unknown cell id: {value}"))
}

fn parse_cells(values: &[String]) -> Result<Vec<CellId>, String> {
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

fn canonicalize_position(position: &Position) -> Result<Position, String> {
    Position::new(
        position.side_to_move(),
        position.black().to_vec(),
        position.white().to_vec(),
    )
    .map_err(|_| "failed to canonicalize position".to_owned())
}

fn result_win(winner: Color, reason: &str) -> GameResultDto {
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

pub fn detect_result(
    position: &Position,
    _history_positions: &[Position],
    turn_index: u16,
) -> Option<GameResultDto> {
    if position.black().len() <= 8 {
        return Some(result_win(Color::White, "black_marbles_reduced_to_eight"));
    }
    if position.white().len() <= 8 {
        return Some(result_win(Color::Black, "white_marbles_reduced_to_eight"));
    }

    if turn_index >= MAX_GAME_TURNS {
        let black = position.black().len();
        let white = position.white().len();
        if black > white {
            return Some(result_win(Color::Black, "max_turns_material_advantage"));
        }
        if white > black {
            return Some(result_win(Color::White, "max_turns_material_advantage"));
        }
        return Some(result_draw("max_turns_even_material"));
    }

    None
}

fn validate_selection(position: &Position, selected: &[CellId]) -> Result<Vec<CellId>, String> {
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    if selected.len() > 3 {
        return Err("selection must contain at most three marbles".to_owned());
    }

    let side = position.side_to_move();
    for cell in selected {
        if position.occupant(*cell) != Some(side) {
            return Err("selection contains a cell not owned by the side to move".to_owned());
        }
    }

    let mut sorted = selected.to_vec();
    sorted.sort_unstable();
    let _ = Move::from_cells(&sorted, Direction::East)
        .map_err(|_| "selection must form a contiguous line of one to three marbles".to_owned())?;
    Ok(sorted)
}

fn translated_destinations(candidate_move: &Move) -> Vec<CellId> {
    candidate_move
        .source_cells()
        .iter()
        .filter_map(|cell| neighbor_cell(*cell, candidate_move.direction()))
        .collect()
}

fn broadside_anchor(candidate_move: &Move) -> CellId {
    let destinations = translated_destinations(candidate_move);
    if destinations.is_empty() {
        return candidate_move.source_cells()[0];
    }

    let geometry = geometry();
    let mut sum_q = 0.0f64;
    let mut sum_r = 0.0f64;
    for cell in &destinations {
        let axial = geometry.cell(*cell).axial;
        sum_q += f64::from(axial.q);
        sum_r += f64::from(axial.r);
    }
    let len = destinations.len() as f64;
    let centroid_q = sum_q / len;
    let centroid_r = sum_r / len;

    destinations
        .into_iter()
        .min_by(|left, right| {
            let left_axial = geometry.cell(*left).axial;
            let right_axial = geometry.cell(*right).axial;
            let left_dist = (f64::from(left_axial.q) - centroid_q).powi(2)
                + (f64::from(left_axial.r) - centroid_r).powi(2);
            let right_dist = (f64::from(right_axial.q) - centroid_q).powi(2)
                + (f64::from(right_axial.r) - centroid_r).powi(2);
            left_dist
                .partial_cmp(&right_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        })
        .unwrap_or(candidate_move.source_cells()[0])
}

fn ejection_anchor(position: &Position, candidate_move: &Move) -> Option<CellId> {
    let front = move_front_cell(candidate_move.source_cells(), candidate_move.direction())?;
    let enemy = position.side_to_move().other();
    let mut current = neighbor_cell(front, candidate_move.direction())?;
    let mut last_enemy = current;

    while position.occupant(current) == Some(enemy) {
        last_enemy = current;
        match neighbor_cell(current, candidate_move.direction()) {
            Some(next) => current = next,
            None => return Some(last_enemy),
        }
    }

    Some(last_enemy)
}

fn move_candidate_from_move(
    position: &Position,
    position_state: &PositionState,
    candidate_move: Move,
) -> MoveCandidateDto {
    let axis = move_group_axis(candidate_move.source_cells());
    let is_inline = axis.is_some_and(|axis| move_is_inline(axis, candidate_move.direction()));
    let side = position.side_to_move();
    let front = move_front_cell(candidate_move.source_cells(), candidate_move.direction());
    let first_ahead = front.and_then(|cell| neighbor_cell(cell, candidate_move.direction()));
    let is_push =
        is_inline && first_ahead.is_some_and(|cell| position.occupant(cell) == Some(side.other()));
    let is_ejection = position_state
        .legal_move_entry(&candidate_move)
        .map(|entry| entry.is_ejection)
        .unwrap_or(false);

    let anchor = if is_inline {
        if is_push {
            if is_ejection {
                ejection_anchor(position, &candidate_move)
                    .unwrap_or_else(|| candidate_move.source_cells()[0])
            } else {
                first_ahead.unwrap_or_else(|| candidate_move.source_cells()[0])
            }
        } else {
            front
                .and_then(|cell| neighbor_cell(cell, candidate_move.direction()))
                .unwrap_or_else(|| candidate_move.source_cells()[0])
        }
    } else {
        broadside_anchor(&candidate_move)
    };

    MoveCandidateDto {
        r#move: candidate_move.to_string(),
        source_cells: candidate_move
            .source_cells()
            .iter()
            .map(ToString::to_string)
            .collect(),
        direction: candidate_move.direction().as_str().to_owned(),
        anchor_cell: anchor.to_string(),
        is_inline,
        is_broadside: !is_inline,
        is_push,
        is_ejection,
    }
}

pub fn legal_moves_for_selection_state(
    session: &SessionState,
    selected_ids: &[String],
) -> Result<Vec<MoveCandidateDto>, String> {
    if session.result.is_some() || selected_ids.is_empty() {
        return Ok(Vec::new());
    }

    let selected = parse_cells(selected_ids)?;
    let sorted = validate_selection(&session.position, &selected)?;
    let position_state = PositionState::new(session.position.clone())
        .map_err(|_| "invalid session position".to_owned())?;
    let show_moves_for_any_group_with_cell = sorted.len() == 1;
    let selected_cell = sorted[0];
    let mut moves = position_state
        .generate_legal_moves()
        .into_iter()
        .filter(|candidate_move| {
            if show_moves_for_any_group_with_cell {
                candidate_move.source_cells().contains(&selected_cell)
            } else {
                candidate_move.source_cells() == sorted.as_slice()
            }
        })
        .map(|candidate_move| {
            move_candidate_from_move(&session.position, &position_state, candidate_move)
        })
        .collect::<Vec<_>>();
    moves.sort_by(|left, right| left.r#move.cmp(&right.r#move));
    Ok(moves)
}

pub fn apply_move_state(
    mut session: SessionState,
    move_text: &str,
) -> Result<SessionState, String> {
    if session.result.is_some() {
        return Err("cannot apply a move in a finished game".to_owned());
    }

    let candidate_move = parse_move(move_text)?;
    let side = session.position.side_to_move();
    let mut position_state = PositionState::new(session.position.clone())
        .map_err(|_| "invalid session position".to_owned())?;
    let move_entry = position_state
        .legal_move_entry(&candidate_move)
        .ok_or_else(|| "illegal move for current position".to_owned())?;
    let snapshot = MoveStackEntry::from_state(&session);

    position_state
        .apply_move(&candidate_move)
        .map_err(|_| "failed to apply move".to_owned())?;

    session.move_stack.push(snapshot);
    session.history_positions.push(session.position.clone());
    session.position = canonicalize_position(position_state.position())?;
    session.no_progress_ply = if move_entry.is_ejection {
        0
    } else {
        session.no_progress_ply.saturating_add(1)
    };
    session.turn_index = session.turn_index.saturating_add(1);
    session.last_move = Some(candidate_move);
    if side == Color::White {
        session.last_engine_reverse_move = reverse_move(candidate_move);
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

pub fn undo_full_turn_state(mut session: SessionState) -> Result<SessionState, String> {
    if session.move_stack.is_empty() {
        return Ok(session);
    }

    let pops = if session.position.side_to_move() == Color::White {
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

pub fn search_best_move_state(
    session: &SessionState,
    depth: u8,
) -> Result<SearchResultDto, String> {
    search_best_move_with_limits_state(session, depth, None)
}

pub fn search_best_move_with_limits_state(
    session: &SessionState,
    depth: u8,
    max_time_ms: Option<u64>,
) -> Result<SearchResultDto, String> {
    search_best_move_with_limits_impl(
        session,
        depth,
        max_time_ms,
        |session, depth, time_ms| match (depth, time_ms) {
            (0, Some(time_ms)) => search_timed_with_turn(
                &session.position,
                &session.history_positions,
                session.no_progress_ply,
                session.turn_index,
                time_ms,
                session.last_engine_reverse_move,
            ),
            (depth, Some(time_ms)) => search_timed_depth_with_turn(
                &session.position,
                &session.history_positions,
                session.no_progress_ply,
                session.turn_index,
                time_ms,
                depth,
                session.last_engine_reverse_move,
            ),
            (depth, None) => search_fixed_depth_with_turn(
                &session.position,
                &session.history_positions,
                session.no_progress_ply,
                session.turn_index,
                depth,
                session.last_engine_reverse_move,
            ),
        },
    )
}

fn search_best_move_with_limits_impl(
    session: &SessionState,
    depth: u8,
    max_time_ms: Option<u64>,
    mut search: impl FnMut(&SessionState, u8, Option<u64>) -> Result<SearchResult, String>,
) -> Result<SearchResultDto, String> {
    if session.result.is_some() {
        return Err("cannot search a finished game".to_owned());
    }

    let score_side = session.position.side_to_move();
    let result = match (depth, max_time_ms.filter(|time_ms| *time_ms > 0)) {
        (0, None) => {
            return Err("at least one search limit must be enabled".to_owned());
        }
        (depth, time_ms) => search(session, depth, time_ms)?,
    };
    let best_move = result
        .best_move
        .ok_or_else(|| "engine search returned no legal move".to_owned())?;
    let white_perspective_score = if score_side == Color::White {
        result.score
    } else {
        -result.score
    };

    Ok(SearchResultDto {
        best_move: best_move.to_string(),
        score: result.score,
        depth: result.depth,
        nodes: result.nodes,
        white_perspective_score,
        black_perspective_score: -white_perspective_score,
    })
}

pub fn session_status_state(session: &SessionState) -> StatusDto {
    StatusDto {
        side_to_move: color_name(session.position.side_to_move()).to_owned(),
        black_count: session.black_count(),
        white_count: session.white_count(),
        turn_index: session.turn_index,
        no_progress_ply: session.no_progress_ply,
        result: session.result.clone(),
        is_game_over: session.result.is_some(),
        can_take_back: !session.move_stack.is_empty(),
    }
}
