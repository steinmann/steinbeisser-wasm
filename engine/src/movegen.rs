use crate::board::{
    ALL_DIRECTIONS, Color, Direction, EngineStateView, LineAxis, Move, MoveError, Position,
    PositionError, geometry,
};
use std::collections::BTreeSet;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UndoSnapshot {
    plan: MovePlan,
    previous_side_to_move: Color,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionState {
    position: Position,
    occupants: [Option<Color>; crate::board::CELL_COUNT],
    black_bits: u64,
    white_bits: u64,
    black_slots: [u8; crate::board::CELL_COUNT],
    white_slots: [u8; crate::board::CELL_COUNT],
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoveApplicationError {
    IllegalMove(Move),
    InvalidMoveShape(MoveError),
    InvalidPosition(PositionError),
}
impl PositionState {
    pub fn new(position: Position) -> Result<Self, PositionError> {
        position.validate()?;
        let occupants = build_occupants(&position);
        let black_bits = position.black().iter().fold(0u64, |accumulator, cell| {
            accumulator | (1u64 << cell.as_u8())
        });
        let white_bits = position.white().iter().fold(0u64, |accumulator, cell| {
            accumulator | (1u64 << cell.as_u8())
        });
        let (black_slots, white_slots) = slot_maps(&position);
        Ok(Self {
            position,
            occupants,
            black_bits,
            white_bits,
            black_slots,
            white_slots,
        })
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    pub fn occupant_fast(&self, cell: crate::board::CellId) -> Option<Color> {
        self.occupants[cell.as_usize()]
    }
    pub fn black_bits(&self) -> u64 {
        self.black_bits
    }
    pub fn white_bits(&self) -> u64 {
        self.white_bits
    }
    pub fn pass_turn(&mut self) -> Color {
        let previous_side_to_move = self.position.side_to_move();
        self.position
            .set_side_to_move(previous_side_to_move.other());
        previous_side_to_move
    }
    pub fn restore_side_to_move(&mut self, side_to_move: Color) {
        self.position.set_side_to_move(side_to_move);
    }
    pub fn generate_legal_moves(&self) -> Vec<Move> {
        let mut moves = BTreeSet::new();
        for group in enumerate_groups(&self.position, self.position.side_to_move()) {
            for direction in ALL_DIRECTIONS {
                let candidate_move = Move::new(group.clone(), direction).unwrap();
                if self.analyze_move(&candidate_move).is_ok() {
                    moves.insert(candidate_move);
                }
            }
        }
        moves.into_iter().collect()
    }
    pub(crate) fn generate_fast_legal_moves(&self, moves: &mut Vec<crate::search::LegalMoveEntry>) {
        let side_bits = self.side_bits(self.position.side_to_move());
        let enemy_bits = self.side_bits(self.position.side_to_move().other());
        let occupied_bits = side_bits | enemy_bits;
        let tables = crate::search::fast_movegen_tables();
        moves.clear();
        for group in &tables.source_groups {
            if side_bits & group.source_mask != group.source_mask {
                continue;
            }
            for direction in group.directions.iter().flatten() {
                let Some(is_ejection) = self.fast_group_direction_legality(
                    group.len as usize,
                    direction,
                    side_bits,
                    enemy_bits,
                    occupied_bits,
                ) else {
                    continue;
                };
                moves.push(crate::search::LegalMoveEntry {
                    candidate_move: direction.candidate_move,
                    is_ejection,
                    history_key: direction.history_key,
                });
            }
        }
    }
    fn side_bits(&self, side: Color) -> u64 {
        match side {
            Color::Black => self.black_bits,
            Color::White => self.white_bits,
        }
    }
    fn fast_group_direction_legality(
        &self,
        len: usize,
        direction: &crate::search::FastGroupDirection,
        side_bits: u64,
        enemy_bits: u64,
        occupied_bits: u64,
    ) -> Option<bool> {
        if direction.inline {
            self.fast_inline_legality(len, direction, side_bits, enemy_bits, occupied_bits)
        } else {
            self.fast_broadside_legality(direction.translated_mask, occupied_bits)
        }
    }
    fn fast_broadside_legality(&self, translated_mask: u64, occupied_bits: u64) -> Option<bool> {
        if occupied_bits & translated_mask == 0 {
            Some(false)
        } else {
            None
        }
    }
    fn fast_inline_legality(
        &self,
        len: usize,
        direction: &crate::search::FastGroupDirection,
        side_bits: u64,
        enemy_bits: u64,
        occupied_bits: u64,
    ) -> Option<bool> {
        let first_bit = direction.ray_bits[0];
        if occupied_bits & first_bit == 0 {
            return Some(false);
        }
        if side_bits & first_bit != 0 {
            return None;
        }
        let second_enemy = direction.ray_bits[1] != 0 && (enemy_bits & direction.ray_bits[1] != 0);
        let third_enemy =
            second_enemy && direction.ray_bits[2] != 0 && (enemy_bits & direction.ray_bits[2] != 0);
        let enemy_count = if third_enemy {
            3
        } else if second_enemy {
            2
        } else {
            1
        };
        if enemy_count >= len {
            return None;
        }
        match direction.landing[enemy_count - 1] {
            Some(cell) => {
                if occupied_bits & (1u64 << cell.as_u8()) != 0 {
                    None
                } else {
                    Some(false)
                }
            }
            None => Some(true),
        }
    }
    pub fn apply_move(
        &mut self,
        candidate_move: &Move,
    ) -> Result<UndoSnapshot, MoveApplicationError> {
        let plan = self.analyze_move(candidate_move)?;
        let previous_side_to_move = self.position.side_to_move();
        apply_plan_in_place(
            &mut self.position,
            &mut self.occupants,
            &mut self.black_bits,
            &mut self.white_bits,
            &mut self.black_slots,
            &mut self.white_slots,
            &plan,
        )?;
        Ok(UndoSnapshot {
            plan,
            previous_side_to_move,
        })
    }
    pub(crate) fn legal_move_entry(
        &self,
        candidate_move: &Move,
    ) -> Option<crate::search::LegalMoveEntry> {
        let plan = self.analyze_move(candidate_move).ok()?;
        Some(crate::search::LegalMoveEntry {
            candidate_move: *candidate_move,
            is_ejection: plan.pushed_cells.len > 0
                && plan
                    .pushed_destinations
                    .iter()
                    .any(|destination| destination.is_none()),
            history_key: crate::search::history_group_key(
                candidate_move.source_cells(),
                candidate_move.direction(),
            ),
        })
    }
    pub fn undo_move(&mut self, undo: UndoSnapshot) {
        undo_plan_in_place(
            &mut self.position,
            &mut self.occupants,
            &mut self.black_bits,
            &mut self.white_bits,
            &mut self.black_slots,
            &mut self.white_slots,
            &undo.plan,
            undo.previous_side_to_move,
        );
    }
    fn analyze_move(&self, candidate_move: &Move) -> Result<MovePlan, MoveApplicationError> {
        let side = self.position.side_to_move();
        let enemy = side.other();
        let occupants = &self.occupants;
        for cell in candidate_move.source_cells() {
            if occupants[cell.as_usize()] != Some(side) {
                return Err(MoveApplicationError::IllegalMove(candidate_move.clone()));
            }
        }
        if candidate_move.len() == 1 {
            return analyze_single_move(candidate_move, occupants, side);
        }
        let axis = group_axis(candidate_move.source_cells())
            .ok_or_else(|| MoveApplicationError::IllegalMove(candidate_move.clone()))?;
        if is_inline(axis, candidate_move.direction()) {
            analyze_inline_move(candidate_move, occupants, side, enemy)
        } else {
            analyze_broadside_move(candidate_move, occupants)
        }
    }
}
impl EngineStateView for PositionState {
    fn position(&self) -> &Position {
        &self.position
    }
}
impl From<MoveError> for MoveApplicationError {
    fn from(value: MoveError) -> Self {
        Self::InvalidMoveShape(value)
    }
}
impl From<PositionError> for MoveApplicationError {
    fn from(value: PositionError) -> Self {
        Self::InvalidPosition(value)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CellList<const N: usize> {
    len: u8,
    data: [crate::board::CellId; N],
}
impl<const N: usize> CellList<N> {
    fn new() -> Self {
        Self {
            len: 0,
            data: [crate::board::CellId::new_unchecked(0); N],
        }
    }
    fn from_slice(cells: &[crate::board::CellId]) -> Self {
        let mut out = Self::new();
        for cell in cells.iter().copied() {
            out.push(cell);
        }
        out
    }
    fn push(&mut self, cell: crate::board::CellId) {
        let index = self.len as usize;
        debug_assert!(index < N);
        self.data[index] = cell;
        self.len += 1;
    }
    fn as_slice(&self) -> &[crate::board::CellId] {
        &self.data[..self.len as usize]
    }
    fn iter(&self) -> std::slice::Iter<'_, crate::board::CellId> {
        self.as_slice().iter()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OptionalCellList<const N: usize> {
    len: u8,
    data: [Option<crate::board::CellId>; N],
}
impl<const N: usize> OptionalCellList<N> {
    fn new() -> Self {
        Self {
            len: 0,
            data: [None; N],
        }
    }
    fn push(&mut self, cell: Option<crate::board::CellId>) {
        let index = self.len as usize;
        debug_assert!(index < N);
        self.data[index] = cell;
        self.len += 1;
    }
    fn iter(&self) -> std::slice::Iter<'_, Option<crate::board::CellId>> {
        self.data[..self.len as usize].iter()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MovePlan {
    source_cells: CellList<3>,
    destination_cells: CellList<3>,
    pushed_cells: CellList<2>,
    pushed_destinations: OptionalCellList<2>,
}
fn enumerate_groups(position: &Position, side: Color) -> Vec<Vec<crate::board::CellId>> {
    let geometry = geometry();
    let marbles = match side {
        Color::Black => position.black(),
        Color::White => position.white(),
    };
    let mut groups: Vec<Vec<crate::board::CellId>> =
        marbles.iter().copied().map(|cell| vec![cell]).collect();
    for axis in [LineAxis::Q, LineAxis::R, LineAxis::S] {
        for line in geometry.lines(axis) {
            let mut run = Vec::new();
            for cell in &line.cells {
                if position.contains(side, *cell) {
                    run.push(*cell);
                } else {
                    emit_groups_from_run(&run, &mut groups);
                    run.clear();
                }
            }
            emit_groups_from_run(&run, &mut groups);
        }
    }
    groups
}
fn emit_groups_from_run(run: &[crate::board::CellId], groups: &mut Vec<Vec<crate::board::CellId>>) {
    for len in 2..=3 {
        if run.len() < len {
            continue;
        }
        for start in 0..=run.len() - len {
            groups.push(run[start..start + len].to_vec());
        }
    }
}
fn analyze_single_move(
    candidate_move: &Move,
    occupants: &[Option<Color>; crate::board::CELL_COUNT],
    side: Color,
) -> Result<MovePlan, MoveApplicationError> {
    let source = candidate_move.source_cells()[0];
    let destination = neighbor_cell(source, candidate_move.direction())
        .ok_or_else(|| MoveApplicationError::IllegalMove(candidate_move.clone()))?;
    if occupants[destination.as_usize()].is_some() {
        return Err(MoveApplicationError::IllegalMove(candidate_move.clone()));
    }
    let _ = side;
    Ok(MovePlan {
        source_cells: CellList::from_slice(&[source]),
        destination_cells: CellList::from_slice(&[destination]),
        pushed_cells: CellList::new(),
        pushed_destinations: OptionalCellList::new(),
    })
}
fn analyze_broadside_move(
    candidate_move: &Move,
    occupants: &[Option<Color>; crate::board::CELL_COUNT],
) -> Result<MovePlan, MoveApplicationError> {
    let mut destinations = CellList::new();
    for source in candidate_move.source_cells() {
        let destination = neighbor_cell(*source, candidate_move.direction())
            .ok_or_else(|| MoveApplicationError::IllegalMove(candidate_move.clone()))?;
        if occupants[destination.as_usize()].is_some() {
            return Err(MoveApplicationError::IllegalMove(candidate_move.clone()));
        }
        destinations.push(destination);
    }
    Ok(MovePlan {
        source_cells: CellList::from_slice(candidate_move.source_cells()),
        destination_cells: destinations,
        pushed_cells: CellList::new(),
        pushed_destinations: OptionalCellList::new(),
    })
}
fn analyze_inline_move(
    candidate_move: &Move,
    occupants: &[Option<Color>; crate::board::CELL_COUNT],
    side: Color,
    enemy: Color,
) -> Result<MovePlan, MoveApplicationError> {
    let front = front_cell(candidate_move.source_cells(), candidate_move.direction())
        .ok_or_else(|| MoveApplicationError::IllegalMove(candidate_move.clone()))?;
    let first_ahead = neighbor_cell(front, candidate_move.direction())
        .ok_or_else(|| MoveApplicationError::IllegalMove(candidate_move.clone()))?;
    match occupants[first_ahead.as_usize()] {
        None => Ok(MovePlan {
            source_cells: CellList::from_slice(candidate_move.source_cells()),
            destination_cells: translated_cells(
                candidate_move.source_cells(),
                candidate_move.direction(),
            )?,
            pushed_cells: CellList::new(),
            pushed_destinations: OptionalCellList::new(),
        }),
        Some(color) if color == side => {
            Err(MoveApplicationError::IllegalMove(candidate_move.clone()))
        }
        Some(color) if color == enemy => {
            let mut enemy_cells = [first_ahead; 2];
            let mut enemy_count = 0usize;
            let mut cursor = Some(first_ahead);
            while let Some(cell) = cursor {
                if occupants[cell.as_usize()] != Some(enemy) {
                    break;
                }
                if enemy_count < enemy_cells.len() {
                    enemy_cells[enemy_count] = cell;
                }
                enemy_count += 1;
                cursor = neighbor_cell(cell, candidate_move.direction());
            }
            if enemy_count >= candidate_move.len() {
                return Err(MoveApplicationError::IllegalMove(candidate_move.clone()));
            }
            let mut pushed_cells = CellList::new();
            let mut pushed_destinations = OptionalCellList::new();
            for index in 0..enemy_count {
                let cell = enemy_cells[index];
                let destination = neighbor_cell(cell, candidate_move.direction());
                if index + 1 == enemy_count {
                    if let Some(next) = destination {
                        if occupants[next.as_usize()].is_some() {
                            return Err(MoveApplicationError::IllegalMove(candidate_move.clone()));
                        }
                    }
                }
                pushed_cells.push(cell);
                pushed_destinations.push(destination);
            }
            Ok(MovePlan {
                source_cells: CellList::from_slice(candidate_move.source_cells()),
                destination_cells: translated_cells(
                    candidate_move.source_cells(),
                    candidate_move.direction(),
                )?,
                pushed_cells,
                pushed_destinations,
            })
        }
        Some(_) => Err(MoveApplicationError::IllegalMove(candidate_move.clone())),
    }
}
fn translated_cells(
    source_cells: &[crate::board::CellId],
    direction: Direction,
) -> Result<CellList<3>, MoveApplicationError> {
    let mut destinations = CellList::new();
    for cell in source_cells.iter().copied() {
        destinations.push(neighbor_cell(cell, direction).ok_or_else(|| {
            MoveApplicationError::IllegalMove(Move::from_cells(source_cells, direction).unwrap())
        })?);
    }
    Ok(destinations)
}
fn remove_from_position(
    cells: &mut Vec<crate::board::CellId>,
    slot_map: &mut [u8; crate::board::CELL_COUNT],
    cell: crate::board::CellId,
    occupants: &mut [Option<Color>; crate::board::CELL_COUNT],
    black_bits: &mut u64,
    white_bits: &mut u64,
    color: Color,
) -> Result<(), MoveApplicationError> {
    let slot = slot_map[cell.as_usize()];
    if slot == u8::MAX {
        return Err(MoveApplicationError::IllegalMove(Move::PLACEHOLDER));
    }
    let index = slot as usize;
    let last = *cells
        .last()
        .ok_or(MoveApplicationError::IllegalMove(Move::PLACEHOLDER))?;
    cells.swap_remove(index);
    slot_map[cell.as_usize()] = u8::MAX;
    if index < cells.len() {
        slot_map[last.as_usize()] = index as u8;
    }
    occupants[cell.as_usize()] = None;
    let bit = 1u64 << cell.as_u8();
    match color {
        Color::Black => *black_bits &= !bit,
        Color::White => *white_bits &= !bit,
    }
    Ok(())
}
fn add_to_position(
    cells: &mut Vec<crate::board::CellId>,
    slot_map: &mut [u8; crate::board::CELL_COUNT],
    cell: crate::board::CellId,
    occupants: &mut [Option<Color>; crate::board::CELL_COUNT],
    black_bits: &mut u64,
    white_bits: &mut u64,
    color: Color,
) {
    if slot_map[cell.as_usize()] == u8::MAX {
        slot_map[cell.as_usize()] = cells.len() as u8;
        cells.push(cell);
    }
    occupants[cell.as_usize()] = Some(color);
    let bit = 1u64 << cell.as_u8();
    match color {
        Color::Black => *black_bits |= bit,
        Color::White => *white_bits |= bit,
    }
}
fn apply_plan_in_place(
    position: &mut Position,
    occupants: &mut [Option<Color>; crate::board::CELL_COUNT],
    black_bits: &mut u64,
    white_bits: &mut u64,
    black_slots: &mut [u8; crate::board::CELL_COUNT],
    white_slots: &mut [u8; crate::board::CELL_COUNT],
    plan: &MovePlan,
) -> Result<(), MoveApplicationError> {
    let side = position.side_to_move();
    let enemy = side.other();
    let (side_slots, enemy_slots) = match side {
        Color::Black => (&mut *black_slots, &mut *white_slots),
        Color::White => (&mut *white_slots, &mut *black_slots),
    };
    {
        let side_cells = position.cells_for_mut(side);
        for cell in plan.source_cells.iter() {
            remove_from_position(
                side_cells, side_slots, *cell, occupants, black_bits, white_bits, side,
            )?;
        }
    }
    {
        let enemy_cells = position.cells_for_mut(enemy);
        for cell in plan.pushed_cells.iter() {
            remove_from_position(
                enemy_cells,
                enemy_slots,
                *cell,
                occupants,
                black_bits,
                white_bits,
                enemy,
            )?;
        }
    }
    {
        let side_cells = position.cells_for_mut(side);
        for destination in plan.destination_cells.iter() {
            add_to_position(
                side_cells,
                side_slots,
                *destination,
                occupants,
                black_bits,
                white_bits,
                side,
            );
        }
    }
    {
        let enemy_cells = position.cells_for_mut(enemy);
        for destination in plan.pushed_destinations.iter() {
            if let Some(destination) = destination {
                add_to_position(
                    enemy_cells,
                    enemy_slots,
                    *destination,
                    occupants,
                    black_bits,
                    white_bits,
                    enemy,
                );
            }
        }
    }
    position.set_side_to_move(enemy);
    Ok(())
}
fn undo_plan_in_place(
    position: &mut Position,
    occupants: &mut [Option<Color>; crate::board::CELL_COUNT],
    black_bits: &mut u64,
    white_bits: &mut u64,
    black_slots: &mut [u8; crate::board::CELL_COUNT],
    white_slots: &mut [u8; crate::board::CELL_COUNT],
    plan: &MovePlan,
    previous_side_to_move: Color,
) {
    let side = previous_side_to_move;
    let enemy = side.other();
    let (side_slots, enemy_slots) = match side {
        Color::Black => (&mut *black_slots, &mut *white_slots),
        Color::White => (&mut *white_slots, &mut *black_slots),
    };
    {
        let side_cells = position.cells_for_mut(side);
        for destination in plan.destination_cells.iter() {
            let _ = remove_from_position(
                side_cells,
                side_slots,
                *destination,
                occupants,
                black_bits,
                white_bits,
                side,
            );
        }
    }
    {
        let enemy_cells = position.cells_for_mut(enemy);
        for destination in plan.pushed_destinations.iter() {
            if let Some(destination) = destination {
                let _ = remove_from_position(
                    enemy_cells,
                    enemy_slots,
                    *destination,
                    occupants,
                    black_bits,
                    white_bits,
                    enemy,
                );
            }
        }
    }
    {
        let side_cells = position.cells_for_mut(side);
        for cell in plan.source_cells.iter() {
            add_to_position(
                side_cells, side_slots, *cell, occupants, black_bits, white_bits, side,
            );
        }
    }
    {
        let enemy_cells = position.cells_for_mut(enemy);
        for cell in plan.pushed_cells.iter() {
            add_to_position(
                enemy_cells,
                enemy_slots,
                *cell,
                occupants,
                black_bits,
                white_bits,
                enemy,
            );
        }
    }
    position.set_side_to_move(side);
}
fn build_occupants(position: &Position) -> [Option<Color>; crate::board::CELL_COUNT] {
    let mut occupants = [None; crate::board::CELL_COUNT];
    for cell in position.black() {
        occupants[cell.as_usize()] = Some(Color::Black);
    }
    for cell in position.white() {
        occupants[cell.as_usize()] = Some(Color::White);
    }
    occupants
}
fn slot_maps(
    position: &Position,
) -> (
    [u8; crate::board::CELL_COUNT],
    [u8; crate::board::CELL_COUNT],
) {
    let mut black_slot_map = [u8::MAX; crate::board::CELL_COUNT];
    let mut white_slot_map = [u8::MAX; crate::board::CELL_COUNT];
    for (i, cell) in position.black().iter().enumerate() {
        black_slot_map[cell.as_usize()] = i as u8;
    }
    for (i, cell) in position.white().iter().enumerate() {
        white_slot_map[cell.as_usize()] = i as u8;
    }
    (black_slot_map, white_slot_map)
}
fn group_axis(source_cells: &[crate::board::CellId]) -> Option<LineAxis> {
    let geometry = geometry();
    let first = geometry.cell(source_cells[0]);
    [LineAxis::Q, LineAxis::R, LineAxis::S]
        .into_iter()
        .find(|axis| {
            let axis_index = axis.index();
            let line_id = first.line_ids[axis_index];
            source_cells
                .iter()
                .all(|cell| geometry.cell(*cell).line_ids[axis_index] == line_id)
        })
}
fn is_inline(axis: LineAxis, direction: Direction) -> bool {
    match axis {
        LineAxis::Q => matches!(direction, Direction::Se | Direction::Nw),
        LineAxis::R => matches!(direction, Direction::East | Direction::West),
        LineAxis::S => matches!(direction, Direction::Ne | Direction::Sw),
    }
}
fn front_cell(
    source_cells: &[crate::board::CellId],
    direction: Direction,
) -> Option<crate::board::CellId> {
    match source_cells {
        [] => None,
        [first] => Some(*first),
        [first, second] => {
            if neighbor_cell(*first, direction) == Some(*second) {
                Some(*second)
            } else {
                Some(*first)
            }
        }
        [first, second, third] => {
            if neighbor_cell(*first, direction) == Some(*second)
                && neighbor_cell(*second, direction) == Some(*third)
            {
                Some(*third)
            } else {
                Some(*first)
            }
        }
        _ => None,
    }
}
fn neighbor_cell(cell: crate::board::CellId, direction: Direction) -> Option<crate::board::CellId> {
    geometry().cell(cell).neighbors[direction.index()]
}
