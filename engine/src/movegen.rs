use crate::board::{
    ALL_DIRECTIONS, Color, Direction, EngineStateView, LineAxis, Move, MoveError, Position,
    PositionError, geometry,
};
use std::collections::BTreeSet;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UndoSnapshot {
    previous_position: Position,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionState {
    position: Position,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoveApplicationError {
    IllegalMove(Move),
    InvalidMoveShape(MoveError),
    InvalidPosition(PositionError),
}
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct CompactEnemyEffect(u16);
impl CompactEnemyEffect {
    const CELL_CODE_MASK: u16 = 0x3f;

    const fn none() -> Self {
        Self(0)
    }
    fn from_toggle_mask(mut toggle: u64) -> Self {
        debug_assert!(toggle.count_ones() <= 2);
        let mut encoded = 0u16;
        if toggle != 0 {
            encoded = toggle.trailing_zeros() as u16 + 1;
            toggle &= toggle - 1;
        }
        if toggle != 0 {
            encoded |= (toggle.trailing_zeros() as u16 + 1) << 6;
            toggle &= toggle - 1;
        }
        debug_assert_eq!(toggle, 0);
        Self(encoded)
    }
    pub(crate) fn toggle_mask(self) -> u64 {
        let first = self.0 & Self::CELL_CODE_MASK;
        let second = (self.0 >> 6) & Self::CELL_CODE_MASK;
        let mut toggle = 0u64;
        if first != 0 {
            toggle |= 1u64 << (first - 1);
        }
        if second != 0 {
            toggle |= 1u64 << (second - 1);
        }
        toggle
    }
}
impl PositionState {
    pub fn new(position: Position) -> Result<Self, PositionError> {
        position.validate()?;
        Ok(Self { position })
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    pub fn occupant_fast(&self, cell: crate::board::CellId) -> Option<Color> {
        self.position.occupant(cell)
    }
    pub fn black_bits(&self) -> u64 {
        self.position.black_bits()
    }
    pub fn white_bits(&self) -> u64 {
        self.position.white_bits()
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
        let mut candidate_group_bits = [0u64; 6];
        let mut anchors = side_bits;
        while anchors != 0 {
            let anchor = anchors.trailing_zeros() as usize;
            let indexed = &tables.anchor_group_bits[anchor];
            for word in 0..candidate_group_bits.len() {
                candidate_group_bits[word] |= indexed[word];
            }
            anchors &= anchors - 1;
        }
        for (word_index, mut groups) in candidate_group_bits.into_iter().enumerate() {
            while groups != 0 {
                let bit_index = groups.trailing_zeros() as usize;
                groups &= groups - 1;
                let group = &tables.source_groups[word_index * 64 + bit_index];
                if side_bits & group.source_mask != group.source_mask {
                    continue;
                }
                for direction in group.directions.iter().flatten() {
                    let Some((is_push, is_ejection, enemy_effect)) = self
                        .fast_group_direction_legality(
                            group.len as usize,
                            direction,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                        )
                    else {
                        continue;
                    };
                    moves.push(crate::search::LegalMoveEntry {
                        candidate_move: direction.candidate_move,
                        is_ejection,
                        is_push,
                        history_key: direction.history_key,
                        plan_index: direction.plan_index,
                        enemy_effect,
                    });
                }
            }
        }
    }
    fn side_bits(&self, side: Color) -> u64 {
        self.position.bits_for(side)
    }
    fn fast_group_direction_legality(
        &self,
        len: usize,
        direction: &crate::search::FastGroupDirection,
        side_bits: u64,
        enemy_bits: u64,
        occupied_bits: u64,
    ) -> Option<(bool, bool, CompactEnemyEffect)> {
        if direction.inline {
            self.fast_inline_legality(len, direction, side_bits, enemy_bits, occupied_bits)
        } else {
            self.fast_broadside_legality(direction.translated_mask, occupied_bits)
        }
    }
    fn fast_broadside_legality(
        &self,
        translated_mask: u64,
        occupied_bits: u64,
    ) -> Option<(bool, bool, CompactEnemyEffect)> {
        if occupied_bits & translated_mask == 0 {
            Some((false, false, CompactEnemyEffect::none()))
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
    ) -> Option<(bool, bool, CompactEnemyEffect)> {
        let first_bit = direction.ray_bits[0];
        if occupied_bits & first_bit == 0 {
            return Some((false, false, CompactEnemyEffect::none()));
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
                let landing_bit = 1u64 << cell.as_u8();
                if occupied_bits & landing_bit != 0 {
                    None
                } else {
                    Some((
                        true,
                        false,
                        CompactEnemyEffect::from_toggle_mask(first_bit | landing_bit),
                    ))
                }
            }
            None => Some((true, true, CompactEnemyEffect::from_toggle_mask(first_bit))),
        }
    }
    pub fn apply_move(
        &mut self,
        candidate_move: &Move,
    ) -> Result<UndoSnapshot, MoveApplicationError> {
        let plan = self.analyze_move(candidate_move)?;
        let undo = UndoSnapshot {
            previous_position: self.position,
        };
        apply_plan_in_place(&mut self.position, &plan);
        Ok(undo)
    }
    pub(crate) fn apply_legal_effect(
        &mut self,
        plan_index: u16,
        enemy_effect: CompactEnemyEffect,
    ) -> UndoSnapshot {
        let undo = UndoSnapshot {
            previous_position: self.position,
        };
        let own_toggle = crate::search::fast_movegen_tables().own_toggle(plan_index);
        self.position
            .apply_toggle_masks(own_toggle, enemy_effect.toggle_mask());
        undo
    }
    pub(crate) fn legal_move_entry(
        &self,
        candidate_move: &Move,
    ) -> Option<crate::search::LegalMoveEntry> {
        let plan = self.analyze_move(candidate_move).ok()?;
        let history_key = crate::search::history_group_key(
            candidate_move.source_cells(),
            candidate_move.direction(),
        );
        let tables = crate::search::fast_movegen_tables();
        let plan_index = tables.plan_index(history_key)?;
        debug_assert_eq!(tables.own_toggle(plan_index), plan.own_toggle());
        Some(crate::search::LegalMoveEntry {
            candidate_move: *candidate_move,
            is_ejection: plan.is_ejection(),
            is_push: plan.is_push(),
            history_key,
            plan_index,
            enemy_effect: plan.enemy_effect(),
        })
    }
    pub fn undo_move(&mut self, undo: UndoSnapshot) {
        self.position = undo.previous_position;
    }
    fn analyze_move(&self, candidate_move: &Move) -> Result<MovePlan, MoveApplicationError> {
        let side = self.position.side_to_move();
        let side_bits = self.position.bits_for(side);
        let enemy_bits = self.position.bits_for(side.other());
        let occupied_bits = side_bits | enemy_bits;
        let source_mask = cells_mask(candidate_move.source_cells());
        if side_bits & source_mask != source_mask {
            return Err(MoveApplicationError::IllegalMove(*candidate_move));
        }
        if candidate_move.len() == 1 {
            return analyze_single_move(candidate_move, occupied_bits);
        }
        let axis = group_axis(candidate_move.source_cells())
            .ok_or_else(|| MoveApplicationError::IllegalMove(*candidate_move))?;
        if is_inline(axis, candidate_move.direction()) {
            analyze_inline_move(candidate_move, enemy_bits, occupied_bits)
        } else {
            analyze_broadside_move(candidate_move, occupied_bits)
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
struct MovePlan {
    own_from: u64,
    own_to: u64,
    enemy_from: u64,
    enemy_to: u64,
}
impl MovePlan {
    const fn quiet(own_from: u64, own_to: u64) -> Self {
        Self {
            own_from,
            own_to,
            enemy_from: 0,
            enemy_to: 0,
        }
    }
    const fn is_push(self) -> bool {
        self.enemy_from != 0
    }
    const fn is_ejection(self) -> bool {
        self.enemy_from.count_ones() > self.enemy_to.count_ones()
    }
    const fn own_toggle(self) -> u64 {
        self.own_from ^ self.own_to
    }
    fn enemy_effect(self) -> CompactEnemyEffect {
        CompactEnemyEffect::from_toggle_mask(self.enemy_from ^ self.enemy_to)
    }
}
fn enumerate_groups(position: &Position, side: Color) -> Vec<Vec<crate::board::CellId>> {
    let geometry = geometry();
    let marbles = position.cells(side);
    let mut groups: Vec<Vec<crate::board::CellId>> =
        marbles.iter().map(|cell| vec![cell]).collect();
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
    occupied_bits: u64,
) -> Result<MovePlan, MoveApplicationError> {
    let source = candidate_move.source_cells()[0];
    let destination = neighbor_cell(source, candidate_move.direction())
        .ok_or_else(|| MoveApplicationError::IllegalMove(*candidate_move))?;
    let destination_bit = 1u64 << destination.as_u8();
    if occupied_bits & destination_bit != 0 {
        return Err(MoveApplicationError::IllegalMove(*candidate_move));
    }
    Ok(MovePlan::quiet(1u64 << source.as_u8(), destination_bit))
}
fn analyze_broadside_move(
    candidate_move: &Move,
    occupied_bits: u64,
) -> Result<MovePlan, MoveApplicationError> {
    let own_to = translated_mask(candidate_move.source_cells(), candidate_move.direction())?;
    if occupied_bits & own_to != 0 {
        return Err(MoveApplicationError::IllegalMove(*candidate_move));
    }
    Ok(MovePlan::quiet(
        cells_mask(candidate_move.source_cells()),
        own_to,
    ))
}
fn analyze_inline_move(
    candidate_move: &Move,
    enemy_bits: u64,
    occupied_bits: u64,
) -> Result<MovePlan, MoveApplicationError> {
    let front = front_cell(candidate_move.source_cells(), candidate_move.direction())
        .ok_or_else(|| MoveApplicationError::IllegalMove(*candidate_move))?;
    let first_ahead = neighbor_cell(front, candidate_move.direction())
        .ok_or_else(|| MoveApplicationError::IllegalMove(*candidate_move))?;
    let first_bit = 1u64 << first_ahead.as_u8();
    let own_from = cells_mask(candidate_move.source_cells());
    let own_to = translated_mask(candidate_move.source_cells(), candidate_move.direction())?;
    if occupied_bits & first_bit == 0 {
        return Ok(MovePlan::quiet(own_from, own_to));
    }
    if enemy_bits & first_bit == 0 {
        return Err(MoveApplicationError::IllegalMove(*candidate_move));
    }

    let mut enemy_from = 0u64;
    let mut enemy_to = 0u64;
    let mut enemy_count = 0usize;
    let mut cursor = Some(first_ahead);
    while let Some(cell) = cursor {
        let bit = 1u64 << cell.as_u8();
        if enemy_bits & bit == 0 {
            break;
        }
        enemy_count += 1;
        enemy_from |= bit;
        cursor = neighbor_cell(cell, candidate_move.direction());
        if let Some(destination) = cursor {
            enemy_to |= 1u64 << destination.as_u8();
        }
    }
    if enemy_count >= candidate_move.len() {
        return Err(MoveApplicationError::IllegalMove(*candidate_move));
    }
    if let Some(landing) = cursor {
        if occupied_bits & (1u64 << landing.as_u8()) != 0 {
            return Err(MoveApplicationError::IllegalMove(*candidate_move));
        }
    }
    Ok(MovePlan {
        own_from,
        own_to,
        enemy_from,
        enemy_to,
    })
}
fn cells_mask(cells: &[crate::board::CellId]) -> u64 {
    cells
        .iter()
        .fold(0u64, |mask, cell| mask | (1u64 << cell.as_u8()))
}
fn translated_mask(
    source_cells: &[crate::board::CellId],
    direction: Direction,
) -> Result<u64, MoveApplicationError> {
    let mut destinations = 0u64;
    for cell in source_cells.iter().copied() {
        let destination = neighbor_cell(cell, direction).ok_or_else(|| {
            MoveApplicationError::IllegalMove(Move::from_cells(source_cells, direction).unwrap())
        })?;
        destinations |= 1u64 << destination.as_u8();
    }
    Ok(destinations)
}
fn apply_plan_in_place(position: &mut Position, plan: &MovePlan) {
    let side = position.side_to_move();
    position.apply_masks(
        side,
        plan.own_from,
        plan.own_to,
        plan.enemy_from,
        plan.enemy_to,
    );
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
