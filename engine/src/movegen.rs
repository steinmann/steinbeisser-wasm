#[path = "move_tables.rs"]
mod tables;
pub(crate) use tables::{
    FastGroupDirection, FastSourceGroup, LegalMoveEntry, fast_movegen_tables, history_group_key,
    move_front_cell, move_group_axis, move_is_inline, neighbor_cell,
};

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

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn specialized_dispatch_matches_slow_legality_and_undo() {
        let starts = [
            include_str!("../../data/positions/classical.fen"),
            include_str!("../../data/positions/belgian-daisy.fen"),
            include_str!("../../data/positions/german-daisy.fen"),
        ];
        let mut state = PositionState::new(Position::from_str(starts[0]).unwrap()).unwrap();
        let mut random = 0x5eed_u64;
        for index in 0..1_000 {
            if index % 100 == 0 {
                state = PositionState::new(
                    Position::from_str(starts[(index / 100) % starts.len()]).unwrap(),
                )
                .unwrap();
            }
            let slow = state.generate_legal_moves();
            let mut fast = Vec::new();
            state.generate_fast_legal_moves(&mut fast);
            let mut fast_moves = fast
                .iter()
                .map(|entry| entry.candidate_move)
                .collect::<Vec<_>>();
            fast_moves.sort_unstable();
            assert_eq!(fast_moves, slow);
            let mut pushes = Vec::new();
            state.generate_fast_push_moves(&mut pushes);
            assert_eq!(
                pushes,
                fast.iter()
                    .copied()
                    .filter(|entry| entry.is_push)
                    .collect::<Vec<_>>()
            );
            for entry in &fast {
                let plan = state.analyze_move(&entry.candidate_move).unwrap();
                assert_eq!(entry.is_push, plan.is_push());
                assert_eq!(entry.is_ejection, plan.is_ejection());
                assert_ne!(entry.history_key, u32::MAX);
            }
            if slow.is_empty() {
                continue;
            }
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let selected = slow[(random >> 32) as usize % slow.len()];
            let before = *state.position();
            let undo = state.apply_move(&selected).unwrap();
            let after = *state.position();
            state.undo_move(undo);
            assert_eq!(*state.position(), before);
            state.apply_move(&selected).unwrap();
            assert_eq!(*state.position(), after);
        }
    }
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
    pub(crate) fn generate_fast_legal_moves(&self, moves: &mut Vec<LegalMoveEntry>) {
        self.generate_fast_moves::<false>(moves);
    }
    pub(crate) fn generate_fast_push_moves(&self, moves: &mut Vec<LegalMoveEntry>) {
        self.generate_fast_moves::<true>(moves);
    }
    // The two entry points share the dispatch rules, but remain separate native
    // monomorphizations. PUSH is resolved at compile time, not per emitted move.
    #[inline(always)]
    fn generate_fast_moves<const PUSH: bool>(&self, moves: &mut Vec<LegalMoveEntry>) {
        let side_bits = self.side_bits(self.position.side_to_move());
        let enemy_bits = self.side_bits(self.position.side_to_move().other());
        let occupied_bits = side_bits | enemy_bits;
        let tables = fast_movegen_tables();
        moves.clear();
        moves.reserve(tables.plan_count());
        let dst = moves.as_mut_ptr();
        let mut written = 0usize;
        let candidate_group_bits = if PUSH {
            tables.owned_groups.generate_push(side_bits, enemy_bits)
        } else {
            tables.owned_groups.generate(side_bits)
        };
        for (word_index, mut groups) in candidate_group_bits.into_iter().enumerate() {
            while groups != 0 {
                let bit_index = groups.trailing_zeros() as usize;
                groups &= groups - 1;
                let id = word_index * 64 + bit_index;
                let mask = tables.source_masks[id];
                if side_bits & mask != mask {
                    continue;
                }
                let group = &tables.source_groups[id];
                if PUSH && group.len < 2 {
                    continue;
                }
                // SAFETY: reserve covers every unique geometric plan; each group
                // is visited once and emits at most its six directions.
                unsafe {
                    match (group.len, group.axis) {
                        (1, 3) => self.emit_group::<1, 3, PUSH>(
                            group,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                            dst,
                            &mut written,
                        ),
                        (2, 0) => self.emit_group::<2, 0, PUSH>(
                            group,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                            dst,
                            &mut written,
                        ),
                        (2, 1) => self.emit_group::<2, 1, PUSH>(
                            group,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                            dst,
                            &mut written,
                        ),
                        (2, 2) => self.emit_group::<2, 2, PUSH>(
                            group,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                            dst,
                            &mut written,
                        ),
                        (3, 0) => self.emit_group::<3, 0, PUSH>(
                            group,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                            dst,
                            &mut written,
                        ),
                        (3, 1) => self.emit_group::<3, 1, PUSH>(
                            group,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                            dst,
                            &mut written,
                        ),
                        (3, 2) => self.emit_group::<3, 2, PUSH>(
                            group,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                            dst,
                            &mut written,
                        ),
                        _ => unreachable!(),
                    }
                }
            }
        }
        // SAFETY: capacity >= every geometric plan; each unique plan is emitted at most once.
        // Every slot below written was initialized by the immediately preceding writes.
        unsafe {
            moves.set_len(written);
        }
    }
    #[inline(always)]
    // SAFETY: dst points into the reserved output buffer; written slots are
    // initialized, with a slot available for each Some direction in g. N/AX match g.
    unsafe fn emit_group<const N: usize, const AX: usize, const PUSH: bool>(
        &self,
        g: &FastSourceGroup,
        own: u64,
        enemy: u64,
        occ: u64,
        dst: *mut LegalMoveEntry,
        written: &mut usize,
    ) {
        // SAFETY: inherited buffer contract; each direction writes zero or one.
        unsafe {
            let group_dst = dst.add(*written);
            let n0 = self.emit_direction::<N, AX, PUSH, 0>(g, own, enemy, occ, group_dst.add(0));
            let n1 = self.emit_direction::<N, AX, PUSH, 1>(g, own, enemy, occ, group_dst.add(n0));
            let n2 =
                self.emit_direction::<N, AX, PUSH, 2>(g, own, enemy, occ, group_dst.add(n0 + n1));
            let n3 = self.emit_direction::<N, AX, PUSH, 3>(
                g,
                own,
                enemy,
                occ,
                group_dst.add(n0 + n1 + n2),
            );
            let n4 = self.emit_direction::<N, AX, PUSH, 4>(
                g,
                own,
                enemy,
                occ,
                group_dst.add(n0 + n1 + n2 + n3),
            );
            let n5 = self.emit_direction::<N, AX, PUSH, 5>(
                g,
                own,
                enemy,
                occ,
                group_dst.add(n0 + n1 + n2 + n3 + n4),
            );
            *written += n0 + n1 + n2 + n3 + n4 + n5;
        }
    }
    #[inline(always)]
    // SAFETY: dst is aligned/writable for one entry when g.directions[D] is Some,
    // otherwise it may be one-past the buffer. N/AX match g and D < 6.
    // Returns the initialized slot count (0 or 1).
    unsafe fn emit_direction<const N: usize, const AX: usize, const PUSH: bool, const D: usize>(
        &self,
        g: &FastSourceGroup,
        own: u64,
        enemy: u64,
        occ: u64,
        dst: *mut LegalMoveEntry,
    ) -> usize {
        let Some(d) = &g.directions[D] else {
            return 0;
        };
        let inline = N > 1
            && ((AX == 0 && (D == Direction::Se.index() || D == Direction::Nw.index()))
                || (AX == 1 && (D == Direction::East.index() || D == Direction::West.index()))
                || (AX == 2 && (D == Direction::Ne.index() || D == Direction::Sw.index())));
        debug_assert_eq!(inline, d.inline);
        if PUSH && (!inline || enemy & d.ray_bits[0] == 0) {
            return 0;
        }
        let target = if inline {
            d.ray_bits[0]
        } else {
            d.translated_mask
        };
        let effect = if occ & target == 0 {
            Some((false, false, CompactEnemyEffect::none()))
        } else if inline {
            self.fast_inline_legality(N, d, own, enemy, occ)
        } else {
            None
        };
        if let Some((is_push, is_ejection, enemy_effect)) = effect {
            unsafe {
                dst.write(LegalMoveEntry {
                    candidate_move: d.candidate_move,
                    is_push,
                    is_ejection,
                    enemy_effect,
                    history_key: d.history_key,
                    plan_index: d.plan_index,
                });
            }
            return 1;
        }
        0
    }
    fn side_bits(&self, side: Color) -> u64 {
        self.position.bits_for(side)
    }

    fn fast_inline_legality(
        &self,
        len: usize,
        direction: &FastGroupDirection,
        side_bits: u64,
        enemy_bits: u64,
        occupied_bits: u64,
    ) -> Option<(bool, bool, CompactEnemyEffect)> {
        let first_bit = direction.ray_bits[0];
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
        let own_toggle = fast_movegen_tables().own_toggle(plan_index);
        self.position
            .apply_toggle_masks(own_toggle, enemy_effect.toggle_mask());
        undo
    }
    pub(crate) fn legal_move_entry(&self, candidate_move: &Move) -> Option<LegalMoveEntry> {
        let plan = self.analyze_move(candidate_move).ok()?;
        let history_key =
            history_group_key(candidate_move.source_cells(), candidate_move.direction());
        let tables = fast_movegen_tables();
        let plan_index = tables.plan_index(history_key)?;
        debug_assert_eq!(tables.own_toggle(plan_index), plan.own_toggle());
        Some(LegalMoveEntry {
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
            .ok_or(MoveApplicationError::IllegalMove(*candidate_move))?;
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
        .ok_or(MoveApplicationError::IllegalMove(*candidate_move))?;
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
        .ok_or(MoveApplicationError::IllegalMove(*candidate_move))?;
    let first_ahead = neighbor_cell(front, candidate_move.direction())
        .ok_or(MoveApplicationError::IllegalMove(*candidate_move))?;
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
    if let Some(landing) = cursor
        && occupied_bits & (1u64 << landing.as_u8()) != 0
    {
        return Err(MoveApplicationError::IllegalMove(*candidate_move));
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
