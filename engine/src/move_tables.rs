//! Precomputed legal group geometry and stable move IDs.
use crate::board::{ALL_DIRECTIONS, Direction, LineAxis, Move, geometry};

const HISTORY_SOURCE_GROUPS_LEN1: usize = crate::board::CELL_COUNT;
const HISTORY_SOURCE_GROUPS_LEN2: usize = combination_count(crate::board::CELL_COUNT, 2);
pub(crate) fn move_group_axis(source_cells: &[crate::board::CellId]) -> Option<LineAxis> {
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
pub(crate) fn move_is_inline(axis: LineAxis, direction: Direction) -> bool {
    match axis {
        LineAxis::Q => matches!(direction, Direction::Se | Direction::Nw),
        LineAxis::R => matches!(direction, Direction::East | Direction::West),
        LineAxis::S => matches!(direction, Direction::Ne | Direction::Sw),
    }
}
pub(crate) fn move_front_cell(
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
// Preserve the direct lookup in hot legality checks after module extraction.
#[inline]
pub(crate) fn neighbor_cell(
    cell: crate::board::CellId,
    direction: Direction,
) -> Option<crate::board::CellId> {
    geometry().cell(cell).neighbors[direction.index()]
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FastGroupDirection {
    pub(crate) candidate_move: Move,
    pub(crate) inline: bool,
    pub(crate) translated_mask: u64,
    pub(crate) history_key: u32,
    pub(crate) plan_index: u16,
    pub(crate) ray_bits: [u64; 3],
    pub(crate) landing: [Option<crate::board::CellId>; 2],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FastSourceGroup {
    pub(crate) axis: u8,
    pub(crate) len: u8,
    pub(crate) source_mask: u64,
    pub(crate) directions: [Option<FastGroupDirection>; 6],
}
#[derive(Clone, Debug)]
pub(crate) struct FastMovegenTables {
    pub(crate) source_masks: Vec<u64>,
    pub(crate) source_groups: Vec<FastSourceGroup>,
    pub(crate) owned_groups: OwnedGroupTables,
    plans: Vec<FastMovePlan>,
    plan_hash: Vec<(u32, u16)>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FastMovePlan {
    own_toggle: u64,
}
impl FastMovegenTables {
    pub(crate) fn plan_count(&self) -> usize {
        self.plans.len()
    }
    pub(crate) fn own_toggle(&self, plan_index: u16) -> u64 {
        self.plans[plan_index as usize].own_toggle
    }
    pub(crate) fn plan_index(&self, k: u32) -> Option<u16> {
        let mut i = k.wrapping_mul(0x9e3779b9) as usize & (self.plan_hash.len() - 1);
        loop {
            let (key, p) = self.plan_hash[i];
            if p == u16::MAX {
                return None;
            }
            if key == k {
                return Some(p);
            }
            i = (i + 1) & (self.plan_hash.len() - 1);
        }
    }
}
pub(crate) fn fast_movegen_tables() -> &'static FastMovegenTables {
    static TABLES: std::sync::OnceLock<FastMovegenTables> = std::sync::OnceLock::new();
    TABLES.get_or_init(build_fast_movegen_tables)
}
fn build_fast_movegen_tables() -> FastMovegenTables {
    let geom = geometry();
    let mut source_groups = Vec::with_capacity(256);
    let mut plans = Vec::with_capacity(2048);
    for cell in geom.cells().iter().map(|cell| cell.index) {
        let cells = [cell, cell, cell];
        let source_mask = 1u64 << cell.as_u8();
        let directions = build_fast_group_directions(&cells, 1, None, source_mask, &mut plans);
        source_groups.push(FastSourceGroup {
            axis: 3,
            len: 1,
            source_mask,
            directions,
        });
    }
    for axis in [LineAxis::Q, LineAxis::R, LineAxis::S] {
        for line in geom.lines(axis) {
            for len in 2..=3 {
                if line.cells.len() < len {
                    continue;
                }
                for start in 0..=line.cells.len() - len {
                    let cells = canonical_group_cells(&line.cells[start..start + len]);
                    let source_mask = fast_source_mask(&cells, len as u8);
                    let directions = build_fast_group_directions(
                        &cells,
                        len as u8,
                        Some(axis),
                        source_mask,
                        &mut plans,
                    );
                    source_groups.push(FastSourceGroup {
                        axis: axis.index() as u8,
                        len: len as u8,
                        source_mask,
                        directions,
                    });
                }
            }
        }
    }
    debug_assert!(source_groups.len() <= 6 * 64);
    assert_eq!(
        plans.len(),
        1734,
        "stable plan IDs must match the search history allocation"
    );
    let mut plan_lookup = source_groups
        .iter()
        .flat_map(|group| group.directions.iter().flatten())
        .map(|direction| (direction.history_key, direction.plan_index))
        .collect::<Vec<_>>();
    plan_lookup.sort_unstable_by_key(|entry| entry.0);
    debug_assert!(plan_lookup.windows(2).all(|pair| pair[0].0 != pair[1].0));
    let owned_groups = OwnedGroupTables::new(&source_groups);
    let mut plan_hash = vec![(0u32, u16::MAX); (plan_lookup.len() * 2).next_power_of_two()];
    for &(k, p) in &plan_lookup {
        let mut i = k.wrapping_mul(0x9e3779b9) as usize & (plan_hash.len() - 1);
        while plan_hash[i].1 != u16::MAX {
            i = (i + 1) & (plan_hash.len() - 1);
        }
        plan_hash[i] = (k, p);
    }
    FastMovegenTables {
        plan_hash,
        owned_groups,
        source_masks: source_groups.iter().map(|g| g.source_mask).collect(),
        source_groups,
        plans,
    }
}
fn build_fast_group_directions(
    cells: &[crate::board::CellId; 3],
    len: u8,
    axis: Option<LineAxis>,
    source_mask: u64,
    plans: &mut Vec<FastMovePlan>,
) -> [Option<FastGroupDirection>; 6] {
    std::array::from_fn(|dir_idx| {
        let direction = ALL_DIRECTIONS[dir_idx];
        let group = &cells[..len as usize];
        let translated = build_fast_translated_cells(group, direction)?;
        let translated_mask = translated
            .iter()
            .flatten()
            .fold(0u64, |mask, cell| mask | (1u64 << cell.as_u8()));
        let (inline, first_step) = match axis {
            None => (false, translated[0]),
            Some(axis) => {
                let inline = move_is_inline(axis, direction);
                let front = move_front_cell(group, direction)?;
                let first_step = if inline {
                    neighbor_cell(front, direction)
                } else {
                    translated[0]
                };
                if inline && first_step.is_none() {
                    return None;
                }
                (inline, first_step)
            }
        };
        let mut ray_bits = [0u64; 3];
        let mut landing = [None; 2];
        let mut current = first_step;
        for index in 0..3 {
            let Some(cell) = current else {
                break;
            };
            ray_bits[index] = 1u64 << cell.as_u8();
            if index > 0 {
                landing[index - 1] = Some(cell);
            }
            current = geometry().cell(cell).neighbors[direction.index()];
        }
        let plan_index = u16::try_from(plans.len()).expect("fast move plan count fits u16");
        plans.push(FastMovePlan {
            own_toggle: source_mask ^ translated_mask,
        });
        Some(FastGroupDirection {
            candidate_move: Move::new_unchecked(group, direction),
            inline,
            translated_mask,
            history_key: history_group_key(group, direction),
            plan_index,
            ray_bits,
            landing,
        })
    })
}
fn canonical_group_cells(group: &[crate::board::CellId]) -> [crate::board::CellId; 3] {
    let mut out = [group[0]; 3];
    for (index, cell) in group.iter().copied().enumerate() {
        out[index] = cell;
    }
    out[..group.len()].sort_unstable();
    out
}
fn build_fast_translated_cells(
    cells: &[crate::board::CellId],
    direction: Direction,
) -> Option<[Option<crate::board::CellId>; 3]> {
    let geom = geometry();
    let mut translated = [None; 3];
    for (index, cell) in cells.iter().copied().enumerate() {
        translated[index] = Some(geom.cell(cell).neighbors[direction.index()]?);
    }
    Some(translated)
}
fn fast_source_mask(cells: &[crate::board::CellId; 3], len: u8) -> u64 {
    cells[..len as usize]
        .iter()
        .fold(0u64, |mask, cell| mask | (1u64 << cell.as_u8()))
}
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct LegalMoveEntry {
    pub(crate) candidate_move: Move,
    pub(crate) is_ejection: bool,
    pub(crate) is_push: bool,
    pub(crate) history_key: u32,
    pub(crate) plan_index: u16,
    pub(crate) enemy_effect: crate::movegen::CompactEnemyEffect,
}
#[derive(Clone, Debug)]
pub(crate) struct OwnedGroupTables {
    shifts: [Vec<(i8, u64)>; 6],
    pub(crate) ids: [[[u16; 61]; 6]; 2],
}
impl OwnedGroupTables {
    #[allow(
        clippy::needless_range_loop,
        reason = "Keep benchmarked hot-path indexing and call structure unchanged."
    )]
    fn new(groups: &[FastSourceGroup]) -> Self {
        let geom = geometry();
        let mut masks = [[0u64; 19]; 6];
        for c in geom.cells() {
            for d in 0..6 {
                if let Some(n) = c.neighbors[d] {
                    let delta = n.as_u8() as i8 - c.index.as_u8() as i8;
                    assert!((-9..=9).contains(&delta));
                    masks[d][(delta + 9) as usize] |= 1u64 << c.index.as_u8();
                }
            }
        }
        let shifts = std::array::from_fn(|d| {
            masks[d]
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| if m == 0 { None } else { Some((i as i8 - 9, m)) })
                .collect()
        });
        let mut ids = [[[u16::MAX; 61]; 6]; 2];
        for (id, g) in groups.iter().enumerate() {
            if g.len < 2 {
                assert_eq!(id, g.source_mask.trailing_zeros() as usize);
                continue;
            }
            let a = g.source_mask.trailing_zeros() as usize;
            let rest = g.source_mask & (g.source_mask - 1);
            let b = rest.trailing_zeros() as u8;
            let d = geom.cells()[a]
                .neighbors
                .iter()
                .position(|x| x.is_some_and(|c| c.as_u8() == b))
                .unwrap();
            ids[g.len as usize - 2][d][a] = id as u16;
        }
        Self { shifts, ids }
    }
    #[inline]
    fn backshift(&self, bits: u64, d: usize) -> u64 {
        let mut out = 0;
        for &(delta, mask) in &self.shifts[d] {
            out |= if delta >= 0 {
                (bits >> (delta as u32)) & mask
            } else {
                (bits << (-delta as u32)) & mask
            };
        }
        out
    }
    pub(crate) fn generate(&self, own: u64) -> [u64; 6] {
        let mut out = [0u64; 6];
        out[0] = own;
        for d in 0..3 {
            let n = self.backshift(own, d);
            for (k, mut active) in [own & n, own & n & self.backshift(n, d)]
                .into_iter()
                .enumerate()
            {
                while active != 0 {
                    let c = active.trailing_zeros() as usize;
                    active &= active - 1;
                    let id = self.ids[k][d][c];
                    if id != u16::MAX {
                        out[id as usize / 64] |= 1u64 << (id as usize % 64);
                    }
                }
            }
        }
        out
    }
}

impl OwnedGroupTables {
    pub(crate) fn generate_push(&self, own: u64, enemy: u64) -> [u64; 6] {
        let mut out = [0u64; 6];
        for d in 0..3 {
            let n = self.backshift(own, d);
            let far = self.backshift(self.backshift(enemy, d), d);
            let near = self.backshift(enemy, (d + 3) % 6);
            for (k, mut active) in [
                (own & n) & (far | near),
                (own & n & self.backshift(n, d)) & (self.backshift(far, d) | near),
            ]
            .into_iter()
            .enumerate()
            {
                while active != 0 {
                    let c = active.trailing_zeros() as usize;
                    active &= active - 1;
                    let id = self.ids[k][d][c];
                    if id != u16::MAX {
                        out[id as usize / 64] |= 1u64 << (id as usize % 64);
                    }
                }
            }
        }
        out
    }
}

const fn combination_count(n: usize, k: usize) -> usize {
    match k {
        0 => 1,
        1 => n,
        2 => (n * (n - 1)) / 2,
        3 => (n * (n - 1) * (n - 2)) / 6,
        _ => 0,
    }
}
pub(crate) fn history_group_key(
    source_cells: &[crate::board::CellId],
    direction: Direction,
) -> u32 {
    (history_source_group_rank(source_cells) * 6 + direction.index()) as u32
}
fn history_source_group_rank(source_cells: &[crate::board::CellId]) -> usize {
    match source_cells {
        [first] => first.as_usize(),
        [first, second] => HISTORY_SOURCE_GROUPS_LEN1 + combination_rank_2(*first, *second),
        [first, second, third] => {
            HISTORY_SOURCE_GROUPS_LEN1
                + HISTORY_SOURCE_GROUPS_LEN2
                + combination_rank_3(*first, *second, *third)
        }
        _ => unreachable!("move source groups must contain 1..=3 cells"),
    }
}
fn combination_rank_2(first: crate::board::CellId, second: crate::board::CellId) -> usize {
    combination_count(first.as_usize(), 1) + combination_count(second.as_usize(), 2)
}
fn combination_rank_3(
    first: crate::board::CellId,
    second: crate::board::CellId,
    third: crate::board::CellId,
) -> usize {
    combination_count(first.as_usize(), 1)
        + combination_count(second.as_usize(), 2)
        + combination_count(third.as_usize(), 3)
}
