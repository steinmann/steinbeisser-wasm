use ::core::cmp::Ordering;
use ::core::fmt;
use ::core::hash::{Hash, Hasher};
use std::array;
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Color {
    Black,
    White,
}
impl Color {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Black => "black",
            Self::White => "white",
        }
    }
    pub const fn other(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "b" | "B" | "black" | "BLACK" => Some(Self::Black),
            "w" | "W" | "white" | "WHITE" => Some(Self::White),
            _ => None,
        }
    }
}
impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

const ROW_LENGTHS: [u8; 9] = [5, 6, 7, 8, 9, 8, 7, 6, 5];
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CellId(u8);
impl CellId {
    pub const fn new(value: u8) -> Option<Self> {
        if value < 61 { Some(Self(value)) } else { None }
    }
    pub(crate) const fn new_unchecked(value: u8) -> Self {
        Self(value)
    }
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
    pub const fn as_u8(self) -> u8 {
        self.0
    }
    pub fn coord(self) -> Coord {
        geometry().cell(self).coord
    }
}
impl fmt::Display for CellId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.coord().fmt(f)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct AxialCoord {
    pub q: i8,
    pub r: i8,
}
impl AxialCoord {
    pub const fn new(q: i8, r: i8) -> Self {
        Self { q, r }
    }
    pub const fn s(self) -> i8 {
        -self.q - self.r
    }
    pub const fn is_on_board(self) -> bool {
        let s = self.s();
        self.q.abs() <= BOARD_RADIUS && self.r.abs() <= BOARD_RADIUS && s.abs() <= BOARD_RADIUS
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Coord {
    row: u8,
    column: u8,
}
impl Coord {
    pub const MIN_ROW: u8 = 0;
    pub const MAX_ROW: u8 = 8;
    pub const fn new(row: u8, column: u8) -> Option<Self> {
        if row > Self::MAX_ROW || column == 0 {
            return None;
        }
        if column > ROW_LENGTHS[row as usize] {
            return None;
        }
        Some(Self { row, column })
    }
    pub const fn row_length(row: u8) -> Option<u8> {
        if row > Self::MAX_ROW {
            None
        } else {
            Some(ROW_LENGTHS[row as usize])
        }
    }
    pub const fn row(self) -> u8 {
        self.row
    }
    pub const fn column(self) -> u8 {
        self.column
    }
    pub const fn row_char(self) -> char {
        (b'A' + self.row) as char
    }
    pub fn parse(value: &str) -> Option<Self> {
        if value.len() < 2 {
            return None;
        }
        let mut chars = value.chars();
        let row_char = chars.next()?.to_ascii_uppercase();
        if !('A'..='I').contains(&row_char) {
            return None;
        }
        let column = chars.as_str().parse::<u8>().ok()?;
        Self::new(row_char as u8 - b'A', column)
    }
}
impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.row_char(), self.column)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Direction {
    East,
    Se,
    Sw,
    West,
    Nw,
    Ne,
}
pub const ALL_DIRECTIONS: [Direction; 6] = [
    Direction::East,
    Direction::Se,
    Direction::Sw,
    Direction::West,
    Direction::Nw,
    Direction::Ne,
];
impl Direction {
    pub const fn index(self) -> usize {
        match self {
            Self::East => 0,
            Self::Se => 1,
            Self::Sw => 2,
            Self::West => 3,
            Self::Nw => 4,
            Self::Ne => 5,
        }
    }
    pub const fn delta(self) -> AxialCoord {
        match self {
            Self::East => AxialCoord::new(1, 0),
            Self::Se => AxialCoord::new(0, 1),
            Self::Sw => AxialCoord::new(-1, 1),
            Self::West => AxialCoord::new(-1, 0),
            Self::Nw => AxialCoord::new(0, -1),
            Self::Ne => AxialCoord::new(1, -1),
        }
    }
    pub const fn opposite(self) -> Self {
        match self {
            Self::East => Self::West,
            Self::Se => Self::Nw,
            Self::Sw => Self::Ne,
            Self::West => Self::East,
            Self::Nw => Self::Se,
            Self::Ne => Self::Sw,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::East => "E",
            Self::Se => "SE",
            Self::Sw => "SW",
            Self::West => "W",
            Self::Nw => "NW",
            Self::Ne => "NE",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "E" => Some(Self::East),
            "SE" => Some(Self::Se),
            "SW" => Some(Self::Sw),
            "W" => Some(Self::West),
            "NW" => Some(Self::Nw),
            "NE" => Some(Self::Ne),
            _ => None,
        }
    }
    pub const fn from_delta(delta: AxialCoord) -> Option<Self> {
        match (delta.q, delta.r) {
            (1, 0) => Some(Self::East),
            (0, 1) => Some(Self::Se),
            (-1, 1) => Some(Self::Sw),
            (-1, 0) => Some(Self::West),
            (0, -1) => Some(Self::Nw),
            (1, -1) => Some(Self::Ne),
            _ => None,
        }
    }
}
impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum LineAxis {
    Q,
    R,
    S,
}
impl LineAxis {
    pub const fn index(self) -> usize {
        match self {
            Self::Q => 0,
            Self::R => 1,
            Self::S => 2,
        }
    }
}

pub const BOARD_RADIUS: i8 = 4;
pub const CELL_COUNT: usize = 61;
pub const LINE_COUNT_PER_AXIS: usize = 9;
pub const SYMMETRY_COUNT: usize = 12;
static GEOMETRY: OnceLock<Geometry> = OnceLock::new();
pub fn geometry() -> &'static Geometry {
    GEOMETRY.get_or_init(Geometry::build)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellGeometry {
    pub index: CellId,
    pub axial: AxialCoord,
    pub coord: Coord,
    pub neighbors: [Option<CellId>; 6],
    pub neighbor_mask: u64,
    pub line_ids: [usize; 3],
    pub center_weight: u8,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Line {
    pub axis: LineAxis,
    pub coordinate: i8,
    pub cells: Vec<CellId>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Symmetry {
    pub rotations: u8,
    pub mirrored: bool,
}
impl Symmetry {
    pub const fn all() -> [Self; SYMMETRY_COUNT] {
        [
            Self::new(0, false),
            Self::new(1, false),
            Self::new(2, false),
            Self::new(3, false),
            Self::new(4, false),
            Self::new(5, false),
            Self::new(0, true),
            Self::new(1, true),
            Self::new(2, true),
            Self::new(3, true),
            Self::new(4, true),
            Self::new(5, true),
        ]
    }
    pub const fn new(rotations: u8, mirrored: bool) -> Self {
        Self {
            rotations: rotations % 6,
            mirrored,
        }
    }
    pub const fn identity() -> Self {
        Self::new(0, false)
    }
}
#[derive(Clone, Debug)]
pub struct Geometry {
    cells: [CellGeometry; CELL_COUNT],
    lines: [Vec<Line>; 3],
    symmetries: [[CellId; CELL_COUNT]; SYMMETRY_COUNT],
    coord_lookup: BTreeMap<Coord, CellId>,
    axial_lookup: BTreeMap<AxialCoord, CellId>,
}
impl Geometry {
    fn build() -> Self {
        let mut cells = Vec::with_capacity(CELL_COUNT);
        let mut coord_lookup = BTreeMap::new();
        let mut axial_lookup = BTreeMap::new();
        for row in 0..=8u8 {
            let r = row as i8 - BOARD_RADIUS;
            let q_min = (-BOARD_RADIUS).max(-r - BOARD_RADIUS);
            let q_max = BOARD_RADIUS.min(-r + BOARD_RADIUS);
            for (offset, q) in (q_min..=q_max).enumerate() {
                let coord = Coord::new(row, offset as u8 + 1).unwrap();
                let axial = AxialCoord::new(q, r);
                let center_weight = Self::center_weight(axial);
                let line_ids = [
                    Self::line_id(axial.q),
                    Self::line_id(axial.r),
                    Self::line_id(axial.s()),
                ];
                let index = CellId::new(cells.len() as u8).unwrap();
                let cell = CellGeometry {
                    index,
                    axial,
                    coord,
                    neighbors: [None; 6],
                    neighbor_mask: 0,
                    line_ids,
                    center_weight,
                };
                cells.push(cell);
                coord_lookup.insert(coord, index);
                axial_lookup.insert(axial, index);
            }
        }
        let mut cells: [CellGeometry; CELL_COUNT] = cells.try_into().unwrap();
        for cell in &mut cells {
            cell.neighbors = array::from_fn(|direction_index| {
                let direction = ALL_DIRECTIONS[direction_index];
                let delta = direction.delta();
                let next = AxialCoord::new(cell.axial.q + delta.q, cell.axial.r + delta.r);
                axial_lookup.get(&next).copied()
            });
            cell.neighbor_mask = cell
                .neighbors
                .iter()
                .flatten()
                .fold(0u64, |mask, nb| mask | (1u64 << nb.as_u8()));
        }
        let lines = Self::build_lines(&cells);
        let symmetries = Self::build_symmetries(&cells, &axial_lookup);
        Self {
            cells,
            lines,
            symmetries,
            coord_lookup,
            axial_lookup,
        }
    }
    fn build_lines(cells: &[CellGeometry; CELL_COUNT]) -> [Vec<Line>; 3] {
        array::from_fn(|axis_index| {
            (-BOARD_RADIUS..=BOARD_RADIUS)
                .map(|coordinate| {
                    let axis = match axis_index {
                        0 => LineAxis::Q,
                        1 => LineAxis::R,
                        _ => LineAxis::S,
                    };
                    let mut line_cells: Vec<CellId> = cells
                        .iter()
                        .filter(|cell| match axis {
                            LineAxis::Q => cell.axial.q == coordinate,
                            LineAxis::R => cell.axial.r == coordinate,
                            LineAxis::S => cell.axial.s() == coordinate,
                        })
                        .map(|cell| cell.index)
                        .collect();
                    line_cells.sort_by_key(|index| {
                        let axial = cells[index.as_usize()].axial;
                        match axis {
                            LineAxis::Q => axial.r,
                            LineAxis::R => axial.q,
                            LineAxis::S => axial.q,
                        }
                    });
                    Line {
                        axis,
                        coordinate,
                        cells: line_cells,
                    }
                })
                .collect()
        })
    }
    fn build_symmetries(
        cells: &[CellGeometry; CELL_COUNT],
        axial_lookup: &BTreeMap<AxialCoord, CellId>,
    ) -> [[CellId; CELL_COUNT]; SYMMETRY_COUNT] {
        let all = Symmetry::all();
        array::from_fn(|symmetry_index| {
            let symmetry = all[symmetry_index];
            array::from_fn(|cell_index| {
                let axial = cells[cell_index].axial;
                let transformed = Self::apply_symmetry(axial, symmetry);
                axial_lookup.get(&transformed).copied().unwrap()
            })
        })
    }
    fn apply_symmetry(axial: AxialCoord, symmetry: Symmetry) -> AxialCoord {
        let mut cube = (axial.q, -axial.q - axial.r, axial.r);
        if symmetry.mirrored {
            cube = (cube.0, cube.2, cube.1);
        }
        for _ in 0..symmetry.rotations {
            cube = (-cube.2, -cube.0, -cube.1);
        }
        AxialCoord::new(cube.0, cube.2)
    }
    fn center_weight(axial: AxialCoord) -> u8 {
        let shell = axial.q.abs().max(axial.r.abs()).max(axial.s().abs());
        (BOARD_RADIUS - shell) as u8
    }
    fn line_id(coordinate: i8) -> usize {
        (coordinate + BOARD_RADIUS) as usize
    }
    pub fn cells(&self) -> &[CellGeometry; CELL_COUNT] {
        &self.cells
    }
    pub fn cell(&self, index: CellId) -> &CellGeometry {
        &self.cells[index.as_usize()]
    }
    pub fn lines(&self, axis: LineAxis) -> &[Line] {
        &self.lines[axis.index()]
    }
    pub fn line(&self, axis: LineAxis, line_id: usize) -> &Line {
        &self.lines[axis.index()][line_id]
    }
    pub fn index_of_coord(&self, coord: Coord) -> Option<CellId> {
        self.coord_lookup.get(&coord).copied()
    }
    pub fn index_of_axial(&self, axial: AxialCoord) -> Option<CellId> {
        self.axial_lookup.get(&axial).copied()
    }
    pub fn symmetry_map(&self, symmetry: Symmetry) -> &[CellId; CELL_COUNT] {
        &self.symmetries[Self::symmetry_slot(symmetry)]
    }
    pub fn transform(&self, index: CellId, symmetry: Symmetry) -> CellId {
        self.symmetry_map(symmetry)[index.as_usize()]
    }
    pub fn transform_direction(&self, direction: Direction, symmetry: Symmetry) -> Direction {
        let tf = Self::apply_symmetry(direction.delta(), symmetry);
        Direction::from_delta(tf).unwrap()
    }
    pub fn inverse_symmetry(&self, symmetry: Symmetry) -> Symmetry {
        let slot = Self::symmetry_slot(symmetry);
        let map = &self.symmetries[slot];
        Symmetry::all()
            .into_iter()
            .find(|candidate| {
                self.symmetry_map(*candidate)
                    .iter()
                    .enumerate()
                    .all(|(index, candidate_target)| {
                        map[candidate_target.as_usize()].as_usize() == index
                    })
            })
            .unwrap()
    }
    fn symmetry_slot(symmetry: Symmetry) -> usize {
        symmetry.rotations as usize + if symmetry.mirrored { 6 } else { 0 }
    }
}

// move shapes and move text
#[derive(Clone, Copy, Debug)]
pub struct Move {
    source_cells: [CellId; 3],
    len: u8,
    direction: Direction,
}
impl Move {
    pub const PLACEHOLDER: Self = Self {
        source_cells: [CellId::new_unchecked(0); 3],
        len: 1,
        direction: Direction::East,
    };
    pub fn new(mut source_cells: Vec<CellId>, direction: Direction) -> Result<Self, MoveError> {
        if source_cells.is_empty() {
            return Err(MoveError::EmptySourceGroup);
        }
        if source_cells.len() > 3 {
            return Err(MoveError::TooManySourceCells(source_cells.len()));
        }
        source_cells.sort_unstable();
        for pair in source_cells.windows(2) {
            if pair[0] == pair[1] {
                return Err(MoveError::DuplicateSourceCell(pair[0].coord()));
            }
        }
        if source_cells.len() > 1 {
            validate_contiguous_group(&source_cells)?;
        }
        let len = source_cells.len() as u8;
        let mut packed_source_cells = [source_cells[0]; 3];
        for (index, cell) in source_cells.into_iter().enumerate() {
            packed_source_cells[index] = cell;
        }
        Ok(Self {
            source_cells: packed_source_cells,
            len,
            direction,
        })
    }
    pub fn from_cells(source_cells: &[CellId], direction: Direction) -> Result<Self, MoveError> {
        if source_cells.is_empty() {
            return Err(MoveError::EmptySourceGroup);
        }
        if source_cells.len() > 3 {
            return Err(MoveError::TooManySourceCells(source_cells.len()));
        }
        let len = source_cells.len() as u8;
        let mut packed_source_cells = [source_cells[0]; 3];
        for (index, cell) in source_cells.iter().copied().enumerate() {
            packed_source_cells[index] = cell;
        }
        packed_source_cells[..source_cells.len()].sort_unstable();
        for pair in packed_source_cells[..source_cells.len()].windows(2) {
            if pair[0] == pair[1] {
                return Err(MoveError::DuplicateSourceCell(pair[0].coord()));
            }
        }
        if source_cells.len() > 1 {
            validate_contiguous_group(&packed_source_cells[..source_cells.len()])?;
        }
        Ok(Self {
            source_cells: packed_source_cells,
            len,
            direction,
        })
    }
    pub fn new_unchecked(source_cells: &[CellId], direction: Direction) -> Self {
        let len = source_cells.len() as u8;
        let mut packed_source_cells = [source_cells[0]; 3];
        match source_cells {
            [first] => {
                packed_source_cells[0] = *first;
            }
            [first, second] => {
                if first <= second {
                    packed_source_cells[0] = *first;
                    packed_source_cells[1] = *second;
                } else {
                    packed_source_cells[0] = *second;
                    packed_source_cells[1] = *first;
                }
            }
            [first, second, third] => {
                let (mut a, mut b, mut c) = (*first, *second, *third);
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                if b > c {
                    std::mem::swap(&mut b, &mut c);
                }
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                packed_source_cells[0] = a;
                packed_source_cells[1] = b;
                packed_source_cells[2] = c;
            }
            _ => unreachable!("move source groups must contain 1..=3 cells"),
        }
        Self {
            source_cells: packed_source_cells,
            len,
            direction,
        }
    }
    pub fn source_cells(&self) -> &[CellId] {
        &self.source_cells[..self.len as usize]
    }
    pub fn direction(&self) -> Direction {
        self.direction
    }
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn transform(&self, symmetry: Symmetry) -> Self {
        let geometry = geometry();
        let mut transformed_cells = [self.source_cells[0]; 3];
        for (index, cell) in self.source_cells().iter().copied().enumerate() {
            transformed_cells[index] = geometry.transform(cell, symmetry);
        }
        Self::from_cells(
            &transformed_cells[..self.len as usize],
            geometry.transform_direction(self.direction, symmetry),
        )
        .unwrap()
    }
}
impl PartialEq for Move {
    fn eq(&self, other: &Self) -> bool {
        self.direction == other.direction && self.source_cells() == other.source_cells()
    }
}
impl Eq for Move {}
impl Hash for Move {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.source_cells().hash(hasher);
        self.direction.hash(hasher);
    }
}
impl PartialOrd for Move {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Move {
    fn cmp(&self, other: &Self) -> Ordering {
        self.source_cells()
            .cmp(other.source_cells())
            .then_with(|| self.direction.cmp(&other.direction))
    }
}
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, cell) in self.source_cells().iter().enumerate() {
            if index > 0 {
                f.write_str(",")?;
            }
            write!(f, "{cell}")?;
        }
        write!(f, ">{}", self.direction)
    }
}
impl FromStr for Move {
    type Err = MoveError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (source_text, direction) = value
            .trim()
            .split_once('>')
            .ok_or_else(|| MoveError::InvalidSyntax(value.to_owned()))?;
        let direction = Direction::parse(direction)
            .ok_or_else(|| MoveError::InvalidDirection(direction.to_owned()))?;
        let mut source_cells = Vec::new();
        for token in source_text.split(',') {
            let coord =
                Coord::parse(token).ok_or_else(|| MoveError::InvalidCoord(token.to_owned()))?;
            let index = geometry().index_of_coord(coord).unwrap();
            source_cells.push(index);
        }
        Self::new(source_cells, direction)
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoveError {
    EmptySourceGroup,
    TooManySourceCells(usize),
    DuplicateSourceCell(Coord),
    NonLinearSourceCells,
    NonContiguousSourceCells,
    InvalidCoord(String),
    InvalidDirection(String),
    InvalidSyntax(String),
}
impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("m")
    }
}
fn validate_contiguous_group(source_cells: &[CellId]) -> Result<(), MoveError> {
    let geometry = geometry();
    let first = geometry.cell(source_cells[0]);
    let shared_axis = [first.line_ids[0], first.line_ids[1], first.line_ids[2]]
        .into_iter()
        .enumerate()
        .find_map(|(axis_index, line_id)| {
            source_cells
                .iter()
                .all(|cell| geometry.cell(*cell).line_ids[axis_index] == line_id)
                .then_some((axis_index, line_id))
        })
        .ok_or(MoveError::NonLinearSourceCells)?;
    let line = &geometry.lines(match shared_axis.0 {
        0 => LineAxis::Q,
        1 => LineAxis::R,
        _ => LineAxis::S,
    })[shared_axis.1];
    let mut positions = source_cells
        .iter()
        .map(|cell| {
            line.cells
                .iter()
                .position(|line_cell| line_cell == cell)
                .unwrap()
        })
        .collect::<Vec<_>>();
    positions.sort_unstable();
    for pair in positions.windows(2) {
        if pair[1] != pair[0] + 1 {
            return Err(MoveError::NonContiguousSourceCells);
        }
    }
    Ok(())
}

// canonical positions and validation
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    side_to_move: Color,
    black: Vec<CellId>,
    white: Vec<CellId>,
}
impl Position {
    pub const MAX_PIECES_PER_SIDE: usize = 14;
    pub fn new(
        side_to_move: Color,
        black: Vec<CellId>,
        white: Vec<CellId>,
    ) -> Result<Self, PositionError> {
        let mut record = Self {
            side_to_move,
            black,
            white,
        };
        record.normalize();
        record.validate()?;
        Ok(record)
    }
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }
    pub fn black(&self) -> &[CellId] {
        &self.black
    }
    pub fn white(&self) -> &[CellId] {
        &self.white
    }
    pub(crate) fn cells_for_mut(&mut self, color: Color) -> &mut Vec<CellId> {
        match color {
            Color::Black => &mut self.black,
            Color::White => &mut self.white,
        }
    }
    pub(crate) fn set_side_to_move(&mut self, side_to_move: Color) {
        self.side_to_move = side_to_move;
    }
    pub fn contains(&self, color: Color, cell: CellId) -> bool {
        self.cells_for(color).binary_search(&cell).is_ok()
    }
    pub fn occupant(&self, cell: CellId) -> Option<Color> {
        if self.black.binary_search(&cell).is_ok() {
            Some(Color::Black)
        } else if self.white.binary_search(&cell).is_ok() {
            Some(Color::White)
        } else {
            None
        }
    }
    pub fn marble_count(&self, color: Color) -> usize {
        self.cells_for(color).len()
    }
    pub fn transform(&self, symmetry: Symmetry) -> Self {
        let geometry = geometry();
        let black = self
            .black
            .iter()
            .copied()
            .map(|cell| geometry.transform(cell, symmetry))
            .collect();
        let white = self
            .white
            .iter()
            .copied()
            .map(|cell| geometry.transform(cell, symmetry))
            .collect();
        Self::new(self.side_to_move, black, white).unwrap()
    }
    pub fn validate(&self) -> Result<(), PositionError> {
        Self::validate_color_list(Color::Black, &self.black)?;
        Self::validate_color_list(Color::White, &self.white)?;
        for black in &self.black {
            if self.white.binary_search(black).is_ok() {
                return Err(PositionError::CellOverlap(black.coord()));
            }
        }
        Ok(())
    }
    pub fn canonical_string(&self) -> String {
        self.to_string()
    }
    fn normalize(&mut self) {
        self.black.sort_unstable();
        self.white.sort_unstable();
    }
    fn cells_for(&self, color: Color) -> &[CellId] {
        match color {
            Color::Black => &self.black,
            Color::White => &self.white,
        }
    }
    fn validate_color_list(color: Color, cells: &[CellId]) -> Result<(), PositionError> {
        if cells.len() > Self::MAX_PIECES_PER_SIDE {
            return Err(PositionError::TooManyPieces {
                color,
                count: cells.len(),
            });
        }
        for pair in cells.windows(2) {
            if pair[0] >= pair[1] {
                return Err(PositionError::NonCanonicalCellOrder(color));
            }
        }
        Ok(())
    }
    fn parse_fen_board(raw: &str) -> Result<(Vec<CellId>, Vec<CellId>), PositionError> {
        let rows = raw.split('/').collect::<Vec<_>>();
        if rows.len() != 9 {
            return Err(PositionError::InvalidFenRowCount(rows.len()));
        }

        let mut black = Vec::new();
        let mut white = Vec::new();
        for (row_index, raw_row) in rows.iter().enumerate() {
            let row = row_index as u8;
            let expected = Coord::row_length(row).unwrap();
            let mut column = 1_u8;
            let mut chars = raw_row.chars().peekable();

            while let Some(char) = chars.next() {
                if char.is_ascii_digit() {
                    let mut empty_count = char.to_digit(10).unwrap() as u8;
                    while let Some(next) = chars.peek().copied() {
                        if !next.is_ascii_digit() {
                            break;
                        }
                        chars.next();
                        empty_count = empty_count
                            .saturating_mul(10)
                            .saturating_add(next.to_digit(10).unwrap() as u8);
                    }
                    if empty_count == 0 {
                        return Err(PositionError::InvalidFenCell {
                            row: (b'A' + row) as char,
                            cell: char,
                        });
                    }
                    column = column.saturating_add(empty_count);
                    continue;
                }

                let color = match char {
                    'S' => Color::Black,
                    's' => Color::White,
                    _ => {
                        return Err(PositionError::InvalidFenCell {
                            row: (b'A' + row) as char,
                            cell: char,
                        });
                    }
                };

                let coord = Coord::new(row, column).ok_or(PositionError::InvalidFenRowLength {
                    row: (b'A' + row) as char,
                    expected,
                    actual: column,
                })?;
                let cell = geometry().index_of_coord(coord).unwrap();
                match color {
                    Color::Black => black.push(cell),
                    Color::White => white.push(cell),
                }
                column = column.saturating_add(1);
            }

            let actual = column.saturating_sub(1);
            if actual != expected {
                return Err(PositionError::InvalidFenRowLength {
                    row: (b'A' + row) as char,
                    expected,
                    actual,
                });
            }
        }

        Ok((black, white))
    }
}
impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows = Vec::with_capacity(9);
        for row in Coord::MIN_ROW..=Coord::MAX_ROW {
            let mut row_text = String::new();
            let mut empty_count = 0_u8;

            for column in 1..=Coord::row_length(row).unwrap() {
                let coord = Coord::new(row, column).unwrap();
                let cell = geometry().index_of_coord(coord).unwrap();
                let marble = match self.occupant(cell) {
                    Some(Color::Black) => Some('S'),
                    Some(Color::White) => Some('s'),
                    None => None,
                };

                if let Some(marble) = marble {
                    if empty_count > 0 {
                        row_text.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    row_text.push(marble);
                } else {
                    empty_count += 1;
                }
            }

            if empty_count > 0 {
                row_text.push_str(&empty_count.to_string());
            }
            rows.push(row_text);
        }

        let side = match self.side_to_move {
            Color::Black => 'b',
            Color::White => 'w',
        };
        let white_ejected = Self::MAX_PIECES_PER_SIDE.saturating_sub(self.white.len());
        let black_ejected = Self::MAX_PIECES_PER_SIDE.saturating_sub(self.black.len());
        write!(
            f,
            "{} 0 0 {} {} {}",
            rows.join("/"),
            side,
            white_ejected,
            black_ejected
        )
    }
}
impl FromStr for Position {
    type Err = PositionError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let tokens = value.trim().split_whitespace().collect::<Vec<_>>();
        let board = tokens.first().ok_or(PositionError::MissingFenBoard)?;
        let side_to_move = tokens
            .get(3)
            .and_then(|token| Color::parse(token))
            .ok_or(PositionError::MissingSideToMove)?;
        let (black, white) = Self::parse_fen_board(board)?;
        Self::new(side_to_move, black, white)
    }
}
pub trait EngineStateView {
    fn position(&self) -> &Position;
    fn side_to_move(&self) -> Color {
        self.position().side_to_move()
    }
    fn validate(&self) -> Result<(), PositionError> {
        self.position().validate()
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PositionError {
    MissingFenBoard,
    MissingSideToMove,
    InvalidFenRowCount(usize),
    InvalidFenRowLength { row: char, expected: u8, actual: u8 },
    InvalidFenCell { row: char, cell: char },
    TooManyPieces { color: Color, count: usize },
    CellOverlap(Coord),
    NonCanonicalCellOrder(Color),
}
impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFenBoard => f.write_str("missing fen board"),
            Self::MissingSideToMove => f.write_str("missing fen side to move"),
            Self::InvalidFenRowCount(count) => write!(f, "fen has {count} rows"),
            Self::InvalidFenRowLength {
                row,
                expected,
                actual,
            } => write!(f, "fen row {row} has {actual} cells, expected {expected}"),
            Self::InvalidFenCell { row, cell } => {
                write!(f, "fen row {row} has invalid cell {cell}")
            }
            Self::TooManyPieces { color, count } => write!(f, "{color} has {count} marbles"),
            Self::CellOverlap(coord) => write!(f, "both sides occupy {coord}"),
            Self::NonCanonicalCellOrder(color) => write!(f, "{color} cells are not canonical"),
        }
    }
}
