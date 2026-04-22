pub mod ac {
    mod _c {
        use core::fmt;
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub enum Co {
            Black,
            White,
        }
        impl Co {
            pub const fn code(self) -> char {
                match self {
                    Self::Black => 'b',
                    Self::White => 'w',
                }
            }
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
        impl fmt::Display for Co {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.name())
            }
        }
    }
    mod coord {
        use super::gm::{gm, Br};
        use core::fmt;
        const ROW_LENGTHS: [u8; 9] = [5, 6, 7, 8, 9, 8, 7, 6, 5];
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub struct Ci(u8);
        impl Ci {
            pub const fn new(value: u8) -> Option<Self> {
                if value < 61 {
                    Some(Self(value))
                } else {
                    None
                }
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
                gm().cell(self).coord
            }
        }
        impl fmt::Display for Ci {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.coord().fmt(f)
            }
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub struct Ax {
            pub q: i8,
            pub r: i8,
        }
        impl Ax {
            pub const fn new(q: i8, r: i8) -> Self {
                Self { q, r }
            }
            pub const fn s(self) -> i8 {
                -self.q - self.r
            }
            pub const fn is_on_board(self) -> bool {
                let s = self.s();
                self.q.abs() <= Br && self.r.abs() <= Br && s.abs() <= Br
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
    }
    mod direction {
        use super::coord::Ax;
        use core::fmt;
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub enum Di {
            East,
            Se,
            Sw,
            West,
            Nw,
            Ne,
        }
        pub const Ad: [Di; 6] = [Di::East, Di::Se, Di::Sw, Di::West, Di::Nw, Di::Ne];
        impl Di {
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
            pub const fn delta(self) -> Ax {
                match self {
                    Self::East => Ax::new(1, 0),
                    Self::Se => Ax::new(0, 1),
                    Self::Sw => Ax::new(-1, 1),
                    Self::West => Ax::new(-1, 0),
                    Self::Nw => Ax::new(0, -1),
                    Self::Ne => Ax::new(1, -1),
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
            pub const fn from_delta(delta: Ax) -> Option<Self> {
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
        impl fmt::Display for Di {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub enum Li {
            Q,
            R,
            S,
        }
        impl Li {
            pub const fn index(self) -> usize {
                match self {
                    Self::Q => 0,
                    Self::R => 1,
                    Self::S => 2,
                }
            }
        }
    }
    mod gm {
        use super::coord::{Ax, Ci, Coord};
        use super::direction::{Ad, Di, Li};
        use std::array;
        use std::collections::BTreeMap;
        use std::sync::OnceLock;
        pub const Br: i8 = 4;
        pub const Cc: usize = 61;
        pub const LINE_COUNT_PER_AXIS: usize = 9;
        pub const Sc: usize = 12;
        static GEOMETRY: OnceLock<Ge> = OnceLock::new();
        pub fn gm() -> &'static Ge {
            GEOMETRY.get_or_init(Ge::build)
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct Cg {
            pub index: Ci,
            pub axial: Ax,
            pub coord: Coord,
            pub ns: [Option<Ci>; 6],
            pub _cw: u64,
            pub line_ids: [usize; 3],
            pub ed: u8,
        }
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct Line {
            pub ax: Li,
            pub _u: i8,
            pub cs: Vec<Ci>,
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub struct Sy {
            pub rotations: u8,
            pub mirrored: bool,
        }
        impl Sy {
            pub const fn all() -> [Self; Sc] {
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
        pub struct Ge {
            cs: [Cg; Cc],
            lines: [Vec<Line>; 3],
            symmetries: [[Ci; Cc]; Sc],
            coord_lookup: BTreeMap<Coord, Ci>,
            _v: BTreeMap<Ax, Ci>,
        }
        impl Ge {
            fn build() -> Self {
                let mut cs = Vec::with_capacity(Cc);
                let mut coord_lookup = BTreeMap::new();
                let mut _v = BTreeMap::new();
                for row in 0..=8u8 {
                    let r = row as i8 - Br;
                    let q_min = (-Br).max(-r - Br);
                    let q_max = Br.min(-r + Br);
                    for (offset, q) in (q_min..=q_max).enumerate() {
                        let coord = Coord::new(row, offset as u8 + 1).unwrap();
                        let axial = Ax::new(q, r);
                        let ed = Self::ed(axial);
                        let line_ids = [
                            Self::line_id(axial.q),
                            Self::line_id(axial.r),
                            Self::line_id(axial.s()),
                        ];
                        let index = Ci::new(cs.len() as u8).unwrap();
                        let cell = Cg {
                            index,
                            axial,
                            coord,
                            ns: [None; 6],
                            _cw: 0,
                            line_ids,
                            ed,
                        };
                        cs.push(cell);
                        coord_lookup.insert(coord, index);
                        _v.insert(axial, index);
                    }
                }
                let mut cs: [Cg; Cc] = cs.try_into().unwrap();
                for cell in &mut cs {
                    cell.ns = array::from_fn(|direction_index| {
                        let direction = Ad[direction_index];
                        let delta = direction.delta();
                        let next = Ax::new(cell.axial.q + delta.q, cell.axial.r + delta.r);
                        _v.get(&next).copied()
                    });
                    cell._cw = cell
                        .ns
                        .iter()
                        .flatten()
                        .fold(0u64, |mask, nb| mask | (1u64 << nb.as_u8()));
                }
                let lines = Self::build_lines(&cs);
                let symmetries = Self::build_symmetries(&cs, &_v);
                Self {
                    cs,
                    lines,
                    symmetries,
                    coord_lookup,
                    _v,
                }
            }
            fn build_lines(cs: &[Cg; Cc]) -> [Vec<Line>; 3] {
                array::from_fn(|ai| {
                    (-Br..=Br)
                        .map(|_u| {
                            let ax = match ai {
                                0 => Li::Q,
                                1 => Li::R,
                                _ => Li::S,
                            };
                            let mut line_cells: Vec<Ci> = cs
                                .iter()
                                .filter(|cell| match ax {
                                    Li::Q => cell.axial.q == _u,
                                    Li::R => cell.axial.r == _u,
                                    Li::S => cell.axial.s() == _u,
                                })
                                .map(|cell| cell.index)
                                .collect();
                            line_cells.sort_by_key(|index| {
                                let axial = cs[index.as_usize()].axial;
                                match ax {
                                    Li::Q => axial.r,
                                    Li::R => axial.q,
                                    Li::S => axial.q,
                                }
                            });
                            Line {
                                ax,
                                _u,
                                cs: line_cells,
                            }
                        })
                        .collect()
                })
            }
            fn build_symmetries(cs: &[Cg; Cc], _v: &BTreeMap<Ax, Ci>) -> [[Ci; Cc]; Sc] {
                let all = Sy::all();
                array::from_fn(|symmetry_index| {
                    let symmetry = all[symmetry_index];
                    array::from_fn(|cell_index| {
                        let axial = cs[cell_index].axial;
                        let tf = Self::apply_symmetry(axial, symmetry);
                        _v.get(&tf).copied().unwrap()
                    })
                })
            }
            fn apply_symmetry(axial: Ax, symmetry: Sy) -> Ax {
                let mut cube = (axial.q, -axial.q - axial.r, axial.r);
                if symmetry.mirrored {
                    cube = (cube.0, cube.2, cube.1);
                }
                for _ in 0..symmetry.rotations {
                    cube = (-cube.2, -cube.0, -cube.1);
                }
                Ax::new(cube.0, cube.2)
            }
            fn ed(axial: Ax) -> u8 {
                let shell = axial.q.abs().max(axial.r.abs()).max(axial.s().abs());
                (Br - shell) as u8
            }
            fn line_id(_u: i8) -> usize {
                (_u + Br) as usize
            }
            pub fn cs(&self) -> &[Cg; Cc] {
                &self.cs
            }
            pub fn cell(&self, index: Ci) -> &Cg {
                &self.cs[index.as_usize()]
            }
            pub fn lines(&self, ax: Li) -> &[Line] {
                &self.lines[ax.index()]
            }
            pub fn line(&self, ax: Li, id: usize) -> &Line {
                &self.lines[ax.index()][id]
            }
            pub fn index_of_coord(&self, coord: Coord) -> Option<Ci> {
                self.coord_lookup.get(&coord).copied()
            }
            pub fn index_of_axial(&self, axial: Ax) -> Option<Ci> {
                self._v.get(&axial).copied()
            }
            pub fn symmetry_map(&self, symmetry: Sy) -> &[Ci; Cc] {
                &self.symmetries[Self::symmetry_slot(symmetry)]
            }
            pub fn transform(&self, index: Ci, symmetry: Sy) -> Ci {
                self.symmetry_map(symmetry)[index.as_usize()]
            }
            pub fn transform_direction(&self, direction: Di, symmetry: Sy) -> Di {
                let tf = Self::apply_symmetry(direction.delta(), symmetry);
                Di::from_delta(tf).unwrap()
            }
            pub fn inverse_symmetry(&self, symmetry: Sy) -> Sy {
                let slot = Self::symmetry_slot(symmetry);
                let map = &self.symmetries[slot];
                Sy::all()
                    .into_iter()
                    .find(|_n| {
                        self.symmetry_map(*_n).iter().enumerate().all(
                            |(index, candidate_target)| {
                                map[candidate_target.as_usize()].as_usize() == index
                            },
                        )
                    })
                    .unwrap()
            }
            fn symmetry_slot(symmetry: Sy) -> usize {
                symmetry.rotations as usize + if symmetry.mirrored { 6 } else { 0 }
            }
        }
    }
    mod mv {
        use super::coord::{Ci, Coord};
        use super::direction::Di;
        use super::gm::{gm, Sy};
        use core::cmp::Ordering;
        use core::fmt;
        use core::hash::{Hash, Hasher};
        use std::str::FromStr;
        #[derive(Clone, Copy, Debug)]
        pub struct Mv {
            _ad: [Ci; 3],
            len: u8,
            direction: Di,
        }
        impl Mv {
            pub const PLACEHOLDER: Self = Self {
                _ad: [Ci::new_unchecked(0); 3],
                len: 1,
                direction: Di::East,
            };
            pub fn new(mut _ad: Vec<Ci>, direction: Di) -> Result<Self, Me> {
                if _ad.is_empty() {
                    return Err(Me::EmptySourceGroup);
                }
                if _ad.len() > 3 {
                    return Err(Me::TooManySourceCells(_ad.len()));
                }
                _ad.sort_unstable();
                for pair in _ad.windows(2) {
                    if pair[0] == pair[1] {
                        return Err(Me::DuplicateSourceCell(pair[0].coord()));
                    }
                }
                if _ad.len() > 1 {
                    validate_contiguous_group(&_ad)?;
                }
                let len = _ad.len() as u8;
                let mut cp = [_ad[0]; 3];
                for (index, cell) in _ad.into_iter().enumerate() {
                    cp[index] = cell;
                }
                Ok(Self {
                    _ad: cp,
                    len,
                    direction,
                })
            }
            pub fn from_cells(_ad: &[Ci], direction: Di) -> Result<Self, Me> {
                if _ad.is_empty() {
                    return Err(Me::EmptySourceGroup);
                }
                if _ad.len() > 3 {
                    return Err(Me::TooManySourceCells(_ad.len()));
                }
                let len = _ad.len() as u8;
                let mut cp = [_ad[0]; 3];
                for (index, cell) in _ad.iter().copied().enumerate() {
                    cp[index] = cell;
                }
                cp[.._ad.len()].sort_unstable();
                for pair in cp[.._ad.len()].windows(2) {
                    if pair[0] == pair[1] {
                        return Err(Me::DuplicateSourceCell(pair[0].coord()));
                    }
                }
                if _ad.len() > 1 {
                    validate_contiguous_group(&cp[.._ad.len()])?;
                }
                Ok(Self {
                    _ad: cp,
                    len,
                    direction,
                })
            }
            pub fn _Y(_ad: &[Ci], direction: Di) -> Self {
                let len = _ad.len() as u8;
                let mut cp = [_ad[0]; 3];
                match _ad {
                    [first] => {
                        cp[0] = *first;
                    }
                    [first, second] => {
                        if first <= second {
                            cp[0] = *first;
                            cp[1] = *second;
                        } else {
                            cp[0] = *second;
                            cp[1] = *first;
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
                        cp[0] = a;
                        cp[1] = b;
                        cp[2] = c;
                    }
                    _ => unreachable!("move source groups must contain 1..=3 cells"),
                }
                Self {
                    _ad: cp,
                    len,
                    direction,
                }
            }
            pub fn _ad(&self) -> &[Ci] {
                &self._ad[..self.len as usize]
            }
            pub fn direction(&self) -> Di {
                self.direction
            }
            pub fn len(&self) -> usize {
                self.len as usize
            }
            pub fn is_empty(&self) -> bool {
                false
            }
            pub fn transform(&self, symmetry: Sy) -> Self {
                let gm = gm();
                let mut tf = [self._ad[0]; 3];
                for (index, cell) in self._ad().iter().copied().enumerate() {
                    tf[index] = gm.transform(cell, symmetry);
                }
                Self::from_cells(
                    &tf[..self.len as usize],
                    gm.transform_direction(self.direction, symmetry),
                )
                .unwrap()
            }
        }
        impl PartialEq for Mv {
            fn eq(&self, other: &Self) -> bool {
                self.direction == other.direction && self._ad() == other._ad()
            }
        }
        impl Eq for Mv {}
        impl Hash for Mv {
            fn hash<H: Hasher>(&self, st: &mut H) {
                self._ad().hash(st);
                self.direction.hash(st);
            }
        }
        impl PartialOrd for Mv {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Mv {
            fn cmp(&self, other: &Self) -> Ordering {
                self._ad()
                    .cmp(other._ad())
                    .then_with(|| self.direction.cmp(&other.direction))
            }
        }
        impl fmt::Display for Mv {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for (index, cell) in self._ad().iter().enumerate() {
                    if index > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{cell}")?;
                }
                write!(f, ">{}", self.direction)
            }
        }
        impl FromStr for Mv {
            type Err = Me;
            fn from_str(value: &str) -> Result<Self, Self::Err> {
                let (cs, direction) = value
                    .trim()
                    .split_once('>')
                    .ok_or_else(|| Me::InvalidSyntax(value.to_owned()))?;
                let direction = Di::parse(direction)
                    .ok_or_else(|| Me::InvalidDirection(direction.to_owned()))?;
                let mut _ad = Vec::new();
                for token in cs.split(',') {
                    let coord =
                        Coord::parse(token).ok_or_else(|| Me::InvalidCoord(token.to_owned()))?;
                    let index = gm().index_of_coord(coord).unwrap();
                    _ad.push(index);
                }
                Self::new(_ad, direction)
            }
        }
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub enum Me {
            EmptySourceGroup,
            TooManySourceCells(usize),
            DuplicateSourceCell(Coord),
            NonLinearSourceCells,
            NonContiguousSourceCells,
            InvalidCoord(String),
            InvalidDirection(String),
            InvalidSyntax(String),
        }
        impl fmt::Display for Me {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("m")
            }
        }
        fn validate_contiguous_group(_ad: &[Ci]) -> Result<(), Me> {
            let gm = gm();
            let first = gm.cell(_ad[0]);
            let shared_axis = [first.line_ids[0], first.line_ids[1], first.line_ids[2]]
                .into_iter()
                .enumerate()
                .find_map(|(ai, line_id)| {
                    _ad.iter()
                        .all(|cell| gm.cell(*cell).line_ids[ai] == line_id)
                        .then_some((ai, line_id))
                })
                .ok_or(Me::NonLinearSourceCells)?;
            let line = &gm.lines(match shared_axis.0 {
                0 => super::direction::Li::Q,
                1 => super::direction::Li::R,
                _ => super::direction::Li::S,
            })[shared_axis.1];
            let mut positions = _ad
                .iter()
                .map(|cell| line.cs.iter().position(|_n| _n == cell).unwrap())
                .collect::<Vec<_>>();
            positions.sort_unstable();
            for pair in positions.windows(2) {
                if pair[1] != pair[0] + 1 {
                    return Err(Me::NonContiguousSourceCells);
                }
            }
            Ok(())
        }
    }
    mod position {
        use super::_c::Co;
        use super::coord::{Ci, Coord};
        use super::gm::{gm, Sy};
        use core::fmt;
        use std::fmt::Write as _;
        use std::str::FromStr;
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct Po {
            _sT: Co,
            black: Vec<Ci>,
            white: Vec<Ci>,
        }
        impl Po {
            pub const MAX_PIECES_PER_SIDE: usize = 14;
            pub fn new(_sT: Co, black: Vec<Ci>, white: Vec<Ci>) -> Result<Self, Pe> {
                let mut record = Self { _sT, black, white };
                record.normalize();
                record.validate()?;
                Ok(record)
            }
            pub fn _sT(&self) -> Co {
                self._sT
            }
            pub fn side_to_move(&self) -> Co {
                self._sT()
            }
            pub fn black(&self) -> &[Ci] {
                &self.black
            }
            pub fn white(&self) -> &[Ci] {
                &self.white
            }
            pub(crate) fn cf(&mut self, _c: Co) -> &mut Vec<Ci> {
                match _c {
                    Co::Black => &mut self.black,
                    Co::White => &mut self.white,
                }
            }
            pub(crate) fn _cG(&mut self, _sT: Co) {
                self._sT = _sT;
            }
            pub fn contains(&self, _c: Co, cell: Ci) -> bool {
                self.cells_for(_c).binary_search(&cell).is_ok()
            }
            pub fn occupant(&self, cell: Ci) -> Option<Co> {
                if self.black.binary_search(&cell).is_ok() {
                    Some(Co::Black)
                } else if self.white.binary_search(&cell).is_ok() {
                    Some(Co::White)
                } else {
                    None
                }
            }
            pub fn mc(&self, _c: Co) -> usize {
                self.cells_for(_c).len()
            }
            pub fn transform(&self, symmetry: Sy) -> Self {
                let gm = gm();
                let black = self
                    .black
                    .iter()
                    .copied()
                    .map(|cell| gm.transform(cell, symmetry))
                    .collect();
                let white = self
                    .white
                    .iter()
                    .copied()
                    .map(|cell| gm.transform(cell, symmetry))
                    .collect();
                Self::new(self._sT, black, white).unwrap()
            }
            pub fn validate(&self) -> Result<(), Pe> {
                Self::validate_color_list(Co::Black, &self.black)?;
                Self::validate_color_list(Co::White, &self.white)?;
                for black in &self.black {
                    if self.white.binary_search(black).is_ok() {
                        return Err(Pe::CellOverlap(black.coord()));
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
            fn cells_for(&self, _c: Co) -> &[Ci] {
                match _c {
                    Co::Black => &self.black,
                    Co::White => &self.white,
                }
            }
            fn validate_color_list(_c: Co, cs: &[Ci]) -> Result<(), Pe> {
                if cs.len() > Self::MAX_PIECES_PER_SIDE {
                    return Err(Pe::TooManyPieces {
                        _c,
                        count: cs.len(),
                    });
                }
                for pair in cs.windows(2) {
                    if pair[0] >= pair[1] {
                        return Err(Pe::NonCanonicalCellOrder(_c));
                    }
                }
                Ok(())
            }
            fn parse_cell_list(raw: &str) -> Result<Vec<Ci>, Pe> {
                if raw == "-" {
                    return Ok(Vec::new());
                }
                let mut cs = Vec::new();
                for token in raw.split(',') {
                    let coord =
                        Coord::parse(token).ok_or_else(|| Pe::InvalidCoord(token.to_owned()))?;
                    let index = gm().index_of_coord(coord).unwrap();
                    cs.push(index);
                }
                Ok(cs)
            }
        }
        impl fmt::Display for Po {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fn write_cells(buffer: &mut String, cs: &[Ci]) {
                    if cs.is_empty() {
                        buffer.push('-');
                        return;
                    }
                    for (idx, cell) in cs.iter().enumerate() {
                        if idx > 0 {
                            buffer.push(',');
                        }
                        let _ = write!(buffer, "{cell}");
                    }
                }
                let mut black = String::new();
                write_cells(&mut black, &self.black);
                let mut white = String::new();
                write_cells(&mut white, &self.white);
                write!(
                    f,
                    "aba-v1;stm={};black={};white={}",
                    self._sT.code(),
                    black,
                    white
                )
            }
        }
        impl FromStr for Po {
            type Err = Pe;
            fn from_str(value: &str) -> Result<Self, Self::Err> {
                let parts: Vec<&str> = value.trim().split(';').collect();
                if parts.len() != 4 || parts[0] != "aba-v1" {
                    return Err(Pe::InvalidFormat);
                }
                let _sT = parts[1]
                    .strip_prefix("stm=")
                    .and_then(Co::parse)
                    .ok_or(Pe::InvalidSideToMove)?;
                let black = parts[2]
                    .strip_prefix("black=")
                    .ok_or(Pe::MissingField("black"))?;
                let white = parts[3]
                    .strip_prefix("white=")
                    .ok_or(Pe::MissingField("white"))?;
                Self::new(
                    _sT,
                    Self::parse_cell_list(black)?,
                    Self::parse_cell_list(white)?,
                )
            }
        }
        pub trait EngineStateView {
            fn position(&self) -> &Po;
            fn _sT(&self) -> Co {
                self.position()._sT()
            }
            fn validate(&self) -> Result<(), Pe> {
                self.position().validate()
            }
        }
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub enum Pe {
            InvalidFormat,
            MissingField(&'static str),
            InvalidSideToMove,
            InvalidCoord(String),
            TooManyPieces { _c: Co, count: usize },
            CellOverlap(Coord),
            NonCanonicalCellOrder(Co),
        }
        impl fmt::Display for Pe {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("p")
            }
        }
    }
    pub use _c::Co;
    pub use coord::{Ax, Ci, Coord};
    pub use direction::{Ad, Di, Li};
    pub use gm::{gm, Br, Cc, Ge, Sc, Sy, LINE_COUNT_PER_AXIS};
    pub use mv::{Me, Mv};
    pub use position::{EngineStateView, Pe, Po};
}
pub mod ar {
    mod st {
        use crate::ac::{gm, Ad, Co, Di, EngineStateView, Li, Me, Mv, Pe, Po};
        use std::collections::BTreeSet;
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct UndoSnapshot {
            plan: Mq,
            pt: Co,
        }
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct Rq {
            position: Po,
            _ae: [Option<Co>; crate::ac::Cc],
            _t: u64,
            _s: u64,
            bi: [u8; crate::ac::Cc],
            wi: [u8; crate::ac::Cc],
        }
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub enum Re {
            Im(Mv),
            InvalidMoveShape(Me),
            InvalidPosition(Pe),
        }
        impl Rq {
            pub fn new(position: Po) -> Result<Self, Pe> {
                position.validate()?;
                let _ae = _ae(&position);
                let _t = position
                    .black()
                    .iter()
                    .fold(0u64, |acc, cell| acc | (1u64 << cell.as_u8()));
                let _s = position
                    .white()
                    .iter()
                    .fold(0u64, |acc, cell| acc | (1u64 << cell.as_u8()));
                let (bi, wi) = slot_maps(&position);
                Ok(Self {
                    position,
                    _ae,
                    _t,
                    _s,
                    bi,
                    wi,
                })
            }
            pub fn position(&self) -> &Po {
                &self.position
            }
            pub fn occupant_fast(&self, cell: crate::ac::Ci) -> Option<Co> {
                self._ae[cell.as_usize()]
            }
            pub fn _t(&self) -> u64 {
                self._t
            }
            pub fn _s(&self) -> u64 {
                self._s
            }
            pub fn _bW(&mut self) -> Co {
                let _ck = self.position._sT();
                self.position._cG(_ck.other());
                _ck
            }
            pub fn _bX(&mut self, _ck: Co) {
                self.position._cG(_ck);
            }
            pub fn generate_legal_moves(&self) -> Vec<Mv> {
                let mut ms = BTreeSet::new();
                for group in enumerate_groups(&self.position, self.position._sT()) {
                    for direction in Ad {
                        let _n = Mv::new(group.clone(), direction).unwrap();
                        if self.analyze_move(&_n).is_ok() {
                            ms.insert(_n);
                        }
                    }
                }
                ms.into_iter().collect()
            }
            pub fn gl(&self, ms: &mut Vec<super::super::_G>) {
                let side_bits = self.v3_side_bits(self.position._sT());
                let enemy_bits = self.v3_side_bits(self.position._sT().other());
                let occupied_bits = side_bits | enemy_bits;
                let tables = super::super::v3_movegen_tables();
                ms.clear();
                for group in &tables.source_groups {
                    if side_bits & group.source_mask != group.source_mask {
                        continue;
                    }
                    for dir in group.dirs.iter().flatten() {
                        let Some(ej) = self.v3_group_dir_legality(
                            group.len as usize,
                            dir,
                            side_bits,
                            enemy_bits,
                            occupied_bits,
                        ) else {
                            continue;
                        };
                        ms.push(super::super::_G {
                            mv: dir.mv,
                            ej,
                            _d: dir.history_key,
                        });
                    }
                }
            }
            fn v3_side_bits(&self, side: Co) -> u64 {
                match side {
                    Co::Black => self._t,
                    Co::White => self._s,
                }
            }
            fn v3_group_dir_legality(
                &self,
                len: usize,
                dir: &super::super::V3GroupDir,
                side_bits: u64,
                enemy_bits: u64,
                occupied_bits: u64,
            ) -> Option<bool> {
                if dir.inline {
                    self.v3_inline_legality(len, dir, side_bits, enemy_bits, occupied_bits)
                } else {
                    self.v3_broadside_legality(dir.translated_mask, occupied_bits)
                }
            }
            fn v3_broadside_legality(
                &self,
                translated_mask: u64,
                occupied_bits: u64,
            ) -> Option<bool> {
                if occupied_bits & translated_mask == 0 {
                    Some(false)
                } else {
                    None
                }
            }
            fn v3_inline_legality(
                &self,
                len: usize,
                dir: &super::super::V3GroupDir,
                side_bits: u64,
                enemy_bits: u64,
                occupied_bits: u64,
            ) -> Option<bool> {
                let first_bit = dir.ray_bits[0];
                if occupied_bits & first_bit == 0 {
                    return Some(false);
                }
                if side_bits & first_bit != 0 {
                    return None;
                }
                let second_enemy = dir.ray_bits[1] != 0 && (enemy_bits & dir.ray_bits[1] != 0);
                let third_enemy =
                    second_enemy && dir.ray_bits[2] != 0 && (enemy_bits & dir.ray_bits[2] != 0);
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
                match dir.landing[enemy_count - 1] {
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
            pub fn apply_move(&mut self, mv: &Mv) -> Result<UndoSnapshot, Re> {
                let plan = self.analyze_move(mv)?;
                let pt = self.position._sT();
                apply_plan_in_place(
                    &mut self.position,
                    &mut self._ae,
                    &mut self._t,
                    &mut self._s,
                    &mut self.bi,
                    &mut self.wi,
                    &plan,
                )?;
                Ok(UndoSnapshot { plan, pt })
            }
            pub fn _bY(&self, mv: &Mv) -> Option<super::super::_G> {
                let plan = self.analyze_move(mv).ok()?;
                Some(super::super::_G {
                    mv: *mv,
                    ej: plan._cl.len > 0 && plan._cm.iter().any(|dst| dst.is_none()),
                    _d: super::super::history_group_key(mv._ad(), mv.direction()),
                })
            }
            pub fn undo_move(&mut self, undo: UndoSnapshot) {
                undo_plan_in_place(
                    &mut self.position,
                    &mut self._ae,
                    &mut self._t,
                    &mut self._s,
                    &mut self.bi,
                    &mut self.wi,
                    &undo.plan,
                    undo.pt,
                );
            }
            fn analyze_move(&self, mv: &Mv) -> Result<Mq, Re> {
                let side = self.position._sT();
                let _e = side.other();
                let _ae = &self._ae;
                for cell in mv._ad() {
                    if _ae[cell.as_usize()] != Some(side) {
                        return Err(Re::Im(mv.clone()));
                    }
                }
                if mv.len() == 1 {
                    return analyze_single_move(mv, &_ae, side);
                }
                let ax = group_axis(mv._ad()).ok_or_else(|| Re::Im(mv.clone()))?;
                if is_inline(ax, mv.direction()) {
                    analyze_inline_move(mv, &_ae, side, _e)
                } else {
                    analyze_broadside_move(mv, &_ae)
                }
            }
        }
        impl EngineStateView for Rq {
            fn position(&self) -> &Po {
                &self.position
            }
        }
        impl From<Me> for Re {
            fn from(value: Me) -> Self {
                Self::InvalidMoveShape(value)
            }
        }
        impl From<Pe> for Re {
            fn from(value: Pe) -> Self {
                Self::InvalidPosition(value)
            }
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        struct _bv<const N: usize> {
            len: u8,
            data: [crate::ac::Ci; N],
        }
        impl<const N: usize> _bv<N> {
            fn new() -> Self {
                Self {
                    len: 0,
                    data: [crate::ac::Ci::new_unchecked(0); N],
                }
            }
            fn from_slice(cells: &[crate::ac::Ci]) -> Self {
                let mut out = Self::new();
                for cell in cells.iter().copied() {
                    out.push(cell);
                }
                out
            }
            fn push(&mut self, cell: crate::ac::Ci) {
                let idx = self.len as usize;
                debug_assert!(idx < N);
                self.data[idx] = cell;
                self.len += 1;
            }
            fn as_slice(&self) -> &[crate::ac::Ci] {
                &self.data[..self.len as usize]
            }
            fn iter(&self) -> std::slice::Iter<'_, crate::ac::Ci> {
                self.as_slice().iter()
            }
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        struct _bw<const N: usize> {
            len: u8,
            data: [Option<crate::ac::Ci>; N],
        }
        impl<const N: usize> _bw<N> {
            fn new() -> Self {
                Self {
                    len: 0,
                    data: [None; N],
                }
            }
            fn push(&mut self, cell: Option<crate::ac::Ci>) {
                let idx = self.len as usize;
                debug_assert!(idx < N);
                self.data[idx] = cell;
                self.len += 1;
            }
            fn iter(&self) -> std::slice::Iter<'_, Option<crate::ac::Ci>> {
                self.data[..self.len as usize].iter()
            }
        }
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        struct Mq {
            ff: _bv<3>,
            ft: _bv<3>,
            _cl: _bv<2>,
            _cm: _bw<2>,
        }
        fn enumerate_groups(position: &Po, side: Co) -> Vec<Vec<crate::ac::Ci>> {
            let gm = gm();
            let _ag = match side {
                Co::Black => position.black(),
                Co::White => position.white(),
            };
            let mut groups: Vec<Vec<crate::ac::Ci>> =
                _ag.iter().copied().map(|cell| vec![cell]).collect();
            for ax in [Li::Q, Li::R, Li::S] {
                for line in gm.lines(ax) {
                    let mut run = Vec::new();
                    for cell in &line.cs {
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
        fn emit_groups_from_run(run: &[crate::ac::Ci], groups: &mut Vec<Vec<crate::ac::Ci>>) {
            for _al in 2..=3 {
                if run.len() < _al {
                    continue;
                }
                for start in 0..=run.len() - _al {
                    groups.push(run[start..start + _al].to_vec());
                }
            }
        }
        fn analyze_single_move(
            mv: &Mv,
            _ae: &[Option<Co>; crate::ac::Cc],
            side: Co,
        ) -> Result<Mq, Re> {
            let source = mv._ad()[0];
            let _af = nb(source, mv.direction()).ok_or_else(|| Re::Im(mv.clone()))?;
            if _ae[_af.as_usize()].is_some() {
                return Err(Re::Im(mv.clone()));
            }
            let _ = side;
            Ok(Mq {
                ff: _bv::from_slice(&[source]),
                ft: _bv::from_slice(&[_af]),
                _cl: _bv::new(),
                _cm: _bw::new(),
            })
        }
        fn analyze_broadside_move(mv: &Mv, _ae: &[Option<Co>; crate::ac::Cc]) -> Result<Mq, Re> {
            let mut destinations = _bv::new();
            for source in mv._ad() {
                let _af = nb(*source, mv.direction()).ok_or_else(|| Re::Im(mv.clone()))?;
                if _ae[_af.as_usize()].is_some() {
                    return Err(Re::Im(mv.clone()));
                }
                destinations.push(_af);
            }
            Ok(Mq {
                ff: _bv::from_slice(mv._ad()),
                ft: destinations,
                _cl: _bv::new(),
                _cm: _bw::new(),
            })
        }
        fn analyze_inline_move(
            mv: &Mv,
            _ae: &[Option<Co>; crate::ac::Cc],
            side: Co,
            _e: Co,
        ) -> Result<Mq, Re> {
            let front = front_cell(mv._ad(), mv.direction()).ok_or_else(|| Re::Im(mv.clone()))?;
            let _ah = nb(front, mv.direction()).ok_or_else(|| Re::Im(mv.clone()))?;
            match _ae[_ah.as_usize()] {
                None => Ok(Mq {
                    ff: _bv::from_slice(mv._ad()),
                    ft: translated_cells(mv._ad(), mv.direction())?,
                    _cl: _bv::new(),
                    _cm: _bw::new(),
                }),
                Some(_c) if _c == side => Err(Re::Im(mv.clone())),
                Some(_c) if _c == _e => {
                    let mut _ci = [_ah; 2];
                    let mut enemy_count = 0usize;
                    let mut cursor = Some(_ah);
                    while let Some(cell) = cursor {
                        if _ae[cell.as_usize()] != Some(_e) {
                            break;
                        }
                        if enemy_count < _ci.len() {
                            _ci[enemy_count] = cell;
                        }
                        enemy_count += 1;
                        cursor = nb(cell, mv.direction());
                    }
                    if enemy_count >= mv.len() {
                        return Err(Re::Im(mv.clone()));
                    }
                    let mut enemy_chain = _bv::new();
                    let mut enemy_destinations = _bw::new();
                    for index in 0..enemy_count {
                        let cell = _ci[index];
                        let _af = nb(cell, mv.direction());
                        if index + 1 == enemy_count {
                            if let Some(next) = _af {
                                if _ae[next.as_usize()].is_some() {
                                    return Err(Re::Im(mv.clone()));
                                }
                            }
                        }
                        enemy_chain.push(cell);
                        enemy_destinations.push(_af);
                    }
                    Ok(Mq {
                        ff: _bv::from_slice(mv._ad()),
                        ft: translated_cells(mv._ad(), mv.direction())?,
                        _cl: enemy_chain,
                        _cm: enemy_destinations,
                    })
                }
                Some(_) => Err(Re::Im(mv.clone())),
            }
        }
        fn translated_cells(_ad: &[crate::ac::Ci], direction: Di) -> Result<_bv<3>, Re> {
            let mut out = _bv::new();
            for cell in _ad.iter().copied() {
                out.push(
                    nb(cell, direction)
                        .ok_or_else(|| Re::Im(Mv::from_cells(_ad, direction).unwrap()))?,
                );
            }
            Ok(out)
        }
        fn _bS(
            cs: &mut Vec<crate::ac::Ci>,
            idx: &mut [u8; crate::ac::Cc],
            cell: crate::ac::Ci,
            _ae: &mut [Option<Co>; crate::ac::Cc],
            _t: &mut u64,
            _s: &mut u64,
            _c: Co,
        ) -> Result<(), Re> {
            let slot = idx[cell.as_usize()];
            if slot == u8::MAX {
                return Err(Re::Im(Mv::PLACEHOLDER));
            }
            let index = slot as usize;
            let last = *cs.last().ok_or(Re::Im(Mv::PLACEHOLDER))?;
            cs.swap_remove(index);
            idx[cell.as_usize()] = u8::MAX;
            if index < cs.len() {
                idx[last.as_usize()] = index as u8;
            }
            _ae[cell.as_usize()] = None;
            let bit = 1u64 << cell.as_u8();
            match _c {
                Co::Black => *_t &= !bit,
                Co::White => *_s &= !bit,
            }
            Ok(())
        }
        fn _bT(
            cs: &mut Vec<crate::ac::Ci>,
            idx: &mut [u8; crate::ac::Cc],
            cell: crate::ac::Ci,
            _ae: &mut [Option<Co>; crate::ac::Cc],
            _t: &mut u64,
            _s: &mut u64,
            _c: Co,
        ) {
            if idx[cell.as_usize()] == u8::MAX {
                idx[cell.as_usize()] = cs.len() as u8;
                cs.push(cell);
            }
            _ae[cell.as_usize()] = Some(_c);
            let bit = 1u64 << cell.as_u8();
            match _c {
                Co::Black => *_t |= bit,
                Co::White => *_s |= bit,
            }
        }
        fn apply_plan_in_place(
            position: &mut Po,
            _ae: &mut [Option<Co>; crate::ac::Cc],
            _t: &mut u64,
            _s: &mut u64,
            bi: &mut [u8; crate::ac::Cc],
            wi: &mut [u8; crate::ac::Cc],
            plan: &Mq,
        ) -> Result<(), Re> {
            let side = position._sT();
            let _e = side.other();
            let (_ag_slots, _ci_slots) = match side {
                Co::Black => (&mut *bi, &mut *wi),
                Co::White => (&mut *wi, &mut *bi),
            };
            {
                let _ag = position.cf(side);
                for cell in plan.ff.iter() {
                    _bS(_ag, _ag_slots, *cell, _ae, _t, _s, side)?;
                }
            }
            {
                let _ci = position.cf(_e);
                for cell in plan._cl.iter() {
                    _bS(_ci, _ci_slots, *cell, _ae, _t, _s, _e)?;
                }
            }
            {
                let _ag = position.cf(side);
                for _af in plan.ft.iter() {
                    _bT(_ag, _ag_slots, *_af, _ae, _t, _s, side);
                }
            }
            {
                let _ci = position.cf(_e);
                for _af in plan._cm.iter() {
                    if let Some(_af) = _af {
                        _bT(_ci, _ci_slots, *_af, _ae, _t, _s, _e);
                    }
                }
            }
            position._cG(_e);
            Ok(())
        }
        fn undo_plan_in_place(
            position: &mut Po,
            _ae: &mut [Option<Co>; crate::ac::Cc],
            _t: &mut u64,
            _s: &mut u64,
            bi: &mut [u8; crate::ac::Cc],
            wi: &mut [u8; crate::ac::Cc],
            plan: &Mq,
            pt: Co,
        ) {
            let side = pt;
            let _e = side.other();
            let (_ag_slots, _ci_slots) = match side {
                Co::Black => (&mut *bi, &mut *wi),
                Co::White => (&mut *wi, &mut *bi),
            };
            {
                let _ag = position.cf(side);
                for _af in plan.ft.iter() {
                    let _ = _bS(_ag, _ag_slots, *_af, _ae, _t, _s, side);
                }
            }
            {
                let _ci = position.cf(_e);
                for _af in plan._cm.iter() {
                    if let Some(_af) = _af {
                        let _ = _bS(_ci, _ci_slots, *_af, _ae, _t, _s, _e);
                    }
                }
            }
            {
                let _ag = position.cf(side);
                for cell in plan.ff.iter() {
                    _bT(_ag, _ag_slots, *cell, _ae, _t, _s, side);
                }
            }
            {
                let _ci = position.cf(_e);
                for cell in plan._cl.iter() {
                    _bT(_ci, _ci_slots, *cell, _ae, _t, _s, _e);
                }
            }
            position._cG(side);
        }
        fn _ae(position: &Po) -> [Option<Co>; crate::ac::Cc] {
            let mut _ae = [None; crate::ac::Cc];
            for cell in position.black() {
                _ae[cell.as_usize()] = Some(Co::Black);
            }
            for cell in position.white() {
                _ae[cell.as_usize()] = Some(Co::White);
            }
            _ae
        }
        fn slot_maps(position: &Po) -> ([u8; crate::ac::Cc], [u8; crate::ac::Cc]) {
            let mut bi = [u8::MAX; crate::ac::Cc];
            let mut wi = [u8::MAX; crate::ac::Cc];
            for (i, cell) in position.black().iter().enumerate() {
                bi[cell.as_usize()] = i as u8;
            }
            for (i, cell) in position.white().iter().enumerate() {
                wi[cell.as_usize()] = i as u8;
            }
            (bi, wi)
        }
        fn group_axis(_ad: &[crate::ac::Ci]) -> Option<Li> {
            let gm = gm();
            let first = gm.cell(_ad[0]);
            [Li::Q, Li::R, Li::S].into_iter().find(|ax| {
                let ai = ax.index();
                let line_id = first.line_ids[ai];
                _ad.iter()
                    .all(|cell| gm.cell(*cell).line_ids[ai] == line_id)
            })
        }
        fn is_inline(ax: Li, direction: Di) -> bool {
            match ax {
                Li::Q => matches!(direction, Di::Se | Di::Nw),
                Li::R => matches!(direction, Di::East | Di::West),
                Li::S => matches!(direction, Di::Ne | Di::Sw),
            }
        }
        fn front_cell(_ad: &[crate::ac::Ci], direction: Di) -> Option<crate::ac::Ci> {
            match _ad {
                [] => None,
                [first] => Some(*first),
                [first, second] => {
                    if nb(*first, direction) == Some(*second) {
                        Some(*second)
                    } else {
                        Some(*first)
                    }
                }
                [first, second, third] => {
                    if nb(*first, direction) == Some(*second)
                        && nb(*second, direction) == Some(*third)
                    {
                        Some(*third)
                    } else {
                        Some(*first)
                    }
                }
                _ => None,
            }
        }
        fn nb(cell: crate::ac::Ci, direction: Di) -> Option<crate::ac::Ci> {
            gm().cell(cell).ns[direction.index()]
        }
    }
    pub use st::{Re, Rq, UndoSnapshot};
}
use crate::ac::{gm, Ad, Co, Coord, Di, Mv, Po};
use crate::ar::Rq;
pub use ac::*;
pub use ar::*;
use std::cmp::Reverse;
use std::io::{self, BufRead, Write};
use std::sync::OnceLock;
use web_time::{Duration, Instant};
pub const MAX_GAME_TURNS: u16 = 350;
const _D: i32 = 100000;
const _C: i32 = _D + 10000;
const TT_SIZE: usize = 1 << 16;
const TT_BUCKET_SIZE: usize = 4;
const EVAL_CACHE_SIZE: usize = 1 << 17;
const MAX_PLY: usize = 96;
const CODINGAME_FIRST_TURN_MS: u64 = 500;
const CODINGAME_TURN_MS: u64 = 45;
const SEARCH_DEADLINE_SLACK_MS: u64 = 4;
const ABORT_POLL_MASK: u64 = 8191;
const RAW_ABORT_POLL_MASK: u64 = 2047;
const ROOT_REVERSE_MOVE_PENALTY: i32 = 200;
const EMERGENCY_EJECTION_BONUS: i32 = 96;
const Aw: i32 = 80;
const _aM: u8 = 4;
const _aN: usize = 4;
const _aO: u8 = 4;
const _aP: usize = 10;
const NULL_MOVE_REDUCTION: u8 = 2;
const NULL_MOVE_MIN_DEPTH: u8 = 5;
const FUTILITY_MARGIN_DEPTH1: i32 = 180;
const FUTILITY_MARGIN_DEPTH2: i32 = 420;
const _at: bool = true;
const EVAL_CACHE_WAYS: usize = 4;
const COUNTERMOVE_TABLE_BITS: usize = 14;
const Cs: usize = 1 << COUNTERMOVE_TABLE_BITS;
const CORRECTION_HISTORY_BITS: usize = 14;
const Cu: usize = 1 << CORRECTION_HISTORY_BITS;
const COUNTERMOVE_ORDER_BONUS: i32 = 1750000;
const _aI: i32 = 900000;
const EJECTION_ORDER_BONUS: i32 = 1250000;
const _ca: i32 = 256;
const HISTORY_LMR_THRESHOLD: i32 = 512;
const CORRECTION_HISTORY_CLAMP: i32 = 192;
const _cb: i32 = 96;
const HISTORY_SOURCE_GROUPS_LEN1: usize = crate::ac::Cc;
const HISTORY_SOURCE_GROUPS_LEN2: usize = _y(crate::ac::Cc, 2);
const HISTORY_SOURCE_GROUPS_LEN3: usize = _y(crate::ac::Cc, 3);
const HISTORY_SOURCE_GROUP_COUNT: usize =
    HISTORY_SOURCE_GROUPS_LEN1 + HISTORY_SOURCE_GROUPS_LEN2 + HISTORY_SOURCE_GROUPS_LEN3;
const Hs: usize = HISTORY_SOURCE_GROUP_COUNT * 6;
const EVAL_CACHE_SEED: u64 = 0xA5A55A5A1F2E3D4C;
const SEARCH_CONTEXT_KEY_SEED: u64 = 0xC6A4A7935BD1E995;
const NO_PROGRESS_KEY_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sr {
    pub bm: Option<Mv>,
    pub score: i32,
    pub dp: u8,
    pub nodes: u64,
}
#[derive(Clone, Copy, Debug, Default)]
struct Diag;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SearchCfg {
    use_fast_movegen: bool,
    use_tt_backfill: bool,
    partial_sort_k: usize,
}
fn search_cfg() -> &'static SearchCfg {
    static CFG: SearchCfg = SearchCfg {
        use_fast_movegen: true,
        use_tt_backfill: false,
        partial_sort_k: 6,
    };
    &CFG
}
#[derive(Clone, Debug, Default)]
struct SearchHistory {
    no_progress: Vec<u16>,
}
impl SearchHistory {
    fn reset(&mut self, _hy: &[_X], _root: _X, _b: u16) {
        let _ = (_hy, _root);
        self.no_progress.clear();
        self.no_progress.push(_b);
    }
    fn push(&mut self, _e: _X, _an: bool) {
        let _ = _e;
        let prev = self.current_no_progress();
        self.no_progress
            .push(if _an { 0 } else { prev.saturating_add(1) });
    }
    fn pop(&mut self) {
        if self.no_progress.len() > 1 {
            self.no_progress.pop();
        }
    }
    fn current_no_progress(&self) -> u16 {
        self.no_progress.last().copied().unwrap_or(0)
    }
    fn search_key(&self, _e: _X, gt: u16) -> u64 {
        let position_key = _K(_e);
        let no_progress_key = u64::from(self.current_no_progress()).wrapping_mul(NO_PROGRESS_KEY_SEED);
        splitmix64(position_key ^ u64::from(gt).wrapping_mul(SEARCH_CONTEXT_KEY_SEED) ^ no_progress_key)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvalCacheEntry {
    key: u64,
    score: i32,
}
#[derive(Clone, Debug)]
struct EvalCache {
    e: Vec<Option<EvalCacheEntry>>,
    sets_mask: usize,
}
impl EvalCache {
    fn new(size: usize) -> Self {
        let sets = (size.max(EVAL_CACHE_WAYS) / EVAL_CACHE_WAYS)
            .max(1)
            .next_power_of_two();
        Self {
            e: vec![None; sets * EVAL_CACHE_WAYS],
            sets_mask: sets - 1,
        }
    }
    fn probe(&self, key: u64) -> Option<i32> {
        let start = ((key as usize) & self.sets_mask) * EVAL_CACHE_WAYS;
        for en in self.e[start..start + EVAL_CACHE_WAYS].iter().flatten() {
            if en.key == key {
                return Some(en.score);
            }
        }
        None
    }
    fn store(&mut self, key: u64, score: i32) {
        let start = ((key as usize) & self.sets_mask) * EVAL_CACHE_WAYS;
        for slot in &mut self.e[start..start + EVAL_CACHE_WAYS] {
            if slot.is_none() || slot.is_some_and(|en| en.key == key) {
                *slot = Some(EvalCacheEntry { key, score });
                return;
            }
        }
        let replacement = start + ((key >> 32) as usize & (EVAL_CACHE_WAYS - 1));
        self.e[replacement] = Some(EvalCacheEntry { key, score });
    }
}
impl Default for EvalCache {
    fn default() -> Self {
        Self::new(EVAL_CACHE_WAYS)
    }
}
struct PersistentContext {
    generation: u8,
    tt: Tt,
    eval_cache: EvalCache,
    hy: [Vec<i16>; 2],
    cr: [Vec<i16>; 2],
    countermoves: [Vec<u64>; 2],
}
impl PersistentContext {
    fn new() -> Self {
        Self {
            generation: 1,
            tt: Tt::new(TT_SIZE),
            eval_cache: EvalCache::new(EVAL_CACHE_SIZE),
            hy: [vec![0; Hs], vec![0; Hs]],
            cr: [vec![0; Cu], vec![0; Cu]],
            countermoves: [vec![0; Cs], vec![0; Cs]],
        }
    }
}
thread_local! {static Pz:std::cell::RefCell<Option<PersistentContext>>=const{std::cell::RefCell::new(None)};}
fn take_persistent_context() -> PersistentContext {
    Pz.with(|cell| {
        cell.borrow_mut()
            .take()
            .unwrap_or_else(PersistentContext::new)
    })
}
fn store_persistent_context(context: PersistentContext) {
    Pz.with(|cell| {
        *cell.borrow_mut() = Some(context);
    });
}
pub fn sp(position: &Po, hy: &[Po], _b: u16, time_ms: u64) -> Result<Sr, String> {
    search_timed(position, hy, _b, time_ms, None)
}
pub fn sp_depth(position: &Po, hy: &[Po], _b: u16, depth: u8) -> Result<Sr, String> {
    search_depth(position, hy, _b, depth, None)
}
pub fn sp_with_gt(position: &Po, hy: &[Po], _b: u16, gt: u16, time_ms: u64) -> Result<Sr, String> {
    search_timed_with_gt(position, hy, _b, gt, time_ms, None)
}
pub fn sp_depth_with_gt(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    depth: u8,
) -> Result<Sr, String> {
    search_depth_with_gt(position, hy, _b, gt, depth, None)
}
fn history_turn_gt(hy_len: usize) -> u16 {
    hy_len.min(MAX_GAME_TURNS as usize) as u16
}
fn search_timed_position(
    position: &Po,
    hy: &[Po],
    _b: u16,
    time_ms: u64,
    root_reverse_move: Option<Mv>,
) -> Result<(Sr, Diag), String> {
    search_timed_position_with_gt(
        position,
        hy,
        _b,
        history_turn_gt(hy.len()),
        time_ms,
        root_reverse_move,
    )
}
fn search_timed_position_with_gt(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    time_ms: u64,
    root_reverse_move: Option<Mv>,
) -> Result<(Sr, Diag), String> {
    let hy = hy.iter().map(_X::fp).collect::<Vec<_>>();
    let mut searcher = Searcher::new_timed(time_ms, root_reverse_move, hy.len() <= 1 && _b == 0);
    let result = searcher.search(position, &hy, _b, gt.min(MAX_GAME_TURNS));
    searcher.persist();
    result
}
fn search_raw_position_with_gt(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    time_ms: u64,
    root_reverse_move: Option<Mv>,
) -> Result<(Sr, Diag), String> {
    let hy = hy.iter().map(_X::fp).collect::<Vec<_>>();
    let mut searcher = Searcher::new_timed_with_poll(
        time_ms,
        root_reverse_move,
        hy.len() <= 1 && _b == 0,
        RAW_ABORT_POLL_MASK,
    );
    let result = searcher.search(position, &hy, _b, gt.min(MAX_GAME_TURNS));
    searcher.persist();
    result
}
fn search_depth_position(
    position: &Po,
    hy: &[Po],
    _b: u16,
    depth: u8,
    root_reverse_move: Option<Mv>,
) -> Result<(Sr, Diag), String> {
    search_depth_position_with_gt(
        position,
        hy,
        _b,
        history_turn_gt(hy.len()),
        depth,
        root_reverse_move,
    )
}
fn search_depth_position_with_gt(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    depth: u8,
    root_reverse_move: Option<Mv>,
) -> Result<(Sr, Diag), String> {
    let hy = hy.iter().map(_X::fp).collect::<Vec<_>>();
    let mut searcher =
        Searcher::new_fixed_depth(depth, root_reverse_move, hy.len() <= 1 && _b == 0);
    let result = searcher.search(position, &hy, _b, gt.min(MAX_GAME_TURNS));
    searcher.persist();
    result
}
fn search_timed(
    position: &Po,
    hy: &[Po],
    _b: u16,
    time_ms: u64,
    root_reverse_move: Option<Mv>,
) -> Result<Sr, String> {
    search_timed_position(position, hy, _b, time_ms, root_reverse_move).map(|(result, _)| result)
}
fn search_timed_with_gt(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    time_ms: u64,
    root_reverse_move: Option<Mv>,
) -> Result<Sr, String> {
    search_timed_position_with_gt(position, hy, _b, gt, time_ms, root_reverse_move)
        .map(|(result, _)| result)
}
fn search_raw_with_gt(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    time_ms: u64,
    root_reverse_move: Option<Mv>,
) -> Result<Sr, String> {
    search_raw_position_with_gt(position, hy, _b, gt, time_ms, root_reverse_move)
        .map(|(result, _)| result)
}
fn search_depth(
    position: &Po,
    hy: &[Po],
    _b: u16,
    depth: u8,
    root_reverse_move: Option<Mv>,
) -> Result<Sr, String> {
    search_depth_position(position, hy, _b, depth, root_reverse_move).map(|(result, _)| result)
}
fn search_depth_with_gt(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    depth: u8,
    root_reverse_move: Option<Mv>,
) -> Result<Sr, String> {
    search_depth_position_with_gt(position, hy, _b, gt, depth, root_reverse_move)
        .map(|(result, _)| result)
}
struct Searcher {
    cfg: SearchCfg,
    deadline: Instant,
    abort_poll_mask: u64,
    fixed_depth: Option<u8>,
    enforce_deadline: bool,
    nodes: u64,
    generation: u8,
    tt: Tt,
    eval_cache: EvalCache,
    acc: Vec<NnueAcc>,
    sh: Vec<_cS>,
    ks: Vec<[Option<Mv>; 2]>,
    hy: [Vec<i16>; 2],
    cr: [Vec<i16>; 2],
    countermoves: [Vec<u64>; 2],
    move_buffers: Vec<Vec<_G>>,
    scored_move_buffers: Vec<Vec<(i32, _G)>>,
    root_reverse_move: Option<Mv>,
    root_turn: u16,
    root_no_progress: u16,
    diag: Diag,
}
impl Searcher {
    fn new_timed(time_ms: u64, root_reverse_move: Option<Mv>, reset_shared: bool) -> Self {
        Self::new_timed_with_poll(time_ms, root_reverse_move, reset_shared, ABORT_POLL_MASK)
    }
    fn new_timed_with_poll(
        time_ms: u64,
        root_reverse_move: Option<Mv>,
        reset_shared: bool,
        abort_poll_mask: u64,
    ) -> Self {
        let safe_time_ms = time_ms.saturating_sub(deadline_slack_ms(time_ms)).max(1);
        Self::with_budget(
            Instant::now() + Duration::from_millis(safe_time_ms),
            abort_poll_mask,
            None,
            true,
            root_reverse_move,
            reset_shared,
        )
    }
    fn new_timed_depth(
        time_ms: u64,
        depth: u8,
        root_reverse_move: Option<Mv>,
        reset_shared: bool,
    ) -> Self {
        let safe_time_ms = time_ms.saturating_sub(deadline_slack_ms(time_ms)).max(1);
        Self::with_budget(
            Instant::now() + Duration::from_millis(safe_time_ms),
            ABORT_POLL_MASK,
            Some(depth.max(1)),
            true,
            root_reverse_move,
            reset_shared,
        )
    }
    fn new_fixed_depth(depth: u8, root_reverse_move: Option<Mv>, reset_shared: bool) -> Self {
        Self::with_budget(
            Instant::now() + Duration::from_secs(24 * 60 * 60),
            ABORT_POLL_MASK,
            Some(depth.max(1)),
            false,
            root_reverse_move,
            reset_shared,
        )
    }
    fn with_budget(
        deadline: Instant,
        abort_poll_mask: u64,
        fixed_depth: Option<u8>,
        enforce_deadline: bool,
        root_reverse_move: Option<Mv>,
        reset_shared: bool,
    ) -> Self {
        let cfg = *search_cfg();
        let shared = if !reset_shared {
            take_persistent_context()
        } else {
            PersistentContext::new()
        };
        Self {
            cfg,
            deadline,
            abort_poll_mask,
            fixed_depth,
            enforce_deadline,
            nodes: 0,
            generation: shared.generation,
            tt: shared.tt,
            eval_cache: shared.eval_cache,
            acc: Vec::with_capacity(MAX_PLY + 2),
            sh: Vec::with_capacity(MAX_PLY + 2),
            ks: vec![[None, None]; MAX_PLY],
            hy: shared.hy,
            cr: shared.cr,
            countermoves: shared.countermoves,
            move_buffers: std::iter::repeat_with(|| Vec::with_capacity(64))
                .take(MAX_PLY + 2)
                .collect(),
            scored_move_buffers: std::iter::repeat_with(|| Vec::with_capacity(64))
                .take(MAX_PLY + 2)
                .collect(),
            root_reverse_move,
            root_turn: 0,
            root_no_progress: 0,
            diag: Diag::default(),
        }
    }
    fn gt(&self, ply: u8) -> u16 {
        self.root_turn.saturating_add(u16::from(ply))
    }
    fn persist(&mut self) {
        store_persistent_context(PersistentContext {
            generation: self.generation,
            tt: std::mem::replace(&mut self.tt, Tt::new(TT_SIZE)),
            eval_cache: std::mem::replace(&mut self.eval_cache, EvalCache::new(EVAL_CACHE_SIZE)),
            hy: std::mem::replace(&mut self.hy, [vec![0; Hs], vec![0; Hs]]),
            cr: std::mem::replace(&mut self.cr, [vec![0; Cu], vec![0; Cu]]),
            countermoves: std::mem::replace(&mut self.countermoves, [vec![0; Cs], vec![0; Cs]]),
        });
    }
    fn take_move_buffer(&mut self, ply: usize) -> Vec<_G> {
        let idx = ply.min(self.move_buffers.len() - 1);
        std::mem::take(&mut self.move_buffers[idx])
    }
    fn recycle_move_buffer(&mut self, ply: usize, mut ms: Vec<_G>) {
        let idx = ply.min(self.move_buffers.len() - 1);
        ms.clear();
        self.move_buffers[idx] = ms;
    }
    fn take_scored_move_buffer(&mut self, ply: usize) -> Vec<(i32, _G)> {
        let idx = ply.min(self.scored_move_buffers.len() - 1);
        std::mem::take(&mut self.scored_move_buffers[idx])
    }
    fn recycle_scored_move_buffer(&mut self, ply: usize, mut scored: Vec<(i32, _G)>) {
        let idx = ply.min(self.scored_move_buffers.len() - 1);
        scored.clear();
        self.scored_move_buffers[idx] = scored;
    }
    fn correction_index(key: u64) -> usize {
        (key as usize) & (Cu - 1)
    }
    fn correction_score(&self, _c: Co, key: u64) -> i32 {
        i32::from(self.cr[_M(_c)][Self::correction_index(key)])
    }
    fn apply_correction(&self, _c: Co, key: u64, eval: i32) -> i32 {
        eval.saturating_add(self.correction_score(_c, key))
    }
    fn corrected_eval(
        &mut self,
        st: &Rq,
        key: u64,
        gt: u16,
        no_progress: u16,
        raw: &mut Option<i32>,
    ) -> i32 {
        let rv = *raw.get_or_insert_with(|| self.evaluate_position(st, gt, no_progress));
        self.apply_correction(st.position()._sT(), key, rv)
    }
    fn update_correction_history(&mut self, _c: Co, key: u64, se: i32, sc: i32, d: u8) {
        let idx = Self::correction_index(key);
        let slot = &mut self.cr[_M(_c)][idx];
        let cur = i32::from(*slot);
        let tgt = (sc - se).clamp(-CORRECTION_HISTORY_CLAMP, CORRECTION_HISTORY_CLAMP);
        let blend = (i32::from(d).max(1) + 2).min(8);
        let upd = cur + (tgt - cur) * blend / 8;
        *slot = upd.clamp(-CORRECTION_HISTORY_CLAMP, CORRECTION_HISTORY_CLAMP) as i16;
    }
    fn search(&mut self, position: &Po, hy: &[_X], _b: u16, gt: u16) -> Result<(Sr, Diag), String> {
        let mut st = Rq::new(position.clone()).map_err(|_| String::new())?;
        let root_fp = _X::fp(position);
        let mut history = SearchHistory::default();
        history.reset(hy, root_fp, _b);
        self.root_turn = gt;
        self.root_no_progress = _b;
        self.acc.clear();
        self.sh.clear();
        self.acc.push(nnue().root_acc(position));
        self.sh.push(_cU(st._t(), st._s()));
        let mut _az = Sr {
            bm: None,
            score: self.evaluate_position(&st, self.gt(0), history.current_no_progress()),
            dp: 0,
            nodes: 0,
        };
        let mut _i = 0u64;
        let mut _cl = 0u64;
        let mut _cq = 0u8;
        if _z(position, 0, self.gt(0)).is_some() {
            _az.score = _z(position, 0, self.gt(0)).unwrap_or(0);
            return Ok((_az, self.diag));
        }
        let mut _f = _az.bm;
        let mut pv = _az.score;
        for d in 1..=u8::MAX {
            self.generation = self.generation.wrapping_add(1);
            if self.generation == 0 {
                self.generation = 1;
            }
            if !self.admit_depth(d, _az.dp, _i, _cl, _cq) {
                break;
            }
            let iteration_started = Instant::now();
            let last_best_before_iteration = _f;
            match self.search_root(&mut st, d, &mut history, _f, pv, _az.dp) {
                Ok((score, bm)) => {
                    _az = Sr {
                        bm,
                        score,
                        dp: d,
                        nodes: self.nodes,
                    };
                    _f = bm;
                    pv = score;
                    _cq = if bm.is_some() && bm == last_best_before_iteration {
                        _cq.saturating_add(1)
                    } else {
                        0
                    };
                }
                Err(Sa) => break,
            }
            _cl = _i;
            _i = iteration_started.elapsed().as_millis() as u64;
        }
        if _az.bm.is_none() {
            let (fallback_move, fallback_score) = self.emergency_root_choice(&mut st);
            _az.bm = fallback_move;
            _az.score = fallback_score;
        }
        _az.nodes = self.nodes;
        Ok((_az, self.diag))
    }
    fn emergency_root_choice(&mut self, st: &mut Rq) -> (Option<Mv>, i32) {
        let side = st.position()._sT();
        let none_killers = [None, None];
        let mut bm = None;
        let mut _g = -_C;
        let mut best_order = i32::MIN;
        let mut ms = self.take_move_buffer(0);
        st.gl(&mut ms);
        for _m in ms.iter().copied() {
            let mv = _m.mv;
            let order = self.move_order_score(side, _m, None, None, None, none_killers, 0);
            self._bP(st, _m);
            let undo = self._bQ(st, _m).unwrap();
            let child_no_progress = if _m.ej {
                0
            } else {
                self.root_no_progress.saturating_add(1)
            };
            let mut score = -_z(st.position(), 1, self.gt(1)).unwrap_or_else(|| {
                self.evaluate_position(st, self.gt(1), child_no_progress)
            });
            if _m.ej {
                score += EMERGENCY_EJECTION_BONUS;
            }
            if self.root_reverse_move == Some(mv) {
                score -= ROOT_REVERSE_MOVE_PENALTY;
            }
            self._bR(st, undo);
            self.pop_acc();
            if score > _g || (score == _g && order > best_order) {
                bm = Some(mv);
                _g = score;
                best_order = order;
            }
        }
        self.recycle_move_buffer(0, ms);
        if bm.is_some() {
            (bm, _g)
        } else {
            (None, _z(st.position(), 0, self.gt(0)).unwrap_or(0))
        }
    }
    fn search_root(
        &mut self,
        st: &mut Rq,
        d: u8,
        history: &mut SearchHistory,
        _f: Option<Mv>,
        pv: i32,
        dp: u8,
    ) -> Result<(i32, Option<Mv>), Sa> {
        let tm = st.position().mc(Co::Black) + st.position().mc(Co::White);
        if dp == 0 || tm == 28 {
            return self.search_root_window(st, d, history, _f, -_C, _C);
        }
        let alpha = (pv - Aw).max(-_C);
        let beta = (pv + Aw).min(_C);
        let mut delta = Aw;
        let mut alpha = alpha;
        let mut beta = beta;
        loop {
            let _n = self.search_root_window(st, d, history, _f, alpha, beta)?;
            if _n.0 > alpha && _n.0 < beta {
                return Ok(_n);
            }
            if delta >= Aw.saturating_mul(4) {
                return self.search_root_window(st, d, history, _f, -_C, _C);
            }
            delta = delta.saturating_mul(2);
            if _n.0 <= alpha {
                alpha = (_n.0 - delta.saturating_mul(2)).max(-_C);
                beta = (_n.0 + delta).min(_C);
            } else {
                alpha = (_n.0 - delta).max(-_C);
                beta = (_n.0 + delta.saturating_mul(2)).min(_C);
            }
        }
    }
    fn search_root_window(
        &mut self,
        st: &mut Rq,
        d: u8,
        history: &mut SearchHistory,
        _f: Option<Mv>,
        mut alpha: i32,
        beta: i32,
    ) -> Result<(i32, Option<Mv>), Sa> {
        self.check_abort()?;
        let _K = history.search_key(_X::fs(st), self.gt(0));
        let tt_move = self.tt.best_move(_K, d);
        let mut _cQ = 0usize;
        let mut _cR = [None; 3];
        let mut _g = -_C;
        let mut bm = None;
        let mut _cK = alpha;
        for mv in [tt_move, _f, None].into_iter().flatten() {
            if _cR[.._cQ].contains(&Some(mv)) {
                continue;
            }
            let Some(_m) = st._bY(&mv) else {
                continue;
            };
            self.check_abort()?;
            let side = st.position()._sT();
            let _aT = _m.ej;
            let is_quiet = !_aT;
            let _cI = if is_quiet {
                self.history_score(side, _m._d)
            } else {
                0
            };
            self._bP(st, _m);
            let undo = self._bQ(st, _m).unwrap();
            let _cn = _X::fs(st);
            history.push(_cn, _m.ej);
            let mut score = match self.search_move_score(
                st,
                d,
                0,
                alpha,
                beta,
                history,
                _cQ,
                is_quiet,
                Some(_m._d),
                _aT,
                _cI,
            ) {
                Ok(score) => score,
                Err(error) => {
                    history.pop();
                    self._bR(st, undo);
                    self.pop_acc();
                    return Err(error);
                }
            };
            history.pop();
            self._bR(st, undo);
            self.pop_acc();
            if self.root_reverse_move == Some(mv) {
                score -= ROOT_REVERSE_MOVE_PENALTY;
            }
            if score > _g {
                _g = score;
                bm = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            _cR[_cQ] = Some(mv);
            _cQ += 1;
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(0, mv);
                    self.reward_history(side, _m._d, d);
                    self.record_countermove(side, None, _m._d);
                }
                self.tt.store(TtEntry {
                    key: _K,
                    d,
                    generation: self.generation,
                    score: et(score, 0),
                    bound: Bk::Lower,
                    bm: Some(mv),
                });
                return Ok((score, Some(mv)));
            }
        }
        _cK = alpha;
        let mut ms = self.take_move_buffer(0);
        st.gl(&mut ms);
        if _cQ > 0 {
            ms.retain(|entry| !_cR.contains(&Some(entry.mv)));
        }
        self.order_moves(st.position()._sT(), &mut ms, _f, tt_move, None, 0);
        if ms.is_empty() {
            self.recycle_move_buffer(0, ms);
            if bm.is_some() {
                return Ok((_g, bm));
            }
            return Ok((_z(st.position(), 0, self.gt(0)).unwrap_or(0), None));
        }
        for (_p, _m) in ms.iter().copied().enumerate() {
            let mv = _m.mv;
            let side = st.position()._sT();
            let _aT = _m.ej;
            let is_quiet = !_aT;
            let _cI = if is_quiet {
                self.history_score(side, _m._d)
            } else {
                0
            };
            self._bP(st, _m);
            let undo = self._bQ(st, _m).unwrap();
            let _cn = _X::fs(st);
            history.push(_cn, _m.ej);
            let mut score = match self.search_move_score(
                st,
                d,
                0,
                alpha,
                beta,
                history,
                _p + _cQ,
                is_quiet,
                Some(_m._d),
                _aT,
                _cI,
            ) {
                Ok(score) => score,
                Err(Sa) if bm.is_some() => {
                    history.pop();
                    self._bR(st, undo);
                    self.pop_acc();
                    self.recycle_move_buffer(0, ms);
                    return Ok((_g, bm));
                }
                Err(error) => {
                    history.pop();
                    self._bR(st, undo);
                    self.pop_acc();
                    self.recycle_move_buffer(0, ms);
                    return Err(error);
                }
            };
            history.pop();
            self._bR(st, undo);
            self.pop_acc();
            if self.root_reverse_move == Some(mv) {
                score -= ROOT_REVERSE_MOVE_PENALTY;
            }
            if score > _g {
                _g = score;
                bm = Some(mv);
            }
            let raised_alpha = score > alpha;
            if raised_alpha {
                alpha = score;
            }
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(0, mv);
                    self.reward_history(side, _m._d, d);
                    self.record_countermove(side, None, _m._d);
                }
                self.tt.store(TtEntry {
                    key: _K,
                    d,
                    generation: self.generation,
                    score: et(score, 0),
                    bound: Bk::Lower,
                    bm: Some(mv),
                });
                self.recycle_move_buffer(0, ms);
                return Ok((score, Some(mv)));
            }
        }
        self.tt.store(TtEntry {
            key: _K,
            d,
            generation: self.generation,
            score: et(_g, 0),
            bound: if _g <= _cK { Bk::Upper } else { Bk::Exact },
            bm,
        });
        self.recycle_move_buffer(0, ms);
        Ok((_g, bm))
    }
    fn search_move_score(
        &mut self,
        st: &mut Rq,
        d: u8,
        ply: u8,
        alpha: i32,
        beta: i32,
        history: &mut SearchHistory,
        _p: usize,
        is_quiet: bool,
        _a: Option<u32>,
        _aS: bool,
        history_score: i32,
    ) -> Result<i32, Sa> {
        if _p == 0 {
            return self
                .negamax(
                    st,
                    d.saturating_sub(1),
                    ply + 1,
                    -beta,
                    -alpha,
                    history,
                    true,
                    _a,
                )
                .map(|score| -score);
        }
        let can_reduce = is_quiet && d >= _aM && _p >= _aN && history_score < HISTORY_LMR_THRESHOLD;
        let base_reduction = if can_reduce { _N(d, _p) } else { 0 };
        let reduction = if can_reduce {
            let mut tuned = base_reduction;
            if history_score <= -2049 {
                tuned = tuned.saturating_add(1);
            } else if history_score > 4096 {
                tuned = tuned.saturating_sub(1);
            }
            tuned.min(d.saturating_sub(2))
        } else {
            base_reduction
        };
        let scout_depth = d.saturating_sub(1 + reduction);
        let mut score = -self.negamax(
            st,
            scout_depth,
            ply + 1,
            -(alpha + 1),
            -alpha,
            history,
            true,
            _a,
        )?;
        if reduction > 0 && score > alpha {
            score = -self.negamax(
                st,
                d.saturating_sub(1),
                ply + 1,
                -(alpha + 1),
                -alpha,
                history,
                true,
                _a,
            )?;
        }
        if score > alpha && score < beta {
            score = -self.negamax(
                st,
                d.saturating_sub(1),
                ply + 1,
                -beta,
                -alpha,
                history,
                true,
                _a,
            )?;
        }
        Ok(score)
    }
    fn negamax(
        &mut self,
        st: &mut Rq,
        d: u8,
        ply: u8,
        mut alpha: i32,
        beta: i32,
        history: &mut SearchHistory,
        allow_null: bool,
        _a: Option<u32>,
    ) -> Result<i32, Sa> {
        self.check_abort()?;
        self.nodes += 1;
        let current = _X::fs(st);
        if let Some(score) = _z(st.position(), ply, self.gt(ply)) {
            return Ok(score);
        }
        let key = history.search_key(current, self.gt(ply));
        let mut beta = beta;
        let mut raw_static = None;
        if let Some(en) = self.tt.probe(key, d) {
            let score = decode_tt_score(en.score, ply);
            match en.bound {
                Bk::Exact => {
                    let raw = *raw_static.get_or_insert_with(|| {
                        self.evaluate_position(st, self.gt(ply), history.current_no_progress())
                    });
                    self.update_correction_history(st.position()._sT(), key, raw, score, d);
                    return Ok(score);
                }
                Bk::Lower => alpha = alpha.max(score),
                Bk::Upper if score <= alpha => {
                    let raw = *raw_static.get_or_insert_with(|| {
                        self.evaluate_position(st, self.gt(ply), history.current_no_progress())
                    });
                    self.update_correction_history(st.position()._sT(), key, raw, score, d);
                    return Ok(score);
                }
                Bk::Upper => beta = beta.min(score),
            }
            if alpha >= beta {
                let raw = *raw_static.get_or_insert_with(|| {
                    self.evaluate_position(st, self.gt(ply), history.current_no_progress())
                });
                self.update_correction_history(st.position()._sT(), key, raw, score, d);
                return Ok(score);
            }
        }
        if d == 0 {
            return Ok(self.corrected_eval(
                st,
                key,
                self.gt(ply),
                history.current_no_progress(),
                &mut raw_static,
            ));
        }
        let ss = if alpha > -_D / 2 && beta < _D / 2 {
            Some(self.corrected_eval(
                st,
                key,
                self.gt(ply),
                history.current_no_progress(),
                &mut raw_static,
            ))
        } else {
            None
        };
        if ply > 0 && d <= 4 {
            let ss = ss.unwrap_or_else(|| {
                self.corrected_eval(
                    st,
                    key,
                    self.gt(ply),
                    history.current_no_progress(),
                    &mut raw_static,
                )
            });
            if ss.saturating_sub(70 + 70 * i32::from(d)) >= beta {
                return Ok(ss);
            }
        }
        if d <= 2 && alpha > -_D / 2 && beta < _D / 2 {
            let ss = ss.unwrap_or_else(|| {
                self.corrected_eval(
                    st,
                    key,
                    self.gt(ply),
                    history.current_no_progress(),
                    &mut raw_static,
                )
            });
            let margin = if d == 1 {
                FUTILITY_MARGIN_DEPTH1
            } else {
                FUTILITY_MARGIN_DEPTH2
            };
            if ss.saturating_add(margin) <= alpha {
                return Ok(ss);
            }
        }
        if allow_null
            && ply > 0
            && d >= NULL_MOVE_MIN_DEPTH
            && d > 3
            && self.gt(ply).saturating_add(1) < MAX_GAME_TURNS
        {
            let null_gate = ss.unwrap_or_else(|| {
                self.corrected_eval(
                    st,
                    key,
                    self.gt(ply),
                    history.current_no_progress(),
                    &mut raw_static,
                )
            });
            if null_gate >= beta.saturating_sub(_cb) {
                let mut null_reduction = NULL_MOVE_REDUCTION.saturating_add(d / 6);
                if null_gate > beta {
                    let bonus = ((null_gate - beta) / 200).clamp(0, 2) as u8;
                    null_reduction = null_reduction.saturating_add(bonus);
                }
                null_reduction = null_reduction.min(d.saturating_sub(2));
                let parent = self.acc.last().cloned().unwrap();
                self.acc.push(parent);
                self.sh.push(*self.sh.last().unwrap());
                let _ck = st._bW();
                let null_fp = _X::fs(st);
                history.push(null_fp, false);
                let null_score = -self.negamax(
                    st,
                    d - 1 - null_reduction,
                    ply + 1,
                    -beta,
                    -beta + 1,
                    history,
                    false,
                    None,
                )?;
                history.pop();
                st._bX(_ck);
                self.pop_acc();
                if null_score >= beta {
                    return Ok(beta);
                }
            }
        }
        let side = st.position()._sT();
        let mut tt_move = self.tt.best_move(key, d);
        if !self.cfg.use_tt_backfill && tt_move.is_none() && d >= 4 {
            tt_move = self.tt.best_move(key, d.saturating_sub(1));
        }
        let _x = self.probe_countermove(side, _a);
        let _cK = alpha;
        let mut _g = -_C;
        let mut bm = None;
        let ks = self.ks.get(ply as usize).copied().unwrap_or([None, None]);
        let mut _cQ = 0usize;
        let mut _cR = [None; 4];
        let mut priority_quiet_tried = [0u32; 4];
        let mut pqt = 0usize;
        for mv in [tt_move, ks[0], ks[1], None].into_iter().flatten() {
            if _cR[.._cQ].contains(&Some(mv)) {
                continue;
            }
            let Some(_m) = st._bY(&mv) else {
                continue;
            };
            self.check_abort()?;
            let _aT = _m.ej;
            let is_quiet = !_aT;
            let _cI = if is_quiet {
                self.history_score(side, _m._d)
            } else {
                0
            };
            self._bP(st, _m);
            let undo = self._bQ(st, _m).unwrap();
            let _cn = _X::fs(st);
            history.push(_cn, _m.ej);
            let score = self.search_move_score(
                st,
                d,
                ply,
                alpha,
                beta,
                history,
                _cQ,
                is_quiet,
                Some(_m._d),
                _aT,
                _cI,
            )?;
            history.pop();
            self._bR(st, undo);
            self.pop_acc();
            if score > _g {
                _g = score;
                bm = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            _cR[_cQ] = Some(mv);
            _cQ += 1;
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(ply as usize, mv);
                    self.reward_history(side, _m._d, d);
                    self.record_countermove(side, _a, _m._d);
                    for _d in priority_quiet_tried.into_iter().take(pqt) {
                        self.penalize_history(side, _d, d);
                    }
                }
                self.tt.store(TtEntry {
                    key,
                    d,
                    generation: self.generation,
                    score: et(score, ply),
                    bound: Bk::Lower,
                    bm: Some(mv),
                });
                if let Some(raw) = raw_static {
                    self.update_correction_history(side, key, raw, score, d);
                }
                return Ok(score);
            }
            if is_quiet && pqt < priority_quiet_tried.len() {
                priority_quiet_tried[pqt] = _m._d;
                pqt += 1;
            }
        }
        let mut ms = self.take_move_buffer(ply as usize);
        st.gl(&mut ms);
        if _cQ > 0 {
            ms.retain(|entry| !_cR.contains(&Some(entry.mv)));
        }
        if ms.is_empty() {
            self.recycle_move_buffer(ply as usize, ms);
            return if bm.is_some() { Ok(_g) } else { Ok(0) };
        }
        self.order_moves(side, &mut ms, None, tt_move, _x, ply);
        let mut quiet_tried = [0u32; 64];
        let mut qt = 0usize;
        for (_p, _m) in ms.iter().copied().enumerate() {
            let mv = _m.mv;
            let _aT = _m.ej;
            let is_quiet = !_aT;
            let _cI = if is_quiet {
                self.history_score(side, _m._d)
            } else {
                0
            };
            if _at
                && ply > 0
                && d <= 4
                && _p >= 3 + d as usize * d as usize
                && is_quiet
                && _cI <= 0
                && Some(mv) != tt_move
                && Some(mv) != ks[0]
                && Some(mv) != ks[1]
                && Some(_m._d) != _x
            {
                continue;
            }
            self._bP(st, _m);
            let undo = self._bQ(st, _m).unwrap();
            let _cn = _X::fs(st);
            history.push(_cn, _m.ej);
            let score = self.search_move_score(
                st,
                d,
                ply,
                alpha,
                beta,
                history,
                _p + _cQ,
                is_quiet,
                Some(_m._d),
                _aT,
                _cI,
            )?;
            history.pop();
            self._bR(st, undo);
            self.pop_acc();
            if score > _g {
                _g = score;
                bm = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(ply as usize, mv);
                    self.reward_history(side, _m._d, d);
                    self.record_countermove(side, _a, _m._d);
                    for _d in priority_quiet_tried.into_iter().take(pqt) {
                        self.penalize_history(side, _d, d);
                    }
                    for _d in quiet_tried.into_iter().take(qt) {
                        self.penalize_history(side, _d, d);
                    }
                }
                self.tt.store(TtEntry {
                    key,
                    d,
                    generation: self.generation,
                    score: et(score, ply),
                    bound: Bk::Lower,
                    bm: Some(mv),
                });
                if let Some(raw) = raw_static {
                    self.update_correction_history(side, key, raw, score, d);
                }
                self.recycle_move_buffer(ply as usize, ms);
                return Ok(score);
            }
            if is_quiet && qt < quiet_tried.len() {
                quiet_tried[qt] = _m._d;
                qt += 1;
            }
        }
        self.tt.store(TtEntry {
            key,
            d,
            generation: self.generation,
            score: et(_g, ply),
            bound: if _g <= _cK { Bk::Upper } else { Bk::Exact },
            bm,
        });
        if let Some(raw) = raw_static {
            self.update_correction_history(side, key, raw, _g, d);
        }
        self.recycle_move_buffer(ply as usize, ms);
        Ok(_g)
    }
    fn order_moves(
        &mut self,
        side: Co,
        ms: &mut Vec<_G>,
        pv_move: Option<Mv>,
        tt_move: Option<Mv>,
        _x: Option<u32>,
        ply: u8,
    ) {
        let ks = self.ks.get(ply as usize).copied().unwrap_or([None, None]);
        let mut scored = self.take_scored_move_buffer(ply as usize);
        scored.extend(ms.iter().copied().map(|_m| {
            (
                self.move_order_score(side, _m, pv_move, tt_move, _x, ks, ply),
                _m,
            )
        }));
        let partial_sort_k = self.cfg.partial_sort_k.min(scored.len());
        if partial_sort_k < scored.len() {
            let pivot = partial_sort_k - 1;
            scored.select_nth_unstable_by_key(pivot, |en| Reverse(en.0));
            scored[..partial_sort_k].sort_unstable_by_key(|en| Reverse(en.0));
        } else {
            scored.sort_unstable_by_key(|en| Reverse(en.0));
        }
        ms.clear();
        ms.extend(scored.iter().map(|en| en.1));
        self.recycle_scored_move_buffer(ply as usize, scored);
    }
    fn move_order_score(
        &self,
        side: Co,
        _m: _G,
        pv_move: Option<Mv>,
        tt_move: Option<Mv>,
        _x: Option<u32>,
        ks: [Option<Mv>; 2],
        _ply: u8,
    ) -> i32 {
        let mv = _m.mv;
        if Some(mv) == pv_move {
            return 4000000;
        }
        if Some(mv) == tt_move {
            return 3000000;
        }
        if Some(mv) == ks[0] {
            return 2000000;
        }
        if Some(mv) == ks[1] {
            return 1000000;
        }
        if Some(_m._d) == _x {
            return COUNTERMOVE_ORDER_BONUS + self.history_score(side, _m._d);
        }
        if _m.ej {
            return EJECTION_ORDER_BONUS + self.history_score(side, _m._d);
        }
        self.history_score(side, _m._d)
    }
    fn history_score(&self, side: Co, _co: u32) -> i32 {
        i32::from(self.hy[_M(side)][_co as usize])
    }
    fn reward_history(&mut self, side: Co, _co: u32, d: u8) {
        let bonus = i16::try_from(i32::from(d) * i32::from(d)).unwrap_or(i16::MAX);
        let slot = &mut self.hy[_M(side)][_co as usize];
        let slot_value = i32::from(*slot);
        let bonus_value = i32::from(bonus);
        let updated = slot_value + bonus_value - ((slot_value * bonus_value) / 16384);
        *slot = updated.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    fn penalize_history(&mut self, side: Co, _co: u32, d: u8) {
        let malus = i16::try_from(i32::from(d) * i32::from(d)).unwrap_or(i16::MAX);
        let slot = &mut self.hy[_M(side)][_co as usize];
        *slot = slot.saturating_sub(malus);
    }
    fn probe_countermove(&self, side: Co, _a: Option<u32>) -> Option<u32> {
        let _a = _a?;
        let en = self.countermoves[_M(side)][_a as usize & (Cs - 1)];
        let stored_previous = (en >> 32) as u32;
        let stored_reply = en as u32;
        if stored_previous == _a && stored_reply != 0 {
            Some(stored_reply - 1)
        } else {
            None
        }
    }
    fn record_countermove(&mut self, side: Co, _a: Option<u32>, reply_key: u32) {
        let Some(_a) = _a else {
            return;
        };
        let index = _a as usize & (Cs - 1);
        self.countermoves[_M(side)][index] =
            (u64::from(_a) << 32) | u64::from(reply_key.saturating_add(1));
    }
    fn current_shape(&mut self, st: &Rq) -> &_cS {
        let _ = st;
        self.sh.last().unwrap()
    }
    fn push_move_state(&mut self, st: &Rq, mv: Mv) {
        let model = nnue();
        let mut next = self.acc.last().cloned().unwrap();
        let mut shape = *self.sh.last().unwrap();
        let position = st.position();
        let side = position._sT();
        let mut own_from = 0u64;
        let mut own_to = 0u64;
        let mut enemy_from = 0u64;
        let mut enemy_to = 0u64;
        for cell in mv._ad() {
            let bit = 1u64 << cell.as_u8();
            own_from |= bit;
            model.ap(&mut next, side, *cell, -1);
            let dst = _cJ(*cell, mv.direction()).unwrap();
            own_to |= 1u64 << dst.as_u8();
            model.ap(&mut next, side, dst, 1);
        }
        if mv.len() > 1 {
            if let Some(ax) = move_group_axis(mv._ad()) {
                if move_is_inline(ax, mv.direction()) {
                    if let Some(front) = move_front_cell(mv._ad(), mv.direction()) {
                        if let Some(_ah) = _cJ(front, mv.direction()) {
                            let _e = side.other();
                            let mut cursor = Some(_ah);
                            while let Some(cell) = cursor {
                                let occupant = st.occupant_fast(cell);
                                if occupant != Some(_e) {
                                    break;
                                }
                                enemy_from |= 1u64 << cell.as_u8();
                                model.ap(&mut next, _e, cell, -1);
                                let dst = _cJ(cell, mv.direction());
                                if let Some(dst) = dst {
                                    enemy_to |= 1u64 << dst.as_u8();
                                    model.ap(&mut next, _e, dst, 1);
                                }
                                cursor = dst;
                            }
                        }
                    }
                }
            }
        }
        self.acc.push(next);
        match side {
            Co::Black => {
                shape.b = _cV(shape.b, st._t(), (st._t() & !own_from) | own_to);
                shape.w = _cV(shape.w, st._s(), (st._s() & !enemy_from) | enemy_to);
            }
            Co::White => {
                shape.w = _cV(shape.w, st._s(), (st._s() & !own_from) | own_to);
                shape.b = _cV(shape.b, st._t(), (st._t() & !enemy_from) | enemy_to);
            }
        }
        self.sh.push(shape);
    }
    fn _bP(&mut self, st: &Rq, _m: _G) {
        self.push_move_state(st, _m.mv);
    }
    fn _bQ(&mut self, st: &mut Rq, _m: _G) -> Result<_bN, Re> {
        st.apply_move(&_m.mv)
    }
    fn _bR(&mut self, st: &mut Rq, undo: _bN) {
        st.undo_move(undo);
    }
    fn pop_acc(&mut self) {
        let _ = self.acc.pop();
        let _ = self.sh.pop();
    }
    fn evaluate_position(&mut self, st: &Rq, _cA: u16, _cB: u16) -> i32 {
        let _e = _X::fs(st);
        let key = _J(_e) ^ ((_cA as u64) << 48) ^ ((_cB as u64) << 32) ^ EVAL_CACHE_SEED;
        if let Some(score) = self.eval_cache.probe(key) {
            return score;
        }
        let shape = *self.current_shape(st);
        let score = nnue().evaluate_with_acc_bits(
            st.position()._sT() == Co::Black,
            &shape,
            st._t(),
            st._s(),
            _cA as f32,
            _cB as f32,
            self.acc.last().unwrap(),
        );
        self.eval_cache.store(key, score);
        score
    }
    fn admit_depth(&self, d: u8, dp: u8, _i: u64, _cl: u64, _cq: u8) -> bool {
        if let Some(limit) = self.fixed_depth {
            if !self.enforce_deadline {
                return d <= limit && dp < limit;
            }
            if d > limit || dp >= limit {
                return false;
            }
        }
        if d <= 1 || dp == 0 || _i == 0 {
            return Instant::now() < self.deadline;
        }
        let remaining_ms = self
            .deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as u64;
        let _ = (_cl, _cq);
        let estimated_next_ms = _i.saturating_mul(39).div_ceil(20).max(_i + 1);
        estimated_next_ms <= remaining_ms.saturating_add(1)
    }
    fn record_killer(&mut self, ply: usize, mv: Mv) {
        if ply >= self.ks.len() || self.ks[ply][0] == Some(mv) {
            return;
        }
        self.ks[ply][1] = self.ks[ply][0];
        self.ks[ply][0] = Some(mv);
    }
    fn check_abort(&self) -> Result<(), Sa> {
        if self.fixed_depth.is_some() && !self.enforce_deadline {
            return Ok(());
        }
        if self.nodes & self.abort_poll_mask == 0 && Instant::now() >= self.deadline {
            Err(Sa)
        } else {
            Ok(())
        }
    }
}
fn deadline_slack_ms(time_ms: u64) -> u64 {
    SEARCH_DEADLINE_SLACK_MS.min((time_ms / 8).max(1))
}
fn move_group_axis(_ad: &[crate::ac::Ci]) -> Option<Li> {
    let gm = gm();
    let first = gm.cell(_ad[0]);
    [Li::Q, Li::R, Li::S].into_iter().find(|ax| {
        let ai = ax.index();
        let line_id = first.line_ids[ai];
        _ad.iter()
            .all(|cell| gm.cell(*cell).line_ids[ai] == line_id)
    })
}
fn move_is_inline(ax: Li, direction: Di) -> bool {
    match ax {
        Li::Q => matches!(direction, Di::Se | Di::Nw),
        Li::R => matches!(direction, Di::East | Di::West),
        Li::S => matches!(direction, Di::Ne | Di::Sw),
    }
}
fn move_front_cell(_ad: &[crate::ac::Ci], direction: Di) -> Option<crate::ac::Ci> {
    match _ad {
        [] => None,
        [first] => Some(*first),
        [first, second] => {
            if _cJ(*first, direction) == Some(*second) {
                Some(*second)
            } else {
                Some(*first)
            }
        }
        [first, second, third] => {
            if _cJ(*first, direction) == Some(*second) && _cJ(*second, direction) == Some(*third) {
                Some(*third)
            } else {
                Some(*first)
            }
        }
        _ => None,
    }
}
fn _cJ(cell: crate::ac::Ci, direction: Di) -> Option<crate::ac::Ci> {
    gm().cell(cell).ns[direction.index()]
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct V3GroupDir {
    mv: Mv,
    inline: bool,
    translated_mask: u64,
    history_key: u32,
    ray_bits: [u64; 3],
    landing: [Option<crate::ac::Ci>; 2],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct V3SourceGroup {
    len: u8,
    source_mask: u64,
    dirs: [Option<V3GroupDir>; 6],
}
#[derive(Clone, Debug)]
struct V3MovegenTables {
    source_groups: Vec<V3SourceGroup>,
}
fn v3_movegen_tables() -> &'static V3MovegenTables {
    static TABLES: std::sync::OnceLock<V3MovegenTables> = std::sync::OnceLock::new();
    TABLES.get_or_init(build_v3_movegen_tables)
}
fn build_v3_movegen_tables() -> V3MovegenTables {
    let geom = gm();
    let mut source_groups = Vec::with_capacity(256);
    for cell in geom.cs().iter().map(|cell| cell.index) {
        let cells = [cell, cell, cell];
        source_groups.push(V3SourceGroup {
            len: 1,
            source_mask: 1u64 << cell.as_u8(),
            dirs: build_v3_group_dirs(&cells, 1, None),
        });
    }
    for ax in [Li::Q, Li::R, Li::S] {
        for line in geom.lines(ax) {
            for len in 2..=3 {
                if line.cs.len() < len {
                    continue;
                }
                for start in 0..=line.cs.len() - len {
                    let cells = canonical_group_cells(&line.cs[start..start + len]);
                    source_groups.push(V3SourceGroup {
                        len: len as u8,
                        source_mask: v3_source_mask(&cells, len as u8),
                        dirs: build_v3_group_dirs(&cells, len as u8, Some(ax)),
                    });
                }
            }
        }
    }
    V3MovegenTables { source_groups }
}
fn build_v3_group_dirs(
    cells: &[crate::ac::Ci; 3],
    len: u8,
    axis: Option<Li>,
) -> [Option<V3GroupDir>; 6] {
    std::array::from_fn(|dir_idx| {
        let direction = Ad[dir_idx];
        let group = &cells[..len as usize];
        let translated = build_v3_translated(group, direction)?;
        let translated_mask = translated
            .iter()
            .flatten()
            .fold(0u64, |mask, cell| mask | (1u64 << cell.as_u8()));
        let (inline, first_step) = match axis {
            None => (false, translated[0]),
            Some(ax) => {
                let inline = move_is_inline(ax, direction);
                let front = move_front_cell(group, direction)?;
                let first_step = if inline {
                    _cJ(front, direction)
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
            current = gm().cell(cell).ns[direction.index()];
        }
        Some(V3GroupDir {
            mv: Mv::_Y(group, direction),
            inline,
            translated_mask,
            history_key: history_group_key(group, direction),
            ray_bits,
            landing,
        })
    })
}
fn canonical_group_cells(group: &[crate::ac::Ci]) -> [crate::ac::Ci; 3] {
    let mut out = [group[0]; 3];
    for (index, cell) in group.iter().copied().enumerate() {
        out[index] = cell;
    }
    out[..group.len()].sort_unstable();
    out
}
fn build_v3_translated(
    cells: &[crate::ac::Ci],
    direction: Di,
) -> Option<[Option<crate::ac::Ci>; 3]> {
    let geom = gm();
    let mut translated = [None; 3];
    for (index, cell) in cells.iter().copied().enumerate() {
        translated[index] = Some(geom.cell(cell).ns[direction.index()]?);
    }
    Some(translated)
}
fn v3_source_mask(cells: &[crate::ac::Ci; 3], len: u8) -> u64 {
    cells[..len as usize]
        .iter()
        .fold(0u64, |mask, cell| mask | (1u64 << cell.as_u8()))
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Sa;
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct _G {
    mv: Mv,
    ej: bool,
    _d: u32,
}
type _bN = UndoSnapshot;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct _X {
    _sT: Co,
    _t: u64,
    _s: u64,
}
impl _X {
    fn fp(position: &Po) -> Self {
        Self {
            _sT: position._sT(),
            _t: bits(position.black()),
            _s: bits(position.white()),
        }
    }
    fn fs(st: &Rq) -> Self {
        Self {
            _sT: st.position()._sT(),
            _t: st._t(),
            _s: st._s(),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bk {
    Exact,
    Lower,
    Upper,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TtEntry {
    key: u64,
    d: u8,
    generation: u8,
    score: i32,
    bound: Bk,
    bm: Option<Mv>,
}
struct Tt {
    buckets: Vec<[Option<TtEntry>; TT_BUCKET_SIZE]>,
}
impl Tt {
    fn new(size: usize) -> Self {
        Self {
            buckets: vec![[None; TT_BUCKET_SIZE]; size.div_ceil(TT_BUCKET_SIZE).max(1)],
        }
    }
    fn probe(&self, key: u64, d: u8) -> Option<TtEntry> {
        let bucket = &self.buckets[key as usize % self.buckets.len()];
        let mut best = None;
        for en in bucket.iter().flatten() {
            if en.key == key && en.d >= d && best.is_none_or(|previous: TtEntry| en.d > previous.d)
            {
                best = Some(*en);
            }
        }
        best
    }
    fn best_move(&self, key: u64, d: u8) -> Option<Mv> {
        let bucket = &self.buckets[key as usize % self.buckets.len()];
        let mut best = None;
        for en in bucket.iter().flatten() {
            if en.key == key
                && (en.d >= d || best.is_none_or(|previous: TtEntry| en.d > previous.d))
            {
                best = Some(*en);
            }
        }
        best.and_then(|en| en.bm)
    }
    fn store(&mut self, _n: TtEntry) {
        let index = _n.key as usize % self.buckets.len();
        let bucket = &mut self.buckets[index];
        for slot in bucket.iter_mut() {
            match slot {
                Some(existing) if existing.key == _n.key => {
                    if existing.d <= _n.d || existing.generation != _n.generation {
                        *slot = Some(_n);
                    }
                    return;
                }
                None => {
                    *slot = Some(_n);
                    return;
                }
                _ => {}
            }
        }
        let mut replacement = 0;
        for index in 1..TT_BUCKET_SIZE {
            let _ao = bucket[index].unwrap();
            let _ap = bucket[replacement].unwrap();
            let _aq = _ao.generation != _n.generation;
            let _ar = _ap.generation != _n.generation;
            if (_aq && !_ar) || (_aq == _ar && (_ao.d < _ap.d)) {
                replacement = index;
            }
        }
        let _as = bucket[replacement].unwrap();
        if _as.generation != _n.generation || _as.d <= _n.d {
            bucket[replacement] = Some(_n);
        }
    }
}
impl Default for Tt {
    fn default() -> Self {
        Self {
            buckets: vec![[None; TT_BUCKET_SIZE]; 1],
        }
    }
}
fn bits(cs: &[crate::ac::Ci]) -> u64 {
    cs.iter()
        .fold(0u64, |acc, cell| acc | (1u64 << cell.as_u8()))
}
fn _J(_e: _X) -> u64 {
    splitmix64(
        _e._t
            ^ _e._s.rotate_left(1)
            ^ match _e._sT {
                Co::Black => 0,
                Co::White => 0x9E3779B97F4A7C15,
            },
    )
}
fn _I(position: &Po) -> Option<Rq> {
    let record = Po::new(
        position._sT().other(),
        position.black().to_vec(),
        position.white().to_vec(),
    )
    .ok()?;
    Rq::new(record).ok()
}
fn _K(_e: _X) -> u64 {
    _J(_e)
}
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}
fn et(score: i32, ply: u8) -> i32 {
    if score >= _D - 1000 {
        score + i32::from(ply)
    } else if score <= -_D + 1000 {
        score - i32::from(ply)
    } else {
        score
    }
}
fn decode_tt_score(score: i32, ply: u8) -> i32 {
    if score >= _D - 1000 {
        score - i32::from(ply)
    } else if score <= -_D + 1000 {
        score + i32::from(ply)
    } else {
        score
    }
}
const fn _y(n: usize, k: usize) -> usize {
    match k {
        0 => 1,
        1 => n,
        2 => (n * (n - 1)) / 2,
        3 => (n * (n - 1) * (n - 2)) / 6,
        _ => 0,
    }
}
fn _H(mv: &Mv) -> usize {
    Hg(mv._ad()) * 6 + mv.direction().index()
}
fn history_group_key(_ad: &[crate::ac::Ci], direction: Di) -> u32 {
    (Hg(_ad) * 6 + direction.index()) as u32
}
fn Hg(_ad: &[crate::ac::Ci]) -> usize {
    match _ad {
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
fn combination_rank_2(first: crate::ac::Ci, second: crate::ac::Ci) -> usize {
    _y(first.as_usize(), 1) + _y(second.as_usize(), 2)
}
fn combination_rank_3(first: crate::ac::Ci, second: crate::ac::Ci, third: crate::ac::Ci) -> usize {
    _y(first.as_usize(), 1) + _y(second.as_usize(), 2) + _y(third.as_usize(), 3)
}
fn _N(d: u8, _p: usize) -> u8 {
    let raw = 1u8
        .saturating_add(d / _aO.max(1))
        .saturating_add((_p / _aP.max(1)) as u8);
    raw.min(d.saturating_sub(2)).max(1)
}
fn _M(side: Co) -> usize {
    match side {
        Co::Black => 0,
        Co::White => 1,
    }
}
fn _z(position: &Po, ply: u8, gt: u16) -> Option<i32> {
    let black = position.black().len();
    let white = position.white().len();
    let winner = if white <= 8 {
        Some(Co::Black)
    } else if black <= 8 {
        Some(Co::White)
    } else if gt < MAX_GAME_TURNS {
        return None;
    } else if black > white {
        Some(Co::Black)
    } else if black < white {
        Some(Co::White)
    } else {
        None
    };
    let base = _D - i32::from(ply);
    Some(match winner {
        Some(_c) if _c == position._sT() => base,
        Some(_) => -base,
        None => 0,
    })
}
const NNUE_MODEL_A85: &str = r##":/4Y@#65P@!<IKH!!!9)!!!'#!!#%[!!",A!!)DV4ZNcJz?pObm?pObm^d%l"J3O#6J3O#6?pObm?pObmJ3O;>zzzzzzzz!!"+^TtU#,h*$L01Hl7R!!'k$s8R('!!'P2!!$(;!!'+Js8NK[!!%p9!!'V#s8Qde!!##=!!)]6s8O^Ws8NNe!!$jL!!%uMs8T:us8N,Cs8PFf!!'2D!!#lts8Ulfs8O&A!!##(!!'.is8T#.!!&5s!!!_]s8Ruo!!".bs8NbCs8P5*!!&P(s8SZ^!!!n]s8P5%!!)'p!!'8D!!(%g!!#_4!!$'_s8Q-'s8Tass8S<$!!'tr!!!3>!!"oH!!)u<!!#4Os8Q'ps8SYJs8On1s8NlB!!(b*!!(BBs8Pe.!!&-/!!(k0!!%iTj8eER578A&g\_*N*Vfj270K[fH3(GQK_jF"o`8D)ZLnn6FRu+&g^DF4oE^?[_X`WrT)6tV&Gm[O.0Z7;DZX#=]_D'r&F8ZNX9.J[iVB#F^\=#fp]TCT)t^Z5625L04mbi;-j\/qam^qh%K6q>T`C_rdL7l()#@D<V#ueOHh*a'HiC2F/,YD$Duk@r:&dEZs81p\d0k[:XTnP)'_HY_V#E5"[/0q*+U7Q72#kr!D#Bt&oBk<#EW3*?lM;Iu"Sm94":GM<n+GW59D`9/)>"aBUC]-W2>$eRbl<RrmJ#TL?N:fB625s1iVQ+;;>$sch=b.TFo[s'gB.Ha.0]hU'E>?=SIp/Wi;G/37eaG'gAU1Q[Ff6m'+D8C[f;-]B(q,YC@smZ\c*?K_Yb_WNr!J]ILlG]&c(D`[Jm]CS,$an!=je^/+X5Z'*'ue<Uujl)"s!n6idc&-NuX*:%V'Tk5):a&,n6KXTRVi<X/L(p&Dl*KalZD*W)B4e,"k.2uFa'R*FmjCCMB^;ZY_UhY+;UU%h"C8c<0%Y5>j1*;RA>JebWpf_9@aO8HQn<VgV+oa-ZW<U.R3OSYOF,4&&8mdsBV&H`XQLB2VlOT:^GBDet8ZMhPugAmEB6Nk4(U'C/f.h1DC9`!rT/,>h?4904(G1SeS.1)XF'EPZ@mJ#00>4ttc/brf_H1mp/1B,)a*"fjd"SQI.GkdF+nb,$;bQ72/YN+:E+ofL[liu/UiXA!A2>`gF.fY1o+TpU"3sn"L)#*4kmK3b(%/>/UaS4g!<t0BX!<rW1!VUsTL&0[ROp>4id.g98%0%1D/GO>dV!kn`pB5()/G`lQ/bUmn1()G1iUgOPrqiDs3q/R:AI+"gkOt_\TDdjEU&2tEM%N__dJnLp\*FCq./fY1<=#c,YkVB$#6e-:B`u3W,l0b5k74*q=6c+Yo_/J"?N4dkMZ_3,W<<+u`;b)\*WHKBhX^d@m/pSn;Z0tq;u?"OJd[8;2Y[dM;Z7jC5khP_r!hHX+T/VJNr>.:^^:PZ*shiF%frp<:@_lcd-CH%#Ql4EMttNbX9GX:0aa'K_>AK]D@aE$A_g/nFS+9!?iG<h"8r#tA.$KQ+8mM\&,`m9&GLPDl2i75pAU*Ic1sQeX:kU;B)+IW.K@Hk-MlTs*XdW!.ec+5E;BM-.NK005PA"[R/j]$aT."bn)_.X<s-/;*<)Mhl3Fd-^'XK:Hi\6m>7cu?d-[e@E:jhW7f(:;LZbT2^BtSQ^\NN]l2#N)Gk?%Jo)a@P%K:JG^&1"UNYg=M%Jm4%3WBrr*VQ!#PQR?Ro^bl_QhP+`OpqO(o'n77q?OG;NWbsH]CstRkPiO*&GZUuJHr[i!=:O`#7"uE@hH!\](Q[D[.u$X5Q5!o4O;RVZ48/Y:&OtiW;(fB`;A3Q&HJOAZh\/.l2@4E8ec^W:AXq`pAVo)I.US;S,IF(=7/Q[#lWNP&+>bn_XH=d'+O(8;uobdQNu+@\bpFHh#-!jK`kun]`ktL_Zd@F<"[.W=S12LK_f6bGlZt,\'%,=q[G"TYlqH3`q3"+_>,DJ"9(UiA,7bUrV6`o<"a3SOSEGfGkpn6j8'qop\I)PipBbT$N98h?N\4QTa&7+Vtp5lWqMSH64Neh[K^":Yk\"s`sLl<$i\H\@e(03HkVVJV>4a.XnW&18GX)9PR-m^>P#?3oDN_1nb="'n+-kOKG9LA>Q+U%>4l\6c4#4(AGG-WmJ`a\U%K?$2%AdJ1\`s>XT"LWg%=G>g&Sh[]`@F%9CA&<)Zfj5V\XFX_=_+;(^HQAEX1#.k4nl\ckX;)#O'qImJ/.956C<e63\/3W:NL_H24*6e,ZK6dJ!M7&I\(%b5#Z@*W3&.+q"\ZZME/G-MsD5_>!!kee3$)Fo0>h.JtRbPQ";`eHCC&qZ23L$fSDa,PnY>\HsM]6hXs[D$tCoWX,@7=nZ,8?58tIo\o<J'_USp63+G%+Tulmg&m`=&cm=WWW2V_ZM!nj!!k.R`VsB@2ZW448.Af2C&AJ6p\\7po_-r:Xpc?NQMF,Y2"e*9H3g__`;/R"m0)e\TE3U4"Rt[KJ-PcJ'D7RYbQq8^)u6E;4S^BDk8QT/+R9+>&G,c.a7FT_a9T?YB*GUCeFco:8,p[1[0$g3Ta=!_2ZP2u-ikGAHkG6;o_Pf`iVm?e'(PAc:B*W]0DLD1WV->V@iO\Bn+SUD#7Gb_WX0U9W:#H8<<2Ll7/I/YS-SEi3!WRfao\mQ<tF=2i:G/'EVXA\r;1:7<s/!np'\,(H1u:Z>l;/1<W)@Thun%^.J!iGIfe^!(_L]gqY<hWg\`Di@eBs#\b,n"XRJ.P;u=)u8eOB#2Y/[3pD;Hq8J'&:F8T5?N;qT<?Ll;80El.u_Z\'_4UNqD6PoJ21B%7_%/U;:VuE7flg[aN$4rO%6iORt:\Y4qhu'/5;#^G-6iEGM;#`?EY7)<RfD?H_])Gat9(F2)$1HCCc0USl$hqC2;#Rg4Zg$l^U^2c#$4He&$46/#]DQ=D9CnD5)Za.R!!UsU?O[PE1(\+(]D4Dr3W)GPJH\sLBC*Y=iXl7mMuj^g62/.k4SOs1fDNr&^\ld_])6jV!u4!)K(lh[rqanGO8eSC;Y=Aq%db_RB):ZjZN97oZh/\A;u;.M>6;k:)\3Y]:(*EeXS8+n(C)l/q#RH,IJGJkj:8^$WVfU"SbN$r0*49>/-j3!]Cr!)Sc#5]1&&!^\,-X&r$\rIMuSYBB__2uqZjY5"8AhihuAr=./0M4^^t/Xf^?ie2>sZVrp`5_PPDccFo1"t9`C%E.1AcC^_J3r552YhYmM`d%003Cn+e@>1`;tih=^L@`;"N_.KdWo<<BE1II/Hu7ehc>f)69;g[U?ug_YJ>#m'r&C&N)FpB*\^N;FXr>5PH!=8.=FC_^%Qp\(<drW"#8q"<_Xq>nJ44S3:i.K2[,.0bG(#_5_Q*r#m3*Y)Bc7L%T_D"dK?ZP]SI"n0M8K_b'Cli^)j@fpcgYkR/Zs7?s+HNL)1(%n!nV$Lu=D#POKKE(N6(':'/)>]BudJ!D-K(pbqq@@!;#5JH+natH&i;8]C=o+`k9_Qg79b#MBk5fJo@.WagjRef2QO_a[EVm'NK_M;hRM8=*ZM<\i8+b@?HLds>4pVT.T);@i`;KoA$NhaN7JQBQmg4ULHN@mTeH/SIfaKtS6i's]0Do_\htYn%JI#lqc2)ntQ1VsDT^`Buhs6pCk4CqAIg!RYXT9R:YP.o.1[IL3ap3S-F7XMX./tq9<Z@&?2"n?g%HG#5R/I<qj:?>:ck#.8p[-]3=oO0SB*S%tYm234eb$`Q8H]%h,nEKJQi;1%n,XL'T`"!LOS([B#kgUrJ-,0%nd;9+%/YAa'7_qHP6<*$Qi^Lm>4^/X6ME)-H2t#Wf)\hR#mms=o+GUHRJfG^+9A&:PQ,.u]a$P"YRe&;XSJ[t=o7+J"8s2+blto[T)a3IbQg]3<snBsSbS*3e,696;"uO;Vt3:YGjh%(V>'u^VZI/7P4`5pZ5FQ&@0fX"R0W`eK_XO4!9t'k7/Rr4YQ#%4(_-H8K+ee&irM:OV#]0Bd/M/e;ZUY2[M"]%S,)^[`V:_PAGQQ!V#YW=/c\rZ2>t)ZT+#<>3r(u7*rl<2QLtpmEUca&"RE5o:&3QFh>Vg/@-[\"E=CGFHO1MZ0+ns#ded8Q/+j;?m/N%SRfZ[l"Ukh:W=04<pA$fV?39-aY5_DucN)Vrd245/^B"E9NV\b,!UOt&-3&DmD#U:%6hn(i&e^u?s7Hg,D#GUc$iX*)*r"mt,5.'.]`)"3-ii`?.J$O7&br:.7g#.Oq[>XfFTEQYVu*;&[0A/gPlS9*Zj2]m'+TBe6iFIt>4h.nDZ_Hi5mKUThu(D%rqloqC&O,#[flEkh]p;;*</)-htN974oiY-g\5X_X8Km68bN#[<><gXA`BQl@K,-]Z27r:fCgNi,l8Pa9EfA.d0rG=<s2b$./e,geHl'[:]Yt\)[%\mit5Dp]D$@G@.t9IA-]jMo)o.'s7/SuDZ&_q-is_A,je=lgF9<(blcQ#N;=t#)us7I2>_V'49<t0mJ/401)+X&b4A.*"S^"3Ab2W'5Q+L[3;jfu8I$4.B+($u;@0Z&eG\(m4:UE_0`Fuj$2l+RU)7ABU[or2DYVKbS-qgjq$6L1(\Iihj7VO5SHUP/biu_2\dCY%&d5!)WrJXam.VP%meU<)"o\2e!;<MjY7JqY*;"C[@/6Vfb4rFRh=`)tm/P]>+p)BK=:U/c,Pe89W<<1q`X;8Gg%\b`B`CL4Pni!Z;u2^\YP,X=S+=Z,6jJSlec^Zjipjbb/I!!a?L/X-%g\'oY6<#9:^"E*_YVX]eG23/8c$3o[em0&+ptO\hYZXNM"(W>SFiHU4SS%K=UIV!IM!"]ZNL@"X9Bg[2Zsip*"n;@oC2Y^AHluY2%bNK./-UF.gD(G_?+0O*=W8eapE1X"80nEPlqj"g[u9RmfpSsSGi0UR0M:14Ub-qaneF!Nr%&n8-B+n?k*gl.J8Q8E<*Z?f(Kpe=neBt')=Tl)$\md5n(s948rt@Es\68r<3Q>XS0U7#7NKp()Y@'lh$kX@JB0p^\pS,2[VYFBaG^G5O\%BT)jTC_#7L+i<;%Ao`8Y,#n&9]E=1q.J,I4X"8Qp.lMMq*_tr*mI.6#*$1tXrK_eCKWqZo?0+FW`3<_\=\I)"%pAU3sS1]3eQhXbpAH!eR64X1u+!Cso]_q1)%eLS5Z2B%R?3A"9r<l*gkjE!t0DnB=g@sq()$FFHH2r=*L^0LBAIga)U]7%d,5I8ihXH$tW:>HAq<SUq#kpOV"oJi;:[uC91(!.gNX/q_rsRoc9DXJQC?-lF%/VL\b67545nHrk1(N+=3rQc$`Wa'D'`.(m'a%;-<=_b"@ef3m,Q+b8rpR5bAd%&lo)(`_#6Tt_HkL]+eG#F0cijCEp\./UZi@f8Zgmqs])7Nb;ub)K`on]XHif*DPR2O+jU(MY/,[Tl8+k9s$iTc-P5>jg-OWiNp'KLMp@h\t*!J4il26,1qZ1a8)\XIi:BMa@*raa^4m%(`L&u;k0)PP6ZiKXu`!_>$Sb.I?/H@4$Jb^,Wr<*B(AHTIPoDl&lo`+Ll/+>"l,l>gbirnH:;@-2"q#3na2>hG+2[Aj[ncd)F,R[TNit#Gu5kll1BD,:?o*[t_57&S=S-G8`[/llrK)r%VQN'>B@k&r,%f1&Oc1htQA+EP&5PZPsA+q2JqtCg*!>/$)`:,hil2+3SGl6J+=SOHJ6i.>mY7G::q@GLY1B0-,iVQ%-JHn+E"961U'aVALS/-AP)>4I1!qaDC&f*J-PQkP$(&#6Q.g$sh`X$,A'^:kK*#Ls>(Bj:3I/[CMk4>5L'`9Hmm/q2(^\Kk\!=uI647U2gn+G6)lM!dMHML#5;#d!c#6ZdQ!tmNc6Np-aj8"c(N!*>qZM?['j9EI%[MAuJ=8A'RD">m^kDJa7PQuaV[K1[S1&2a\q$BF["mU!odKX%XaUfKg&I*mgh=V`g,5JbYJ,NID0_rQQjU,8nMXeanHhj92TCtJJSG95qLB;l*0F&rrkm[FaHM-G)%JCq;Q3AlJ2Y@[pCBp"+)\e#,q"0%8HM]Q/#P1k"IKdJ":'knFSa7Ktn-=18jn9AT1C&s2NL>"fCBlok,QYFSW;/d\4SCH&?iP0Clj`LlfBrkY63'@AL\*7ng%k.U>54iq!YWrI-OF#R"TQ:S@dlk^irmU>^[%^$lOHnmDANN]0D:,'+QMPhX7Gca&djI#:^a#t=u(R8kll5-4R_P.8dLCW@0LNFr!-!h>mZu#A,O(8Msng\LC=F`<=tE#'C7P!,kA)6Hh,Yd#5VjHFp[jJ=9m.B^'h7<:B&!@f(k[1WU1T8K`/OaPRHgXr=oJ<natfBec-cjEq0M?`<&+&_$4'@ZL*AtFT^e):%Tk'R0Em!=oMhF>m3t(4V4Rj]_UddV=m+[n+1N%!<d<Df^*V_Wq6_qcM`P)551$P@frM<D$:RC*=tU4B)a^ol05SLlg%gaWWhYdM[g'pL_;cM;u,/PT`J`j8,;BKp]8M8IgI%ps7@]Z,QM!B7I9F9"U>S4V>%/8rs"2RmMD8hU\IsgjT%[Pk424Jh#mc[.e*WD8GfRgV?<.r*:&Im+9Bt23scl"56osCBD\q=i:EW9C@H*KKE^H8faW`:D\(P$1&CGI;[3p%%fArtl3QtZQ5.=1+p<*"]D<`>5O$noC]r>e9DV<N<X3=@K+;)e>55c5(BPo`$3#;I&dq;1iVj]%+9@Q#8-/l(<Ttt=AGpHD1'j0?\I1q4(\_/8h"kmeZNb=V8d3->eHi)NK+E2+d.\@B`U_%^A-g?RN!MoRZ3tsa'*)SA4:^NNFT,G5VAj=LPlbY+eFm8LFRd$K#5W]@_YW9rg%`Pe3"tQSMsCiTSF0b_A,IDL&G?20?N?f#p]k4#fb$sPY6"\)f)(]sp&=IgT_YDsR0iEYlOoung\Xh+N;-6]Eu*('cNPj*D$IBN;#4i'e.j)%L[4Qt*"/eGC'%uGW;KQnVsf9HkkbntQj8*B7J*qH%1_k,`p/<[2sPA_Hh_LH]CR3Mir?Lj3sJIIKb7tag]3'.?hGl9c2U0?V#&OWRh;V!JfNhM:&<E'<VMC])?\4gXTsUdiX>GTj733Pn-u/fR-ri!H3W$sXU[,^F9th=h")?eW;YiU6LhqfYl;/jIM`Rm8EV*"G;GT[_Y1_0Dtr#uU\6MI,72EV.h'0A\+E5kU@XrIQ2;F7/,gIs)%U9Z;]A1,?20l]pAK=E^]Dmf<X'!68Ij2L"R3&r9+'/HD=StCQO',ZL(-WHJdS1C=oMY$@JS"95MchH[fZ9kdhVf[X75HdO7u3>UA"-Q_>83ea92kR3t.&9#7%9m(]?T)/GaVe8)U3Q#PXtpQO1\);A$A=A+r.];?H.dXT7)YOp4hdCC-(':@*'6U^PrXr9B47S->#WBa0FoZjIoM[L3]"&GY/SkNB>e-hP+MlOMJEdclStZ1s.TiqNlc0D-Y*XT^ci1'btkW"L`P%06/.JFbPji7>IL#Qu"F`t6f"AJU7*Q1kdg2>gGMJcB'\;?D=L0aPu&,OuQj\,eG`5OGHJdfWnSQNHUe/e^8D6km<#(&;;35PWbGOn(I3jU?kD'D68Fc2hh]*!G@44775#XnFCQ;?\?=W!RDG'a,-OD>"&$J(RL,1'=fZ!#)(-k8;&:=8T!!63WM:f_oacrrR6BlOX@&=T(qt4o53N@IBlh?j;D^NVrtRmM!tGrsF5Lk4j,m\c&]+XSK-pPRa_m>m:361BK?/e-&,Fcfs$d-N0P43!eI:Wt91C5Pn"R$Mtq63W#oa<WM%_Gme$XQj^>#C]!,K#OmZ]f+,k!K*C93Q3hF/FT2ID,9<EPG5l.Emg-8rW<2)^6M78U/bkV1c1lnb^A]/\iq!lhQkJW[!pP+@=RfYqB)Pm4+R`P<,QcNt9*W]<9FjJ`H3'K6QhlC+NqnJ;irSuZ9FO_p:()jWHMj8d7eX+uH3Po'SHXuT>ROj9c2"^H]-BN>/b[QW7g#Rb<s:2Mkl%@Pr;.!+nbS=Cjp#)nM"P!=*YUa47H@8IUZ58K;#+Vh)>/IcVYk!+#6k&!cj35.rW1LBN;>I,nEo]RU&)t9AI5gGp^W5]8c>:<p&Ft!.fjPj<<lt8g(AZ,7Jc3L+<92>]_(OW=ojriklCtY,mXtUY5lT?lMB62M#"mfZLP]rr=jk`n)@1C8F;!'7f!/Vme$emMZVi<+9rbU4UF:6+8RDgRf3WkD;_E)(B2,Q)A-!L58*no>lMh2Sc^>L49O1:YQa^q+;0#"U\FHb_%#Q/\+>1ID#ug#\,aG6!uR[(9aKkJU\KlB8a6Kl/b)7)=qX%"*q<\?QM'/3Y5<PHdIbutoDm;>^^0`.Ih"1'iW+Z7A,9X.51C>Vjod[H#SjN=@hY@4D#Mu^#QmX(EWD=+pB"7m$k>/rTCUM:"pGM9k44B2'*;J>aoLZ"4sZNjfF*)nn+WOHkkI"EoC8^Uo*gr\Ern6%Vt]$J62uE>e*ul4T)JHam0DPDTF3CM1'(2JO8R,gLXEL;GQ_+JobMfPU)%qPR/&'))ut-ZR/VR6AcgulZk1"Q_Y^SI`VXTNVt*O_X8n(QB)9(59d9l?cNZl>C\ba9%LP8i7J;6)E=Q+K&dJFG!<(jR)$ZW7qVq8QRJrKKVZY0?`sddujS[d5>5cV6&,%CVg&r5oC(Nc5#maB&]_XkPA+&aqo)ejg^BGGU0a[1D1&a<7$la<dGQUJ,V$'Hg&HL8^B)g9T[K+nZ+o%Z%49(cO%.MOJC(VrlhY)p8L@/sTR0.a3M!!k*Reano=p5lA3"O7&Q2ebtFoBhZ<qI7=!sV^*.h'`%J.)SZrVB%<o^4pP1]rO2('A=QL^V?&)[(0V@3P+9m/3aOMufgSBEK=T*s#790FE./d.o6gQNLCn8b7?7CDWZ?B_R2cD"U*lo)f$l)"(H0E;]A+oE7V[FV&?]GQ!O$r<$*oNnrG#;?Lk;Hk6qjoa_Q4Ieu1c/+TkRReN?Nk6e?jU(MGBBDTC;\J\-:4S^5t`;,l#D?17E'bCQGHP$5TGPS336j8,@\F\8Rq[W/p7e>^E?1^YeK)q>4<qE[&Rf@L/BEkF2/.ANO)uQZ98,k[@qUbTRq?-Zs$l8F33YG`uV>Oorqu!5u^AFr+Ad'[[f+VTkjS1D1U'%t$rV9+dn,<[mU%rF&L*7E1<!tSbi:V<e/-X&cId]f-W!bK]E<&rBGlCM>.g"W3+n?WWandalBEqi<Xp]1Dj8ZIc_Z4`H"5W)H3<FO/M%r\\iY;.b`V90$SGa''oD4mN;$*']Se-t`&,kVlJcOa'MYK(6L\^!.b5CN"h&.(.Ta"?h_tIs>E<?R9d.%\F/dh1G+:$'sOo@6AYlL*g\I],7OT)Zdp[011P4?s'BE4e)DZcUBRL@IAAH04=`UDU_FR`ZC%i5]R3WH2ZW;a:3pAfgHJbBTTH3?P5C]--r`r-bK&,s`O62d&Vo^m#(@epB0ch.bX9D]8+V?M#A$2QIXnG#EDBa-?r\c!!6a:*:c7f)cX+nl*I\*G^N5QGBk0+3mPg'-@1)#b*F47Ztg&Fg;7mir);B*+G!!!+\U0`^24cL\^LdKDK+li!:TB_)B-nbRe:2?"C#\+tgONVdMf$ML/)bkYELZNg[9"T-jQC%>g/ap.>8o%QD(q@9S)5ktT`K(:T('^6;9GPKG\<!ms]SHV%Mn,hYJ=mo)dR.\2q@2k../.-4mpAF[`j8HFbl1l@r.KZFDPks,548VSJU\W+?6N.>tnbX7+f);uS0DMOWS,R1+eIKt,-2HL%EUt1?V[6r:jOBps5Rgd";u_7C70%iFRI+bq')>Z.ZNWH!Z3RZ/!X3ZDh<1+">P-GFr!,"bmLL!.ht*oG61cJtEVDfr(^PNee+N,-Gl9u9@Jcf!VYVGB/GRfiZht14lh5Z:M#qAp;A&<krpu6P\FSSGM@'%[']e6M*t2rmnc9.!D>tIZYjiA5O8NhpecN_pZj:%7XoX1M2<WKgG5"`:g'3<=V[\X\V>G3.i:-F;-hRK=nHIM>](=#C8bc6nSbAEbH2,>Z)>]Euj88fVY4]C=*s#:FWXO=LK_$Pk>m<"N4URka(^GBdJI\%6bPn:6Xn:fblgG&b=qTHc,R#n/-j\i19)<#r\FU.5Vt;8>as/D.FTGYB-i-^C\d5J;Hgb#:a9/UFRKDU^k5ghB7/[8i4Sp64_"iWdQi%'cR/$RcY513g,6C=6Y5EABT(MIEKa6Z>cN?c6nIS%`"TQ"2puH]0C@B@P2[8jMd0>aFcO-3<K`Y`flhVq8lM-2;O=#PJWrJjgI/pA2:BS]=,4opXq[+>B%ffT6<r;7Y@/3+ij8A3OHM=!1deit@T(l4p_Yr!ie-/bQUA3@;5kQDm[Kum2fCr>9-kEK^irRO,eE@e^dJOF[*X1$c'F>WiK`PQWO914>]CH",c1)_K%N2JBDZo"onGaQhNt*<5?hW(GNX.62`r0ZUYjKR/:\e9$)?:$Lb4kr%.0%Hd)>?f80EPkqaq#g$=R@X@cgVqIp]d/e#2VB6^'i6ef`c?l+nUO$./@ulLB;u(0FM=ta786ebQja?-h"5C4Ru5@=q19.G7DssPkG^Z>9VSkJ+B942$]fSaSed8qX;`4[JUjZZMj%oJbq.f8Gs;%V"THQ+9opnE>ON5?L!^Uo^:c18."\[*7DIlL(1d$?j6#mGkuIjH1*-gJH*4^R0NQqN:n=doE3_HDX5gOWV)SU&I16(rXCdX"T:X\DAM.-d.;qhGR+NQ>59lR485'L5Q,[EK_aa8Y5K=A3rA:BjRpn2NW3;lKbCclQ0o.\FTT_MT+%e'K[a0%GRUAUn.7&J_uI^_M!sF5'E2;8T*9*2G5VX>`<$_MX6R+h\G3W:I/L&O/dQIlYPMf&HhHIrM#'X?oa#1*NV'@Tn+c_hSFpXt/Ge`5iV?:T;#(1pi:`iQkPbMlFV2%Yd-m>%lMg8>71,1\`";86\.E`K`;!d6..%<LGlCVE9`b=mmK4jHF9MaBqYuQRWp^N"qtTmmrX^RFQMu4/1AFiD)ZGZp(?^>YK)hbNJGm5-Vtl/C:&BtH:B33X_"cgqYPO7T)u3A3E<2U=;?ilqVZ1N7%eV7D-4$%BB*tC1":a,_-iP(ZL?LK,g[?^'r#AT4@g(p+4Tp9/;?@F*]D4SnYP\n55namqS-4lGB`UfaOnX&(])Q=I3s?H"KE6E-hu3';^@PP%deS%jb5*Xf*qJds=SoE*m/Z&L`=6SblME18mdQt?0FFoaJd8XTg)q.2$Nj]5:+34W5kR5IfbNiON<IuK_>qaQPS;40;u"]Prq?!U)@Yls<rT<!\HJ_h`Vr"&VY"g/?k,X(3WIt@'_WsO+7r2-<W^qY[eYdU%eF*.j7['g_uof@7i:Us#4=\Z*q+CF(C$c/IfolLQ5*9mfa#=g;"d!bp^NSrcj:clh??%JL[1u6D@\o)5P>a$BCf<r3!Qk]T*EsADYcWdE!Z.6.I`9#7L4#)=8iC.qX$`Pbk_hVZ2uBWWqf?W5PpZ@n+MnQklm.URMcJ:hs?p<jQm-F8-aV8$NeWKj9O$-@h+.]]_<W`D=bCG<s#*"Ym5RXcg!4ZXU2l@>OQ5/T_3m1AdQ!E$k+EUI/l\-.2kJOI.Kl48Hq'Rc2FdMLZoN[kP^_s_"ldjrVoFba8+*VL@p<2o`Dl2B+j,',k>gVh?$mP$k.Xcd->oT$jZ(p%ii[TW<'(3"RCaOQ2fYE4oK*ld-GfIOp45JD=)cNoClnsirM[So`b*X/H8cIOVXA`K_UK+kR.@lr;9Ou&a^_O)uWGKlMd[],m"2MT(S!>Fn[BqN;ite*u9,&S+[Tu;Z`BRlNXs(-17K*'*n:;0`[40JHmG0^&ds8E;FqXdem\I4otEKZ33Yo_s?dd55%tY<r-hA>m-,ZT)[77^A\`Aa8EdO%L>>kAc"RnEp^"BMXZ93Gl1hA8bkgbgAXnR>PH;J4ol9:&J]]gPiS>6L^9dCR0LM7_[+3YV[KWs.f$aG8cDm$FT+-#9G.p4D$Ude>6Mb9KE448"mbmTQM=>bM$*/3.fT>GE;XhT])QCFQ0:.+@0&@fZMI`TD@d'P49RkGM?'(J%/ck+qt)?;c1r=LMuVH:!t"Q!-2m!+"n/)RZNf\&jp5]!Fp];bQLsqGHh@g\=7KZ&FWRjdeHMobZMK^V#6\*'FSM./^AcL^ZO+2/XobKpJHH&G2tW+4#6F];4T[tNPQ"c&-4FSc_rG\7+94&!l2<49/+@l]W;E[]2#_Y)*te&DnGPNP\Eo4GV[\.Zk6:YXR1N-i&G2grj7cRgIJ"*]R4-)Rj9ANlbO[1VeI,FEf(lT]h#2'dTElM,KEOm@pBamh!;=eK!"*lW*<a:Z*WVf8=pTQU1@A0G)$]((=8VpueaZ!'4o1<=jSu^q(D^Eb0DH1c2=!$f*X#U<$46q-SJ96AVYlhY-NYe2VtpQ'8..Q`e-=&(C%PX#&/)/9g\jj?q"Zc<UB^,Udf;Z+H3#)_V>GiM?1t`</e%dZ>Q!^dJ-&s[P4r#c&dlGWp%MJn&bghX&+l\'0a0HBJ.h;UNUk@1a7YDc#mP;1?jS4S^_g&Cnck?P8+$*ar</o$g%uKj#lfkZJ*qgTHP68R;Yb\HT(btJ0*ma05n@JqhZX_kfD4;)B(G0jrr`Ar`r&X:o)>Nl[.k13Xo?lXf_OJ0M>NG.M"UHLj8g\6oFiAPIf(`2/-.3qB+^R(+Tu?S0F6D%h'SNS>kCN$`q&g6@ebloFT/uD_se`@gC4\g"nI$@-M_3Gl3Yf(%1bu*Ifkf43=:K=Wr;u#`<N(A+8rDG!V7fM^\tJA49Qo2V?'3iV"@+MJbPZD>Q`FVNtr0-KDn.A$jt;Q'a[P1P5tma)$9F.3?I8<cM?H4>OG8o1%nlU/Fd`Nb47%T_$"KHXR<7`@ItW@Ooab;a:5fQUAqb(jT.sMWVsF-Ig=L+\c(jpmdO]sp%n:b6iaFR3;_kAr:I`+J+_\)iW?_)JeR&A.erf>CC#IZJHg0#q>q]>j8F6(Fq^hTk5M1d&cZ,A=o"3^-3FtO1@t)1;Z]efMYW_8;Y=B%h>)^*3='$YU\u;3+o*&RNrO"m!=%fdpAgZg#P4c(iU-mM`;XZ3@/A7U=oQ,98G?L'h>^Xr-PWNYGP"3&EXS`;JHmJ(-4$RKQ4nuG+8@VZXT+Rp1&^\F[JuHa8.=;a3='<no`.\gQf?Nnh"]k.0Eq12-hq*IV#RLl;?5YSg$VQM'`dLn[JGe$BbF>MFpQmuYl6Q^-hX#52Y2;!4Sd.jbQ,3XJI&Uce-3)T[IclZ!sIWfd0KUEN"5+F$Mc=IB)?9R1&"rV)?^'5p'Oq;TD6S(ZNZ3`5k+(>kPR@I_#uU^VYlVShuX;_$i@%3>4LJj"9o,0dJksNS.[je_r/E1^B1YAA,a[T:%H[.JGL#D?2Kij+:No+.fB_Mqt0LN*!;'%1'o)q;@@F?2Z;#!meE.^47eC7Rh%^S<XfWL>47dOJHr[V6hJ\-\c2.$Y6IDgbl)MUq\$pA!Vio,ZMg:G<<ZJ''D\%%h[SiMlJrHVgBc10C]BS&c1naCDYodkkQ#3#;A;D&[e@iG!VhoSO9N8u[gC:P=qIP.q>_?-*qD#tYP63LbR)Jg;\34@2=Es(<==0@h"`i7<W@CDLC"mr0Ds5lO:,q%/c:6cdf2oJNruTb_>)^`QO(A37/%Q#P8^1qeG*qkK(p>\XnPs$'a2Ya[1YM,Fn1b5-2s8*o)2De$P"TWfEL%&8dMd!rr%WcHgsE+jp>l'rX3082>'#rI1qh?b5'0V"p(1c<se$^-NNc04:XsY]^n&XeF;c$;$>VUPPT>'V?qY^lLmdE5R_`=W;.eGEVNT9(&!UpliT0SJI'R)%/)Y"C'nkOE=usZ,739%LB_/T_Z,2]m.:St$2AKN/-A;.Yl$'@T(Zae[0dKC0_6Ihq"S7k;$S$1\.WZM:BVj:IJI4BfD;<;E;AkmjnjZ5(]Prn'_g_pq?F&6fD0@rV"EO9B(YZud04Cl*t\GR7/3YoD%%$W$O.gW%0SHWjos*$LCi/"s7Pst]CVg'hu4K"Gi+8B[f9_-1Bo6,p$_VRh=;!S"U8o3*u_6aTDeijE<"8P(B;YT+9R6)[/FM8!rBG=aSG6*-NPap5PCiVk4SNE^%N9N>7/X\jqPQ1L\9'tmgPTTe-2uWJd;;KpAtEn:]mWZebc'Q9DJ,bq"VH*;!U(3e+iS*ncW>!an&1%RIc+91]_sjJeTL*oD6W.QM>k!lMIRWBEUF"@fCELAH)iG3:XKZUAM4H<W;Rpc2Hf&:A4#[0`hjs5nihc*q+FNIgiIFEsCn5/-Oc.1^-tXbl&jl*<3>S_>NO#%/JBKkO>_EYO,fpW!,]]#61q,U@HG1/H@!sf*KgkchVSVo_RS@M?,F.'aT6c_#X<9W<3&0$M")p?N"g%(]-i0&-=clIe=rqXSlW6&JDGIl1?t5U(?G],RCUR,mBD//dk;FQMp"90*->.6ig^%C&=S"Ta:T:WW9^--3XkT?g8m5:]%j'oE4=E,Q"hM_\.[sNW?Bef'_l'`s()h8,UdTW"n.uiWE<^^%rcL$Mc.R,4SG2(&QP@6iO@jdL.0%K)dA$E;jnE"p98TR0d^#OTppC=oCAX(';Phf)+Y56NSA-klkB&BCNY1`s9!Mrp0(S+o/8@r<#LbnbN[iY8cg*Zhse,IIGW$Fph+?h>%0hPnEKgnD6J3=UYuNLB?066iX:enbEaRR/r<R#o$2D;Y\N;/c2^E^&GqP4:-i8_$X*9XoAD"p@Itk1]Ckc_[2%rLB^NZ-h!/ih?pseA+,re>Q'iedKXRm%f:kRIOiYA/G`ic'_nL2=p/=,([kr;2\O(5WpUW%G8eX2M#TRKT*H&)#5^7O3XF=AoG?Bh&GATuK)u2O%01VY8IDBu0`eNQ)ZX@Dear)BC\M`KRJM45\HIlt#57QSYS(I5:@MZm<Vn?E0aId]aS>u1ndDJlbk3[ph"*uHqtH*F)t\1Qd1SYMFSm``k9HE!3qhV=3Wap'.JiB:V$hJSr"9_H<V^ABeHIDrblcZ$nH[><DZ`Dq)u),a]`&E[`pr0k8--.)E<55#PQ&8c-4nr3[eXh:hY7-L:'^^p/KZ81!s2*h"8Y%]Yk9IN3V4NdcLjm;#7ZLu9D0n<Q5FfN2YL>\(A00\(&13(CC29pEXp=sN:O[q?5=d\56&n2D?Tb(Op/>]eHaCkNqg(!U%o]43s2;k\,-R%W<Pif@M>+*3qpDh1A@+5jo_gZ2%i%P,kitghY#.l<r?S3L[H%9>Q4@!mg(l_%f0-.0*R^H`Tq^Y5Q"%GFnOSuW=Y[/\Ids,QL\ksg^tk-C]lj(>mB3V5Qejml3*sM_"m%,WVU`@L]PQuaR3p9Z3J,=`XV_Bh"n&PQh)U,o`IJLXqr)_JbB0Cjn^%s_Yc:kF8^q5V#ZDU+p@3Dnat$"*V[_4Q1XeU!rg+3./3HCEt!U$+VS,\%d_FAfFfCrJHl;\2$d^hbPu#JM#*)07/Xq6RK#2TG5cslU][%UU@k5S.g:(lg](UY5Pkce,m<?'N"C!k3;A."C&d`#A,][+d0ntC:]DaFq["S^hs3N*dIpqq]_\o=hu`$D6MgioL]9F9U_.)0)>/=V5RS@rkR49i3t4R[b60?tD#aS/=9J(+W<))fk4LM/i;e`R2#OcSQjS6@W9hb*,P;B@8cla!q?#.D'*7=np'#jADtZ!nnH%qNfE%<HYnf)$3;>B6o`:<Xeb9sbr8_u)\bQ*[7eY"'a:&Y!KDN.Y'`t)of'p-P'*aQZhZBAL$2s/f2#=cpbl%,6a9&C>&I*jo2tLt^JH^&oea"g^KD$2[V?7V?M$!bQq].9kH3##k>O:5O6O"t=c0B!J?ke_1;=3!7q$(F<')jctFT2UKh=_lYAdK4Zrt+B0ZM.H'quYaGna4a5Q3AH"F9K&R)?-c8)"lf#MtjaXVZWIgM$5.8`;-Y1#n#Yjn`sZamIrC62?e[@g'!r:pC:s;Vtjj*=n1VV.g%C+BAk$=^_CAhNqTmV#9+3YJG\:1q>I,cjnQ%Xh%cC`(_\>)5kbcn_ZgGCcKVG6M?quGbll;aOnN)YiqHIk2>2pkT(kJI[0,q)q#LZmBb<o3Mt.M\D"VH<C]t@R\dj2WS-m%.bP2hN,Pc]jaRpkb<pKqklO_SSB`8tN.3.UP`Tp&!q!Z,o^\FW,LDJkRM\l6q"o#/'l4+@)%fR7K<![:,nbg)lVu\FCir,GeXmaL+^&YVEXpH]ZkRUhj_\#]I-j-pOUA:\djT^86\K(,0%0]E#+oAbKchZ5tfC4D"iq:%c3Y)Z5B)QcPo*HB9pY'0b_=;Ci$Mf/XarYL-`=YZI"nHF58JW!<-O;^?6j7Gn;uW`g-Na/A>5:&iHN.%FJcsC$^&6+D;A3%.V[\4W/H\iMan^/ZeHS>0D[ldK<WNs6?22kJ"oA,u2#j`qQiOMq"q"lSqY'pcQN8o8:u_Qa8H54FEV\&j`>.h3cOts2.de_e<YQMA3tEhB+U,CW[/GUTQh29p`VB!:YQGaP636WT@Je+=/H(;!/dfes8Hd6<_>7I5ZiB+UI19rSqYIYqir-(\A,/IjMudJl%K5&\<!q(ap@XIC4o\U9a8sEd<;bD<@/5Ze\-K>'"Vh"5l1ZA&2[tE,edR99q$.H&#R'i)K_u&L(]JUgiW?gq_Yjf=X90[Yci"RSVZsF4..-O$Sb6sp$j.@DKE;;T!Y=l"1_!R_p%XIFRK4f4"Uc"D.2R=D2YYVr$3Q7kYOOOYFRE93b4_V(7/^?e8.-+V#6INu)[VPi%0*F;I0N((nc=^W')O?]defFB!W3W6n,P9AnHOaX_>W?fK*s%5:$hfTan/X/Y67K+bQ7_:)?^BHGQ\QXHh.[H\,rc3>jWa7!Z*[(r:k.(ZjcI1'E:l7POI0$=7eQUOUKM:HOQ\oCAWGVNsmK'pZ]d&4ULiSQN@m,p@RPHWqrP*:\C1C$Ng%p*<m>jErm!J4:U-NU@`U/"n>%WEWjbtAHgfWBa#4L<WA`jIJI1Enbs:-@H>`2n.P@$R/Z(DRKb\QC&A_O<9WrHg@K1e2AgT8I1Zf*Onsq:'FV5A&FkqcH3=?DK*,<[c2[D5"o'kd_=rlSOnhNIO:%-R&.f-HH40WH2$.%a_"@^(J-(c1cj"ga,QmQ7SGX#ns8*o<K`QZ*jRqR/EXkt[iW'qu@0hVf`oN3bea"pR<q^241`Q&WaV$<8W;8%HZ4Z'g%0Dk!Jd#]S*W9LKTDnE^DYl4$Vtm.h.KFGg`WN:=FqIdSTEoZ7FTq[>CA$$O2?V@uZ3^U(OoW#gc26f*0D6@d-i%*O@K7DF[1!'L@/^`;5QgESDsren%f2gY@eN_++;c$SBb7NMU%Aim4V9X@^BBr1p&jIMV>gGmVYb9.0)@mIh#309ecJV?HNOlG@g0FAZibgRSd4'P<qd7%Rf;O<K*u)YMZ,O.!<>UgV>3^ai;aK$a7t5hGRNC@')KoK"odc_D@nl9Du:[R(&>HNj:+r>0b;)!C\G[HQ4jYn<!$&*:CV(/Q3@?qa8(ngWrA:n0`\oiFoPeCdfW_`Wq:*%'F(W>j7P\8qXYd,ao_#.C]OqC3W@G%'G0^5+S\D-oE#L!:^_:C\.M.?rqbFGj8n6H#Ph1'YOQ-&m.C_gWW-)k!!aVc*WIJb]*i36+7Ym*"q[Tt^Ab,BVXu81AH3\\K`!1h>l=<fAI-*`p%KX-o*ad_9DP(UEUoCl3WEdm;#a/cf)69X_$q(W(&GrKD[9YG\Ht=jh%;7Ks7b+CklRRbV=s9]K(N7K]_-dYmfl_cT+'HtM#UQU\I-ODLBI#JBF$3Sk6:G_l2!gF-2VKk*s.3$N<"_/3Xb6PQN@Wq(CE>!ci*,/.eC4F-N.KMK_pT'DY/)JoEtZtC&h!&PQ"&B,R!u58Ihg3Gl=H=/HOH3)$>d%i;^:^8GDEU(DMnp_[k'-m/=!eL(2)bqtVcF])^4I&H!%BW;j@%eFuuXo(npP3V\7*#6]2P&dQ&Jh[%-mCBf=jdI7M<jnD79EsM(,?hj`tJHg`&6Moa<UAj'L,QVQQM@4S?mJZJN-iE!5:_D.[1&QCs&G@jb<=t&>N=aA;9DWN+5n[2bScq7[,QX8+[J5slQ2J_ubPEh&N;Y%+Yl265)$Jmu_?1eY,R=hQWs\CqZ1O@Yk4MU;YmT+U$2^.^dKkg1LAEJ$9*$dLq>Gd;$k+KtCAA;;Du]M5U_-<+I/VXf_=hgV2[lDF#S?G)L&7kL2@u;JoE66#>X&=Olp!&M!'fiGUq_h*.6[h7a1[n4(-bsk5ek>-kJ*-XYe>6]2F#eD#sPk]$pQn91-[*F*PR2@)SgiC7R0rnV8$D1CHjHGe3:h9ID.6+!C.l)o"5VD&O/I>"M]Pe*58&3-,6n8O?YR/1r&]kaM0)c@(lqWl9:C:;a8._Jj6#7nN<ck\\?6$5!>tHAj6e1oK9Fr9"kuqgq?oN.R1W;QTp3!/\d)=GX"4c#X;K$GJB^'J\EUhKg5,'34qJ[d6AVfUq^4:%)"27NBS2=aZaIO]K`6Bf=kTP&jJko.m7_BL-DG-2niC@a?5,J/j>-mLcjrIO$7N-con*iPeMp[Lq^eM>.mlfUHPA`ZTSt+YWUc8)F#>Q#JYZHJ3I-RF$;-Gd6<I0m_1K=e@di4A3QGU,sL.%4h`@9bJ)Nk!^2(Gr4YWNWBH$V_E1%*Y.g[''>70Te\5;<Y!#MNnib@Wrk/NI+?l>i/A5\o\it).pq%[&/3`BQpc\:'00Hsm\j!t-]tBKH7_JL,PJ1f+G.i\NniN(,:-V#,%DOe[XZR5s&AKMKBY8V$9YOVJ8%kAUn31+p<k_pZ$,"l-n[lt@[^d.uS\%^OF1eHe'>0__34r:)r&_G[ah+VBZb+T-TtBX%RloL569A)T(HUKGpH(]j2S1^*Nk?C]Y<,e".D4o>qDt\+$p+1HTKG<XbIbod9Y,QNr&]%TnN2Yc?T[8a&3OC.OZ^7S_Ro7kP.Q;J,J4Mj3P6uW"$FRiG<8@'*B`ihe%9][0t]Ui>ru)N*kZt]]t7rt&Na+TM`f[uN4U@irjoKr`4MUim^p0=KY%]f$+sCS&3OT+I_==p%_GCNE')'SGWWJ1qE"o#(H[VD$p5;VqR^(BHG$lUMS*L'OLr3'h`3tB$p:/P5!,r_KtQj-"?UXQ1H^>,*'6KbjL]Io1qZ9]hn/L\9=\9>h`K"p8A)gd<kQSs)*A.RRQTDVm^lt@-9Rn!s#Wb8c+>r/#<`qijh'"Lhmt$d$9L<USij2$$p8PZqDo-E.D.FVS2q,#nN11M1-DiM(;/0$%m6PRot/T!`4L0PUcd@IV7>1]b.73VZTE5$Ge9m&5<LJKJNJ&u"$BQ[K07h(r&`6H:;"Y7:coH8?TV\LJ%Rn50g#Qk)8%tM^:?`iC-7,_cT;q;W]ML?!'SNSFh0d,cTKeK4M4e3ZFlU?jug/&::ib$'YVh"o"$-c$GGf8>eFp!#<ct@;EY><::k"K&Nrca76ShH69V(RP<E[K^qH%sRQ@qh'0?]TFuucITKH#T5eZhGI_<gPpcG>f7D(V_+MF6+0Y:qdY<6jBYWC+V8\FUeJ%Wcq$9bkd$9c@/\\*#Yk.@4JS352s(d1N:$U+NZ#J==0%D7PfWk2f22a%R(#Wr31c9-,$CqP:B,JCadEBAN376M3e:H_P"NBBo#rjt)rd6-&8bdnt)FM#R)_n,uog:D-K&Np9/OM-%XP!,n(gV1b&ot?L".6dA;s1Ra*1-MQ/n@]9>8[i!lOhPBu83E<R4?d]tZT\66?,!\WaZb(5-b^W<,ejnLO?_Y4SNaN.7_]e3M`n"u!PZ7(ffj,Q!koFK&AM0YMnXRe^qLet&O,9jf0'34L:r[nJA$@d,/8%(Tfs*p!'Z@hm(M3/dle;/N]sIV4$HM]oKG^]#JNtl"[4U@KY.gm%),UH^qMu4CqPm0#f#6W*^/\3r4VZm5sP.@$,/mUBtOrII(k?R(d;>[K>+Z&%DFobUcuK-"$_F5<kV[/eA,Gt%7'k*cbQ?`b!,MEQG4,4bWn7RlG#.;n\7!^T>*I`4i!/Q)o1G8`^"0h#XR9;'u111$GT*V%R7r!&\Ub'SNnsbT>->l90gTd)aRW(Kg:%u(;S?XahMf0pcthnOhUcg5<ctW7R..WGJD5mr]enY'gXnXA&%VI'L;#S\%q:I_a*qMjM+Pa$,@2I%7&pJID6I73Bp*M#<sg^m6FoG>X7`qUVF9_SA6/T4M[QC#Jc#c!^Z=1.mP1Ydlt*bGX-0'.6ntVOMG-@:VS_Ldm-oAVa"Rn!5Z)sQG6o4F$>o(UqU&^ffeiGpV#NP69^f@3PJE#B0I@R!PVG>,!Fsg;Eb=T$U,L&.DEQ\!5<D>P<QFPrk3Q(KYG+;:d2YqYWWN2fY,/[2SObC;SKi)<P?&I6,$L,gH@3&QbJ)Ebrnlf#/1V/(VXGo83MRFK>')H'#">J]KAHsMS=fZQTfGtm^\up-pFIMJj.]_OLsg)*kouK_EB/M&AA-RKYH`Um6%C6`4T1EEB^=HL-:0Y]t61k9g5tC,//0n!^0#49>0_F#JN1@?9[($!Bur>1VT`<loT&p&&4h47Qs';!!(3FZ92E^'gUZ%Sj!2g$pVdH83`,:G!Q&%rkR;c&\U0?Va%;&eNXpSKY=?</3on6`P9To>JQ4;gV/X/"@+'$#J^(cMSF4*J\C82M85m>B>:Q"e3EEsg;(+,m(Ub/#=-;M<'M-Js$)SNmlrW[4$[iHdCtg4KKnC('>MPup:hLD%1E(:qYnhK#mBu.#ON!FnHnL(nF5NM$j6;*liRJ%#5@Te"TA`Gp&,@+%N>-9n&+Z$!sT;=z!!!?)p&+Uer;S;M!rqltp@n_*rrDlc!<`3"nG`apqZ?j%p$i%T#mUP2!V$Wsnd53hquR#ps5!PSz!!)uk%f["8$4652rqd*#qZ[9'$3CV4"mZcu%KHV7%1N:9%Lr4+r<iW(oEb4(#p/d6p!34;rqR0*z!!)ESr;H*h'+Fg;(A?/e&GsiLqWAFX-0F=l#7CV?r<NE>dI$6%oa:U%#jq3]oDK(:$-`nb,ldNNz!!!3!#5&?6q>1*jrtP_=q%W\q"mcZP!Wri+";(J*r:pp(%Is?"p@\e"!!*B$&+]bQlj*LrrT=4[z!!!QAm0i7inF%F]&I[h3$h<WsoCM>6k7dFtmgSgXkjf#pmJ.1u%.ao=%eU>"m1BpD&g7hL'+OLDzz!!*$!!!*$!!!!$"rrE'!!!!$!!!!$!rrE'!zs8W*!!!!$"s8N'"!!*$!!<3$"zz!!!E5%ikBNqsW>,"pk//$jlP7qr$o8jU)J$%0Ptkq=Osn":tn1rq6^+rtG\@#lP,;#SmF.#5A0&z!!)g!q#)K6p&<kOklq4Wn,W"Uo*Z$;q\/Vmo_T1(&.&U'"T\u4%1E+,oDSRj#QtM'm1oR"nHK'cz!!*$!s8W-!s8W*!rrE'!!WW3"s8W,us8N*!s8N*#s8N*"s8N'!s8W*!s8W-!!<<*!rr<'!s8W)uz!!)!Bs7H?U!!;9T":,M(nFZYXmf<Lq#6Pk?r<ri.n-03n#RD.8!X\-#mKN1i!sftW!W<kcq$I!-z!!)Zuroam["pOZ0qYUTt#m9_u!XT4m&+oVO'b19RqY^Nn$189-#O(a^$Ma_c#Rh(Tq?I$7qss[oz!!!fP!s8Q<r;Y:O!!):#&-:&f*<#p1q#9pp"muEm$i]c%mKMk[#4rE#&dAg5q?ZTn"qBQ9$O$%mz!!!*$&L.8O"n^[5"8E'&lPK0^"7YjJlh_%P&*N3H#l!od&0:fOpAG3V"<mgS$k*"9'-7)Hp43>Lz!!!Z1&GZJ6k5F*?$j[d_#6#VD!#Q<7qYpcfrs@rf(B=^=#NYp]%H[`Sp@&b!#p0$Er;$1&!V?@#z!!)fmqt'X`"98r-rrMN^q#UBr"8;C#r;Hfur;I*'lMgDVq?mB2m0!4g"UY)$r=8r#":,(u3"?#kz!!)BQ#Q!]D*<FS!r<a,2-78<Xm,.48',(B3!;tX`me-qY)$Km"qUOs*pDro"!#+Djkn"R>&,6%sz!!!9*&+]K-isFlI#l=6%"8hTi'*Jg%pB2$6jST8Q$P)S/q@E_o":>P7q>:R<!;$Bsr=o.q"Uk)/z!!)EVp?N7\#SK'1qsk$ho^r^U%M]`S)u9O%s8EW:#S?q+!UKFc!Whidq=F1^p^m9*qr.;d!!EB%z!!!iPr=Si?$N02iq#1"&!XA;lnGrC]&,tr5"TJ)nlLjfV"Qet[mICea*<-<=quu0ao+:O#$jch"z!!!-(q?m3,o_T$i&I\FA#lXVurVu]9q#gEroa10aqtC@#q"b@.&+fr4"ni?!q]?CY!=/u,%1N:Kz!!)?U%e1;$&eEn2l1#AQmM6'"'Eo%$'CZG"&+(/E&J=^#&/"[$lk8jQoF^@#%ghdmo'GiTh;o(1z!!!>srpKaXr!*Q1"SDj1n,E1M!=/r$"7Z1>qYU6i$j$/>q>Kq'k5k2m$N9YuoChq[%0Hb9$kNC>z!!)T_&+pY/%1Vb=oC`[eoF;E'%gWX\'ClV*%.t);%1;h!$jlY$ndY<aoF(4"$jZ=lp#bfXoC!%Xz!!)WV#4h]fr<i8j$M<rl!$D18#R(MF(@_q>pAYR/#6ao6mIpVKhu*!E$M3`_obIE=#jr0)q<Rk\z!!!-1k5G,T$Mt\D!X&9"#P79`li@%n!Xe,n!X8]4nFHnlpYu#[&,#i,"S2KToa:[2%/Bo6!"AGtz!!)?c!VQ!S"U>,-q;qtU!q?BY&+0cQ!;>aQ$1Rp!#6aDe"o%WeoD\LR":+Mt!?(J5$O[%7i:Zs>z!!*#sq?I'%$L\f(!XJZ'"T7isoCMAOpBgQ`%0H(so'?M[!Vc]hlLY5Ur<WYrrWi,s#7C_;q[iN*z!!!60nG_GOl3.%Ys8;*\"RG[l!>"o4nHf:4k5Ykeq%<W7lN%=c(^^NN#k%j!nJDW>"Ub_$!tY/,z!!)s&lh(2[#Qa/rkPkGYp[o:#!sSc9!=Sr%"mZ[!r;-0U!!2itr;H]V$O$Rm!t+hn!@.%=p\"dTz!!!<&!r`Q.%0>hrr!s#2qY^^,&JWs]"Tdie!X\uAeGKUL!r`6/#3u?fp]gp2!X02-q"jOo&e=s@z!!)Eap%\I_rVuQd$N'kqk54WL#Rph<q@E?*"S;ik!=o/5r"T*,"9ei7jT+KBq[jMB#l+MdquZm)z!!",A!!)&ki&O-982\n-&AgR=2q1::^!=L9_:l2NgOD)s7)$<!.Fe"##;
const NNQ_MAGIC: [u8; 4] = *b"NNQ1";
const CURRENT_NNQ_VERSION: u16 = 7;
const _cC: u8 = 23;
const PREVIOUS_NNQ_VERSION: u16 = 5;
const OFFSET_NNQ_VERSION: u16 = 7;
const LEGACY_NNQ_VERSION: u16 = 4;
const NNUE_SCORE_LIMIT: f32 = 5000.0;
const NNUE_MAX_OUT: f32 = 1.0;
const NNUE_SPARSE: usize = 122;
const _E: usize = 8;
const Nh: usize = 58;
const NNUE_H1: usize = 32;
const NNUE_H0_PAD: usize = 64;
const NNUE_H1_PAD: usize = 32;
#[derive(Clone)]
struct NnueAcc {
    black: [i32; Nh],
    white: [i32; Nh],
}
struct _W {
    sparse_scale: f32,
    df: [f32; _E],
    ds: [f32; _E],
    output_bias: f32,
    act0: f32,
    act1: f32,
    _cp: [i32; Nh],
    sparse_weights: Box<[i16]>,
    dense_weights: Box<[f32]>,
    hidden_scale: f32,
    hidden_bias: [f32; NNUE_H1],
    hidden_weights: Box<[i8]>,
    output_scale: f32,
    output_weights: Box<[i8]>,
}
impl _W {
    fn load() -> Self {
        let bytes = decode_a85(NNUE_MODEL_A85);
        let mut p = 0usize;
        assert_eq!(&bytes[p..p + 4], &NNQ_MAGIC);
        p += 4;
        let version = take_u16(&bytes, &mut p);
        assert!(
            version == LEGACY_NNQ_VERSION
                || version == PREVIOUS_NNQ_VERSION
                || version == CURRENT_NNQ_VERSION
        );
        assert_eq!(take_u8(&bytes, &mut p), _cC);
        assert_eq!(take_u8(&bytes, &mut p), 1);
        assert_eq!(take_u8(&bytes, &mut p), 1);
        if version >= OFFSET_NNQ_VERSION {
            assert_eq!(take_u8(&bytes, &mut p), 1);
        } else {
            assert_eq!(take_u8(&bytes, &mut p), 0);
        }
        assert_eq!(take_u32(&bytes, &mut p) as usize, NNUE_SPARSE);
        assert_eq!(take_u32(&bytes, &mut p) as usize, _E);
        assert_eq!(take_u32(&bytes, &mut p), 2);
        assert_eq!(take_u32(&bytes, &mut p) as usize, Nh);
        assert_eq!(take_u32(&bytes, &mut p) as usize, NNUE_H1);
        let sparse_scale = take_f32(&bytes, &mut p);
        let mut df = [0.0; _E];
        let mut ds = [1.0; _E];
        if _E > 7 {
            ds[7] = 64.0;
        }
        if version >= OFFSET_NNQ_VERSION {
            let _cE = take_u32(&bytes, &mut p) as usize;
            assert!(_cE <= _E);
            let mut i = 0;
            while i < _cE {
                ds[i] = take_f32(&bytes, &mut p).max(1.0);
                i += 1;
            }
            let _cF = take_u32(&bytes, &mut p) as usize;
            assert!(_cF <= _E);
            let mut i = 0;
            while i < _cF {
                df[i] = take_f32(&bytes, &mut p);
                i += 1;
            }
        } else if version >= PREVIOUS_NNQ_VERSION {
            let _cE = take_u32(&bytes, &mut p) as usize;
            assert!(_cE <= _E);
            let mut i = 0;
            while i < _cE {
                ds[i] = take_f32(&bytes, &mut p).max(1.0);
                i += 1;
            }
        } else {
            if _E > 0 {
                ds[0] = take_f32(&bytes, &mut p).max(1.0);
            }
            if _E > 1 {
                ds[1] = take_f32(&bytes, &mut p).max(1.0);
            }
            if _E > 4 {
                ds[4] = take_f32(&bytes, &mut p).max(1.0);
            }
            if _E > 5 {
                ds[5] = take_f32(&bytes, &mut p).max(1.0);
            }
        }
        let output_bias = take_f32(&bytes, &mut p);
        let act0 = take_f32(&bytes, &mut p);
        let act1 = take_f32(&bytes, &mut p);
        let mut _cp = [0; Nh];
        for slot in &mut _cp {
            *slot = take_i32(&bytes, &mut p);
        }
        let sparse_weights = take_i16_box(&bytes, &mut p, NNUE_SPARSE * Nh);
        let dense_weights = take_f32_box(&bytes, &mut p, _E * Nh);
        assert_eq!(take_u32(&bytes, &mut p) as usize, NNUE_H0_PAD);
        let hidden_scale = take_f32(&bytes, &mut p);
        let mut hidden_bias = [0.0; NNUE_H1];
        for slot in &mut hidden_bias {
            *slot = take_f32(&bytes, &mut p);
        }
        let hidden_weights = take_i8_box(&bytes, &mut p, NNUE_H0_PAD * NNUE_H1);
        assert_eq!(take_u32(&bytes, &mut p) as usize, NNUE_H1_PAD);
        let output_scale = take_f32(&bytes, &mut p);
        let output_weights = take_i8_box(&bytes, &mut p, NNUE_H1_PAD);
        assert_eq!(p, bytes.len());
        Self {
            sparse_scale,
            df,
            ds,
            output_bias,
            act0,
            act1,
            _cp,
            sparse_weights,
            dense_weights,
            hidden_scale,
            hidden_bias,
            hidden_weights,
            output_scale,
            output_weights,
        }
    }
    fn root_acc(&self, position: &Po) -> NnueAcc {
        let mut acc = NnueAcc {
            black: self._cp,
            white: self._cp,
        };
        for cell in position.black() {
            self.ap(&mut acc, Co::Black, *cell, 1);
        }
        for cell in position.white() {
            self.ap(&mut acc, Co::White, *cell, 1);
        }
        acc
    }
    fn ap(&self, acc: &mut NnueAcc, _c: Co, cell: crate::ac::Ci, delta: i32) {
        let raw = cell.as_usize();
        let black_row = if _c == Co::Black {
            raw
        } else {
            self::ac::Cc + raw
        };
        let white_row = if _c == Co::Black {
            self::ac::Cc + raw
        } else {
            raw
        };
        let mut j = 0;
        while j < Nh {
            acc.black[j] += delta * i32::from(self.sparse_weights[black_row * Nh + j]);
            acc.white[j] += delta * i32::from(self.sparse_weights[white_row * Nh + j]);
            j += 1;
        }
    }
    fn evaluate_with_acc_bits(
        &self,
        sb: bool,
        sh: &_cS,
        _t: u64,
        _s: u64,
        _cB: f32,
        _cN: f32,
        acc: &NnueAcc,
    ) -> i32 {
        let dense = dense_features(sb, sh, _t, _s, _cB, _cN, &self.df, &self.ds);
        let base = if sb { &acc.black } else { &acc.white };
        let mut h0 = [0.0; Nh];
        let mut j = 0;
        while j < Nh {
            h0[j] = base[j] as f32 * self.sparse_scale;
            j += 1;
        }
        let mut d = 0;
        while d < _E {
            let v = dense[d];
            if v != 0.0 {
                let row = d * Nh;
                let mut k = 0;
                while k < Nh {
                    h0[k] += self.dense_weights[row + k] * v;
                    k += 1;
                }
            }
            d += 1;
        }
        let mut q0 = [0i8; NNUE_H0_PAD];
        j = 0;
        while j < Nh {
            q0[j] = quantize_relu(h0[j], self.act0);
            j += 1;
        }
        let mut h1 = [0.0; NNUE_H1];
        let mut i = 0;
        while i < NNUE_H1 {
            let row = i * NNUE_H0_PAD;
            let dot = dot_i8_h0(&q0, &self.hidden_weights[row..row + NNUE_H0_PAD]);
            h1[i] = (dot as f32 * self.act0 * self.hidden_scale + self.hidden_bias[i]).max(0.0);
            i += 1;
        }
        let mut q1 = [0i8; NNUE_H1_PAD];
        i = 0;
        while i < NNUE_H1 {
            q1[i] = quantize_relu(h1[i], self.act1);
            i += 1;
        }
        let dot = dot_i8_32(&q1, &self.output_weights);
        ((dot as f32 * self.act1 * self.output_scale + self.output_bias)
            .clamp(-NNUE_MAX_OUT, NNUE_MAX_OUT)
            * NNUE_SCORE_LIMIT)
            .round() as i32
    }
}
#[inline(always)]
fn dot_i8_h0(a: &[i8; NNUE_H0_PAD], b: &[i8]) -> i32 {
    let mut dot = 0i32;
    let mut i = 0usize;
    while i < NNUE_H0_PAD {
        dot += i32::from(a[i]) * i32::from(b[i]);
        i += 1;
    }
    dot
}
#[inline(always)]
fn dot_i8_32(a: &[i8; NNUE_H1_PAD], b: &[i8]) -> i32 {
    let mut dot = 0i32;
    let mut i = 0usize;
    while i < NNUE_H1_PAD {
        dot += i32::from(a[i]) * i32::from(b[i]);
        i += 1;
    }
    dot
}
fn nnue() -> &'static _W {
    static MODEL: OnceLock<_W> = OnceLock::new();
    MODEL.get_or_init(_W::load)
}
fn dense_features(
    sb: bool,
    sh: &_cS,
    _t: u64,
    _s: u64,
    _cB: f32,
    _cN: f32,
    df: &[f32; _E],
    ds: &[f32; _E],
) -> [f32; _E] {
    let (own, opp, orient) = if sb {
        (sh.b, sh.w, 1.0)
    } else {
        (sh.w, sh.b, -1.0)
    };
    let _ = (_t, _s);
    let mut dense = [0.0; _E];
    let mut raw = [0.0; _E];
    if _E > 0 {
        raw[0] = (i32::from(sh.b.edge) - i32::from(sh.w.edge)) as f32 * orient;
    }
    if _E > 1 {
        raw[1] = (i32::from(sh.b.cp) - i32::from(sh.w.cp)) as f32 * orient;
    }
    if _E > 2 {
        raw[2] = 0.0;
    }
    if _E > 3 {
        raw[3] = 0.0;
    }
    if _E > 4 {
        raw[4] = (i32::from(sh.b.liberty_count) - i32::from(sh.w.liberty_count)) as f32 * orient;
    }
    if _E > 5 {
        raw[5] = (i32::from(own.singletons) - i32::from(opp.singletons)) as f32;
    }
    if _E > 6 {
        raw[6] = _cD(_cB);
    }
    if _E > 7 {
        raw[7] = _cN.clamp(0.0, 350.0);
    }
    let mut i = 0;
    while i < _E {
        dense[i] = clamp_feature((raw[i] - df[i]) / ds[i].max(1.0));
        i += 1;
    }
    dense
}
fn _cD(_cB: f32) -> f32 {
    let clamped = _cB.clamp(0.0, 350.0);
    ((350.0 - clamped) / 350.0).clamp(0.0, 1.0)
}
#[derive(Clone, Copy, Default)]
struct _cs {
    pc: u8,
    edge: u8,
    cp: u8,
    liberty_count: u8,
    singletons: u8,
}
#[derive(Clone, Copy, Default)]
struct _cS {
    b: _cs,
    w: _cs,
}
fn _cU(_t: u64, _s: u64) -> _cS {
    _cS {
        b: _cT(_t),
        w: _cT(_s),
    }
}
fn _cT(bits: u64) -> _cs {
    let cells = gm().cs();
    let mut out = _cs {
        pc: bits.count_ones() as u8,
        .._cs::default()
    };
    let mut active = bits;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let g = &cells[bit.trailing_zeros() as usize];
        let neighbors = g._cw;
        out.edge = out.edge.saturating_add(g.ed);
        out.cp = out.cp.saturating_add((neighbors & bits).count_ones() as u8);
    }
    (out.liberty_count, out.singletons) = _cW(bits);
    out
}
fn _cW(bits: u64) -> (u8, u8) {
    let cells = gm().cs();
    let mut active = bits;
    let mut liberty_mask = 0u64;
    let mut singletons = 0u8;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let neighbors = cells[bit.trailing_zeros() as usize]._cw;
        liberty_mask |= neighbors & !bits;
        if neighbors & bits == 0 {
            singletons = singletons.saturating_add(1);
        }
    }
    (
        liberty_mask.count_ones().min(u8::MAX as u32) as u8,
        singletons,
    )
}
fn _cV(mut side: _cs, old: u64, new: u64) -> _cs {
    if old == new {
        return side;
    }
    let cells = gm().cs();
    let rem = old & !new;
    let add = new & !old;
    let mut cur = old;
    let mut active = rem;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let g = &cells[bit.trailing_zeros() as usize];
        side.edge -= g.ed;
        side.cp -= ((g._cw & cur).count_ones() as u8) << 1;
        cur &= !bit;
    }
    active = add;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let g = &cells[bit.trailing_zeros() as usize];
        side.edge += g.ed;
        side.cp += ((g._cw & cur).count_ones() as u8) << 1;
        cur |= bit;
    }
    side.pc = new.count_ones() as u8;
    (side.liberty_count, side.singletons) = _cW(new);
    side
}
fn clamp_feature(v: f32) -> f32 {
    v.clamp(-1.0, 1.0)
}
fn quantize_relu(v: f32, scale: f32) -> i8 {
    ((v.max(0.0) / if scale > 0.0 { scale } else { 1.0 / 127.0 })
        .round()
        .clamp(0.0, 127.0)) as i8
}
fn decode_a85(input: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 4 / 5);
    let mut chunk = [0u32; 5];
    let mut len = 0usize;
    for byte in input.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'z' {
            assert_eq!(len, 0);
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        chunk[len] = u32::from(byte - 33);
        len += 1;
        if len == 5 {
            let mut value = 0u32;
            for digit in chunk {
                value = value * 85 + digit;
            }
            out.extend_from_slice(&value.to_be_bytes());
            len = 0;
        }
    }
    if len != 0 {
        let mut i = len;
        while i < 5 {
            chunk[i] = 84;
            i += 1;
        }
        let mut value = 0u32;
        for digit in chunk {
            value = value * 85 + digit;
        }
        out.extend_from_slice(&value.to_be_bytes()[..len - 1]);
    }
    out
}
fn take_u8(bytes: &[u8], p: &mut usize) -> u8 {
    let value = bytes[*p];
    *p += 1;
    value
}
fn take_u16(bytes: &[u8], p: &mut usize) -> u16 {
    let value = u16::from_le_bytes(bytes[*p..*p + 2].try_into().unwrap());
    *p += 2;
    value
}
fn take_u32(bytes: &[u8], p: &mut usize) -> u32 {
    let value = u32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
    *p += 4;
    value
}
fn take_i32(bytes: &[u8], p: &mut usize) -> i32 {
    let value = i32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
    *p += 4;
    value
}
fn take_f32(bytes: &[u8], p: &mut usize) -> f32 {
    let value = f32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
    *p += 4;
    value
}
fn take_i16_box(bytes: &[u8], p: &mut usize, len: usize) -> Box<[i16]> {
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        out.push(i16::from_le_bytes(bytes[*p..*p + 2].try_into().unwrap()));
        *p += 2;
        i += 1;
    }
    out.into_boxed_slice()
}
fn take_i8_box(bytes: &[u8], p: &mut usize, len: usize) -> Box<[i8]> {
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        out.push(bytes[*p] as i8);
        *p += 1;
        i += 1;
    }
    out.into_boxed_slice()
}
fn take_f32_box(bytes: &[u8], p: &mut usize, len: usize) -> Box<[f32]> {
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        out.push(f32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap()));
        *p += 4;
        i += 1;
    }
    out.into_boxed_slice()
}
#[derive(Clone, Debug)]
struct _B {
    raw: String,
    values: [i32; 5],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct _A;
const CODINGAME_DIRECTION_MAP: [Di; 6] = [Di::East, Di::Ne, Di::Nw, Di::West, Di::Sw, Di::Se];
impl _A {
    fn parse_action(&self, action: &_B) -> Option<Mv> {
        let row0 = u8::try_from(action.values[1]).ok()?;
        let col0 = u8::try_from(action.values[0] + 1).ok()?;
        let start = gm().index_of_coord(Coord::new(row0, col0)?)?;
        let row1 = u8::try_from(action.values[3]).ok()?;
        let col1 = u8::try_from(action.values[2] + 1).ok()?;
        let end = gm().index_of_coord(Coord::new(row1, col1)?)?;
        let direction = *CODINGAME_DIRECTION_MAP.get(action.values[4] as usize)?;
        let cs = cells_between(start, end)?;
        Mv::from_cells(&cs, direction).ok()
    }
}
#[derive(Debug)]
struct _U {
    _cy: Co,
    prior_positions: Vec<Po>,
    last_position: Option<Po>,
    lp: Option<Po>,
    last_total_score: Option<i32>,
    lr: Option<Mv>,
    _b: u16,
    _cv: usize,
}
impl _U {
    fn new(_cy: Co) -> Self {
        Self {
            _cy,
            prior_positions: Vec::new(),
            last_position: None,
            lp: None,
            last_total_score: None,
            lr: None,
            _b: 0,
            _cv: 0,
        }
    }
    fn observe_position(&mut self, position: &Po, total_score: i32) {
        if let Some(previous) = self.last_position.replace(position.clone()) {
            self.prior_positions.push(previous);
            self._b = if self.last_total_score == Some(total_score) {
                self._b.saturating_add(2)
            } else {
                0
            };
        }
        self.last_total_score = Some(total_score);
    }
    fn time_budget_ms(&self) -> u64 {
        if self._cv == 0 {
            CODINGAME_FIRST_TURN_MS
        } else {
            CODINGAME_TURN_MS
        }
    }
    fn turns_played(&self) -> u16 {
        let own_turns_played = self._cv.saturating_mul(2);
        let side_offset = if self._cy == Co::White { 1 } else { 0 };
        own_turns_played
            .saturating_add(side_offset)
            .min(MAX_GAME_TURNS as usize) as u16
    }
    fn finish_turn(&mut self) {
        self._cv = self._cv.saturating_add(1);
    }
    fn record_own_move(&mut self, position: &Po, mv: Option<Mv>) {
        self.lp = mv.and_then(|mv| _P(position, mv));
        self.lr = mv.and_then(reverse_move);
    }
}
struct _V {
    raw: String,
    mv: Option<Mv>,
}
pub fn main() {
    let _ = run_codingame();
}
fn run_codingame() -> Result<(), String> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let my_id = _cz(&mut lines)?
        .trim()
        .parse::<u8>()
        .map_err(|_| String::new())?;
    let _cy = match my_id {
        1 => Co::White,
        2 => Co::Black,
        _ => return Err(String::new()),
    };
    let mut st = _U::new(_cy);
    let _ = gm();
    let _ = nnue();
    while let Some(score_line) = next_optional_line(&mut lines)? {
        let turn_started = Instant::now();
        let scores = parse_i32s(&score_line)?;
        if scores.len() != 2 {
            return Err(String::new());
        }
        let sh = _Q(&mut lines, st._cy)?;
        let _j = _cz(&mut lines)?;
        let legal_actions_count = _cz(&mut lines)?
            .trim()
            .parse::<usize>()
            .map_err(|_| String::new())?;
        let mut _k = Vec::with_capacity(legal_actions_count);
        for _ in 0..legal_actions_count {
            _k.push(_T(&_cz(&mut lines)?)?);
        }
        let position = sh;
        let turn_budget_ms = st.time_budget_ms();
        let pre_search_ms = turn_started.elapsed().as_millis() as u64;
        let search_time_ms = turn_budget_ms
            .saturating_sub(pre_search_ms)
            .saturating_sub(4)
            .max(1);
        st.observe_position(&position, scores[0].saturating_add(scores[1]));
        let gt = st.turns_played();
        let chosen = _R(
            &position,
            &st.prior_positions,
            st._b,
            gt,
            search_time_ms,
            st.lr,
            &_k,
        )?;
        println!("{}", chosen.raw);
        io::stdout().flush().map_err(|_| String::new())?;
        st.record_own_move(&position, chosen.mv);
        st.finish_turn();
    }
    Ok(())
}
fn _R(
    position: &Po,
    hy: &[Po],
    _b: u16,
    gt: u16,
    time_ms: u64,
    _h: Option<Mv>,
    _k: &[_B],
) -> Result<_V, String> {
    if _k.is_empty() {
        return Ok(_V {
            raw: "0 0 0 0 0".to_owned(),
            mv: None,
        });
    }
    let codec = _A;
    let result = search_raw_with_gt(position, hy, _b, gt, time_ms, _h)?;
    if let Some(bm) = result.bm {
        for action in _k {
            if codec.parse_action(action) == Some(bm) {
                return Ok(_V {
                    raw: action.raw.clone(),
                    mv: Some(bm),
                });
            }
        }
    }
    Ok(_V {
        raw: _k[0].raw.clone(),
        mv: codec.parse_action(&_k[0]),
    })
}
fn _T(line: &str) -> Result<_B, String> {
    let values = parse_i32s(line)?;
    let values: [i32; 5] = values.try_into().map_err(|_| String::new())?;
    Ok(_B {
        raw: line.trim().to_owned(),
        values,
    })
}
fn _Q(lines: &mut impl Iterator<Item = io::Result<String>>, _sT: Co) -> Result<Po, String> {
    let mut black = Vec::new();
    let mut white = Vec::new();
    let gm = gm();
    for row in 0..=8u8 {
        let raw = _cz(lines)?;
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
            let cell = gm.index_of_coord(coord).unwrap();
            match value {
                b'1' => white.push(cell),
                b'2' => black.push(cell),
                _ => {}
            }
        }
    }
    Po::new(_sT, black, white).map_err(|_| String::new())
}
fn reverse_move(mv: Mv) -> Option<Mv> {
    let gm = gm();
    let mut destination_cells = [mv._ad()[0]; 3];
    for (index, cell) in mv._ad().iter().copied().enumerate() {
        destination_cells[index] = gm.cell(cell).ns[mv.direction().index()]?;
    }
    Mv::from_cells(&destination_cells[..mv.len()], mv.direction().opposite()).ok()
}
fn _P(position: &Po, mv: Mv) -> Option<Po> {
    let mut st = Rq::new(position.clone()).ok()?;
    st.apply_move(&mv).ok()?;
    Some(st.position().clone())
}
fn cells_between(start: crate::ac::Ci, end: crate::ac::Ci) -> Option<Vec<crate::ac::Ci>> {
    if start == end {
        return Some(vec![start]);
    }
    let gm = gm();
    for direction in Ad {
        let mut cs = vec![start];
        let mut current = start;
        while cs.len() < 3 {
            let Some(next) = gm.cell(current).ns[direction.index()] else {
                break;
            };
            current = next;
            cs.push(current);
            if current == end {
                return Some(cs);
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
fn _cz(lines: &mut impl Iterator<Item = io::Result<String>>) -> Result<String, String> {
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
