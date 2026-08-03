use crate::board::{self, Color, Position, geometry};
use std::sync::OnceLock;

const NNUE_MODEL_A85: &str = include_str!("net.mlp");
const NNQ_MAGIC: [u8; 4] = *b"NNQ1";
const CURRENT_NNQ_VERSION: u16 = 7;
const NNUE_BOARD_RADIUS_MARKER: u8 = 23;
const PREVIOUS_NNQ_VERSION: u16 = 5;
const OFFSET_NNQ_VERSION: u16 = 7;
const LEGACY_NNQ_VERSION: u16 = 4;
const NNUE_SCORE_LIMIT: f32 = 5000.0;
const NNUE_MAX_OUT: f32 = 1.0;
const NNUE_SPARSE: usize = 122;
const DENSE_FEATURE_COUNT: usize = 8;
const SPARSE_FEATURE_COUNT: usize = 84;
const NNUE_H1: usize = 50;
const NNUE_H0_PAD: usize = 96;
const NNUE_H1_PAD: usize = 64;
#[derive(Clone)]
pub(crate) struct NnueAccumulator {
    black: [i32; SPARSE_FEATURE_COUNT],
    white: [i32; SPARSE_FEATURE_COUNT],
}
pub(crate) struct NnueModel {
    sparse_scale: f32,
    dense_offsets: [f32; DENSE_FEATURE_COUNT],
    dense_scales: [f32; DENSE_FEATURE_COUNT],
    output_bias: f32,
    act0: f32,
    act1: f32,
    baseline_accumulator: [i32; SPARSE_FEATURE_COUNT],
    sparse_weights: Box<[i16]>,
    dense_weights: Box<[f32]>,
    hidden_scale: f32,
    hidden_bias: [f32; NNUE_H1],
    hidden_weights: Box<[i8]>,
    output_scale: f32,
    output_weights: Box<[i8]>,
}
impl NnueModel {
    fn load() -> Self {
        let bytes = decode_a85(NNUE_MODEL_A85);
        let mut cursor = 0usize;
        assert_eq!(&bytes[cursor..cursor + 4], &NNQ_MAGIC);
        cursor += 4;
        let version = take_u16(&bytes, &mut cursor);
        assert!(
            version == LEGACY_NNQ_VERSION
                || version == PREVIOUS_NNQ_VERSION
                || version == CURRENT_NNQ_VERSION
        );
        assert_eq!(take_u8(&bytes, &mut cursor), NNUE_BOARD_RADIUS_MARKER);
        assert_eq!(take_u8(&bytes, &mut cursor), 1);
        assert_eq!(take_u8(&bytes, &mut cursor), 1);
        if version >= OFFSET_NNQ_VERSION {
            assert_eq!(take_u8(&bytes, &mut cursor), 1);
        } else {
            assert_eq!(take_u8(&bytes, &mut cursor), 0);
        }
        assert_eq!(take_u32(&bytes, &mut cursor) as usize, NNUE_SPARSE);
        assert_eq!(take_u32(&bytes, &mut cursor) as usize, DENSE_FEATURE_COUNT);
        assert_eq!(take_u32(&bytes, &mut cursor), 2);
        assert_eq!(take_u32(&bytes, &mut cursor) as usize, SPARSE_FEATURE_COUNT);
        assert_eq!(take_u32(&bytes, &mut cursor) as usize, NNUE_H1);
        let sparse_scale = take_f32(&bytes, &mut cursor);
        let mut dense_offsets = [0.0; DENSE_FEATURE_COUNT];
        let mut dense_scales = [1.0; DENSE_FEATURE_COUNT];
        if DENSE_FEATURE_COUNT > 7 {
            dense_scales[7] = 64.0;
        }
        if version >= OFFSET_NNQ_VERSION {
            let scale_count = take_u32(&bytes, &mut cursor) as usize;
            assert!(scale_count <= DENSE_FEATURE_COUNT);
            let mut i = 0;
            while i < scale_count {
                dense_scales[i] = take_f32(&bytes, &mut cursor).max(1.0);
                i += 1;
            }
            let offset_count = take_u32(&bytes, &mut cursor) as usize;
            assert!(offset_count <= DENSE_FEATURE_COUNT);
            let mut i = 0;
            while i < offset_count {
                dense_offsets[i] = take_f32(&bytes, &mut cursor);
                i += 1;
            }
        } else if version >= PREVIOUS_NNQ_VERSION {
            let scale_count = take_u32(&bytes, &mut cursor) as usize;
            assert!(scale_count <= DENSE_FEATURE_COUNT);
            let mut i = 0;
            while i < scale_count {
                dense_scales[i] = take_f32(&bytes, &mut cursor).max(1.0);
                i += 1;
            }
        } else {
            if DENSE_FEATURE_COUNT > 0 {
                dense_scales[0] = take_f32(&bytes, &mut cursor).max(1.0);
            }
            if DENSE_FEATURE_COUNT > 1 {
                dense_scales[1] = take_f32(&bytes, &mut cursor).max(1.0);
            }
            if DENSE_FEATURE_COUNT > 4 {
                dense_scales[4] = take_f32(&bytes, &mut cursor).max(1.0);
            }
            if DENSE_FEATURE_COUNT > 5 {
                dense_scales[5] = take_f32(&bytes, &mut cursor).max(1.0);
            }
        }
        let output_bias = take_f32(&bytes, &mut cursor);
        let act0 = take_f32(&bytes, &mut cursor);
        let act1 = take_f32(&bytes, &mut cursor);
        let mut baseline_accumulator = [0; SPARSE_FEATURE_COUNT];
        for slot in &mut baseline_accumulator {
            *slot = take_i32(&bytes, &mut cursor);
        }
        let sparse_weights = take_i16_box(&bytes, &mut cursor, NNUE_SPARSE * SPARSE_FEATURE_COUNT);
        let dense_weights = take_f32_box(
            &bytes,
            &mut cursor,
            DENSE_FEATURE_COUNT * SPARSE_FEATURE_COUNT,
        );
        assert_eq!(take_u32(&bytes, &mut cursor) as usize, NNUE_H0_PAD);
        let hidden_scale = take_f32(&bytes, &mut cursor);
        let mut hidden_bias = [0.0; NNUE_H1];
        for slot in &mut hidden_bias {
            *slot = take_f32(&bytes, &mut cursor);
        }
        let hidden_weights = take_i8_box(&bytes, &mut cursor, NNUE_H0_PAD * NNUE_H1);
        assert_eq!(take_u32(&bytes, &mut cursor) as usize, NNUE_H1_PAD);
        let output_scale = take_f32(&bytes, &mut cursor);
        let output_weights = take_i8_box(&bytes, &mut cursor, NNUE_H1_PAD);
        assert_eq!(cursor, bytes.len());
        Self {
            sparse_scale,
            dense_offsets,
            dense_scales,
            output_bias,
            act0,
            act1,
            baseline_accumulator,
            sparse_weights,
            dense_weights,
            hidden_scale,
            hidden_bias,
            hidden_weights,
            output_scale,
            output_weights,
        }
    }
    pub(crate) fn root_accumulator(&self, position: &Position) -> NnueAccumulator {
        let mut accumulator = NnueAccumulator {
            black: self.baseline_accumulator,
            white: self.baseline_accumulator,
        };
        for cell in position.black() {
            self.apply_sparse_delta(&mut accumulator, Color::Black, cell, 1);
        }
        for cell in position.white() {
            self.apply_sparse_delta(&mut accumulator, Color::White, cell, 1);
        }
        accumulator
    }
    #[cfg_attr(target_arch = "aarch64", allow(unsafe_code))]
    pub(crate) fn apply_sparse_delta(
        &self,
        accumulator: &mut NnueAccumulator,
        color: Color,
        cell: crate::board::CellId,
        delta: i32,
    ) {
        let raw = cell.as_usize();
        let black_row = if color == Color::Black {
            raw
        } else {
            board::CELL_COUNT + raw
        };
        let white_row = if color == Color::Black {
            board::CELL_COUNT + raw
        } else {
            raw
        };
        let black_base = black_row * SPARSE_FEATURE_COUNT;
        let white_base = white_row * SPARSE_FEATURE_COUNT;
        #[cfg(target_arch = "aarch64")]
        {
            let mut feature_index = 0usize;
            // SAFETY: each base starts an in-range weight row, and the loop
            // condition leaves at least four elements for every load/store.
            unsafe {
                use std::arch::aarch64::*;
                let delta = vdupq_n_s32(delta);
                while feature_index + 4 <= SPARSE_FEATURE_COUNT {
                    let black_weights = vmovl_s16(vld1_s16(
                        self.sparse_weights.as_ptr().add(black_base + feature_index),
                    ));
                    let white_weights = vmovl_s16(vld1_s16(
                        self.sparse_weights.as_ptr().add(white_base + feature_index),
                    ));
                    let black_accumulator =
                        vld1q_s32(accumulator.black.as_ptr().add(feature_index));
                    let white_accumulator =
                        vld1q_s32(accumulator.white.as_ptr().add(feature_index));
                    vst1q_s32(
                        accumulator.black.as_mut_ptr().add(feature_index),
                        vmlaq_s32(black_accumulator, black_weights, delta),
                    );
                    vst1q_s32(
                        accumulator.white.as_mut_ptr().add(feature_index),
                        vmlaq_s32(white_accumulator, white_weights, delta),
                    );
                    feature_index += 4;
                }
            }
            while feature_index < SPARSE_FEATURE_COUNT {
                accumulator.black[feature_index] +=
                    delta * i32::from(self.sparse_weights[black_base + feature_index]);
                accumulator.white[feature_index] +=
                    delta * i32::from(self.sparse_weights[white_base + feature_index]);
                feature_index += 1;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut feature_index = 0;
            while feature_index < SPARSE_FEATURE_COUNT {
                accumulator.black[feature_index] +=
                    delta * i32::from(self.sparse_weights[black_base + feature_index]);
                accumulator.white[feature_index] +=
                    delta * i32::from(self.sparse_weights[white_base + feature_index]);
                feature_index += 1;
            }
        }
    }
    pub(crate) fn evaluate_with_accumulator_bits(
        &self,
        black_to_move: bool,
        shape: &FeatureShape,
        black_bits: u64,
        white_bits: u64,
        turn_index: f32,
        no_progress_ply: f32,
        accumulator: &NnueAccumulator,
    ) -> i32 {
        let dense = dense_features(
            black_to_move,
            shape,
            black_bits,
            white_bits,
            turn_index,
            no_progress_ply,
            &self.dense_offsets,
            &self.dense_scales,
        );
        let base = if black_to_move {
            &accumulator.black
        } else {
            &accumulator.white
        };
        let mut sparse_hidden = [0.0; SPARSE_FEATURE_COUNT];
        let mut sparse_index = 0;
        while sparse_index < SPARSE_FEATURE_COUNT {
            sparse_hidden[sparse_index] = base[sparse_index] as f32 * self.sparse_scale;
            sparse_index += 1;
        }
        let mut dense_index = 0;
        while dense_index < DENSE_FEATURE_COUNT {
            let feature_value = dense[dense_index];
            if feature_value != 0.0 {
                let row_start = dense_index * SPARSE_FEATURE_COUNT;
                let mut sparse_index = 0;
                while sparse_index < SPARSE_FEATURE_COUNT {
                    sparse_hidden[sparse_index] +=
                        self.dense_weights[row_start + sparse_index] * feature_value;
                    sparse_index += 1;
                }
            }
            dense_index += 1;
        }
        let mut quantized_sparse = [0i8; NNUE_H0_PAD];
        sparse_index = 0;
        while sparse_index < SPARSE_FEATURE_COUNT {
            quantized_sparse[sparse_index] = quantize_relu(sparse_hidden[sparse_index], self.act0);
            sparse_index += 1;
        }
        let mut hidden_values = [0.0; NNUE_H1];
        let mut hidden_index = 0;
        while hidden_index < NNUE_H1 {
            let row_start = hidden_index * NNUE_H0_PAD;
            let dot = dot_i8_sparse_hidden(
                &quantized_sparse,
                &self.hidden_weights[row_start..row_start + NNUE_H0_PAD],
            );
            hidden_values[hidden_index] = (dot as f32 * self.act0 * self.hidden_scale
                + self.hidden_bias[hidden_index])
                .max(0.0);
            hidden_index += 1;
        }
        let mut quantized_hidden = [0i8; NNUE_H1_PAD];
        hidden_index = 0;
        while hidden_index < NNUE_H1 {
            quantized_hidden[hidden_index] = quantize_relu(hidden_values[hidden_index], self.act1);
            hidden_index += 1;
        }
        let dot = dot_i8_hidden(&quantized_hidden, &self.output_weights);
        let nnue_score = ((dot as f32 * self.act1 * self.output_scale + self.output_bias)
            .clamp(-NNUE_MAX_OUT, NNUE_MAX_OUT)
            * NNUE_SCORE_LIMIT)
            .round() as i32;
        let turn = turn_index.clamp(0.0, 350.0) as i32;
        let remaining = (350 - turn).max(1);
        let mut score = nnue_score;

        let material = i32::from(shape.black.piece_count) - i32::from(shape.white.piece_count);
        let material = if black_to_move { material } else { -material };
        let center = i32::from(shape.black.edge) - i32::from(shape.white.edge);
        let center = if black_to_move { center } else { -center };
        let black_score = i32::from(Position::MAX_PIECES_PER_SIDE as u8 - shape.white.piece_count);
        let white_score = i32::from(Position::MAX_PIECES_PER_SIDE as u8 - shape.black.piece_count);
        let progress = black_score * black_score - white_score * white_score;
        let progress = if black_to_move { progress } else { -progress };
        let liberties = i32::from(shape.black.liberty_count) - i32::from(shape.white.liberty_count);
        let liberties = if black_to_move { liberties } else { -liberties };

        score = score.saturating_add(material.saturating_mul(4096 / remaining));
        score = score.saturating_sub(center.saturating_mul(256 / remaining));
        score = score.saturating_add(progress.saturating_mul(512 / remaining));
        score = score.saturating_add(liberties.saturating_mul(256 / remaining));
        score
    }
}
#[inline(always)]
fn dot_i8_sparse_hidden(left: &[i8; NNUE_H0_PAD], right: &[i8]) -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        assert_eq!(right.len(), NNUE_H0_PAD);
        // SAFETY: the emitted module requires SIMD and unsupported runtimes
        // reject it before this call. Both inputs contain exactly NNUE_H0_PAD
        // bytes, a multiple of one v128 load.
        return unsafe { dot_i8_wasm_simd::<NNUE_H0_PAD>(left.as_ptr(), right.as_ptr()) };
    }
    #[cfg(not(target_arch = "wasm32"))]
    dot_i8_scalar::<NNUE_H0_PAD>(left, right)
}
#[inline(always)]
fn dot_i8_hidden(left: &[i8; NNUE_H1_PAD], right: &[i8]) -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        assert_eq!(right.len(), NNUE_H1_PAD);
        // SAFETY: see dot_i8_sparse_hidden. NNUE_H1_PAD is also a multiple
        // of one v128 load and both inputs contain that many bytes.
        return unsafe { dot_i8_wasm_simd::<NNUE_H1_PAD>(left.as_ptr(), right.as_ptr()) };
    }
    #[cfg(not(target_arch = "wasm32"))]
    dot_i8_scalar::<NNUE_H1_PAD>(left, right)
}
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn dot_i8_scalar<const LEN: usize>(left: &[i8; LEN], right: &[i8]) -> i32 {
    let mut dot = 0i32;
    let mut index = 0usize;
    while index < LEN {
        dot += i32::from(left[index]) * i32::from(right[index]);
        index += 1;
    }
    dot
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[allow(unsafe_code)]
unsafe fn dot_i8_wasm_simd<const LEN: usize>(left: *const i8, right: *const i8) -> i32 {
    use core::arch::wasm32::*;

    assert_eq!(LEN % 16, 0);
    let mut sums = i32x4_splat(0);
    let mut index = 0usize;
    while index < LEN {
        // SAFETY: callers guarantee LEN readable bytes at both pointers, and
        // the loop advances in complete 16-byte chunks while index < LEN.
        let (left_bytes, right_bytes) = unsafe {
            (
                v128_load(left.add(index).cast::<v128>()),
                v128_load(right.add(index).cast::<v128>()),
            )
        };
        sums = i32x4_add(
            sums,
            i32x4_dot_i16x8(
                i16x8_extend_low_i8x16(left_bytes),
                i16x8_extend_low_i8x16(right_bytes),
            ),
        );
        sums = i32x4_add(
            sums,
            i32x4_dot_i16x8(
                i16x8_extend_high_i8x16(left_bytes),
                i16x8_extend_high_i8x16(right_bytes),
            ),
        );
        index += 16;
    }
    i32x4_extract_lane::<0>(sums)
        + i32x4_extract_lane::<1>(sums)
        + i32x4_extract_lane::<2>(sums)
        + i32x4_extract_lane::<3>(sums)
}
pub(crate) fn nnue() -> &'static NnueModel {
    static MODEL: OnceLock<NnueModel> = OnceLock::new();
    MODEL.get_or_init(NnueModel::load)
}
fn dense_features(
    black_to_move: bool,
    shape: &FeatureShape,
    black_bits: u64,
    white_bits: u64,
    turn_index: f32,
    no_progress_ply: f32,
    dense_offsets: &[f32; DENSE_FEATURE_COUNT],
    dense_scales: &[f32; DENSE_FEATURE_COUNT],
) -> [f32; DENSE_FEATURE_COUNT] {
    let (friendly_shape, enemy_shape, side_orientation) = if black_to_move {
        (shape.black, shape.white, 1.0)
    } else {
        (shape.white, shape.black, -1.0)
    };
    let _ = (black_bits, white_bits);
    let mut dense = [0.0; DENSE_FEATURE_COUNT];
    let mut raw = [0.0; DENSE_FEATURE_COUNT];
    if DENSE_FEATURE_COUNT > 0 {
        raw[0] =
            (i32::from(shape.black.edge) - i32::from(shape.white.edge)) as f32 * side_orientation;
    }
    if DENSE_FEATURE_COUNT > 1 {
        raw[1] = (i32::from(shape.black.contact_pairs) - i32::from(shape.white.contact_pairs))
            as f32
            * side_orientation;
    }
    if DENSE_FEATURE_COUNT > 2 {
        raw[2] = 0.0;
    }
    if DENSE_FEATURE_COUNT > 3 {
        raw[3] = 0.0;
    }
    if DENSE_FEATURE_COUNT > 4 {
        raw[4] = (i32::from(shape.black.liberty_count) - i32::from(shape.white.liberty_count))
            as f32
            * side_orientation;
    }
    if DENSE_FEATURE_COUNT > 5 {
        raw[5] = (i32::from(friendly_shape.singletons) - i32::from(enemy_shape.singletons)) as f32;
    }
    if DENSE_FEATURE_COUNT > 6 {
        raw[6] = remaining_turn_feature(turn_index);
    }
    if DENSE_FEATURE_COUNT > 7 {
        raw[7] = no_progress_ply.clamp(0.0, 350.0);
    }
    let mut feature_index = 0;
    while feature_index < DENSE_FEATURE_COUNT {
        dense[feature_index] = clamp_feature(
            (raw[feature_index] - dense_offsets[feature_index])
                / dense_scales[feature_index].max(1.0),
        );
        feature_index += 1;
    }
    dense
}
fn remaining_turn_feature(turn_index: f32) -> f32 {
    let clamped = turn_index.clamp(0.0, 350.0);
    ((350.0 - clamped) / 350.0).clamp(0.0, 1.0)
}
#[derive(Clone, Copy, Default)]
pub(crate) struct SideFeatureShape {
    piece_count: u8,
    edge: u8,
    contact_pairs: u8,
    liberty_count: u8,
    singletons: u8,
}
#[derive(Clone, Copy, Default)]
pub(crate) struct FeatureShape {
    pub(crate) black: SideFeatureShape,
    pub(crate) white: SideFeatureShape,
}
pub(crate) fn build_feature_shape(black_bits: u64, white_bits: u64) -> FeatureShape {
    FeatureShape {
        black: side_feature_shape(black_bits),
        white: side_feature_shape(white_bits),
    }
}
fn side_feature_shape(bits: u64) -> SideFeatureShape {
    let cells = geometry().cells();
    let mut shape = SideFeatureShape {
        piece_count: bits.count_ones() as u8,
        ..SideFeatureShape::default()
    };
    let mut active = bits;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let cell_geometry = &cells[bit.trailing_zeros() as usize];
        let neighbors = cell_geometry.neighbor_mask;
        shape.edge = shape.edge.saturating_add(cell_geometry.center_weight);
        shape.contact_pairs = shape
            .contact_pairs
            .saturating_add((neighbors & bits).count_ones() as u8);
    }
    (shape.liberty_count, shape.singletons) = edge_span(bits);
    shape
}
fn edge_span(bits: u64) -> (u8, u8) {
    let cells = geometry().cells();
    let mut active = bits;
    let mut liberty_mask = 0u64;
    let mut singletons = 0u8;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let neighbors = cells[bit.trailing_zeros() as usize].neighbor_mask;
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
pub(crate) fn update_side_feature_shape(
    mut side: SideFeatureShape,
    old: u64,
    new: u64,
) -> SideFeatureShape {
    if old == new {
        return side;
    }
    let cells = geometry().cells();
    let removed_bits = old & !new;
    let added_bits = new & !old;
    let mut current_bits = old;
    let mut active = removed_bits;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let cell_geometry = &cells[bit.trailing_zeros() as usize];
        side.edge -= cell_geometry.center_weight;
        side.contact_pairs -=
            ((cell_geometry.neighbor_mask & current_bits).count_ones() as u8) << 1;
        current_bits &= !bit;
    }
    active = added_bits;
    while active != 0 {
        let bit = active & active.wrapping_neg();
        active ^= bit;
        let cell_geometry = &cells[bit.trailing_zeros() as usize];
        side.edge += cell_geometry.center_weight;
        side.contact_pairs +=
            ((cell_geometry.neighbor_mask & current_bits).count_ones() as u8) << 1;
        current_bits |= bit;
    }
    side.piece_count = new.count_ones() as u8;
    (side.liberty_count, side.singletons) = edge_span(new);
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
fn take_u8(bytes: &[u8], cursor: &mut usize) -> u8 {
    let value = bytes[*cursor];
    *cursor += 1;
    value
}
fn take_u16(bytes: &[u8], cursor: &mut usize) -> u16 {
    let value = u16::from_le_bytes(bytes[*cursor..*cursor + 2].try_into().unwrap());
    *cursor += 2;
    value
}
fn take_u32(bytes: &[u8], cursor: &mut usize) -> u32 {
    let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    value
}
fn take_i32(bytes: &[u8], cursor: &mut usize) -> i32 {
    let value = i32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    value
}
fn take_f32(bytes: &[u8], cursor: &mut usize) -> f32 {
    let value = f32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    value
}
fn take_i16_box(bytes: &[u8], cursor: &mut usize, len: usize) -> Box<[i16]> {
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        out.push(i16::from_le_bytes(
            bytes[*cursor..*cursor + 2].try_into().unwrap(),
        ));
        *cursor += 2;
        i += 1;
    }
    out.into_boxed_slice()
}
fn take_i8_box(bytes: &[u8], cursor: &mut usize, len: usize) -> Box<[i8]> {
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        out.push(bytes[*cursor] as i8);
        *cursor += 1;
        i += 1;
    }
    out.into_boxed_slice()
}
fn take_f32_box(bytes: &[u8], cursor: &mut usize, len: usize) -> Box<[f32]> {
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        out.push(f32::from_le_bytes(
            bytes[*cursor..*cursor + 4].try_into().unwrap(),
        ));
        *cursor += 4;
        i += 1;
    }
    out.into_boxed_slice()
}
