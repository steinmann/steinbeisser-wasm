use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use steinbeisser::{CELL_COUNT, CellId as CellIndex, Position, geometry};

pub(crate) struct TargetBlend {
    pub score: f32,
    pub result: f32,
    pub lambda_mix: f64,
}

pub(crate) fn combined_target(blend: TargetBlend) -> f64 {
    blend.lambda_mix * linear_clip_target(blend.score)
        + (1.0 - blend.lambda_mix) * f64::from(blend.result)
}

pub(crate) fn huber_loss(error: f64) -> f64 {
    let absolute = error.abs();
    if absolute <= 1.0 {
        0.5 * error * error
    } else {
        absolute - 0.5
    }
}

pub const FEATURE_SET_NAME: &str = "steinbeisser_nnue_features";
pub const FEATURE_SCHEMA_ROW_LENGTHS: [usize; 9] = [5, 6, 7, 8, 9, 8, 7, 6, 5];
pub const SPARSE_FEATURE_COUNT: usize = CELL_COUNT * 2;
pub const MAX_ACTIVE_FEATURES: usize = Position::MAX_PIECES_PER_SIDE * 2;
pub const DENSE_SCALAR_FEATURE_COUNT: usize = 8;
pub const INPUT_COUNT: usize = SPARSE_FEATURE_COUNT + DENSE_SCALAR_FEATURE_COUNT;
pub const TURNS_TO_LIMIT_SCALE: f32 = 1.0;
pub const NO_PROGRESS_PLIES_SCALE: f32 = 64.0;

pub const DENSE_FEATURE_NAMES: [&str; DENSE_SCALAR_FEATURE_COUNT] = [
    "edge_balance_norm",
    "compact_balance_norm",
    "reserved_group_slot_norm",
    "reserved_largest_group_slot_norm",
    "liberty_balance_norm",
    "isolated_balance_norm",
    "turns_to_limit_norm",
    "no_progress_plies_norm",
];

pub const DENSE_FEATURE_SCALES: [f32; DENSE_SCALAR_FEATURE_COUNT] = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    TURNS_TO_LIMIT_SCALE,
    NO_PROGRESS_PLIES_SCALE,
];

#[derive(Clone, Debug, Serialize)]
pub struct FeatureSchema {
    pub name: &'static str,
    pub row_lengths: &'static [usize],
    pub cell_count: usize,
    pub sparse_feature_count: usize,
    pub dense_feature_count: usize,
    pub input_count: usize,
    pub max_active_features: usize,
    pub dense_feature_names: &'static [&'static str],
    pub dense_feature_scales: &'static [f32],
}

pub const fn current_feature_schema() -> FeatureSchema {
    FeatureSchema {
        name: FEATURE_SET_NAME,
        row_lengths: &FEATURE_SCHEMA_ROW_LENGTHS,
        cell_count: CELL_COUNT,
        sparse_feature_count: SPARSE_FEATURE_COUNT,
        dense_feature_count: DENSE_SCALAR_FEATURE_COUNT,
        input_count: INPUT_COUNT,
        max_active_features: MAX_ACTIVE_FEATURES,
        dense_feature_names: &DENSE_FEATURE_NAMES,
        dense_feature_scales: &DENSE_FEATURE_SCALES,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DenseFeatureNormalization {
    #[serde(default)]
    pub dense_feature_offsets: Vec<f32>,
    pub dense_feature_scales: Vec<f32>,
}

impl DenseFeatureNormalization {
    pub fn from_offsets_and_scales(offsets: Vec<f32>, scales: Vec<f32>) -> Self {
        Self {
            dense_feature_offsets: offsets,
            dense_feature_scales: scales,
        }
    }

    pub const fn identity() -> Self {
        Self {
            dense_feature_offsets: Vec::new(),
            dense_feature_scales: Vec::new(),
        }
    }

    pub fn resolved_scales(&self, feature_set: NnueFeatureSet) -> Vec<f32> {
        let mut scales = feature_set.default_dense_feature_scales().to_vec();
        for (target, source) in scales.iter_mut().zip(self.dense_feature_scales.iter()) {
            *target = source.abs().max(1.0);
        }
        scales
    }

    pub fn resolved_offsets(&self, feature_set: NnueFeatureSet) -> Vec<f32> {
        let mut offsets = vec![0.0; feature_set.dense_input_count()];
        for (target, source) in offsets.iter_mut().zip(self.dense_feature_offsets.iter()) {
            *target = *source;
        }
        offsets
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NnueFeatureSet {
    Steinbeisser,
}

impl NnueFeatureSet {
    pub const fn current() -> Self {
        Self::Steinbeisser
    }

    pub const fn input_count(self) -> usize {
        match self {
            Self::Steinbeisser => INPUT_COUNT,
        }
    }

    pub const fn sparse_input_count(self) -> usize {
        match self {
            Self::Steinbeisser => SPARSE_FEATURE_COUNT,
        }
    }

    pub const fn dense_input_count(self) -> usize {
        match self {
            Self::Steinbeisser => DENSE_SCALAR_FEATURE_COUNT,
        }
    }

    pub const fn default_dense_feature_scales(self) -> &'static [f32] {
        match self {
            Self::Steinbeisser => &DENSE_FEATURE_SCALES,
        }
    }

    pub fn feature_vector_from_bitboards_with_context(
        self,
        side_to_move_is_black: bool,
        black_bits: u64,
        white_bits: u64,
        game_turns_played: f32,
        no_progress_plies: f32,
        normalization: DenseFeatureNormalization,
    ) -> SparseFeatureVector {
        self.feature_vector_from_state_components(
            side_to_move_is_black,
            black_bits,
            white_bits,
            game_turns_played,
            no_progress_plies,
            normalization,
        )
    }

    fn feature_vector_from_state_components(
        self,
        side_to_move_is_black: bool,
        black_bits: u64,
        white_bits: u64,
        game_turns_played: f32,
        no_progress_plies: f32,
        normalization: DenseFeatureNormalization,
    ) -> SparseFeatureVector {
        let mut features = SparseFeatureVector::new(self);
        fill_piece_cell_features(side_to_move_is_black, black_bits, white_bits, &mut features);
        let dense_values = normalized_dense_feature_values_from_bitboards(
            self,
            side_to_move_is_black,
            black_bits,
            white_bits,
            game_turns_played,
            no_progress_plies,
            normalization,
        );
        for value in dense_values {
            features.push_dense(value);
        }
        features
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseFeatureVector {
    feature_set: NnueFeatureSet,
    active_indices: [u16; MAX_ACTIVE_FEATURES],
    len: usize,
    dense_values: [f32; DENSE_SCALAR_FEATURE_COUNT],
    dense_len: usize,
}

impl SparseFeatureVector {
    pub const fn new(feature_set: NnueFeatureSet) -> Self {
        Self {
            feature_set,
            active_indices: [0; MAX_ACTIVE_FEATURES],
            len: 0,
            dense_values: [0.0; DENSE_SCALAR_FEATURE_COUNT],
            dense_len: 0,
        }
    }

    pub const fn feature_set(&self) -> NnueFeatureSet {
        self.feature_set
    }

    pub fn active_indices(&self) -> &[u16] {
        &self.active_indices[..self.len]
    }

    pub fn dense_values(&self) -> &[f32] {
        &self.dense_values[..self.dense_len]
    }

    fn push(&mut self, index: u16) {
        debug_assert!(self.len < self.active_indices.len());
        self.active_indices[self.len] = index;
        self.len += 1;
    }

    fn push_dense(&mut self, value: f32) {
        debug_assert!(self.dense_len < self.dense_values.len());
        self.dense_values[self.dense_len] = value;
        self.dense_len += 1;
    }
}

fn fill_piece_cell_features(
    side_to_move_is_black: bool,
    black_bits: u64,
    white_bits: u64,
    out: &mut SparseFeatureVector,
) {
    let (own_bits, opp_bits) = if side_to_move_is_black {
        (black_bits, white_bits)
    } else {
        (white_bits, black_bits)
    };
    push_piece_cell_segment(own_bits, 0, out);
    push_piece_cell_segment(opp_bits, CELL_COUNT as u16, out);
}

fn push_piece_cell_segment(bits: u64, offset: u16, out: &mut SparseFeatureVector) {
    let mut active = bits;
    while active != 0 {
        let raw = active.trailing_zeros() as u8;
        let cell = CellIndex::new(raw).expect("active feature bit should identify a board cell");
        out.push(offset + cell.as_u8() as u16);
        active &= active - 1;
    }
}

fn normalized_dense_feature_values_from_bitboards(
    feature_set: NnueFeatureSet,
    side_to_move_is_black: bool,
    black_bits: u64,
    white_bits: u64,
    game_turns_played: f32,
    no_progress_plies: f32,
    normalization: DenseFeatureNormalization,
) -> [f32; DENSE_SCALAR_FEATURE_COUNT] {
    let raw = raw_dense_feature_values_from_bitboards(
        feature_set,
        side_to_move_is_black,
        black_bits,
        white_bits,
        game_turns_played,
        no_progress_plies,
    );
    normalize_dense_values(feature_set, raw, normalization)
}

fn normalize_dense_values(
    feature_set: NnueFeatureSet,
    raw: [f32; DENSE_SCALAR_FEATURE_COUNT],
    normalization: DenseFeatureNormalization,
) -> [f32; DENSE_SCALAR_FEATURE_COUNT] {
    let mut values = [0.0_f32; DENSE_SCALAR_FEATURE_COUNT];
    let offsets = normalization.resolved_offsets(feature_set);
    let scales = normalization.resolved_scales(feature_set);
    for index in 0..feature_set.dense_input_count() {
        values[index] = clamp_feature((raw[index] - offsets[index]) / scales[index].max(1.0));
    }
    values
}

fn raw_dense_feature_values_from_bitboards(
    _feature_set: NnueFeatureSet,
    side_to_move_is_black: bool,
    black_bits: u64,
    white_bits: u64,
    game_turns_played: f32,
    no_progress_plies: f32,
) -> [f32; DENSE_SCALAR_FEATURE_COUNT] {
    let (own_analysis, opp_analysis) = if side_to_move_is_black {
        (analyze_side(black_bits), analyze_side(white_bits))
    } else {
        (analyze_side(white_bits), analyze_side(black_bits))
    };
    let mut raw = [0.0_f32; DENSE_SCALAR_FEATURE_COUNT];
    raw[0] = own_analysis.edge_total as f32 - opp_analysis.edge_total as f32;
    raw[1] = own_analysis.compact_total as f32 - opp_analysis.compact_total as f32;
    raw[2] = 0.0;
    raw[3] = 0.0;
    raw[4] = own_analysis.liberty_count as f32 - opp_analysis.liberty_count as f32;
    raw[5] = own_analysis.isolated_count as f32 - opp_analysis.isolated_count as f32;
    raw[6] = turns_to_limit_norm(game_turns_played);
    raw[7] = no_progress_plies.clamp(0.0, 350.0);
    raw
}

#[derive(Clone, Copy, Debug, Default)]
struct SideAnalysis {
    edge_total: i32,
    compact_total: i32,
    liberty_count: u8,
    isolated_count: u8,
}

fn analyze_side(bits: u64) -> SideAnalysis {
    let board = geometry();
    let mut analysis = SideAnalysis::default();
    let mut active = bits;
    while active != 0 {
        let raw = active.trailing_zeros() as u8;
        let cell = CellIndex::new(raw).expect("bit index should stay on board");
        let cell_geometry = board.cell(cell);
        analysis.edge_total += i32::from(cell_geometry.center_weight);

        let mut friendly_neighbors = 0_i32;
        for neighbor in cell_geometry.neighbors.iter().flatten() {
            if bits & (1_u64 << neighbor.as_u8()) != 0 {
                friendly_neighbors += 1;
            }
        }
        analysis.compact_total += friendly_neighbors;
        active &= active - 1;
    }

    let mut remaining = bits;
    let mut liberty_mask = 0_u64;
    while remaining != 0 {
        let mut frontier = vec![remaining.trailing_zeros() as u8];
        let mut group_bits = 0_u64;
        let mut group_size = 0_u8;
        while let Some(raw) = frontier.pop() {
            let bit = 1_u64 << raw;
            if bits & bit == 0 || group_bits & bit != 0 {
                continue;
            }
            group_bits |= bit;
            group_size = group_size.saturating_add(1);
            let cell = CellIndex::new(raw).expect("group bit should identify a board cell");
            for neighbor in board.cell(cell).neighbors.iter().flatten() {
                let neighbor_raw = neighbor.as_u8();
                let neighbor_bit = 1_u64 << neighbor_raw;
                if bits & neighbor_bit != 0 {
                    if group_bits & neighbor_bit == 0 {
                        frontier.push(neighbor_raw);
                    }
                } else {
                    liberty_mask |= neighbor_bit;
                }
            }
        }
        remaining &= !group_bits;
        if group_size == 1 {
            analysis.isolated_count = analysis.isolated_count.saturating_add(1);
        }
    }
    analysis.liberty_count = liberty_mask.count_ones().min(u8::MAX as u32) as u8;
    analysis
}

fn turns_to_limit_norm(game_turns_played: f32) -> f32 {
    let clamped = game_turns_played.clamp(0.0, 350.0);
    ((350.0 - clamped) / 350.0).clamp(0.0, 1.0)
}

fn clamp_feature(value: f32) -> f32 {
    value.clamp(-1.0, 1.0)
}

// Quantized runtime model ----------------------------------------------------

const MAX_MODEL_OUTPUT: f32 = 1.0;
const ENGINE_SCORE_LIMIT: i32 = 5_000;

const NNQ_MAGIC: [u8; 4] = *b"NNQ1";
const NNQ_VERSION: u16 = 7;
const NNUE_BOARD_RADIUS_MARKER: u8 = 23;
const CLIPPED_RELU_ACTIVATION: u8 = 1;
const SCALAR_BACKEND_ID: u8 = 1;
const TARGET_TRANSFORM_LINEAR_CLIP_V1: u8 = 1;

const SPARSE_WEIGHT_QUANT_RANGE: f32 = 32_760.0;
const DENSE_WEIGHT_QUANT_RANGE: f32 = 127.0;
const ACTIVATION_QUANT_RANGE: f32 = 127.0;
const DOT_PRODUCT_CHUNK: usize = 16;

pub(crate) fn linear_clip_target(score: impl Into<f64>) -> f64 {
    let score = score.into();
    if score >= 90_000.0 {
        1.0
    } else if score <= -90_000.0 {
        -1.0
    } else {
        score.clamp(-(ENGINE_SCORE_LIMIT as f64), ENGINE_SCORE_LIMIT as f64)
            / ENGINE_SCORE_LIMIT as f64
    }
}

pub(crate) fn linear_clip_score(output: f32) -> i32 {
    (output.clamp(-MAX_MODEL_OUTPUT, MAX_MODEL_OUTPUT) * ENGINE_SCORE_LIMIT as f32).round() as i32
}

fn expect_linear_clip_transform_id(value: u8) -> Result<()> {
    if value != TARGET_TRANSFORM_LINEAR_CLIP_V1 {
        bail!("unsupported NNUE target transform id {value}; expected linear_clip_v1");
    }
    Ok(())
}

fn expect_linear_clip_transform_json(raw: Option<&Value>) -> Result<()> {
    match raw.and_then(Value::as_str) {
        Some("linear_clip_v1") => Ok(()),
        Some(other) => bail!("unsupported NNUE target transform {other}; expected linear_clip_v1"),
        None => bail!("NNUE model is missing target_transform"),
    }
}

#[derive(Clone, Debug)]
pub struct SparseMlpModel {
    feature_set: NnueFeatureSet,
    dense_input_count: usize,
    dense_normalization: DenseFeatureNormalization,
    hidden_sizes: Vec<usize>,
    sparse_weight_scale: f32,
    input_layer_sparse_weights: Vec<i16>,
    input_layer_dense_weights: Vec<f32>,
    input_layer_biases: Vec<i32>,
    hidden_layers: Vec<QuantizedDenseLayer>,
    output_layer: QuantizedOutputLayer,
    activation_scales: Vec<f32>,
}

#[derive(Clone, Debug)]
struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

#[derive(Clone, Debug)]
struct QuantizedDenseLayer {
    input_size: usize,
    padded_input_size: usize,
    output_size: usize,
    weights: Vec<i8>,
    biases: Vec<f32>,
    weight_scale: f32,
}

#[derive(Clone, Debug)]
struct QuantizedOutputLayer {
    input_size: usize,
    padded_input_size: usize,
    weights: Vec<i8>,
    bias: f32,
    weight_scale: f32,
}

#[derive(Clone, Debug)]
struct FloatModel {
    feature_set: NnueFeatureSet,
    dense_input_count: usize,
    dense_normalization: DenseFeatureNormalization,
    hidden_sizes: Vec<usize>,
    input_layer_sparse_weights: Vec<f32>,
    input_layer_dense_weights: Vec<f32>,
    input_layer_biases: Vec<f32>,
    hidden_layers: Vec<DenseLayer>,
    output_weights: Vec<f32>,
    output_bias: f32,
    activation_scales: Vec<f32>,
}

impl SparseMlpModel {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read NNUE model {}", path.display()))?;
        if bytes.starts_with(&NNQ_MAGIC) {
            return Self::from_nnq_bytes(&bytes)
                .with_context(|| format!("failed to parse NNUE runtime {}", path.display()));
        }
        let raw = serde_json::from_slice::<Value>(&bytes)
            .with_context(|| format!("failed to parse NNUE model JSON {}", path.display()))?;
        Self::from_value(raw)
    }

    pub fn feature_set(&self) -> NnueFeatureSet {
        self.feature_set
    }

    pub fn dense_normalization(&self) -> DenseFeatureNormalization {
        self.dense_normalization.clone()
    }

    pub fn raw_output(&self, features: &SparseFeatureVector) -> f32 {
        debug_assert_eq!(features.feature_set(), self.feature_set);
        let first_hidden_size = self.hidden_sizes[0];
        let mut accumulator = self.input_layer_biases.clone();
        for &index in features.active_indices() {
            let row_start = usize::from(index) * first_hidden_size;
            let row = &self.input_layer_sparse_weights[row_start..row_start + first_hidden_size];
            add_i16_row(&mut accumulator, row);
        }

        let mut current = vec![0.0_f32; first_hidden_size];
        for (slot, &value) in current.iter_mut().zip(accumulator.iter()) {
            *slot = value as f32 * self.sparse_weight_scale;
        }
        for (dense_index, &dense_value) in features
            .dense_values()
            .iter()
            .take(self.dense_input_count)
            .enumerate()
        {
            add_dense_row(
                &mut current,
                &self.input_layer_dense_weights,
                dense_index,
                dense_value,
            );
        }
        relu_in_place(&mut current);

        let mut current_scale = self.activation_scales[0];
        let mut current_q = quantize_activations(&current, current_scale);
        for (layer_index, layer) in self.hidden_layers.iter().enumerate() {
            current = evaluate_dense_layer(layer, &current_q, current_scale);
            current_scale = self.activation_scales[layer_index + 1];
            current_q = quantize_activations(&current, current_scale);
        }
        evaluate_output_layer(&self.output_layer, &current_q, current_scale)
    }

    fn from_value(raw: Value) -> Result<Self> {
        let float_model = parse_float_model(raw)?;
        Ok(Self::quantize_from_float(float_model))
    }

    fn quantize_from_float(float_model: FloatModel) -> Self {
        let max_abs = float_model
            .input_layer_sparse_weights
            .iter()
            .fold(0.0_f32, |max_abs, value| max_abs.max(value.abs()));
        let sparse_weight_scale = if max_abs > 0.0 {
            max_abs / SPARSE_WEIGHT_QUANT_RANGE
        } else {
            1.0 / SPARSE_WEIGHT_QUANT_RANGE
        };
        let input_layer_sparse_weights = float_model
            .input_layer_sparse_weights
            .iter()
            .map(|value| quantize_i16(*value, sparse_weight_scale))
            .collect::<Vec<_>>();
        let input_layer_biases = float_model
            .input_layer_biases
            .iter()
            .map(|value| (*value / sparse_weight_scale).round() as i32)
            .collect::<Vec<_>>();
        let hidden_layers = float_model
            .hidden_layers
            .iter()
            .map(|layer| quantize_dense_layer(layer, layer.input_size))
            .collect::<Vec<_>>();
        let output_input_size = float_model
            .hidden_layers
            .last()
            .map(|layer| layer.output_size)
            .unwrap_or(float_model.hidden_sizes[0]);
        let output_layer = quantize_output_layer(
            &float_model.output_weights,
            float_model.output_bias,
            output_input_size,
        );

        Self {
            feature_set: float_model.feature_set,
            dense_input_count: float_model.dense_input_count,
            dense_normalization: float_model.dense_normalization,
            hidden_sizes: float_model.hidden_sizes,
            sparse_weight_scale,
            input_layer_sparse_weights,
            input_layer_dense_weights: float_model.input_layer_dense_weights,
            input_layer_biases,
            hidden_layers,
            output_layer,
            activation_scales: float_model.activation_scales,
        }
    }

    fn from_nnq_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);
        let magic = read_array::<4>(&mut cursor)?;
        if magic != NNQ_MAGIC {
            bail!("invalid NNUE runtime magic");
        }
        let version = read_u16(&mut cursor)?;
        if version != NNQ_VERSION {
            bail!("unsupported NNUE runtime version {version}; expected {NNQ_VERSION}");
        }
        let feature_marker = read_u8(&mut cursor)?;
        if feature_marker != NNUE_BOARD_RADIUS_MARKER {
            bail!("unsupported NNUE feature marker {feature_marker}");
        }
        let activation = read_u8(&mut cursor)?;
        if activation != CLIPPED_RELU_ACTIVATION {
            bail!("unsupported NNUE activation id {activation}; expected relu");
        }
        let backend = read_u8(&mut cursor)?;
        if backend != SCALAR_BACKEND_ID {
            bail!("unsupported NNUE backend id {backend}; expected scalar");
        }
        expect_linear_clip_transform_id(read_u8(&mut cursor)?)?;
        let feature_set = NnueFeatureSet::current();

        let sparse_input_count = read_u32(&mut cursor)? as usize;
        let dense_input_count = read_u32(&mut cursor)? as usize;
        let hidden_layer_count = read_u32(&mut cursor)? as usize;
        if hidden_layer_count == 0 {
            bail!("NNUE runtime must contain at least one hidden layer");
        }
        let mut hidden_sizes = Vec::with_capacity(hidden_layer_count);
        for _ in 0..hidden_layer_count {
            hidden_sizes.push(read_u32(&mut cursor)? as usize);
        }
        validate_model_shape(
            feature_set,
            sparse_input_count,
            dense_input_count,
            &hidden_sizes,
        )?;

        let sparse_weight_scale = read_f32(&mut cursor)?;
        let scale_count = read_u32(&mut cursor)? as usize;
        let scales = read_f32_vec(&mut cursor, scale_count)?;
        let offset_count = read_u32(&mut cursor)? as usize;
        let offsets = read_f32_vec(&mut cursor, offset_count)?;
        let dense_normalization = DenseFeatureNormalization::from_offsets_and_scales(
            offsets,
            resolve_normalization_values(scales, dense_input_count, "dense_feature_scales")?,
        );
        let output_bias = read_f32(&mut cursor)?;
        let activation_scales = read_f32_vec(&mut cursor, hidden_sizes.len())?;
        validate_activation_scales(&activation_scales, hidden_sizes.len())?;

        let first_hidden_size = hidden_sizes[0];
        let input_layer_biases = read_i32_vec(&mut cursor, first_hidden_size)?;
        let input_layer_sparse_weights =
            read_i16_vec(&mut cursor, sparse_input_count * first_hidden_size)?;
        let input_layer_dense_weights =
            read_f32_vec(&mut cursor, dense_input_count * first_hidden_size)?;

        let mut hidden_layers = Vec::new();
        let mut previous_size = first_hidden_size;
        for &hidden_size in hidden_sizes.iter().skip(1) {
            let padded_input_size = read_u32(&mut cursor)? as usize;
            validate_padded_width(previous_size, padded_input_size, "dense layer")?;
            let weight_scale = read_f32(&mut cursor)?;
            let biases = read_f32_vec(&mut cursor, hidden_size)?;
            let weights = read_i8_vec(&mut cursor, padded_input_size * hidden_size)?;
            hidden_layers.push(QuantizedDenseLayer {
                input_size: previous_size,
                padded_input_size,
                output_size: hidden_size,
                weights,
                biases,
                weight_scale,
            });
            previous_size = hidden_size;
        }

        let output_padded_input_size = read_u32(&mut cursor)? as usize;
        validate_padded_width(previous_size, output_padded_input_size, "output layer")?;
        let output_weight_scale = read_f32(&mut cursor)?;
        let output_weights = read_i8_vec(&mut cursor, output_padded_input_size)?;
        if cursor.position() != bytes.len() as u64 {
            bail!("NNUE runtime contains trailing bytes");
        }

        Ok(Self {
            feature_set,
            dense_input_count,
            dense_normalization,
            hidden_sizes,
            sparse_weight_scale,
            input_layer_sparse_weights,
            input_layer_dense_weights,
            input_layer_biases,
            hidden_layers,
            output_layer: QuantizedOutputLayer {
                input_size: previous_size,
                padded_input_size: output_padded_input_size,
                weights: output_weights,
                bias: output_bias,
                weight_scale: output_weight_scale,
            },
            activation_scales,
        })
    }
}

fn parse_float_model(raw: Value) -> Result<FloatModel> {
    let feature_set = match parse_string_field(&raw, "feature_set")? {
        FEATURE_SET_NAME => NnueFeatureSet::current(),
        other => bail!("unsupported NNUE feature set {other}"),
    };
    let activation = parse_string_field(&raw, "activation")?;
    if activation != "relu" {
        bail!("unsupported NNUE activation {activation}; expected relu");
    }
    let norm = parse_string_field(&raw, "norm")?;
    if norm != "none" {
        bail!("unsupported NNUE norm {norm}");
    }
    let block_type = parse_string_field(&raw, "block_type")?;
    if block_type != "plain" {
        bail!("unsupported NNUE block_type {block_type}");
    }

    let hidden_sizes = parse_hidden_sizes(&raw)?;
    if hidden_sizes.is_empty() {
        bail!("NNUE model must define at least one hidden layer");
    }
    let sparse_input_count = parse_usize_field(&raw, "input_count_sparse")?;
    let dense_input_count = parse_usize_field(&raw, "input_count_dense")?;
    validate_model_shape(
        feature_set,
        sparse_input_count,
        dense_input_count,
        &hidden_sizes,
    )?;

    let input_count = parse_usize_field(&raw, "input_count")?;
    if input_count != sparse_input_count + dense_input_count {
        bail!(
            "NNUE model input_count {} does not match sparse+dense counts {}",
            input_count,
            sparse_input_count + dense_input_count
        );
    }

    let dense_normalization = parse_dense_normalization(&raw, dense_input_count)?;
    expect_linear_clip_transform_json(raw.get("target_transform"))?;
    let input_layer_sparse_weights =
        parse_matrix_field(&raw, "w1_sparse", sparse_input_count, hidden_sizes[0])?;
    let input_layer_dense_weights =
        parse_matrix_field(&raw, "w1_dense", dense_input_count, hidden_sizes[0])?;
    let input_layer_biases = parse_vector_field(&raw, "b1", hidden_sizes[0])?;

    let mut hidden_layers = Vec::new();
    let mut previous_size = hidden_sizes[0];
    for (index, &hidden_size) in hidden_sizes.iter().enumerate().skip(1) {
        let layer_index = index + 1;
        hidden_layers.push(DenseLayer {
            input_size: previous_size,
            output_size: hidden_size,
            weights: parse_matrix_field(
                &raw,
                &format!("w{layer_index}"),
                previous_size,
                hidden_size,
            )?,
            biases: parse_vector_field(&raw, &format!("b{layer_index}"), hidden_size)?,
        });
        previous_size = hidden_size;
    }

    let output_layer_index = hidden_sizes.len() + 1;
    let output_weights =
        parse_output_weights_field(&raw, &format!("w{output_layer_index}"), previous_size)?;
    let output_bias = parse_scalar_field(&raw, &format!("b{output_layer_index}"))?;
    let activation_scales = raw
        .get("runtime_activation_scales")
        .ok_or_else(|| anyhow::anyhow!("NNUE model is missing runtime_activation_scales"))
        .and_then(|value| parse_activation_scales(value, hidden_sizes.len()))?;
    validate_activation_scales(&activation_scales, hidden_sizes.len())?;

    Ok(FloatModel {
        feature_set,
        dense_input_count,
        dense_normalization,
        hidden_sizes,
        input_layer_sparse_weights,
        input_layer_dense_weights,
        input_layer_biases,
        hidden_layers,
        output_weights,
        output_bias,
        activation_scales,
    })
}

fn validate_model_shape(
    feature_set: NnueFeatureSet,
    sparse_input_count: usize,
    dense_input_count: usize,
    hidden_sizes: &[usize],
) -> Result<()> {
    if hidden_sizes.contains(&0) {
        bail!("NNUE hidden layer sizes must be positive");
    }
    if sparse_input_count != feature_set.sparse_input_count() {
        bail!(
            "NNUE sparse input count {} does not match current feature count {}",
            sparse_input_count,
            feature_set.sparse_input_count()
        );
    }
    if dense_input_count != feature_set.dense_input_count() {
        bail!(
            "NNUE dense input count {} does not match current feature count {}",
            dense_input_count,
            feature_set.dense_input_count()
        );
    }
    Ok(())
}

fn validate_activation_scales(scales: &[f32], expected_len: usize) -> Result<()> {
    if scales.len() != expected_len {
        bail!(
            "NNUE runtime activation scale count {} does not match hidden layer count {}",
            scales.len(),
            expected_len
        );
    }
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        bail!("NNUE runtime activation scales must be finite positive values");
    }
    Ok(())
}

fn validate_padded_width(
    input_size: usize,
    padded_input_size: usize,
    layer_name: &str,
) -> Result<()> {
    let expected = padded_width(input_size);
    if padded_input_size != expected {
        bail!(
            "NNUE runtime {layer_name} padded input size {padded_input_size} does not match expected {expected}"
        );
    }
    Ok(())
}

fn parse_hidden_sizes(raw: &Value) -> Result<Vec<usize>> {
    let values = raw
        .get("hidden_sizes")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("NNUE model is missing hidden_sizes metadata"))?;
    values
        .iter()
        .map(|value| parse_usize_value(value, "hidden_sizes entry"))
        .collect()
}

fn parse_dense_normalization(
    raw: &Value,
    dense_input_count: usize,
) -> Result<DenseFeatureNormalization> {
    let scales = raw
        .get("dense_feature_scales")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("NNUE model is missing dense_feature_scales"))?
        .iter()
        .map(|value| parse_f32_value(value, "dense_feature_scales entry"))
        .collect::<Result<Vec<_>>>()?;
    let offsets = raw
        .get("dense_feature_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("NNUE model is missing dense_feature_offsets"))?
        .iter()
        .map(|value| parse_f32_value(value, "dense_feature_offsets entry"))
        .collect::<Result<Vec<_>>>()?;
    Ok(DenseFeatureNormalization::from_offsets_and_scales(
        resolve_normalization_values(offsets, dense_input_count, "dense_feature_offsets")?,
        resolve_normalization_values(scales, dense_input_count, "dense_feature_scales")?,
    ))
}

fn resolve_normalization_values(
    values: Vec<f32>,
    expected_len: usize,
    label: &str,
) -> Result<Vec<f32>> {
    if values.len() != expected_len {
        bail!(
            "NNUE model field {label} has {} values, expected {}",
            values.len(),
            expected_len
        );
    }
    Ok(values)
}

fn parse_string_field<'a>(raw: &'a Value, field: &str) -> Result<&'a str> {
    raw.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("NNUE model field {field} must be a string"))
}

fn parse_usize_field(raw: &Value, field: &str) -> Result<usize> {
    raw.get(field)
        .ok_or_else(|| anyhow::anyhow!("NNUE model is missing {field}"))
        .and_then(|value| parse_usize_value(value, field))
}

fn parse_usize_value(value: &Value, label: &str) -> Result<usize> {
    let number = value.as_u64().ok_or_else(|| {
        anyhow::anyhow!("NNUE model field {label} must be a non-negative integer")
    })?;
    usize::try_from(number)
        .map_err(|_| anyhow::anyhow!("NNUE model field {label} is too large for usize"))
}

fn parse_matrix_field(
    raw: &Value,
    field: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<f32>> {
    let rows = raw
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("NNUE model field {field} must be a matrix"))?;
    if rows.len() != expected_rows {
        bail!(
            "NNUE model field {field} has {} rows, expected {}",
            rows.len(),
            expected_rows
        );
    }

    let mut flattened = Vec::with_capacity(expected_rows * expected_cols);
    for (row_index, row) in rows.iter().enumerate() {
        let cols = row.as_array().ok_or_else(|| {
            anyhow::anyhow!("NNUE model field {field}[{row_index}] must be an array")
        })?;
        if cols.len() != expected_cols {
            bail!(
                "NNUE model field {field}[{row_index}] has {} columns, expected {}",
                cols.len(),
                expected_cols
            );
        }
        for value in cols {
            flattened.push(parse_f32_value(value, field)?);
        }
    }
    Ok(flattened)
}

fn parse_vector_field(raw: &Value, field: &str, expected_len: usize) -> Result<Vec<f32>> {
    let values = raw
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("NNUE model field {field} must be an array"))?;
    if values.len() != expected_len {
        bail!(
            "NNUE model field {field} has {} values, expected {}",
            values.len(),
            expected_len
        );
    }
    values
        .iter()
        .map(|value| parse_f32_value(value, field))
        .collect()
}

fn parse_output_weights_field(raw: &Value, field: &str, expected_len: usize) -> Result<Vec<f32>> {
    let value = raw
        .get(field)
        .ok_or_else(|| anyhow::anyhow!("NNUE model is missing field {field}"))?;
    if let Some(values) = value.as_array() {
        if values.len() == expected_len && values.iter().all(Value::is_number) {
            return values
                .iter()
                .map(|entry| parse_f32_value(entry, field))
                .collect();
        }
        if values.len() == expected_len {
            let mut flattened = Vec::with_capacity(expected_len);
            for (row_index, row) in values.iter().enumerate() {
                let cols = row.as_array().ok_or_else(|| {
                    anyhow::anyhow!("NNUE model field {field}[{row_index}] must be an array")
                })?;
                if cols.len() != 1 {
                    bail!(
                        "NNUE model field {field}[{row_index}] has {} columns, expected 1",
                        cols.len()
                    );
                }
                flattened.push(parse_f32_value(&cols[0], field)?);
            }
            return Ok(flattened);
        }
    }
    bail!(
        "NNUE model field {field} must be a vector of length {} or a {}x1 column matrix",
        expected_len,
        expected_len
    );
}

fn parse_scalar_field(raw: &Value, field: &str) -> Result<f32> {
    let value = raw
        .get(field)
        .ok_or_else(|| anyhow::anyhow!("NNUE model is missing field {field}"))?;
    if value.is_number() {
        return parse_f32_value(value, field);
    }
    if let Some(values) = value.as_array()
        && let [entry] = values.as_slice()
    {
        return parse_f32_value(entry, field);
    }
    bail!("NNUE model field {field} must be a scalar or a single-element array");
}

fn parse_activation_scales(value: &Value, expected_len: usize) -> Result<Vec<f32>> {
    let values = value.as_array().ok_or_else(|| {
        anyhow::anyhow!("NNUE model field runtime_activation_scales must be an array")
    })?;
    if values.len() != expected_len {
        bail!(
            "NNUE model field runtime_activation_scales has {} values, expected {}",
            values.len(),
            expected_len
        );
    }
    values
        .iter()
        .map(|entry| parse_f32_value(entry, "runtime_activation_scales"))
        .collect()
}

fn parse_f32_value(value: &Value, field: &str) -> Result<f32> {
    let number = value
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("NNUE model field {field} must contain numeric values"))?;
    Ok(number as f32)
}

fn add_dense_row(target: &mut [f32], weights: &[f32], dense_index: usize, dense_value: f32) {
    if dense_value == 0.0 {
        return;
    }
    let row_start = dense_index * target.len();
    let row = &weights[row_start..row_start + target.len()];
    for (slot, &weight) in target.iter_mut().zip(row.iter()) {
        *slot += dense_value * weight;
    }
}

fn evaluate_dense_layer(
    layer: &QuantizedDenseLayer,
    activations: &[i8],
    activation_scale: f32,
) -> Vec<f32> {
    debug_assert_eq!(layer.padded_input_size, padded_width(layer.input_size));
    debug_assert_eq!(activations.len(), layer.padded_input_size);
    let dot_scale = activation_scale * layer.weight_scale;
    let mut output = vec![0.0_f32; layer.output_size];
    for (output_index, target_slot) in output.iter_mut().enumerate() {
        let row_start = output_index * layer.padded_input_size;
        let row = &layer.weights[row_start..row_start + layer.padded_input_size];
        let dot = dot_i8_vectors(activations, row);
        *target_slot = (dot as f32 * dot_scale + layer.biases[output_index]).max(0.0);
    }
    output
}

fn evaluate_output_layer(
    layer: &QuantizedOutputLayer,
    activations: &[i8],
    activation_scale: f32,
) -> f32 {
    debug_assert_eq!(layer.padded_input_size, padded_width(layer.input_size));
    debug_assert_eq!(activations.len(), layer.padded_input_size);
    let dot = dot_i8_vectors(activations, &layer.weights);
    dot as f32 * activation_scale * layer.weight_scale + layer.bias
}

fn relu_in_place(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

fn quantize_activations(values: &[f32], scale: f32) -> Vec<i8> {
    let effective_scale = if scale > 0.0 {
        scale
    } else {
        1.0 / ACTIVATION_QUANT_RANGE
    };
    let inverse_scale = 1.0 / effective_scale;
    let mut quantized = vec![0_i8; padded_width(values.len())];
    for (slot, &value) in quantized.iter_mut().zip(values.iter()) {
        *slot = (value.max(0.0) * inverse_scale)
            .round()
            .clamp(-ACTIVATION_QUANT_RANGE, ACTIVATION_QUANT_RANGE) as i8;
    }
    quantized
}

fn quantize_i16(value: f32, scale: f32) -> i16 {
    (value / scale)
        .round()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

fn quantize_i8(value: f32, scale: f32) -> i8 {
    (value / scale)
        .round()
        .clamp(-DENSE_WEIGHT_QUANT_RANGE, DENSE_WEIGHT_QUANT_RANGE) as i8
}

fn quantized_weight_scale(values: &[f32]) -> f32 {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |current, value| current.max(value.abs()));
    if max_abs > 0.0 {
        max_abs / DENSE_WEIGHT_QUANT_RANGE
    } else {
        1.0 / DENSE_WEIGHT_QUANT_RANGE
    }
}

fn quantize_dense_layer(layer: &DenseLayer, input_size: usize) -> QuantizedDenseLayer {
    let padded_input_size = padded_width(input_size);
    let weight_scale = quantized_weight_scale(&layer.weights);
    let mut weights = vec![0_i8; layer.output_size * padded_input_size];
    for output_index in 0..layer.output_size {
        let output_row_start = output_index * padded_input_size;
        for input_index in 0..input_size {
            let source_index = input_index * layer.output_size + output_index;
            weights[output_row_start + input_index] =
                quantize_i8(layer.weights[source_index], weight_scale);
        }
    }
    QuantizedDenseLayer {
        input_size,
        padded_input_size,
        output_size: layer.output_size,
        weights,
        biases: layer.biases.clone(),
        weight_scale,
    }
}

fn quantize_output_layer(weights: &[f32], bias: f32, input_size: usize) -> QuantizedOutputLayer {
    let padded_input_size = padded_width(input_size);
    let weight_scale = quantized_weight_scale(weights);
    let mut quantized = vec![0_i8; padded_input_size];
    for (slot, &value) in quantized.iter_mut().zip(weights.iter()) {
        *slot = quantize_i8(value, weight_scale);
    }
    QuantizedOutputLayer {
        input_size,
        padded_input_size,
        weights: quantized,
        bias,
        weight_scale,
    }
}

fn add_i16_row(accumulator: &mut [i32], row: &[i16]) {
    for (value, &weight) in accumulator.iter_mut().zip(row.iter()) {
        *value += i32::from(weight);
    }
}

fn dot_i8_vectors(lhs: &[i8], rhs: &[i8]) -> i32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| i32::from(a) * i32::from(b))
        .sum()
}

const fn padded_width(width: usize) -> usize {
    if width == 0 {
        0
    } else {
        width.div_ceil(DOT_PRODUCT_CHUNK) * DOT_PRODUCT_CHUNK
    }
}

fn read_array<const N: usize>(cursor: &mut Cursor<&[u8]>) -> Result<[u8; N]> {
    let mut bytes = [0_u8; N];
    cursor.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
    Ok(read_array::<1>(cursor)?[0])
}

fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
    Ok(u16::from_le_bytes(read_array::<2>(cursor)?))
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array::<4>(cursor)?))
}

fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    Ok(f32::from_le_bytes(read_array::<4>(cursor)?))
}

fn read_i16_vec(cursor: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<i16>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(i16::from_le_bytes(read_array::<2>(cursor)?));
    }
    Ok(values)
}

fn read_i32_vec(cursor: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<i32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(i32::from_le_bytes(read_array::<4>(cursor)?));
    }
    Ok(values)
}

fn read_i8_vec(cursor: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<i8>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_u8(cursor)? as i8);
    }
    Ok(values)
}

fn read_f32_vec(cursor: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_f32(cursor)?);
    }
    Ok(values)
}
