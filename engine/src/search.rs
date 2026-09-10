use crate::board::{ALL_DIRECTIONS, Color, Direction, LineAxis, Move, Position, geometry};
use crate::eval::{
    FeatureShape, NnueAccumulator, build_feature_shape, nnue, update_side_feature_shape,
};
use crate::movegen::{MoveApplicationError, PositionState, UndoSnapshot};
use std::cmp::Reverse;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
pub const MAX_GAME_TURNS: u16 = 350;
pub const WIN_SCORE: i32 = 100000;
const SEARCH_SCORE_BOUND: i32 = WIN_SCORE + 10000;
const TRANSPOSITION_TABLE_SIZE: usize = 1 << 16;
const EVAL_CACHE_SIZE: usize = 1 << 17;
const MAX_PLY: usize = 96;
const SHORT_SEARCH_DEADLINE_SLACK_MS: u64 = 4;
const LONG_SEARCH_DEADLINE_SLACK_MS: u64 = 15;
const LONG_SEARCH_DEADLINE_SLACK_THRESHOLD_MS: u64 = 1000;
#[cfg(not(target_arch = "wasm32"))]
const DEPTH_ADMISSION_MARGIN_MS: u64 = 4;
#[cfg(target_arch = "wasm32")]
const DEPTH_ADMISSION_MARGIN_MS: u64 = 1;
const ABORT_POLL_MASK: u64 = 8191;
const RAW_ABORT_POLL_MASK: u64 = 2047;
const ROOT_REVERSE_MOVE_PENALTY: i32 = 50;
const EMERGENCY_EJECTION_BONUS: i32 = 96;
const ASPIRATION_WINDOW: i32 = 80;
const LMR_MIN_DEPTH: u8 = 4;
const LMR_MIN_MOVE_INDEX: usize = 2;
const LMR_DEPTH_DIVISOR: u8 = 4;
const LMR_MOVE_DIVISOR: usize = 12;
const NULL_MOVE_REDUCTION: u8 = 3;
const NULL_MOVE_MIN_DEPTH: u8 = 5;
const FUTILITY_MARGIN_DEPTH1: i32 = 208;
const FUTILITY_MARGIN_DEPTH2: i32 = 495;
const USE_LATE_MOVE_PRUNING: bool = true;
const EVAL_CACHE_WAYS: usize = 4;
const COUNTERMOVE_TABLE_BITS: usize = 15;
const COUNTERMOVE_TABLE_SIZE: usize = 1 << COUNTERMOVE_TABLE_BITS;
const CORRECTION_HISTORY_BITS: usize = 14;
const CORRECTION_HISTORY_SIZE: usize = 1 << CORRECTION_HISTORY_BITS;
const COUNTERMOVE_ORDER_BONUS: i32 = 1750000;
const FOLLOWUP_ORDER_BONUS: i32 = 1500000;
const EJECTION_ORDER_BONUS: i32 = 1250000;
const HISTORY_LMR_THRESHOLD: i32 = 512;
const CORRECTION_HISTORY_CLAMP: i32 = 192;
const NULL_MOVE_MARGIN: i32 = 96;
const PUSH_ORDER_BONUS: i32 = 650000;
const HISTORY_SOURCE_GROUPS_LEN1: usize = crate::board::CELL_COUNT;
const HISTORY_SOURCE_GROUPS_LEN2: usize = combination_count(crate::board::CELL_COUNT, 2);
const HISTORY_SCORE_TABLE_SIZE: usize = 1734;
const EVAL_CACHE_SEED: u64 = 0xA5A55A5A1F2E3D4C;
const SEARCH_CONTEXT_KEY_SEED: u64 = 0xC6A4A7935BD1E995;
const NO_PROGRESS_KEY_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SearchDiagnostics {
    pub last_iteration_ms: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SearchConfig {
    use_fast_movegen: bool,
    use_transposition_backfill: bool,
    partial_sort_k: usize,
}
fn search_config() -> &'static SearchConfig {
    static CFG: SearchConfig = SearchConfig {
        use_fast_movegen: true,
        use_transposition_backfill: false,
        partial_sort_k: 8,
    };
    &CFG
}
#[derive(Clone, Debug, Default)]
struct SearchHistory {
    no_progress: u16,
}
impl SearchHistory {
    fn reset(
        &mut self,
        history_positions: &[PositionKey],
        root_position: PositionKey,
        no_progress_ply: u16,
    ) {
        let _ = (history_positions, root_position);
        self.no_progress = no_progress_ply;
    }
    fn child(&self, made_progress: bool) -> Self {
        Self {
            no_progress: if made_progress {
                0
            } else {
                self.no_progress.saturating_add(1)
            },
        }
    }

    fn current_no_progress(&self) -> u16 {
        self.no_progress
    }
    fn search_key_from_position_hash(&self, position_hash: u64, turn_index: u16) -> u64 {
        let no_progress_key =
            u64::from(self.current_no_progress()).wrapping_mul(NO_PROGRESS_KEY_SEED);
        splitmix64(
            position_hash
                ^ u64::from(turn_index).wrapping_mul(SEARCH_CONTEXT_KEY_SEED)
                ^ no_progress_key,
        )
    }
}

#[derive(Clone, Debug)]
struct EvalCache {
    tagged_buckets: Vec<TaggedEvalBucket>,
    index_bits: u32,
    full_key_buckets: Vec<FullKeyEvalBucket>,
    sets_mask: usize,
}
impl EvalCache {
    fn new(size: usize) -> Self {
        let sets = (size.max(EVAL_CACHE_WAYS) / EVAL_CACHE_WAYS)
            .max(1)
            .next_power_of_two();
        Self {
            tagged_buckets: if sets > 1 {
                vec![
                    TaggedEvalBucket {
                        tags: [0; 4],
                        scores: [0; 4]
                    };
                    sets
                ]
            } else {
                Vec::new()
            },
            index_bits: sets.trailing_zeros(),
            full_key_buckets: if sets == 1 {
                vec![
                    FullKeyEvalBucket {
                        keys: [0; 4],
                        scores: [0; 4],
                        valid: 0
                    };
                    1
                ]
            } else {
                Vec::new()
            },
            sets_mask: sets - 1,
        }
    }
    fn probe(&self, key: u64) -> Option<i32> {
        if self.index_bits == 0 {
            return self.probe_full_key(key);
        }
        let b = &self.tagged_buckets[key as usize & self.sets_mask];
        let tag = (key >> self.index_bits) | (1u64 << 63);
        for w in 0..4 {
            if b.tags[w] == tag {
                return Some(b.scores[w]);
            }
        }
        None
    }
    fn store(&mut self, key: u64, score: i32) {
        if self.index_bits == 0 {
            return self.store_full_key(key, score);
        }
        let b = &mut self.tagged_buckets[key as usize & self.sets_mask];
        let tag = (key >> self.index_bits) | (1u64 << 63);
        let w = (0..4)
            .find(|&w| b.tags[w] == 0 || b.tags[w] == tag)
            .unwrap_or((key >> 32) as usize & 3);
        b.tags[w] = tag;
        b.scores[w] = score;
    }
}
impl Default for EvalCache {
    fn default() -> Self {
        Self::new(EVAL_CACHE_WAYS)
    }
}
struct PersistentContext {
    generation: u8,
    transposition_table: TranspositionTable,
    eval_cache: EvalCache,
    history_scores: [Vec<i16>; 2],
    correction_history: [Vec<i16>; 2],
    countermoves: [Vec<u64>; 2],
}
impl PersistentContext {
    fn new_with_tt_size(transposition_table_size: usize) -> Self {
        Self {
            generation: 1,
            transposition_table: TranspositionTable::new(transposition_table_size),
            eval_cache: EvalCache::new(EVAL_CACHE_SIZE),
            history_scores: [
                vec![0; HISTORY_SCORE_TABLE_SIZE],
                vec![0; HISTORY_SCORE_TABLE_SIZE],
            ],
            correction_history: [
                vec![0; CORRECTION_HISTORY_SIZE],
                vec![0; CORRECTION_HISTORY_SIZE],
            ],
            countermoves: [
                vec![0; COUNTERMOVE_TABLE_SIZE],
                vec![0; COUNTERMOVE_TABLE_SIZE],
            ],
        }
    }
}
thread_local! {
    static PERSISTENT_CONTEXT: std::cell::RefCell<Option<PersistentContext>> =
        const { std::cell::RefCell::new(None) };
}
fn take_persistent_context_with_tt_size(transposition_table_size: usize) -> PersistentContext {
    PERSISTENT_CONTEXT.with(|cell| {
        let mut context = cell
            .borrow_mut()
            .take()
            .unwrap_or_else(|| PersistentContext::new_with_tt_size(transposition_table_size));
        if context.transposition_table.entry_capacity()
            != TranspositionTable::entry_capacity_for_size(transposition_table_size)
        {
            context.transposition_table = TranspositionTable::new(transposition_table_size);
        }
        context
    })
}
fn store_persistent_context(context: PersistentContext) {
    PERSISTENT_CONTEXT.with(|cell| {
        *cell.borrow_mut() = Some(context);
    });
}

fn search_timed_position_with_turn(
    position: &Position,
    position_history: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    time_ms: u64,
    root_reverse_move: Option<Move>,
) -> Result<(SearchResult, SearchDiagnostics), String> {
    let position_history_keys = position_history
        .iter()
        .map(PositionKey::from_position)
        .collect::<Vec<_>>();
    let mut searcher = Searcher::new_timed(
        time_ms,
        root_reverse_move,
        position_history.len() <= 1 && no_progress_ply == 0,
    );
    let result = searcher.search(
        position,
        &position_history_keys,
        no_progress_ply,
        turn_index.min(MAX_GAME_TURNS),
    );
    searcher.persist();
    result
}
fn search_raw_position_with_turn(
    position: &Position,
    position_history: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    time_ms: u64,
    root_reverse_move: Option<Move>,
) -> Result<(SearchResult, SearchDiagnostics), String> {
    let position_history_keys = position_history
        .iter()
        .map(PositionKey::from_position)
        .collect::<Vec<_>>();
    let mut searcher = Searcher::new_timed_with_poll(
        time_ms,
        root_reverse_move,
        position_history.len() <= 1 && no_progress_ply == 0,
        RAW_ABORT_POLL_MASK,
    );
    let result = searcher.search(
        position,
        &position_history_keys,
        no_progress_ply,
        turn_index.min(MAX_GAME_TURNS),
    );
    searcher.persist();
    result
}

fn search_fixed_depth_position_with_turn(
    position: &Position,
    position_history: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    depth: u8,
    root_reverse_move: Option<Move>,
) -> Result<(SearchResult, SearchDiagnostics), String> {
    let position_history_keys = position_history
        .iter()
        .map(PositionKey::from_position)
        .collect::<Vec<_>>();
    let mut searcher = Searcher::new_fixed_depth(
        depth,
        root_reverse_move,
        position_history.len() <= 1 && no_progress_ply == 0,
    );
    let result = searcher.search(
        position,
        &position_history_keys,
        no_progress_ply,
        turn_index.min(MAX_GAME_TURNS),
    );
    searcher.persist();
    result
}

pub fn search_timed_with_turn(
    position: &Position,
    position_history: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    time_ms: u64,
    root_reverse_move: Option<Move>,
) -> Result<SearchResult, String> {
    search_timed_position_with_turn(
        position,
        position_history,
        no_progress_ply,
        turn_index,
        time_ms,
        root_reverse_move,
    )
    .map(|(result, _)| result)
}
pub fn search_timed_depth_with_turn(
    position: &Position,
    position_history: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    time_ms: u64,
    depth: u8,
    root_reverse_move: Option<Move>,
) -> Result<SearchResult, String> {
    let position_history_keys = position_history
        .iter()
        .map(PositionKey::from_position)
        .collect::<Vec<_>>();
    let mut searcher = Searcher::new_timed_depth(
        time_ms,
        depth,
        root_reverse_move,
        position_history.len() <= 1 && no_progress_ply == 0,
    );
    let result = searcher.search(
        position,
        &position_history_keys,
        no_progress_ply,
        turn_index.min(MAX_GAME_TURNS),
    );
    searcher.persist();
    result.map(|(result, _)| result)
}
pub(crate) fn search_raw_with_turn(
    position: &Position,
    position_history: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    time_ms: u64,
    root_reverse_move: Option<Move>,
) -> Result<SearchResult, String> {
    search_raw_position_with_turn(
        position,
        position_history,
        no_progress_ply,
        turn_index,
        time_ms,
        root_reverse_move,
    )
    .map(|(result, _)| result)
}

pub fn search_fixed_depth_with_turn(
    position: &Position,
    position_history: &[Position],
    no_progress_ply: u16,
    turn_index: u16,
    depth: u8,
    root_reverse_move: Option<Move>,
) -> Result<SearchResult, String> {
    search_fixed_depth_position_with_turn(
        position,
        position_history,
        no_progress_ply,
        turn_index,
        depth,
        root_reverse_move,
    )
    .map(|(result, _)| result)
}
pub(crate) struct Searcher {
    config: SearchConfig,
    deadline: Instant,
    abort_poll_mask: u64,
    fixed_depth: Option<u8>,
    enforce_deadline: bool,
    nodes: u64,
    generation: u8,
    transposition_table: TranspositionTable,
    eval_cache: EvalCache,
    accumulators: Vec<NnueAccumulator>,
    feature_shapes: Vec<FeatureShape>,
    killer_keys: Vec<[Option<u32>; 2]>,
    killers: Vec<[Option<Move>; 2]>,
    static_eval_marks: Vec<Option<i32>>,
    history_scores: [Vec<i16>; 2],
    correction_history: [Vec<i16>; 2],
    countermoves: [Vec<u64>; 2],
    followups: [Vec<u64>; 2],
    move_buffers: Vec<Vec<LegalMoveEntry>>,
    scored_move_buffers: Vec<Vec<(i32, LegalMoveEntry)>>,
    root_reverse_move: Option<Move>,
    root_turn: u16,
    root_no_progress: u16,
    root_position_hash: u64,
    diagnostics: SearchDiagnostics,
}
impl Searcher {
    fn new_timed(time_ms: u64, root_reverse_move: Option<Move>, reset_shared: bool) -> Self {
        Self::new_timed_with_poll(time_ms, root_reverse_move, reset_shared, ABORT_POLL_MASK)
    }
    fn new_timed_with_poll(
        time_ms: u64,
        root_reverse_move: Option<Move>,
        reset_shared: bool,
        abort_poll_mask: u64,
    ) -> Self {
        Self::new_timed_with_poll_and_tt_size(
            time_ms,
            root_reverse_move,
            reset_shared,
            abort_poll_mask,
            TRANSPOSITION_TABLE_SIZE,
        )
    }
    fn new_timed_with_poll_and_tt_size(
        time_ms: u64,
        root_reverse_move: Option<Move>,
        reset_shared: bool,
        abort_poll_mask: u64,
        transposition_table_size: usize,
    ) -> Self {
        let safe_time_ms = time_ms.saturating_sub(deadline_slack_ms(time_ms)).max(1);
        Self::with_budget(
            Instant::now() + Duration::from_millis(safe_time_ms),
            abort_poll_mask,
            None,
            true,
            root_reverse_move,
            reset_shared,
            transposition_table_size,
        )
    }
    pub(crate) fn new_timed_depth(
        time_ms: u64,
        depth: u8,
        root_reverse_move: Option<Move>,
        reset_shared: bool,
    ) -> Self {
        Self::new_timed_depth_with_tt_size(
            time_ms,
            depth,
            root_reverse_move,
            reset_shared,
            TRANSPOSITION_TABLE_SIZE,
        )
    }
    fn new_timed_depth_with_tt_size(
        time_ms: u64,
        depth: u8,
        root_reverse_move: Option<Move>,
        reset_shared: bool,
        transposition_table_size: usize,
    ) -> Self {
        let safe_time_ms = time_ms.saturating_sub(deadline_slack_ms(time_ms)).max(1);
        Self::with_budget(
            Instant::now() + Duration::from_millis(safe_time_ms),
            ABORT_POLL_MASK,
            Some(depth.max(1)),
            true,
            root_reverse_move,
            reset_shared,
            transposition_table_size,
        )
    }
    fn new_fixed_depth(depth: u8, root_reverse_move: Option<Move>, reset_shared: bool) -> Self {
        Self::with_budget(
            Instant::now() + Duration::from_secs(24 * 60 * 60),
            ABORT_POLL_MASK,
            Some(depth.max(1)),
            false,
            root_reverse_move,
            reset_shared,
            TRANSPOSITION_TABLE_SIZE,
        )
    }
    fn with_budget(
        deadline: Instant,
        abort_poll_mask: u64,
        fixed_depth: Option<u8>,
        enforce_deadline: bool,
        root_reverse_move: Option<Move>,
        reset_shared: bool,
        transposition_table_size: usize,
    ) -> Self {
        let config = *search_config();
        let mut shared = if !reset_shared {
            take_persistent_context_with_tt_size(transposition_table_size)
        } else {
            PersistentContext::new_with_tt_size(transposition_table_size)
        };
        for table in &mut shared.correction_history {
            for score in table {
                *score = (i32::from(*score) * 113 / 128) as i16;
            }
        }
        let mut history_scores = shared.history_scores;
        for table in &mut history_scores {
            for score in table {
                *score = (i32::from(*score) * 7 / 8) as i16;
            }
        }
        Self {
            config,
            deadline,
            abort_poll_mask,
            fixed_depth,
            enforce_deadline,
            nodes: 0,
            generation: shared.generation,
            transposition_table: shared.transposition_table,
            eval_cache: shared.eval_cache,
            accumulators: Vec::with_capacity(MAX_PLY + 2),
            feature_shapes: Vec::with_capacity(MAX_PLY + 2),
            killer_keys: vec![[None, None]; MAX_PLY],
            killers: vec![[None, None]; MAX_PLY],
            static_eval_marks: vec![None; MAX_PLY + 2],
            history_scores,
            correction_history: shared.correction_history,
            countermoves: shared.countermoves,
            followups: [
                vec![0; COUNTERMOVE_TABLE_SIZE],
                vec![0; COUNTERMOVE_TABLE_SIZE],
            ],
            move_buffers: std::iter::repeat_with(|| Vec::with_capacity(64))
                .take(MAX_PLY + 2)
                .collect(),
            scored_move_buffers: std::iter::repeat_with(|| Vec::with_capacity(64))
                .take(MAX_PLY + 2)
                .collect(),
            root_reverse_move,
            root_turn: 0,
            root_no_progress: 0,
            root_position_hash: 0,
            diagnostics: SearchDiagnostics::default(),
        }
    }
    #[inline]
    fn probe_transposition(&self, key: u64, depth: u8) -> Option<TranspositionEntry> {
        self.transposition_table.probe(key, depth)
    }
    #[inline]
    fn best_transposition_move(&self, key: u64, depth: u8) -> Option<Move> {
        self.transposition_table
            .best_move_entry(key, depth)
            .and_then(|entry| entry.best_move)
    }
    #[inline]
    fn best_root_transposition_move(&self, key: u64, depth: u8) -> Option<Move> {
        self.transposition_table
            .best_move_entry(key, depth)
            .and_then(|entry| entry.best_move)
    }
    #[inline]
    fn store_transposition(&mut self, entry: TranspositionEntry) {
        self.transposition_table.store(entry);
    }
    #[inline]
    fn store_root_transposition(&mut self, entry: TranspositionEntry) {
        self.transposition_table.store(entry);
    }
    #[inline]
    fn turn_at_ply(&self, ply: u8) -> u16 {
        self.root_turn.saturating_add(u16::from(ply))
    }
    #[inline]
    fn terminal_depth_limit(&self) -> u8 {
        MAX_GAME_TURNS
            .saturating_sub(self.root_turn)
            .max(1)
            .min(u16::from(u8::MAX)) as u8
    }
    #[inline]
    fn reaches_terminal_horizon(&self, ply: u8, depth: u8) -> bool {
        self.turn_at_ply(ply).saturating_add(u16::from(depth)) >= MAX_GAME_TURNS
    }
    #[inline]
    fn terminal_horizon_requires_exact_search(&self, ply: u8, depth: u8) -> bool {
        self.reaches_terminal_horizon(ply, depth)
    }
    pub(crate) fn persist(self) {
        store_persistent_context(PersistentContext {
            generation: self.generation,
            transposition_table: self.transposition_table,
            eval_cache: self.eval_cache,
            history_scores: self.history_scores,
            correction_history: self.correction_history,
            countermoves: self.countermoves,
        });
    }
    fn take_move_buffer(&mut self, ply: usize) -> Vec<LegalMoveEntry> {
        let index = ply.min(self.move_buffers.len() - 1);
        std::mem::take(&mut self.move_buffers[index])
    }
    fn recycle_move_buffer(&mut self, ply: usize, mut move_entries: Vec<LegalMoveEntry>) {
        let index = ply.min(self.move_buffers.len() - 1);
        move_entries.clear();
        self.move_buffers[index] = move_entries;
    }
    fn take_scored_move_buffer(&mut self, ply: usize) -> Vec<(i32, LegalMoveEntry)> {
        let index = ply.min(self.scored_move_buffers.len() - 1);
        std::mem::take(&mut self.scored_move_buffers[index])
    }
    fn recycle_scored_move_buffer(&mut self, ply: usize, mut scored: Vec<(i32, LegalMoveEntry)>) {
        let index = ply.min(self.scored_move_buffers.len() - 1);
        scored.clear();
        self.scored_move_buffers[index] = scored;
    }
    fn correction_index(key: u64) -> usize {
        (key as usize) & (CORRECTION_HISTORY_SIZE - 1)
    }
    fn correction_score(&self, side: Color, key: u64) -> i32 {
        i32::from(self.correction_history[side_index(side)][Self::correction_index(key)])
    }
    fn apply_correction(&self, side: Color, key: u64, eval: i32) -> i32 {
        eval.saturating_add(self.correction_score(side, key))
    }
    fn corrected_eval(
        &mut self,
        position_state: &PositionState,
        key: u64,
        position_hash: u64,
        turn_index: u16,
        no_progress: u16,
        raw: &mut Option<i32>,
    ) -> i32 {
        let rv = *raw.get_or_insert_with(|| {
            self.evaluate_position_with_hash(position_state, position_hash, turn_index, no_progress)
        });
        self.apply_correction(position_state.position().side_to_move(), key, rv)
    }
    fn update_correction_history(
        &mut self,
        side: Color,
        key: u64,
        raw_eval: i32,
        searched_score: i32,
        depth: u8,
    ) {
        let index = Self::correction_index(key);
        let slot = &mut self.correction_history[side_index(side)][index];
        let current_correction = i32::from(*slot);
        let target_correction =
            (searched_score - raw_eval).clamp(-CORRECTION_HISTORY_CLAMP, CORRECTION_HISTORY_CLAMP);
        let blend = (i32::from(depth).max(1) + 2).min(8);
        let updated_correction =
            current_correction + (target_correction - current_correction) * blend / 8;
        *slot =
            updated_correction.clamp(-CORRECTION_HISTORY_CLAMP, CORRECTION_HISTORY_CLAMP) as i16;
    }
    fn update_tt_cutoff_correction(
        &mut self,
        position_state: &PositionState,
        key: u64,
        position_hash: u64,
        score: i32,
        depth: u8,
        turn_index: u16,
        no_progress: u16,
        raw: &mut Option<i32>,
    ) {
        let raw_eval = *raw.get_or_insert_with(|| {
            self.evaluate_position_with_hash(position_state, position_hash, turn_index, no_progress)
        });
        self.update_correction_history(
            position_state.position().side_to_move(),
            key,
            raw_eval,
            score,
            depth,
        );
    }
    pub(crate) fn search(
        &mut self,
        position: &Position,
        history_scores: &[PositionKey],
        no_progress_ply: u16,
        turn_index: u16,
    ) -> Result<(SearchResult, SearchDiagnostics), String> {
        let mut position_state = PositionState::new(position.clone()).map_err(|_| String::new())?;
        let root_position_key = PositionKey::from_position(position);
        self.root_position_hash = position_hash(root_position_key);
        let mut history = SearchHistory::default();
        history.reset(history_scores, root_position_key, no_progress_ply);
        self.root_turn = turn_index;
        self.root_no_progress = no_progress_ply;
        self.accumulators.clear();
        self.feature_shapes.clear();

        self.accumulators.push(nnue().root_accumulator(position));
        self.feature_shapes.push(build_feature_shape(
            position_state.black_bits(),
            position_state.white_bits(),
        ));
        let mut best_result = SearchResult {
            best_move: None,
            score: self.evaluate_position_with_hash(
                &position_state,
                self.root_position_hash,
                self.turn_at_ply(0),
                history.current_no_progress(),
            ),
            depth: 0,
            nodes: 0,
        };
        let mut last_iteration_ms = 0u64;
        let mut previous_iteration_ms = 0u64;
        let mut stable_best_iterations = 0u8;
        if terminal_score(position, 0, self.turn_at_ply(0)).is_some() {
            best_result.score = terminal_score(position, 0, self.turn_at_ply(0)).unwrap_or(0);
            return Ok((best_result, self.diagnostics));
        }
        self.advance_generation();
        let mut previous_best_move = best_result.best_move;
        let mut previous_score = best_result.score;
        for depth in 1..=u8::MAX {
            if !self.admit_depth(
                depth,
                best_result.depth,
                last_iteration_ms,
                previous_iteration_ms,
                stable_best_iterations,
            ) {
                break;
            }
            let iteration_started = Instant::now();
            let last_best_before_iteration = previous_best_move;
            match self.search_root(
                &mut position_state,
                depth,
                &mut history,
                previous_best_move,
                previous_score,
                best_result.depth,
            ) {
                Ok((score, best_move)) => {
                    best_result = SearchResult {
                        best_move: best_move,
                        score,
                        depth: depth,
                        nodes: self.nodes,
                    };
                    previous_best_move = best_move;
                    previous_score = score;
                    stable_best_iterations =
                        if best_move.is_some() && best_move == last_best_before_iteration {
                            stable_best_iterations.saturating_add(1)
                        } else {
                            0
                        };
                }
                Err(SearchAbort) => break,
            }
            previous_iteration_ms = last_iteration_ms;
            last_iteration_ms = iteration_started.elapsed().as_millis() as u64;
        }
        if best_result.best_move.is_none() {
            let (fallback_move, fallback_score) = self.emergency_root_choice(&mut position_state);
            best_result.best_move = fallback_move;
            best_result.score = fallback_score;
        }
        self.diagnostics.last_iteration_ms = last_iteration_ms;
        best_result.nodes = self.nodes;
        Ok((best_result, self.diagnostics))
    }
    fn emergency_root_choice(&mut self, position_state: &mut PositionState) -> (Option<Move>, i32) {
        let side = position_state.position().side_to_move();
        let none_killers = [None, None];
        let mut best_move = None;
        let mut best_score = -SEARCH_SCORE_BOUND;
        let mut best_order = i32::MIN;
        let mut move_entries = self.take_move_buffer(0);
        position_state.generate_fast_legal_moves(&mut move_entries);
        for move_entry in move_entries.iter().copied() {
            let candidate_move = move_entry.candidate_move;
            let order =
                self.move_order_score(side, move_entry, None, None, None, None, none_killers);
            self.push_move_entry(position_state, move_entry);
            let undo = self.apply_move_entry(position_state, move_entry).unwrap();
            let child_no_progress = if move_entry.is_ejection {
                0
            } else {
                self.root_no_progress.saturating_add(1)
            };
            let mut score = -terminal_score(position_state.position(), 1, self.turn_at_ply(1))
                .unwrap_or_else(|| {
                    self.evaluate_position(position_state, self.turn_at_ply(1), child_no_progress)
                });
            if move_entry.is_ejection {
                score += EMERGENCY_EJECTION_BONUS;
            }
            if self.root_reverse_move == Some(candidate_move) {
                score -= ROOT_REVERSE_MOVE_PENALTY;
            }
            self.undo_move_entry(position_state, undo);
            self.pop_acc();
            if score > best_score || (score == best_score && order > best_order) {
                best_move = Some(candidate_move);
                best_score = score;
                best_order = order;
            }
        }
        self.recycle_move_buffer(0, move_entries);
        if best_move.is_some() {
            (best_move, best_score)
        } else {
            (
                None,
                terminal_score(position_state.position(), 0, self.turn_at_ply(0)).unwrap_or(0),
            )
        }
    }
    fn search_root(
        &mut self,
        position_state: &mut PositionState,
        depth: u8,
        history: &mut SearchHistory,
        previous_best_move: Option<Move>,
        previous_score: i32,
        completed_depth: u8,
    ) -> Result<(i32, Option<Move>), SearchAbort> {
        let total_marbles = position_state.position().marble_count(Color::Black)
            + position_state.position().marble_count(Color::White);
        if completed_depth == 0 || total_marbles == 28 {
            return self.search_root_window(
                position_state,
                depth,
                history,
                previous_best_move,
                -SEARCH_SCORE_BOUND,
                SEARCH_SCORE_BOUND,
            );
        }
        if self.terminal_horizon_requires_exact_search(0, depth) {
            return self.search_root_window(
                position_state,
                depth,
                history,
                previous_best_move,
                -SEARCH_SCORE_BOUND,
                SEARCH_SCORE_BOUND,
            );
        }
        let aspiration_window = self.root_aspiration_window();
        let alpha = (previous_score - aspiration_window).max(-SEARCH_SCORE_BOUND);
        let beta = (previous_score + aspiration_window).min(SEARCH_SCORE_BOUND);
        let mut delta = aspiration_window;
        let mut alpha = alpha;
        let mut beta = beta;
        loop {
            let window_result = self.search_root_window(
                position_state,
                depth,
                history,
                previous_best_move,
                alpha,
                beta,
            )?;
            if window_result.0 > alpha && window_result.0 < beta {
                return Ok(window_result);
            }
            if delta >= aspiration_window.saturating_mul(4) {
                return self.search_root_window(
                    position_state,
                    depth,
                    history,
                    previous_best_move,
                    -SEARCH_SCORE_BOUND,
                    SEARCH_SCORE_BOUND,
                );
            }
            delta = delta.saturating_mul(2);
            if window_result.0 <= alpha {
                alpha = (window_result.0 - delta.saturating_mul(2)).max(-SEARCH_SCORE_BOUND);
                beta = (window_result.0 + delta).min(SEARCH_SCORE_BOUND);
            } else {
                alpha = (window_result.0 - delta).max(-SEARCH_SCORE_BOUND);
                beta = (window_result.0 + delta.saturating_mul(2)).min(SEARCH_SCORE_BOUND);
            }
        }
    }
    fn root_aspiration_window(&self) -> i32 {
        ASPIRATION_WINDOW
    }
    fn advance_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.generation = 1;
        }
    }
    fn search_root_window(
        &mut self,
        position_state: &mut PositionState,
        depth: u8,
        history: &mut SearchHistory,
        previous_best_move: Option<Move>,
        mut alpha: i32,
        beta: i32,
    ) -> Result<(i32, Option<Move>), SearchAbort> {
        self.check_abort()?;
        let root_key =
            history.search_key_from_position_hash(self.root_position_hash, self.turn_at_ply(0));
        let transposition_move = self.best_root_transposition_move(root_key, depth);
        let mut priority_count = 0usize;
        let mut priority_moves = [None; 4];
        let mut best_score = -SEARCH_SCORE_BOUND;
        let mut best_move = None;
        let original_alpha;
        for candidate_move in [transposition_move, previous_best_move, None]
            .into_iter()
            .flatten()
        {
            if priority_moves[..priority_count].contains(&Some(candidate_move)) {
                continue;
            }
            let Some(move_entry) = position_state.legal_move_entry(&candidate_move) else {
                continue;
            };
            self.check_abort()?;
            let side = position_state.position().side_to_move();
            let is_ejection = move_entry.is_ejection;
            let is_quiet = !is_ejection;
            let history_score = if is_quiet {
                self.history_score(side, move_entry.plan_index)
            } else {
                0
            };
            self.push_move_entry(position_state, move_entry);
            let undo = self.apply_move_entry(position_state, move_entry).unwrap();
            let mut child_history = history.child(move_entry.is_ejection);
            let history = &mut child_history;

            let mut score = match self.search_move_score(
                position_state,
                depth,
                0,
                alpha,
                beta,
                history,
                priority_count,
                is_quiet,
                Some(move_entry.history_key),
                is_ejection,
                history_score,
                None,
            ) {
                Ok(score) => score,
                Err(error) => {
                    self.undo_move_entry(position_state, undo);
                    self.pop_acc();
                    return Err(error);
                }
            };

            self.undo_move_entry(position_state, undo);
            self.pop_acc();
            if self.root_reverse_move == Some(candidate_move) {
                score -= ROOT_REVERSE_MOVE_PENALTY;
            }
            if score > best_score {
                best_score = score;
                best_move = Some(candidate_move);
            }
            if score > alpha {
                alpha = score;
            }
            priority_moves[priority_count] = Some(candidate_move);
            priority_count += 1;
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(0, candidate_move);
                }
                self.store_root_transposition(TranspositionEntry {
                    key: root_key,
                    depth: depth,
                    generation: self.generation,
                    score: encode_tt_score(score, 0),
                    bound: BoundKind::Lower,
                    best_move: Some(candidate_move),
                    exact_terminal: self.terminal_horizon_requires_exact_search(0, depth),
                });
                return Ok((score, Some(candidate_move)));
            }
        }
        original_alpha = alpha;
        let mut move_entries = self.take_move_buffer(0);
        position_state.generate_fast_legal_moves(&mut move_entries);
        if priority_count > 0 {
            move_entries.retain(|entry| !priority_moves.contains(&Some(entry.candidate_move)));
        }
        let move_entries = self.order_moves_scored(
            position_state.position().side_to_move(),
            &mut move_entries,
            previous_best_move,
            transposition_move,
            None,
            None,
            0,
        );
        if move_entries.is_empty() {
            self.recycle_scored_move_buffer(0, move_entries);
            if best_move.is_some() {
                return Ok((best_score, best_move));
            }
            return Ok((
                terminal_score(position_state.position(), 0, self.turn_at_ply(0)).unwrap_or(0),
                None,
            ));
        }
        for (move_index, move_entry) in move_entries.iter().map(|x| x.1).enumerate() {
            let candidate_move = move_entry.candidate_move;
            let side = position_state.position().side_to_move();
            let is_ejection = move_entry.is_ejection;
            let is_quiet = !is_ejection;
            let history_score = if is_quiet {
                self.history_score(side, move_entry.plan_index)
            } else {
                0
            };
            self.push_move_entry(position_state, move_entry);
            let undo = self.apply_move_entry(position_state, move_entry).unwrap();
            let mut child_history = history.child(move_entry.is_ejection);
            let history = &mut child_history;

            let mut score = match self.search_move_score(
                position_state,
                depth,
                0,
                alpha,
                beta,
                history,
                move_index + priority_count,
                is_quiet,
                Some(move_entry.history_key),
                is_ejection,
                history_score,
                None,
            ) {
                Ok(score) => score,
                Err(error) => {
                    self.undo_move_entry(position_state, undo);
                    self.pop_acc();
                    self.recycle_scored_move_buffer(0, move_entries);
                    return Err(error);
                }
            };

            self.undo_move_entry(position_state, undo);
            self.pop_acc();
            if self.root_reverse_move == Some(candidate_move) {
                score -= ROOT_REVERSE_MOVE_PENALTY;
            }
            if score > best_score {
                best_score = score;
                best_move = Some(candidate_move);
            }
            let raised_alpha = score > alpha;
            if raised_alpha {
                alpha = score;
            }
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(0, candidate_move);
                }
                self.store_root_transposition(TranspositionEntry {
                    key: root_key,
                    depth: depth,
                    generation: self.generation,
                    score: encode_tt_score(score, 0),
                    bound: BoundKind::Lower,
                    best_move: Some(candidate_move),
                    exact_terminal: self.terminal_horizon_requires_exact_search(0, depth),
                });
                self.recycle_scored_move_buffer(0, move_entries);
                return Ok((score, Some(candidate_move)));
            }
        }
        self.store_root_transposition(TranspositionEntry {
            key: root_key,
            depth: depth,
            generation: self.generation,
            score: encode_tt_score(best_score, 0),
            bound: if best_score <= original_alpha {
                BoundKind::Upper
            } else {
                BoundKind::Exact
            },
            best_move: best_move,
            exact_terminal: self.terminal_horizon_requires_exact_search(0, depth),
        });
        self.recycle_scored_move_buffer(0, move_entries);
        Ok((best_score, best_move))
    }
    fn search_move_score(
        &mut self,
        position_state: &mut PositionState,
        depth: u8,
        ply: u8,
        alpha: i32,
        beta: i32,
        history: &mut SearchHistory,
        move_index: usize,
        is_quiet: bool,
        previous_history_key: Option<u32>,
        _is_ejection: bool,
        history_score: i32,
        own_previous_key: Option<u32>,
    ) -> Result<i32, SearchAbort> {
        if move_index == 0 {
            return self
                .negamax(
                    position_state,
                    depth.saturating_sub(1),
                    ply + 1,
                    -beta,
                    -alpha,
                    history,
                    true,
                    previous_history_key,
                    own_previous_key,
                )
                .map(|score| -score);
        }
        let can_reduce = is_quiet
            && depth >= LMR_MIN_DEPTH
            && move_index >= LMR_MIN_MOVE_INDEX
            && history_score < HISTORY_LMR_THRESHOLD;
        let reduction = if can_reduce {
            let mut tuned = late_move_reduction(depth, move_index);
            if history_score <= -2049 {
                tuned = tuned.saturating_add(1);
            } else if history_score > 4096 {
                tuned = tuned.saturating_sub(1);
            }
            let tuned = tuned.min(depth.saturating_sub(2));
            if self.terminal_horizon_requires_exact_search(ply, depth) {
                0
            } else {
                tuned
            }
        } else {
            0
        };
        let scout_depth = depth.saturating_sub(1 + reduction);
        let mut score = -self.negamax(
            position_state,
            scout_depth,
            ply + 1,
            -(alpha + 1),
            -alpha,
            history,
            true,
            previous_history_key,
            own_previous_key,
        )?;
        if reduction > 0 && score > alpha {
            score = -self.negamax(
                position_state,
                depth.saturating_sub(1),
                ply + 1,
                -(alpha + 1),
                -alpha,
                history,
                true,
                previous_history_key,
                own_previous_key,
            )?;
        }
        if score > alpha && score < beta {
            score = -self.negamax(
                position_state,
                depth.saturating_sub(1),
                ply + 1,
                -beta,
                -alpha,
                history,
                true,
                previous_history_key,
                own_previous_key,
            )?;
        }
        Ok(score)
    }
    fn negamax(
        &mut self,
        position_state: &mut PositionState,
        depth: u8,
        ply: u8,
        mut alpha: i32,
        beta: i32,
        history: &mut SearchHistory,
        allow_null: bool,
        previous_history_key: Option<u32>,
        own_previous_key: Option<u32>,
    ) -> Result<i32, SearchAbort> {
        self.check_abort()?;
        self.nodes += 1;
        let current = PositionKey::from_state(position_state);
        if let Some(score) = terminal_score(position_state.position(), ply, self.turn_at_ply(ply)) {
            return Ok(score);
        }
        let mut beta = beta;
        let current_position_hash = position_hash(current);
        let key =
            history.search_key_from_position_hash(current_position_hash, self.turn_at_ply(ply));
        let exact_terminal_horizon = self.terminal_horizon_requires_exact_search(ply, depth);
        let mut raw_static = None;
        if let Some(entry) = self.probe_transposition(key, depth) {
            let score = decode_tt_score(entry.score, ply);
            let tt_cutoffs_allowed = !exact_terminal_horizon || entry.exact_terminal;
            match entry.bound {
                BoundKind::Exact if tt_cutoffs_allowed => {
                    self.update_tt_cutoff_correction(
                        position_state,
                        key,
                        current_position_hash,
                        score,
                        depth,
                        self.turn_at_ply(ply),
                        history.current_no_progress(),
                        &mut raw_static,
                    );
                    return Ok(score);
                }
                BoundKind::Lower if tt_cutoffs_allowed => alpha = alpha.max(score),
                BoundKind::Upper if tt_cutoffs_allowed && score <= alpha => {
                    self.update_tt_cutoff_correction(
                        position_state,
                        key,
                        current_position_hash,
                        score,
                        depth,
                        self.turn_at_ply(ply),
                        history.current_no_progress(),
                        &mut raw_static,
                    );
                    return Ok(score);
                }
                BoundKind::Upper if tt_cutoffs_allowed => beta = beta.min(score),
                _ => {}
            }
            if alpha >= beta && tt_cutoffs_allowed {
                self.update_tt_cutoff_correction(
                    position_state,
                    key,
                    current_position_hash,
                    score,
                    depth,
                    self.turn_at_ply(ply),
                    history.current_no_progress(),
                    &mut raw_static,
                );
                return Ok(score);
            }
        }
        if depth == 0 {
            return self.tactical_leaf_score(
                position_state,
                ply,
                alpha,
                beta,
                history,
                key,
                current_position_hash,
                &mut raw_static,
            );
        }
        let static_score = if alpha > -WIN_SCORE / 2 && beta < WIN_SCORE / 2 {
            Some(self.corrected_eval(
                position_state,
                key,
                current_position_hash,
                self.turn_at_ply(ply),
                history.current_no_progress(),
                &mut raw_static,
            ))
        } else {
            None
        };
        if let Some(slot) = self.static_eval_marks.get_mut(ply as usize) {
            *slot = static_score;
        }
        let improving = ply >= 2
            && match (
                static_score,
                self.static_eval_marks
                    .get(ply as usize - 2)
                    .copied()
                    .flatten(),
            ) {
                (Some(current), Some(previous)) => current > previous,
                _ => false,
            };
        if !exact_terminal_horizon && ply > 0 && depth <= 4 {
            let static_score = static_score.unwrap_or_else(|| {
                self.corrected_eval(
                    position_state,
                    key,
                    current_position_hash,
                    self.turn_at_ply(ply),
                    history.current_no_progress(),
                    &mut raw_static,
                )
            });
            if static_score
                .saturating_sub(90 + 90 * i32::from(depth) - if improving { 90 } else { 0 })
                >= beta
            {
                return Ok(static_score);
            }
        }
        let mut futility_prune_quiets = false;
        if !exact_terminal_horizon && depth <= 2 && alpha > -WIN_SCORE / 2 && beta < WIN_SCORE / 2 {
            let static_score = static_score.unwrap_or_else(|| {
                self.corrected_eval(
                    position_state,
                    key,
                    current_position_hash,
                    self.turn_at_ply(ply),
                    history.current_no_progress(),
                    &mut raw_static,
                )
            });
            let margin = if depth == 1 {
                FUTILITY_MARGIN_DEPTH1
            } else {
                FUTILITY_MARGIN_DEPTH2
            };
            if static_score.saturating_add(margin) <= alpha {
                futility_prune_quiets = true;
            }
        }
        if allow_null
            && !exact_terminal_horizon
            && ply > 0
            && depth >= NULL_MOVE_MIN_DEPTH
            && depth > 3
            && self.turn_at_ply(ply).saturating_add(1) < MAX_GAME_TURNS
        {
            let null_gate = static_score.unwrap_or_else(|| {
                self.corrected_eval(
                    position_state,
                    key,
                    current_position_hash,
                    self.turn_at_ply(ply),
                    history.current_no_progress(),
                    &mut raw_static,
                )
            });
            if null_gate >= beta.saturating_sub(NULL_MOVE_MARGIN) {
                let mut null_reduction = NULL_MOVE_REDUCTION.saturating_add(depth / 6);
                if null_gate > beta {
                    let bonus = ((null_gate - beta) / 200).clamp(0, 2) as u8;
                    null_reduction = null_reduction.saturating_add(bonus);
                }
                null_reduction = null_reduction.min(depth.saturating_sub(2));
                let parent = self.accumulators.last().cloned().unwrap();
                self.accumulators.push(parent);
                self.feature_shapes
                    .push(*self.feature_shapes.last().unwrap());
                let previous_side_to_move = position_state.pass_turn();
                let mut child_history = history.child(false);
                let history = &mut child_history;

                let null_score = -self.negamax(
                    position_state,
                    depth - 1 - null_reduction,
                    ply + 1,
                    -beta,
                    -beta + 1,
                    history,
                    false,
                    None,
                    previous_history_key,
                )?;

                position_state.restore_side_to_move(previous_side_to_move);
                self.pop_acc();
                if null_score >= beta {
                    return Ok(beta);
                }
            }
        }
        let side = position_state.position().side_to_move();
        let mut transposition_move = self.best_transposition_move(key, depth);
        if !self.config.use_transposition_backfill && transposition_move.is_none() && depth >= 4 {
            transposition_move = self.best_transposition_move(key, depth.saturating_sub(1));
        }
        let countermove_key = self.probe_countermove(side, previous_history_key);
        let followup_key = self.probe_followup(side, own_previous_key);
        let original_alpha = alpha;
        let mut best_score = -SEARCH_SCORE_BOUND;
        let mut best_move = None;
        let killers = self
            .killers
            .get(ply as usize)
            .copied()
            .unwrap_or([None, None]);
        let tt_key = transposition_move.map(|m| history_group_key(m.source_cells(), m.direction()));
        let captured_killer_keys = self
            .killer_keys
            .get(ply as usize)
            .copied()
            .unwrap_or([None, None]);
        let mut priority_ids = [u16::MAX; 4];
        let mut priority_count = 0usize;
        let mut priority_moves = [None; 4];
        let mut priority_quiet_tried = [0u16; 4];
        let mut priority_quiet_tried_count = 0usize;
        for candidate_move in [transposition_move, killers[0], killers[1], None]
            .into_iter()
            .flatten()
        {
            if priority_moves[..priority_count].contains(&Some(candidate_move)) {
                continue;
            }
            let Some(move_entry) = position_state.legal_move_entry(&candidate_move) else {
                continue;
            };
            self.check_abort()?;
            if futility_prune_quiets && !move_entry.is_push {
                continue;
            }
            let is_ejection = move_entry.is_ejection;
            let is_quiet = !is_ejection;
            let history_score = if is_quiet {
                self.history_score(side, move_entry.plan_index)
            } else {
                0
            };
            self.push_move_entry(position_state, move_entry);
            let undo = self.apply_move_entry(position_state, move_entry).unwrap();
            let mut child_history = history.child(move_entry.is_ejection);
            let history = &mut child_history;

            let score = self.search_move_score(
                position_state,
                depth,
                ply,
                alpha,
                beta,
                history,
                priority_count,
                is_quiet,
                Some(move_entry.history_key),
                is_ejection,
                history_score,
                previous_history_key,
            )?;

            self.undo_move_entry(position_state, undo);
            self.pop_acc();
            if score > best_score {
                best_score = score;
                best_move = Some(candidate_move);
            }
            if score > alpha {
                alpha = score;
            }
            priority_ids[priority_count] = move_entry.plan_index;
            priority_moves[priority_count] = Some(candidate_move);
            priority_count += 1;
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(ply as usize, candidate_move);
                    self.reward_history(side, move_entry.plan_index, depth);
                    self.record_countermove(side, previous_history_key, move_entry.history_key);
                    self.record_followup(side, own_previous_key, move_entry.history_key);
                    for history_key in priority_quiet_tried
                        .into_iter()
                        .take(priority_quiet_tried_count)
                    {
                        self.penalize_history(side, history_key, depth);
                    }
                }
                self.store_transposition(TranspositionEntry {
                    key,
                    depth: depth,
                    generation: self.generation,
                    score: encode_tt_score(score, ply),
                    bound: BoundKind::Lower,
                    best_move: Some(candidate_move),
                    exact_terminal: exact_terminal_horizon,
                });
                if let Some(raw) = raw_static {
                    self.update_correction_history(side, key, raw, score, depth);
                }
                return Ok(beta);
            }
            if is_quiet && priority_quiet_tried_count < priority_quiet_tried.len() {
                priority_quiet_tried[priority_quiet_tried_count] = move_entry.plan_index;
                priority_quiet_tried_count += 1;
            }
        }
        let mut move_entries = self.take_move_buffer(ply as usize);
        position_state.generate_fast_legal_moves(&mut move_entries);
        if priority_count > 0 {
            move_entries
                .retain(|entry| !priority_ids[..priority_count].contains(&entry.plan_index));
        }
        if move_entries.is_empty() {
            self.recycle_move_buffer(ply as usize, move_entries);
            return if best_move.is_some() {
                Ok(best_score)
            } else {
                Ok(0)
            };
        }
        let move_entries = self.order_moves_scored(
            side,
            &mut move_entries,
            None,
            transposition_move,
            countermove_key,
            followup_key,
            ply,
        );
        let mut quiet_tried = [0u16; 64];
        let mut quiet_tried_count = 0usize;
        for (move_index, move_entry) in move_entries.iter().map(|x| x.1).enumerate() {
            let candidate_move = move_entry.candidate_move;
            let is_ejection = move_entry.is_ejection;
            let is_quiet = !is_ejection;
            let history_score = if is_quiet {
                self.history_score(side, move_entry.plan_index)
            } else {
                0
            };
            if futility_prune_quiets && !move_entry.is_push {
                continue;
            }
            if USE_LATE_MOVE_PRUNING
                && !self.terminal_horizon_requires_exact_search(ply, depth)
                && ply > 0
                && depth <= 5
                && move_index
                    >= (3 + depth as usize * depth as usize) / (if improving { 1 } else { 2 })
                && is_quiet
                && history_score <= 0
                && Some(move_entry.history_key) != tt_key
                && Some(move_entry.history_key) != captured_killer_keys[0]
                && Some(move_entry.history_key) != captured_killer_keys[1]
                && Some(move_entry.history_key) != countermove_key
            {
                continue;
            }
            self.push_move_entry(position_state, move_entry);
            let undo = self.apply_move_entry(position_state, move_entry).unwrap();
            let mut child_history = history.child(move_entry.is_ejection);
            let history = &mut child_history;

            let score = self.search_move_score(
                position_state,
                depth,
                ply,
                alpha,
                beta,
                history,
                move_index + priority_count,
                is_quiet,
                Some(move_entry.history_key),
                is_ejection,
                history_score,
                previous_history_key,
            )?;

            self.undo_move_entry(position_state, undo);
            self.pop_acc();
            if score > best_score {
                best_score = score;
                best_move = Some(candidate_move);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if is_quiet {
                    self.record_killer(ply as usize, candidate_move);
                    self.reward_history(side, move_entry.plan_index, depth);
                    self.record_countermove(side, previous_history_key, move_entry.history_key);
                    self.record_followup(side, own_previous_key, move_entry.history_key);
                    for history_key in priority_quiet_tried
                        .into_iter()
                        .take(priority_quiet_tried_count)
                    {
                        self.penalize_history(side, history_key, depth);
                    }
                    for history_key in quiet_tried.into_iter().take(quiet_tried_count) {
                        self.penalize_history(side, history_key, depth);
                    }
                }
                self.store_transposition(TranspositionEntry {
                    key,
                    depth: depth,
                    generation: self.generation,
                    score: encode_tt_score(score, ply),
                    bound: BoundKind::Lower,
                    best_move: Some(candidate_move),
                    exact_terminal: exact_terminal_horizon,
                });
                if let Some(raw) = raw_static {
                    self.update_correction_history(side, key, raw, score, depth);
                }
                self.recycle_scored_move_buffer(ply as usize, move_entries);
                return Ok(beta);
            }
            if is_quiet && quiet_tried_count < quiet_tried.len() {
                quiet_tried[quiet_tried_count] = move_entry.plan_index;
                quiet_tried_count += 1;
            }
        }
        if best_move.is_none() {
            self.recycle_scored_move_buffer(ply as usize, move_entries);
            return Ok(alpha);
        }
        self.store_transposition(TranspositionEntry {
            key,
            depth: depth,
            generation: self.generation,
            score: encode_tt_score(best_score, ply),
            bound: if best_score <= original_alpha {
                BoundKind::Upper
            } else {
                BoundKind::Exact
            },
            best_move: best_move,
            exact_terminal: exact_terminal_horizon,
        });
        if let Some(raw) = raw_static {
            self.update_correction_history(side, key, raw, best_score, depth);
        }
        self.recycle_scored_move_buffer(ply as usize, move_entries);
        Ok(best_score)
    }
    fn tactical_leaf_score(
        &mut self,
        position_state: &mut PositionState,
        ply: u8,
        mut alpha: i32,
        beta: i32,
        history: &mut SearchHistory,
        key: u64,
        position_hash: u64,
        raw_static: &mut Option<i32>,
    ) -> Result<i32, SearchAbort> {
        let mut best = self.corrected_eval(
            position_state,
            key,
            position_hash,
            self.turn_at_ply(ply),
            history.current_no_progress(),
            raw_static,
        );
        if best >= beta || self.terminal_horizon_requires_exact_search(ply, 0) {
            return Ok(best);
        }
        if best > alpha {
            alpha = best;
        }

        let side = position_state.position().side_to_move();
        let mut move_entries = self.take_move_buffer(ply as usize);
        position_state.generate_fast_push_moves(&mut move_entries);
        let move_entries =
            self.order_moves_scored(side, &mut move_entries, None, None, None, None, ply);

        for move_entry in move_entries.iter().map(|x| x.1) {
            self.check_abort()?;
            self.push_move_entry(position_state, move_entry);
            let undo = self.apply_move_entry(position_state, move_entry).unwrap();
            let mut child_history = history.child(move_entry.is_ejection);
            let history = &mut child_history;

            let score = -terminal_score(
                position_state.position(),
                ply + 1,
                self.turn_at_ply(ply + 1),
            )
            .unwrap_or_else(|| {
                self.evaluate_position(
                    position_state,
                    self.turn_at_ply(ply + 1),
                    history.current_no_progress(),
                )
            });

            self.undo_move_entry(position_state, undo);
            self.pop_acc();

            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }

        self.recycle_scored_move_buffer(ply as usize, move_entries);
        Ok(best)
    }

    fn order_moves_scored(
        &mut self,
        side: Color,
        move_entries: &mut Vec<LegalMoveEntry>,
        principal_variation_move: Option<Move>,
        transposition_move: Option<Move>,
        countermove_key: Option<u32>,
        followup_key: Option<u32>,
        ply: u8,
    ) -> Vec<(i32, LegalMoveEntry)> {
        let key = |m: Move| history_group_key(m.source_cells(), m.direction());
        let principal_variation_move = principal_variation_move.map(key);
        let transposition_move = transposition_move.map(key);
        let killers = self
            .killer_keys
            .get(ply as usize)
            .copied()
            .unwrap_or([None, None]);
        let mut scored = self.take_scored_move_buffer(ply as usize);
        let ctx = MoveOrderContext {
            history: &self.history_scores[side_index(side)],
            keys: [
                principal_variation_move,
                transposition_move,
                killers[0],
                killers[1],
                countermove_key,
                followup_key,
            ]
            .map(|k| k.unwrap_or(u32::MAX)),
        };
        scored.extend(
            move_entries
                .iter()
                .copied()
                .map(|e| (context_move_order_score(&ctx, e), e)),
        );
        let partial_sort_k = self.config.partial_sort_k.min(scored.len());
        if partial_sort_k < scored.len() {
            let pivot = partial_sort_k - 1;
            scored.select_nth_unstable_by_key(pivot, |entry| Reverse(entry.0));
            scored[..partial_sort_k].sort_unstable_by_key(|entry| Reverse(entry.0));
        } else {
            scored.sort_unstable_by_key(|entry| Reverse(entry.0));
        }
        self.recycle_move_buffer(ply as usize, std::mem::take(move_entries));
        scored
    }

    fn move_order_score(
        &self,
        side: Color,
        move_entry: LegalMoveEntry,
        principal_variation_move: Option<Move>,
        transposition_move: Option<Move>,
        countermove_key: Option<u32>,
        followup_key: Option<u32>,
        killers: [Option<Move>; 2],
    ) -> i32 {
        let key = |m: Move| history_group_key(m.source_cells(), m.direction());
        self.move_order_score_keys(
            side,
            move_entry,
            principal_variation_move.map(key),
            transposition_move.map(key),
            countermove_key,
            followup_key,
            killers.map(|m| m.map(key)),
        )
    }
    fn move_order_score_keys(
        &self,
        side: Color,
        move_entry: LegalMoveEntry,
        principal_variation_move: Option<u32>,
        transposition_move: Option<u32>,
        countermove_key: Option<u32>,
        followup_key: Option<u32>,
        killers: [Option<u32>; 2],
    ) -> i32 {
        let candidate_move = move_entry.history_key;
        if Some(candidate_move) == principal_variation_move {
            return 4000000;
        }
        if Some(candidate_move) == transposition_move {
            return 3000000;
        }
        if Some(candidate_move) == killers[0] {
            return 2000000;
        }
        if Some(candidate_move) == killers[1] {
            return 1000000;
        }
        if Some(move_entry.history_key) == countermove_key {
            return COUNTERMOVE_ORDER_BONUS + self.history_score(side, move_entry.plan_index);
        }
        if Some(move_entry.history_key) == followup_key {
            return FOLLOWUP_ORDER_BONUS + self.history_score(side, move_entry.plan_index);
        }
        if move_entry.is_ejection {
            return EJECTION_ORDER_BONUS + self.history_score(side, move_entry.plan_index);
        }
        if move_entry.is_push {
            return PUSH_ORDER_BONUS
                + i32::try_from(move_entry.candidate_move.len()).unwrap_or(0) * 10000
                + self.history_score(side, move_entry.plan_index);
        }
        self.history_score(side, move_entry.plan_index)
    }
    fn history_score(&self, side: Color, plan_index: u16) -> i32 {
        i32::from(self.history_scores[side_index(side)][plan_index as usize])
    }
    fn reward_history(&mut self, side: Color, plan_index: u16, depth: u8) {
        let bonus = i16::try_from(i32::from(depth) * i32::from(depth)).unwrap_or(i16::MAX);
        {
            let slot = &mut self.history_scores[side_index(side)][plan_index as usize];
            let slot_value = i32::from(*slot);
            let bonus_value = i32::from(bonus);
            let updated = slot_value + bonus_value - ((slot_value * bonus_value) / 16384);
            *slot = updated.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
    }
    fn penalize_history(&mut self, side: Color, plan_index: u16, depth: u8) {
        let malus = i16::try_from(i32::from(depth) * i32::from(depth)).unwrap_or(i16::MAX);
        {
            let slot = &mut self.history_scores[side_index(side)][plan_index as usize];
            let slot_value = i32::from(*slot);
            let malus_value = i32::from(malus);
            let updated = slot_value - malus_value - ((slot_value * malus_value) / 16384);
            *slot = updated.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
    }
    fn probe_countermove(&self, side: Color, previous_history_key: Option<u32>) -> Option<u32> {
        let previous_history_key = previous_history_key?;
        let entry = self.countermoves[side_index(side)]
            [previous_history_key as usize & (COUNTERMOVE_TABLE_SIZE - 1)];
        let stored_previous = (entry >> 32) as u32;
        let stored_reply = entry as u32;
        if stored_previous == previous_history_key && stored_reply != 0 {
            Some(stored_reply - 1)
        } else {
            None
        }
    }
    fn record_countermove(
        &mut self,
        side: Color,
        previous_history_key: Option<u32>,
        reply_key: u32,
    ) {
        let Some(previous_history_key) = previous_history_key else {
            return;
        };
        let index = previous_history_key as usize & (COUNTERMOVE_TABLE_SIZE - 1);
        self.countermoves[side_index(side)][index] =
            (u64::from(previous_history_key) << 32) | u64::from(reply_key.saturating_add(1));
    }
    fn probe_followup(&self, side: Color, own_previous_key: Option<u32>) -> Option<u32> {
        let own_previous_key = own_previous_key?;
        let entry = self.followups[side_index(side)]
            [own_previous_key as usize & (COUNTERMOVE_TABLE_SIZE - 1)];
        let stored_previous = (entry >> 32) as u32;
        let stored_reply = entry as u32;
        if stored_previous == own_previous_key && stored_reply != 0 {
            Some(stored_reply - 1)
        } else {
            None
        }
    }
    fn record_followup(&mut self, side: Color, own_previous_key: Option<u32>, reply_key: u32) {
        let Some(own_previous_key) = own_previous_key else {
            return;
        };
        let index = own_previous_key as usize & (COUNTERMOVE_TABLE_SIZE - 1);
        self.followups[side_index(side)][index] =
            (u64::from(own_previous_key) << 32) | u64::from(reply_key.saturating_add(1));
    }
    fn current_shape(&mut self, position_state: &PositionState) -> &FeatureShape {
        let _ = position_state;
        self.feature_shapes.last().unwrap()
    }
    fn push_move_state(&mut self, position_state: &PositionState, move_entry: LegalMoveEntry) {
        let model = nnue();
        let mut next = self.accumulators.last().cloned().unwrap();
        let mut shape = *self.feature_shapes.last().unwrap();
        let position = position_state.position();
        let side = position.side_to_move();
        let own_bits = position.bits_for(side);
        let enemy_bits = position.bits_for(side.other());
        let own_toggle = fast_movegen_tables().own_toggle(move_entry.plan_index);
        let enemy_toggle = move_entry.enemy_effect.toggle_mask();
        for (color, before, toggle) in [
            (side, own_bits, own_toggle),
            (side.other(), enemy_bits, enemy_toggle),
        ] {
            let mut removed = before & toggle;
            while removed != 0 {
                let cell = crate::board::CellId::new_unchecked(removed.trailing_zeros() as u8);
                model.apply_sparse_delta(&mut next, color, cell, -1);
                removed &= removed - 1;
            }
            let mut added = !before & toggle;
            while added != 0 {
                let cell = crate::board::CellId::new_unchecked(added.trailing_zeros() as u8);
                model.apply_sparse_delta(&mut next, color, cell, 1);
                added &= added - 1;
            }
        }
        let own_after = own_bits ^ own_toggle;
        let enemy_after = enemy_bits ^ enemy_toggle;
        self.accumulators.push(next);
        match side {
            Color::Black => {
                shape.black = update_side_feature_shape(shape.black, own_bits, own_after);
                if enemy_toggle != 0 {
                    shape.white = update_side_feature_shape(shape.white, enemy_bits, enemy_after);
                }
            }
            Color::White => {
                shape.white = update_side_feature_shape(shape.white, own_bits, own_after);
                if enemy_toggle != 0 {
                    shape.black = update_side_feature_shape(shape.black, enemy_bits, enemy_after);
                }
            }
        }
        self.feature_shapes.push(shape);
    }
    fn push_move_entry(&mut self, position_state: &PositionState, move_entry: LegalMoveEntry) {
        self.push_move_state(position_state, move_entry);
    }
    fn apply_move_entry(
        &mut self,
        position_state: &mut PositionState,
        move_entry: LegalMoveEntry,
    ) -> Result<UndoSnapshot, MoveApplicationError> {
        debug_assert_eq!(
            position_state.legal_move_entry(&move_entry.candidate_move),
            Some(move_entry)
        );
        Ok(position_state.apply_legal_effect(move_entry.plan_index, move_entry.enemy_effect))
    }
    fn undo_move_entry(&mut self, position_state: &mut PositionState, undo: UndoSnapshot) {
        position_state.undo_move(undo);
    }
    fn pop_acc(&mut self) {
        let _ = self.accumulators.pop();
        let _ = self.feature_shapes.pop();
    }
    fn evaluate_position(
        &mut self,
        position_state: &PositionState,
        turn_index: u16,
        no_progress_ply: u16,
    ) -> i32 {
        let position_key = PositionKey::from_state(position_state);
        self.evaluate_position_with_hash(
            position_state,
            position_hash(position_key),
            turn_index,
            no_progress_ply,
        )
    }
    fn evaluate_position_with_hash(
        &mut self,
        position_state: &PositionState,
        position_hash: u64,
        turn_index: u16,
        no_progress_ply: u16,
    ) -> i32 {
        let key = position_hash
            ^ ((turn_index as u64) << 48)
            ^ ((no_progress_ply as u64) << 32)
            ^ EVAL_CACHE_SEED;
        if let Some(score) = self.eval_cache.probe(key) {
            return score;
        }
        let shape = *self.current_shape(position_state);
        let score = nnue().evaluate_with_accumulator_bits(
            position_state.position().side_to_move() == Color::Black,
            &shape,
            position_state.black_bits(),
            position_state.white_bits(),
            turn_index as f32,
            no_progress_ply as f32,
            self.accumulators.last().unwrap(),
        );
        self.eval_cache.store(key, score);
        score
    }
    fn admit_depth(
        &self,
        depth: u8,
        completed_depth: u8,
        last_iteration_ms: u64,
        previous_iteration_ms: u64,
        stable_best_iterations: u8,
    ) -> bool {
        let terminal_limit = self.terminal_depth_limit();
        if depth > terminal_limit || completed_depth >= terminal_limit {
            return false;
        }
        if let Some(limit) = self.fixed_depth {
            if !self.enforce_deadline {
                return depth <= limit && completed_depth < limit;
            }
            if depth > limit || completed_depth >= limit {
                return false;
            }
        }
        if depth <= 1 || completed_depth == 0 || last_iteration_ms == 0 {
            return Instant::now() < self.deadline;
        }
        let remaining_ms = self
            .deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as u64;
        let ratio_estimate = if previous_iteration_ms > 0 {
            last_iteration_ms
                .saturating_mul(last_iteration_ms)
                .div_ceil(previous_iteration_ms)
                .clamp(
                    last_iteration_ms.saturating_mul(7).div_ceil(4),
                    last_iteration_ms.saturating_mul(5).div_ceil(2),
                )
        } else {
            last_iteration_ms.saturating_mul(39).div_ceil(20)
        };
        let estimated_next_ms = ratio_estimate.max(last_iteration_ms + 1);
        let _ = stable_best_iterations;
        estimated_next_ms <= remaining_ms.saturating_add(DEPTH_ADMISSION_MARGIN_MS)
    }
    fn record_killer(&mut self, ply: usize, candidate_move: Move) {
        if ply >= self.killers.len() || self.killers[ply][0] == Some(candidate_move) {
            return;
        }
        self.killer_keys[ply][1] = self.killer_keys[ply][0];
        self.killer_keys[ply][0] = Some(history_group_key(
            candidate_move.source_cells(),
            candidate_move.direction(),
        ));
        self.killers[ply][1] = self.killers[ply][0];
        self.killers[ply][0] = Some(candidate_move);
    }
    fn check_abort(&self) -> Result<(), SearchAbort> {
        if self.fixed_depth.is_some() && !self.enforce_deadline {
            return Ok(());
        }
        if self.nodes & self.abort_poll_mask == 0 && Instant::now() >= self.deadline {
            Err(SearchAbort)
        } else {
            Ok(())
        }
    }
}
fn deadline_slack_ms(time_ms: u64) -> u64 {
    let slack_cap = if time_ms >= LONG_SEARCH_DEADLINE_SLACK_THRESHOLD_MS {
        LONG_SEARCH_DEADLINE_SLACK_MS
    } else {
        SHORT_SEARCH_DEADLINE_SLACK_MS
    };
    slack_cap.min((time_ms / 8).max(1))
}
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SearchAbort;
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct LegalMoveEntry {
    pub(crate) candidate_move: Move,
    pub(crate) is_ejection: bool,
    pub(crate) is_push: bool,
    pub(crate) history_key: u32,
    pub(crate) plan_index: u16,
    pub(crate) enemy_effect: crate::movegen::CompactEnemyEffect,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PositionKey {
    side_to_move: Color,
    black_bits: u64,
    white_bits: u64,
}
impl PositionKey {
    pub(crate) fn from_position(position: &Position) -> Self {
        Self {
            side_to_move: position.side_to_move(),
            black_bits: position.black_bits(),
            white_bits: position.white_bits(),
        }
    }
    fn from_state(position_state: &PositionState) -> Self {
        Self {
            side_to_move: position_state.position().side_to_move(),
            black_bits: position_state.black_bits(),
            white_bits: position_state.white_bits(),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundKind {
    Exact,
    Lower,
    Upper,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TranspositionEntry {
    key: u64,
    depth: u8,
    generation: u8,
    score: i32,
    bound: BoundKind,
    best_move: Option<Move>,
    exact_terminal: bool,
}
#[derive(Clone)]
struct TranspositionTable {
    index_mask: Option<usize>,
    keys: Vec<[u64; 2]>,
    payloads: Vec<[Option<TranspositionPayload>; 2]>,
}
impl TranspositionTable {
    #[inline]
    fn index(&self, k: u64) -> usize {
        match self.index_mask {
            Some(m) => k as usize & m,
            None => k as usize % self.keys.len(),
        }
    }
    fn entry_capacity_for_size(size: usize) -> usize {
        size.div_ceil(2).max(1) * 2
    }
    fn new(size: usize) -> Self {
        let n = size.div_ceil(2).max(1);
        Self {
            index_mask: n.is_power_of_two().then_some(n - 1),
            keys: vec![[0; 2]; n],
            payloads: vec![[None; 2]; n],
        }
    }
    fn entry_capacity(&self) -> usize {
        self.keys.len() * 2
    }
    fn probe(&self, key: u64, depth: u8) -> Option<TranspositionEntry> {
        let b = self.index(key);
        let mut best = None;
        for w in 0..2 {
            if self.keys[b][w] == key {
                if let Some(p) = self.payloads[b][w] {
                    let e = p.entry(key);
                    if e.depth >= depth
                        && best.is_none_or(|old: TranspositionEntry| e.depth > old.depth)
                    {
                        best = Some(e);
                    }
                }
            }
        }
        best
    }
    fn best_move_entry(&self, key: u64, depth: u8) -> Option<TranspositionEntry> {
        let b = self.index(key);
        let mut best = None;
        for w in 0..2 {
            if self.keys[b][w] == key {
                if let Some(p) = self.payloads[b][w] {
                    let e = p.entry(key);
                    if e.depth >= depth
                        || best.is_none_or(|old: TranspositionEntry| e.depth > old.depth)
                    {
                        best = Some(e);
                    }
                }
            }
        }
        best
    }
    fn store(&mut self, e: TranspositionEntry) {
        let b = self.index(e.key);
        let mut selected = None;
        for w in 0..2 {
            match self.payloads[b][w] {
                Some(p) if self.keys[b][w] == e.key => {
                    if p.depth <= e.depth || p.generation != e.generation {
                        selected = Some(w);
                    }
                    if let Some(w) = selected {
                        self.keys[b][w] = e.key;
                        self.payloads[b][w] = Some(TranspositionPayload::from(e));
                    }
                    return;
                }
                None => {
                    selected = Some(w);
                    break;
                }
                _ => {}
            }
        }
        if selected.is_none() {
            let a = self.payloads[b][0].unwrap();
            let z = self.payloads[b][1].unwrap();
            let sa = a.generation != e.generation;
            let sz = z.generation != e.generation;
            let w = usize::from((sz && !sa) || (sz == sa && z.depth < a.depth));
            let p = self.payloads[b][w].unwrap();
            if p.generation != e.generation || p.depth <= e.depth {
                selected = Some(w);
            }
        }
        if let Some(w) = selected {
            self.keys[b][w] = e.key;
            self.payloads[b][w] = Some(TranspositionPayload::from(e));
        }
    }
}
impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new(2)
    }
}

fn position_hash(position_key: PositionKey) -> u64 {
    splitmix64(
        position_key.black_bits
            ^ position_key.white_bits.rotate_left(1)
            ^ match position_key.side_to_move {
                Color::Black => 0,
                Color::White => 0x9E3779B97F4A7C15,
            },
    )
}
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}
fn encode_tt_score(score: i32, ply: u8) -> i32 {
    if score >= WIN_SCORE - 1000 {
        score + i32::from(ply)
    } else if score <= -WIN_SCORE + 1000 {
        score - i32::from(ply)
    } else {
        score
    }
}
fn decode_tt_score(score: i32, ply: u8) -> i32 {
    if score >= WIN_SCORE - 1000 {
        score - i32::from(ply)
    } else if score <= -WIN_SCORE + 1000 {
        score + i32::from(ply)
    } else {
        score
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
fn late_move_reduction(depth: u8, move_index: usize) -> u8 {
    let raw = 1u8
        .saturating_add(depth / LMR_DEPTH_DIVISOR.max(1))
        .saturating_add((move_index / LMR_MOVE_DIVISOR.max(1)) as u8);
    raw.min(depth.saturating_sub(2)).max(1)
}
fn side_index(side: Color) -> usize {
    match side {
        Color::Black => 0,
        Color::White => 1,
    }
}
fn terminal_score(position: &Position, ply: u8, turn_index: u16) -> Option<i32> {
    let black = position.marble_count(Color::Black);
    let white = position.marble_count(Color::White);
    let winner = if white <= 8 {
        Some(Color::Black)
    } else if black <= 8 {
        Some(Color::White)
    } else if turn_index < MAX_GAME_TURNS {
        return None;
    } else if black > white {
        Some(Color::Black)
    } else if black < white {
        Some(Color::White)
    } else {
        None
    };
    let base = WIN_SCORE - i32::from(ply);
    Some(match winner {
        Some(side) if side == position.side_to_move() => base,
        Some(_) => -base,
        None => 0,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct OwnedGroupTables {
    shifts: [Vec<(i8, u64)>; 6],
    ids: [[[u16; 61]; 6]; 2],
}
impl OwnedGroupTables {
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

struct MoveOrderContext<'a> {
    history: &'a [i16],
    keys: [u32; 6],
}
fn context_move_order_score(ctx: &MoveOrderContext, move_entry: LegalMoveEntry) -> i32 {
    let mut mask = 0u32;
    for i in 0..6 {
        mask |= u32::from(move_entry.history_key == ctx.keys[i]) << i;
    }
    match mask.trailing_zeros() {
        0 => return 4000000,
        1 => return 3000000,
        2 => return 2000000,
        3 => return 1000000,
        4 => {
            return COUNTERMOVE_ORDER_BONUS + i32::from(ctx.history[move_entry.plan_index as usize]);
        }
        5 => return FOLLOWUP_ORDER_BONUS + i32::from(ctx.history[move_entry.plan_index as usize]),
        _ => {}
    }
    if move_entry.is_ejection {
        return EJECTION_ORDER_BONUS + i32::from(ctx.history[move_entry.plan_index as usize]);
    }
    if move_entry.is_push {
        return PUSH_ORDER_BONUS
            + i32::try_from(move_entry.candidate_move.len()).unwrap_or(0) * 10000
            + i32::from(ctx.history[move_entry.plan_index as usize]);
    }
    i32::from(ctx.history[move_entry.plan_index as usize])
}

#[derive(Clone, Copy)]
struct TranspositionPayload {
    score: i32,
    depth: u8,
    generation: u8,
    bound: BoundKind,
    best_move: Option<Move>,
    exact_terminal: bool,
}
impl TranspositionPayload {
    fn from(e: TranspositionEntry) -> Self {
        Self {
            score: e.score,
            depth: e.depth,
            generation: e.generation,
            bound: e.bound,
            best_move: e.best_move,
            exact_terminal: e.exact_terminal,
        }
    }
    fn entry(self, key: u64) -> TranspositionEntry {
        TranspositionEntry {
            key,
            score: self.score,
            depth: self.depth,
            generation: self.generation,
            bound: self.bound,
            best_move: self.best_move,
            exact_terminal: self.exact_terminal,
        }
    }
}

#[derive(Clone, Debug)]
struct FullKeyEvalBucket {
    keys: [u64; 4],
    scores: [i32; 4],
    valid: u8,
}

impl EvalCache {
    fn probe_full_key(&self, key: u64) -> Option<i32> {
        let b = &self.full_key_buckets[key as usize & self.sets_mask];
        for w in 0..4 {
            if b.valid & (1 << w) != 0 && b.keys[w] == key {
                return Some(b.scores[w]);
            }
        }
        None
    }
    fn store_full_key(&mut self, key: u64, score: i32) {
        let b = &mut self.full_key_buckets[key as usize & self.sets_mask];
        let w = (0..4)
            .find(|&w| b.valid & (1 << w) == 0 || b.keys[w] == key)
            .unwrap_or((key >> 32) as usize & 3);
        b.keys[w] = key;
        b.scores[w] = score;
        b.valid |= 1 << w;
    }
}

#[derive(Clone, Debug)]
struct TaggedEvalBucket {
    tags: [u64; 4],
    scores: [i32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    const TRANSPOSITION_BUCKET_SIZE: usize = 2;
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct EvalCacheEntry {
        key: u64,
        score: i32,
    }
    fn probe_transposition_bucket(
        bucket: &[Option<TranspositionEntry>; TRANSPOSITION_BUCKET_SIZE],
        key: u64,
        depth: u8,
    ) -> Option<TranspositionEntry> {
        let mut best = None;
        for entry in bucket.iter().flatten() {
            if entry.key == key
                && entry.depth >= depth
                && best.is_none_or(|previous: TranspositionEntry| entry.depth > previous.depth)
            {
                best = Some(*entry);
            }
        }
        best
    }
    fn best_transposition_move_entry_in_bucket(
        bucket: &[Option<TranspositionEntry>; TRANSPOSITION_BUCKET_SIZE],
        key: u64,
        depth: u8,
    ) -> Option<TranspositionEntry> {
        let mut best = None;
        for entry in bucket.iter().flatten() {
            if entry.key == key
                && (entry.depth >= depth
                    || best.is_none_or(|previous: TranspositionEntry| entry.depth > previous.depth))
            {
                best = Some(*entry);
            }
        }
        best
    }
    fn store_transposition_bucket(
        bucket: &mut [Option<TranspositionEntry>; TRANSPOSITION_BUCKET_SIZE],
        entry: TranspositionEntry,
    ) {
        for slot in bucket.iter_mut() {
            match slot {
                Some(existing) if existing.key == entry.key => {
                    if existing.depth <= entry.depth || existing.generation != entry.generation {
                        *slot = Some(entry);
                    }
                    return;
                }
                None => {
                    *slot = Some(entry);
                    return;
                }
                _ => {}
            }
        }
        let mut replacement = 0;
        for index in 1..TRANSPOSITION_BUCKET_SIZE {
            let candidate = bucket[index].unwrap();
            let current_replacement = bucket[replacement].unwrap();
            let candidate_is_stale = candidate.generation != entry.generation;
            let replacement_is_stale = current_replacement.generation != entry.generation;
            if (candidate_is_stale && !replacement_is_stale)
                || (candidate_is_stale == replacement_is_stale
                    && (candidate.depth < current_replacement.depth))
            {
                replacement = index;
            }
        }
        let replaced = bucket[replacement].unwrap();
        if replaced.generation != entry.generation || replaced.depth <= entry.depth {
            bucket[replacement] = Some(entry);
        }
    }
    impl TranspositionTable {
        fn bucket(&self, b: usize) -> [Option<TranspositionEntry>; 2] {
            std::array::from_fn(|w| self.payloads[b][w].map(|p| p.entry(self.keys[b][w])))
        }
        fn snapshot_buckets(&self) -> Vec<[Option<TranspositionEntry>; 2]> {
            (0..self.keys.len()).map(|i| self.bucket(i)).collect()
        }
    }
    impl EvalCache {
        fn snapshot_entries(&self) -> Vec<Option<EvalCacheEntry>> {
            if self.index_bits > 0 {
                return self
                    .tagged_buckets
                    .iter()
                    .enumerate()
                    .flat_map(|(set, b)| {
                        (0..4).map(move |w| {
                            if b.tags[w] == 0 {
                                None
                            } else {
                                Some(EvalCacheEntry {
                                    key: ((b.tags[w] & !(1u64 << 63)) << self.index_bits)
                                        | set as u64,
                                    score: b.scores[w],
                                })
                            }
                        })
                    })
                    .collect();
            }
            self.full_key_buckets
                .iter()
                .flat_map(|b| {
                    (0..4).map(move |w| {
                        if b.valid & (1 << w) != 0 {
                            Some(EvalCacheEntry {
                                key: b.keys[w],
                                score: b.scores[w],
                            })
                        } else {
                            None
                        }
                    })
                })
                .collect()
        }
    }
    #[test]
    fn compact_layouts_and_canonical_groups() {
        let t = fast_movegen_tables();
        for k in 0..2 {
            for d in 3..6 {
                assert!(t.owned_groups.ids[k][d].iter().all(|&v| v == u16::MAX));
            }
        }
        assert_eq!(std::mem::size_of::<Move>(), 4);
        assert_eq!(std::mem::size_of::<Option<Move>>(), 4);
        assert_eq!(std::mem::size_of::<LegalMoveEntry>(), 16);
        assert_eq!(std::mem::size_of::<(i32, LegalMoveEntry)>(), 20);
        assert!(std::mem::size_of::<Option<TranspositionPayload>>() <= 16);
    }
    #[test]
    fn split_transposition_table_matches_reference() {
        let mut seed = 17u64;
        let mut t = TranspositionTable::new(6);
        let mut old = vec![[None; 2]; 3];
        for n in 0..20000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let key = (seed >> 32) % 15;
            let e = TranspositionEntry {
                key,
                depth: ((seed >> 8) % 16) as u8,
                generation: (n / 23) as u8,
                score: n,
                bound: BoundKind::Exact,
                best_move: None,
                exact_terminal: n % 7 == 0,
            };
            t.store(e);
            store_transposition_bucket(&mut old[t.index(key)], e);
            assert_eq!(t.snapshot_buckets(), old);
            for k in 0..15 {
                for d in [0, 3, 15] {
                    assert_eq!(
                        t.probe(k, d),
                        probe_transposition_bucket(&old[t.index(k)], k, d)
                    );
                    assert_eq!(
                        t.best_move_entry(k, d),
                        best_transposition_move_entry_in_bucket(&old[t.index(k)], k, d)
                    );
                }
            }
        }
    }
    #[test]
    fn lossless_eval_cache_matches_reference() {
        assert_eq!(std::mem::size_of::<TaggedEvalBucket>(), 48);
        for bits in 1..=15 {
            for set in 0u64..(1u64 << bits) {
                for hi in [0u64, u64::MAX, 1 << 63, 1 << 32] {
                    let key = (hi & !((1u64 << bits) - 1)) | set;
                    let tag = (key >> bits) | (1u64 << 63);
                    assert_ne!(tag, 0);
                    assert_eq!(((tag & !(1u64 << 63)) << bits) | set, key);
                }
            }
        }
        for bits in 0..=5 {
            let sets = 1usize << bits;
            let mut a = EvalCache::new(sets * 4);
            let mut reference = vec![
                FullKeyEvalBucket {
                    keys: [0; 4],
                    scores: [0; 4],
                    valid: 0
                };
                sets
            ];
            let mut x = 17u64;
            for n in 0..5000 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = if n % 3 == 0 { x % 128 } else { x };
                let ix = key as usize & (sets - 1);
                let b = &mut reference[ix];
                let expected = (0..4)
                    .find(|&w| b.valid & (1 << w) != 0 && b.keys[w] == key)
                    .map(|w| b.scores[w]);
                assert_eq!(a.probe(key), expected);
                a.store(key, n);
                let w = (0..4)
                    .find(|&w| b.valid & (1 << w) == 0 || b.keys[w] == key)
                    .unwrap_or((key >> 32) as usize & 3);
                b.keys[w] = key;
                b.scores[w] = n;
                b.valid |= 1 << w;
                let logical: Vec<_> = reference
                    .iter()
                    .flat_map(|b| {
                        (0..4).map(move |w| {
                            if b.valid & (1 << w) == 0 {
                                None
                            } else {
                                Some(EvalCacheEntry {
                                    key: b.keys[w],
                                    score: b.scores[w],
                                })
                            }
                        })
                    })
                    .collect();
                assert_eq!(a.snapshot_entries(), logical);
            }
        }
    }
}
