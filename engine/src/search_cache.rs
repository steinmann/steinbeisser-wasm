//! Lossless evaluation cache and split transposition storage. Layout is benchmarked.
use super::EVAL_CACHE_WAYS;
use crate::board::Move;

#[derive(Clone, Debug)]
pub(super) struct EvalCache {
    pub(super) tagged_buckets: Vec<TaggedEvalBucket>,
    pub(super) index_bits: u32,
    pub(super) full_key_buckets: Vec<FullKeyEvalBucket>,
    pub(super) sets_mask: usize,
}
impl EvalCache {
    pub(super) fn new(size: usize) -> Self {
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
    // Keep the former same-module inlining after separating cache ownership.
    #[inline]
    pub(super) fn probe(&self, key: u64) -> Option<i32> {
        if self.index_bits == 0 {
            return self.probe_full_key(key);
        }
        let b = &self.tagged_buckets[key as usize & self.sets_mask];
        // The bucket index retains the removed low bits. For nonzero index_bits,
        // shifting frees the top bit for validity without dropping key information.
        let tag = (key >> self.index_bits) | (1u64 << 63);
        for w in 0..4 {
            if b.tags[w] == tag {
                return Some(b.scores[w]);
            }
        }
        None
    }
    #[inline]
    pub(super) fn store(&mut self, key: u64, score: i32) {
        if self.index_bits == 0 {
            return self.store_full_key(key, score);
        }
        let b = &mut self.tagged_buckets[key as usize & self.sets_mask];
        // The bucket index retains the removed low bits. For nonzero index_bits,
        // shifting frees the top bit for validity without dropping key information.
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BoundKind {
    Exact,
    Lower,
    Upper,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TranspositionEntry {
    pub(super) key: u64,
    pub(super) depth: u8,
    pub(super) generation: u8,
    pub(super) score: i32,
    pub(super) bound: BoundKind,
    pub(super) best_move: Option<Move>,
    pub(super) exact_terminal: bool,
}
#[derive(Clone)]
pub(super) struct TranspositionTable {
    pub(super) index_mask: Option<usize>,
    pub(super) keys: Vec<[u64; 2]>,
    pub(super) payloads: Vec<[Option<TranspositionPayload>; 2]>,
}
impl TranspositionTable {
    #[inline]
    pub(super) fn index(&self, k: u64) -> usize {
        match self.index_mask {
            Some(m) => k as usize & m,
            None => k as usize % self.keys.len(),
        }
    }
    pub(super) fn entry_capacity_for_size(size: usize) -> usize {
        size.div_ceil(2).max(1) * 2
    }
    pub(super) fn new(size: usize) -> Self {
        let n = size.div_ceil(2).max(1);
        Self {
            index_mask: n.is_power_of_two().then_some(n - 1),
            keys: vec![[0; 2]; n],
            payloads: vec![[None; 2]; n],
        }
    }
    pub(super) fn entry_capacity(&self) -> usize {
        self.keys.len() * 2
    }
    pub(super) fn probe(&self, key: u64, depth: u8) -> Option<TranspositionEntry> {
        let b = self.index(key);
        let mut best = None;
        for w in 0..2 {
            if self.keys[b][w] == key
                && let Some(p) = self.payloads[b][w]
            {
                let e = p.entry(key);
                if e.depth >= depth
                    && best.is_none_or(|old: TranspositionEntry| e.depth > old.depth)
                {
                    best = Some(e);
                }
            }
        }
        best
    }
    // Preserve the former call boundary instead of duplicating this lookup in
    // the root and recursive search loops after module extraction.
    #[inline(never)]
    pub(super) fn best_move_entry(&self, key: u64, depth: u8) -> Option<TranspositionEntry> {
        let b = self.index(key);
        let mut best = None;
        for w in 0..2 {
            if self.keys[b][w] == key
                && let Some(p) = self.payloads[b][w]
            {
                let e = p.entry(key);
                if e.depth >= depth
                    || best.is_none_or(|old: TranspositionEntry| e.depth > old.depth)
                {
                    best = Some(e);
                }
            }
        }
        best
    }
    pub(super) fn store(&mut self, e: TranspositionEntry) {
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

#[derive(Clone, Copy)]
pub(super) struct TranspositionPayload {
    pub(super) score: i32,
    pub(super) depth: u8,
    pub(super) generation: u8,
    pub(super) bound: BoundKind,
    pub(super) best_move: Option<Move>,
    pub(super) exact_terminal: bool,
}
impl TranspositionPayload {
    pub(super) fn from(e: TranspositionEntry) -> Self {
        Self {
            score: e.score,
            depth: e.depth,
            generation: e.generation,
            bound: e.bound,
            best_move: e.best_move,
            exact_terminal: e.exact_terminal,
        }
    }
    pub(super) fn entry(self, key: u64) -> TranspositionEntry {
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
pub(super) struct FullKeyEvalBucket {
    pub(super) keys: [u64; 4],
    pub(super) scores: [i32; 4],
    pub(super) valid: u8,
}

impl EvalCache {
    pub(super) fn probe_full_key(&self, key: u64) -> Option<i32> {
        let b = &self.full_key_buckets[key as usize & self.sets_mask];
        for w in 0..4 {
            if b.valid & (1 << w) != 0 && b.keys[w] == key {
                return Some(b.scores[w]);
            }
        }
        None
    }
    pub(super) fn store_full_key(&mut self, key: u64, score: i32) {
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
pub(super) struct TaggedEvalBucket {
    pub(super) tags: [u64; 4],
    pub(super) scores: [i32; 4],
}
