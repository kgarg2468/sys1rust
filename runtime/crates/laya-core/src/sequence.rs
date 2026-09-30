//! `common.build_sequence` / `collate_items` reimplemented over [`LayaTokenizer`].
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: `collate_to`.

use crate::backend::Batch;
use crate::question::{QType, Question};
use crate::tokenizer::LayaTokenizer;
use crate::{pyjson, Error, Result};
use serde_json::Value;

/// One encoded question row before collation.
#[derive(Debug, Clone)]
pub struct EncodedItem {
    pub ids: Vec<u32>,
    pub markers: Vec<u32>,
    pub qtype: QType,
}

/// `serialize_state`: strings pass through, anything else is `json.dumps(..., ensure_ascii=False)`.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other),
    }
}

/// Tokenized state text, computed once per `predict` call and shared by all questions.
pub struct StateTokens {
    pub ids: Vec<u32>,
    pub truncate_left: bool,
}

impl StateTokens {
    pub fn new(tok: &LayaTokenizer, state: &Value) -> Result<Self> {
        let text = serialize_state(state).replace(&tok.mask_token, " ");
        Ok(Self {
            ids: tok.encode(&text)?,
            truncate_left: state.is_array(),
        })
    }
}

/// Format: `[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`.
pub fn build_sequence(
    tok: &LayaTokenizer,
    state: &StateTokens,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let ins = q.instructions.replace(&tok.mask_token, " ");
    let mut head_ids = tok.encode(&format!("{} question: {}", q.qtype.name(), ins))?;
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(q.options.len());
    for opt in &q.options {
        let mut ids = tok.encode(&format!(" {}", opt.replace(&tok.mask_token, " ")))?;
        ids.truncate(48);
        let mut o = Vec::with_capacity(ids.len() + 1);
        o.push(tok.mask_id);
        o.extend(ids);
        opt_ids.push(o);
    }
    let total: usize = opt_ids.iter().map(Vec::len).sum();
    let mut opt_budget = head_max_len as i64 - total as i64;
    if opt_budget < 16 {
        let per = ((head_max_len as i64 - 16) / opt_ids.len().max(1) as i64).max(4) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        let total: usize = opt_ids.iter().map(Vec::len).sum();
        opt_budget = head_max_len as i64 - total as i64;
    }
    head_ids.truncate(opt_budget.max(8) as usize);

    let mut ids = Vec::with_capacity(max_len);
    ids.push(tok.cls_id);
    ids.extend(&head_ids);
    ids.push(tok.sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len() as u32);
        ids.extend(o);
    }
    ids.push(tok.sep_id);
    let room = (max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
    let st = &state.ids;
    let st: &[u32] = if state.truncate_left {
        &st[st.len().saturating_sub(room)..]
    } else {
        &st[..room.min(st.len())]
    };
    ids.extend(st);
    ids.push(tok.sep_id);
    ids.truncate(max_len);
    markers.retain(|&m| (m as usize) < max_len);
    Ok((ids, markers))
}

/// Tokenize one state against every question (`Agent._encode_state`).
pub fn encode_state(
    tok: &LayaTokenizer,
    state: &Value,
    questions: &[Question],
    max_len: usize,
    head_max_len: usize,
) -> Result<Vec<EncodedItem>> {
    let st = StateTokens::new(tok, state)?;
    let mut items = Vec::with_capacity(questions.len());
    for q in questions {
        let (ids, markers) = build_sequence(tok, &st, q, max_len, head_max_len)?;
        if markers.len() != q.options.len() {
            return Err(Error::Question(format!(
                "question {} options exceed head_max_len={}",
                pyjson::repr_str(&q.id),
                head_max_len
            )));
        }
        items.push(EncodedItem {
            ids,
            markers,
            qtype: q.qtype,
        });
    }
    Ok(items)
}

/// `collate_items`: right-pad rows with `pad_id`.
pub fn collate(items: &[EncodedItem], pad_id: u32) -> Batch {
    collate_to(items, pad_id, 0)
}

/// [`collate`] with rows padded to at least `min_len` tokens (for fixed-size shape buckets).
pub fn collate_to(items: &[EncodedItem], pad_id: u32, min_len: usize) -> Batch {
    let n = items.len();
    let len = items
        .iter()
        .map(|it| it.ids.len())
        .max()
        .unwrap_or(0)
        .max(min_len);
    let kmax = items.iter().map(|it| it.markers.len()).max().unwrap_or(0);
    let mut input_ids = vec![pad_id; n * len];
    let mut attention_mask = vec![0u32; n * len];
    let mut marker_pos = vec![0u32; n * kmax];
    let mut seq_lens = Vec::with_capacity(n);
    let mut marker_count = Vec::with_capacity(n);
    let mut qtype = Vec::with_capacity(n);
    for (i, it) in items.iter().enumerate() {
        input_ids[i * len..i * len + it.ids.len()].copy_from_slice(&it.ids);
        attention_mask[i * len..i * len + it.ids.len()].fill(1);
        marker_pos[i * kmax..i * kmax + it.markers.len()].copy_from_slice(&it.markers);
        seq_lens.push(it.ids.len());
        marker_count.push(it.markers.len());
        qtype.push(it.qtype.index() as u32);
    }
    Batch {
        n,
        len,
        kmax,
        input_ids,
        attention_mask,
        seq_lens,
        marker_pos,
        marker_count,
        qtype,
    }
}
