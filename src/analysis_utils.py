"""Helpers for the SUMMIT analysis notebooks (kept out of the summit package).

compute_reference   per-probe reference mu (for centering X yourself)
eval_blocks         random probe blocks for held-out scoring
block_data          cut those blocks out of X once
heldout_error       hidden-block MSE of many W, same blocks, same rule
top_columns         top-N probes of each signature (by |H|)
stability_table     top-N overlap of matched signatures between runs
parse_name          'r0.5_s1' -> (0.5, 1), 'fullens_s2' -> ('fullens', 2)
"""
import numpy as np
import pandas as pd

from summit.consensus import match_components


def _in_row_order(values, index):
    # A labeled Series is lined up with the row names; anything else is taken in row order.
    if isinstance(values, pd.Series) and not values.index.equals(pd.RangeIndex(len(values))):
        values = values.reindex(index)
        if values.isna().any():
            raise ValueError("labels are missing for some rows")
    return np.asarray(values)


def _weighted_median_rows(values, weights, tol=1e-9):
    # Per-row weighted median. Sort each row, add up the weights in that order,
    # take the first value where the total reaches 0.5 (average with the next
    # value if it lands exactly on 0.5, so equal weights give the plain median).
    order = np.argsort(values, axis=1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=1)
    running = np.cumsum(weights[order], axis=1)
    last = values.shape[1] - 1
    lo = np.minimum((running < 0.5 - tol).sum(axis=1), last)
    hi = np.minimum((running <= 0.5 + tol).sum(axis=1), last)
    rows = np.arange(values.shape[0])
    return 0.5 * (ordered[rows, lo] + ordered[rows, hi])


def compute_reference(X, groups, group_weight=None, chunk=20000):
    """Per-probe reference mu from a samples x probes DataFrame.

    1. Median of each probe within each group (e.g. tissue).
    2. Combine the group medians: plain median (group_weight=None, every
       group counts once) or weighted median (group_weight = Series or dict
       by group label, e.g. the capped weight_share per tissue).
    groups: one label per row (array in row order, or Series by sample name).
    Returns a Series indexed by probe name. Center with X - mu.
    """
    codes, labels = pd.factorize(_in_row_order(groups, X.index))
    rows_of = [np.flatnonzero(codes == g) for g in range(len(labels))]
    weights = None
    if group_weight is not None:
        weights = pd.Series(group_weight).reindex(labels).to_numpy(dtype=float)
        if np.isnan(weights).any():
            raise ValueError("group_weight is missing some groups")
        weights = weights / weights.sum()
    parts = []
    for start in range(0, X.shape[1], chunk):
        block = X.iloc[:, start:start + chunk].to_numpy(dtype=np.float64)
        medians = np.stack([np.median(block[r], axis=0) for r in rows_of], axis=1)
        parts.append(np.median(medians, axis=1) if weights is None
                     else _weighted_median_rows(medians, weights))
    return pd.Series(np.concatenate(parts), index=X.columns, name="mu")


def eval_blocks(n_cols, ratio, n_blocks, seed):
    """n_blocks random blocks of round(ratio * n_cols) probes, each split in
    half: (kept, hidden) column positions. Blocks are drawn independently."""
    rng = np.random.default_rng(seed)
    size = int(round(ratio * n_cols))
    half = size // 2
    blocks = []
    for _ in range(n_blocks):
        cols = rng.choice(n_cols, size=size, replace=False)
        blocks.append((np.sort(cols[half:]), np.sort(cols[:half])))
    return blocks


def block_data(X, train, hidden, blocks):
    """Cut the blocks out of X once. Per block (D, B, C, T):
                        kept probes   hidden probes
        training rows       D              C
        hidden rows         B              T (target)
    """
    tr, hd = X.index.get_indexer(train), X.index.get_indexer(hidden)

    def cut(rows, cols):
        return X.iloc[rows, cols].to_numpy(dtype=np.float32)

    return [(cut(tr, k), cut(hd, k), cut(tr, h), cut(hd, h).astype(np.float64))
            for k, h in blocks]


def _mse(resid, w, name):
    per_row = np.einsum("ij,ij->i", resid, resid) / resid.shape[1]   # MSE of each row
    out = {name: float(per_row.mean())}
    if w is not None:
        w = np.asarray(w, dtype=float)
        out[name + "_w"] = float((w * per_row).sum() / w.sum())
    return out


def heldout_error(solver, data, W_dict, w=None):
    """Hidden-block MSE of every W, on the same blocks, by the same rule as
    the consensus prediction in summit.bcv:

        H_kept = transform_H(D, W)          kept probes
        W_hid  = transform_W(B, H=H_kept)   hidden samples
        H_hid  = transform_H(C, W)          hidden probes
        error  = T - W_hid @ H_hid

    solver : a SUMMIT model fitted on the training rows (gives the row
        weights for transform_H and the settings for transform_W).
    data : from block_data.
    W_dict : {name: W}, each W = training rows x k, in training order.
    w : weights of the hidden rows (adds mse_w and mse0_w).
    Returns one row per (name, block): mse, mse0 (= predicting zero) and the
    weighted versions.
    """
    rows = []
    for name, W in W_dict.items():
        for b, (D, B, C, T) in enumerate(data):
            H_kept = solver.transform_H(D, W=W)
            resid = T - solver.transform_W(B, H=H_kept) @ solver.transform_H(C, W=W)
            rows.append({"name": name, "block": b, **_mse(resid, w, "mse"), **_mse(T, w, "mse0")})
    return pd.DataFrame(rows)


def top_columns(H, top_n):
    """Positions of the top_n probes by |H| for each signature (k x top_n)."""
    return np.argpartition(-np.abs(np.asarray(H)), top_n - 1, axis=1)[:, :top_n]


def stability_table(W, top, pairs):
    """For each pair of runs (a, b): match signatures one-to-one by W
    correlation (Hungarian), then top-N overlap of each matched pair
    = shared top-N probes / N.

    W, top : {run name: W} and {run name: top_columns(H)}.
    Returns (one row per pair, {pair: one row per signature of a}).
    """
    rows, per_component = [], {}
    for a, b in pairs:
        order, w_corr = match_components(W[b], W[a])        # b's order[j] matches a's j
        n = top[a].shape[1]
        overlap = np.array([len(np.intersect1d(top[a][j], top[b][order[j]])) / n
                            for j in range(len(order))])
        per_component[(a, b)] = pd.DataFrame({"comp_a": np.arange(len(order)), "comp_b": order,
                                              "w_corr": w_corr, "top_overlap": overlap})
        rows.append({"a": a, "b": b, "w_corr_median": float(np.median(w_corr)),
                     "top_overlap_median": float(np.median(overlap))})
    return pd.DataFrame(rows), per_component


def parse_name(name):
    """'r0.5_s1' -> (0.5, 1); 'fullens_s2' -> ('fullens', 2); 'single_s0' -> ('single', 0)."""
    kind, seed = name.rsplit("_s", 1)
    return (float(kind[1:]) if kind.startswith("r") else kind), int(seed)
