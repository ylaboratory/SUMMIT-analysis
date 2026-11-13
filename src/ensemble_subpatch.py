import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import pickle
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import wilcoxon

    from summit import SUMMITEnsemble
    from analysis_utils import (block_data, compute_reference, eval_blocks, heldout_error,
                                parse_name, stability_table, top_columns)
    from ontology_weights import organ_system_weights, read_ontology, weight_tables

    return (
        Path,
        SUMMITEnsemble,
        block_data,
        compute_reference,
        eval_blocks,
        heldout_error,
        itertools,
        mo,
        np,
        organ_system_weights,
        parse_name,
        pd,
        pickle,
        read_ontology,
        stability_table,
        top_columns,
        torch,
        weight_tables,
        wilcoxon,
    )


@app.cell
def _(Path):
    # Input files (change to your own paths)
    META = "meta_train.csv"                      # one row per sample: Sample, training.ID, training.Name, Dataset
    MVAL = "train.parquet"                       # samples x probes
    ONTOLOGY = "full_ontology.edgelist"          # tissue ontology for the sample weights
    SPLIT = "split.csv"                          # columns: sample, hidden (True / False)
    OUT = Path("ensemble_subpatch")              # all outputs go here
    OUT.mkdir(parents=True, exist_ok=True)

    # Settings shared by every run
    K = 50
    DEVICE = "cuda"
    RATIOS = (0.01, 0.1, 0.25, 0.5, 0.75, 1.0)   # share of training samples per patch
    ENSEMBLES = tuple(range(1, 11))              # independent ensembles (= random_state)
    N_FULL_FITS = 10                             # full ensemble e uses seeds (e-1)*10 .. e*10-1
    N_SINGLE = 3                                 # single fits = first 3 members of full ensemble 1
    EVAL_SEED = 12345                            # probes of the shared held-out blocks
    EVAL_COL_RATIO = 0.02                        # each block: 1% kept + 1% hidden probes
    N_BLOCKS = 10
    TOP_N = 1000                                 # top-N probes for stability / consistency
    ERR = "mse_w"                                # error used in the tables: "mse" or "mse_w"

    model_settings = dict(init="soft_kmeans", init_responsibility=0.8,
                          max_iter=5000, device=DEVICE)
    patch_settings = dict(n_patches=10, col_ratio=0.01, replace=True)   # 1% of probes per patch
    return (
        DEVICE,
        ENSEMBLES,
        ERR,
        EVAL_COL_RATIO,
        EVAL_SEED,
        K,
        META,
        MVAL,
        N_BLOCKS,
        N_FULL_FITS,
        N_SINGLE,
        ONTOLOGY,
        OUT,
        RATIOS,
        SPLIT,
        TOP_N,
        model_settings,
        patch_settings,
    )


@app.cell
def _(META, MVAL, ONTOLOGY, np, organ_system_weights, pd, read_ontology, weight_tables):
    meta = pd.read_csv(META)
    mval = pd.read_parquet(MVAL).loc[meta["Sample"], :]

    G = read_ontology(ONTOLOGY, header=False)
    id_to_name = dict(zip(meta["training.ID"], meta["training.Name"]))
    tissue_table, _ = weight_tables(meta["training.ID"], G, names=id_to_name, cap=5)
    gw = tissue_table.set_index("tissue")["weight_share"]          # capped share per tissue
    w = organ_system_weights(meta["training.ID"], G, cap=5)
    w = pd.Series(np.asarray(w, dtype=float), index=mval.index)    # one weight per sample
    meta = meta.set_index("Sample")
    return gw, meta, mval, w


@app.cell
def _(SPLIT, compute_reference, gw, meta, mval, np, pd):
    # Training / hidden samples, and mu from the training samples only
    # (so the hidden samples are never used before scoring).
    split = pd.read_csv(SPLIT)
    train = split.loc[~split["hidden"], "sample"].tolist()
    hidden = split.loc[split["hidden"], "sample"].tolist()

    mu = compute_reference(mval.loc[train], meta.loc[train, "training.Name"], gw)
    Xc = mval - mu.astype(np.float32)
    Xtr = Xc.loc[train]
    print(f"{len(train)} training and {len(hidden)} hidden samples, {Xc.shape[1]} probes")
    return Xc, Xtr, hidden, train


@app.cell
def _(
    DEVICE,
    K,
    SUMMITEnsemble,
    TOP_N,
    Xtr,
    meta,
    model_settings,
    np,
    top_columns,
    torch,
    train,
    w,
):
    def fit_ensemble(**settings):
        """One SUMMITEnsemble on the training samples, with its peak GPU memory."""
        cuda = torch.cuda.is_available() and str(DEVICE).startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats(DEVICE)
        model = SUMMITEnsemble(K, progress=False, **model_settings, **settings)
        model.fit(Xtr, row_weight=w.loc[train], row_groups=meta.loc[train, "Dataset"])
        model.gpu_peak_gib_ = torch.cuda.max_memory_allocated(DEVICE) / 2**30 if cuda else np.nan
        return model

    def summarize(model):
        """Keep only what the analysis needs (small), so the model can be freed."""
        c = model.consensus_
        return {"W": model.W, "top": top_columns(model.H, TOP_N), "patch_W": c["patch_W"],
                "patches": c["patches"], "summary": c["summary"],
                "patch_rows": c["patch_rows"], "gpu_peak_gib": model.gpu_peak_gib_}

    return fit_ensemble, summarize


@app.cell
def _(
    ENSEMBLES,
    OUT,
    RATIOS,
    fit_ensemble,
    patch_settings,
    pickle,
    summarize,
    torch,
):
    # Patch ensembles: every sample ratio x every independent ensemble.
    # The same random_state gives the same probe patches at every ratio.
    runs = {}
    for _r in RATIOS:
        for _s in ENSEMBLES:
            print(f"===== row_ratio {_r}, ensemble {_s} =====", flush=True)
            runs[(_r, _s)] = summarize(fit_ensemble(row_ratio=_r, random_state=_s, **patch_settings))
            torch.cuda.empty_cache()

    with open(OUT / "patch_ensembles.pkl", "wb") as _f:
        pickle.dump(runs, _f)
    return (runs,)


@app.cell
def _(ENSEMBLES, N_FULL_FITS, OUT, fit_ensemble, pickle, summarize, torch):
    # Reference: ensembles of 10 full fits (all training samples x all probes).
    full_runs, solver = {}, None
    for _e in ENSEMBLES:
        print(f"===== full-fit ensemble {_e} =====", flush=True)
        _m = fit_ensemble(row_ratio=1.0, n_patches=N_FULL_FITS, col_ratio=1.0, replace=True,
                          random_state=(_e - 1) * N_FULL_FITS)
        full_runs[_e] = summarize(_m)
        if solver is None:
            solver = _m          # kept: solves H and W for any W of the training samples
        del _m
        torch.cuda.empty_cache()

    with open(OUT / "full_fit_ensembles.pkl", "wb") as _f:
        pickle.dump(full_runs, _f)
    return full_runs, solver


@app.cell
def _(ENSEMBLES, N_SINGLE, TOP_N, Xtr, full_runs, runs, solver, top_columns):
    # Every W to compare, named "<type>_s<seed>":
    #   r0.5_s1      consensus of the 50% patch ensemble 1
    #   fullens_s1   consensus of full-fit ensemble 1 (seeds 0-9)
    #   single_s0    one plain full fit (seeds 0-2, members of fullens_s1)
    W = {f"r{_r}_s{_s}": _o["W"] for (_r, _s), _o in runs.items()}
    TOP = {f"r{_r}_s{_s}": _o["top"] for (_r, _s), _o in runs.items()}
    for _e, _o in full_runs.items():
        W[f"fullens_s{_e}"], TOP[f"fullens_s{_e}"] = _o["W"], _o["top"]

    _first = full_runs[ENSEMBLES[0]]
    SINGLE_SEEDS = _first["patches"]["model_seed"].tolist()[:N_SINGLE]
    for _p, _seed in enumerate(SINGLE_SEEDS):
        W[f"single_s{_seed}"] = _first["patch_W"][_p]
        TOP[f"single_s{_seed}"] = top_columns(solver.transform_H(Xtr, W=_first["patch_W"][_p]), TOP_N)
    return SINGLE_SEEDS, TOP, W


@app.cell
def _(
    EVAL_COL_RATIO,
    EVAL_SEED,
    N_BLOCKS,
    Xc,
    block_data,
    eval_blocks,
    hidden,
    train,
):
    # The shared held-out blocks: the same for every W.
    blocks = eval_blocks(Xc.shape[1], EVAL_COL_RATIO, N_BLOCKS, EVAL_SEED)
    data = block_data(Xc, train, hidden, blocks)
    return (data,)


@app.cell
def _(OUT, W, data, heldout_error, hidden, parse_name, pickle, solver, w):
    # Held-out error of EVERY W on the same blocks, by the same rule.
    heldout = heldout_error(solver, data, W, w=w.loc[hidden])
    heldout["kind"], heldout["ensemble"] = zip(*heldout["name"].map(parse_name))

    with open(OUT / "heldout_error.pkl", "wb") as _f:
        pickle.dump(heldout, _f)
    heldout.groupby("name")[["mse", "mse_w", "mse0", "mse0_w"]].agg(["mean", "sem"])
    return (heldout,)


@app.cell
def _(OUT, data, heldout_error, hidden, pd, pickle, runs, solver, w):
    # Held-out error of EACH patch's own W (fitted for the patch samples,
    # filled in for the rest), same blocks and same rule as above.
    _parts = []
    for (_r, _s), _o in runs.items():
        _Wp = {f"p{_p}": _Wm for _p, _Wm in enumerate(_o["patch_W"])}
        _t = heldout_error(solver, data, _Wp, w=w.loc[hidden])
        _parts.append(_t.assign(ratio=_r, ensemble=_s, fit_patch=_t["name"].str[1:].astype(int)))
        print(f"patch W scored: ratio {_r}, ensemble {_s}", flush=True)
    heldout_patch = pd.concat(_parts, ignore_index=True)

    with open(OUT / "heldout_error_patch.pkl", "wb") as _f:
        pickle.dump(heldout_patch, _f)
    return (heldout_patch,)


@app.cell
def _(ERR, heldout, heldout_patch, np, pd):
    # One patch alone vs the consensus of its ensemble (lower = better).
    #   one_patch       mean over all single patches of that ratio
    #   best_patch      mean over ensembles of the best patch in each ensemble
    #   consensus       mean over the consensus W of that ratio
    #   share_cons_wins share of (ensemble, patch, block) where the consensus
    #                   beats that patch on the same block
    _cons = heldout[heldout["kind"].map(lambda k: not isinstance(k, str))].rename(
        columns={"kind": "ratio"})
    _cons["ratio"] = _cons["ratio"].astype(float)
    _per_W = heldout_patch.groupby(["ratio", "ensemble", "fit_patch"])[ERR].mean().reset_index()
    _m = heldout_patch.merge(_cons[["ratio", "ensemble", "block", ERR]],
                             on=["ratio", "ensemble", "block"], suffixes=("", "_cons"))

    compare_members = pd.DataFrame({
        "one_patch": _per_W.groupby("ratio")[ERR].mean(),
        "one_patch_se": _per_W.groupby("ratio")[ERR].std(ddof=1)
                        / np.sqrt(_per_W.groupby("ratio").size()),
        "best_patch": _per_W.groupby(["ratio", "ensemble"])[ERR].min().groupby("ratio").mean(),
        "consensus": _cons.groupby("ratio")[ERR].mean(),
        "share_cons_wins": (_m[f"{ERR}_cons"] < _m[ERR]).groupby(_m["ratio"]).mean(),
    })
    compare_members
    return


@app.cell
def _(
    ENSEMBLES,
    OUT,
    RATIOS,
    SINGLE_SEEDS,
    TOP,
    TOP_N,
    W,
    itertools,
    pickle,
    stability_table,
):
    _ens = list(ENSEMBLES)
    _pairs = itertools.combinations
    pairs = (
        # stability: independent ensembles of the same kind
        [(f"r{_r}_s{a}", f"r{_r}_s{b}") for _r in RATIOS for a, b in _pairs(_ens, 2)]
        + [(f"fullens_s{a}", f"fullens_s{b}") for a, b in _pairs(_ens, 2)]
        + [(f"single_s{a}", f"single_s{b}") for a, b in _pairs(SINGLE_SEEDS, 2)]
        # consistency: each patch ensemble vs each full-fit ensemble
        + [(f"r{_r}_s{a}", f"fullens_s{b}") for _r in RATIOS for a in _ens for b in _ens]
        # single fits vs the full-fit ensembles they are NOT part of
        + [(f"single_s{a}", f"fullens_s{b}") for a in SINGLE_SEEDS for b in _ens[1:]]
    )
    stab_table, per_component = stability_table(W, TOP, pairs)

    with open(OUT / "stability.pkl", "wb") as _f:
        pickle.dump({"table": stab_table, "per_component": per_component, "top_n": TOP_N}, _f)
    stab_table
    return (per_component,)


@app.cell
def _(ERR, np, parse_name, pd):
    # Helpers: cost table and summary table
    def _label(r):
        if isinstance(r, str):
            return {"fullens": "full ens", "single": "single fit"}[r]
        return f"{r * 100:g}%"

    def _med(v):
        return float(np.median(v)) if len(v) else np.nan

    def overlaps(per_component, a, b, col="top_overlap"):
        """All per-component values for pairs of type a and b (a ratio like
        0.5, 'fullens' or 'single'), in either order."""
        vals = [d[col].to_numpy() for (x, y), d in per_component.items()
                if (parse_name(x)[0], parse_name(y)[0]) in ((a, b), (b, a))]
        return np.concatenate(vals) if vals else np.array([])

    def _held(heldout, kind, ensemble=None):
        v = heldout[heldout["kind"] == kind]
        if ensemble is not None:
            v = v[v["ensemble"] == ensemble]
        return float(v[ERR].mean()), float(v[ERR].sem())

    def cost_table(runs, full_runs):
        """One row per ensemble (and per single full fit): seconds by step,
        peak GPU memory, convergence."""
        rows = []

        def add(setting, e, o, single=False):
            p = o["patches"]
            fit, fill, total = p["fit_seconds"].sum(), p["fill_seconds"].sum(), o["summary"]["total_seconds"]
            if single:
                fit = total = p["fit_seconds"].mean()
                fill = 0.0
            rows.append({"setting": setting, "ensemble": e, "fit": fit, "fill": fill,
                         "consensus": total - fit - fill, "total": total,
                         "gpu_peak_gib": o["gpu_peak_gib"],
                         "share_converged": p["converged"].mean(),
                         "n_iter_median": p["n_iter"].median()})

        for (r, s), o in runs.items():
            add(r, s, o)
        for e, o in full_runs.items():
            add("fullens", e, o)
            add("single", e, o, single=True)
        return pd.DataFrame(rows)

    def subpatch_summary(runs, full_runs, heldout, per_component):
        """One row per setting: time, speed-up vs the full-fit ensemble,
        GPU memory, stability, consistency, held-out error, convergence."""
        cost = cost_table(runs, full_runs)
        ref_time = cost.loc[cost["setting"] == "fullens", "total"].mean()
        rows = []
        for r in sorted({r for r, _ in runs}) + ["fullens", "single"]:
            c = cost[cost["setting"] == r]
            err, err_se = _held(heldout, r)
            if r == "fullens":
                cons = np.array([])
            else:
                cons = overlaps(per_component, r, "fullens")
            rows.append({
                "setting": {"fullens": "full ensemble (10 fits)",
                            "single": "single full fit"}.get(r, _label(r)),
                "seconds": c["total"].mean(),
                "speed_up_vs_full_ensemble": ref_time / c["total"].mean(),
                "gpu_peak_gib": c["gpu_peak_gib"].max(),
                "stability": _med(overlaps(per_component, r, r)),
                "consistency_vs_full_ensemble": _med(cons),
                "heldout_error": err, "heldout_error_se": err_se,
                "share_converged": c["share_converged"].mean(),
                "n_iter_median": c["n_iter_median"].median(),
            })
        return pd.DataFrame(rows).set_index("setting")

    return cost_table, subpatch_summary


@app.cell
def _(cost_table, full_runs, runs):
    # Time (seconds) and peak GPU memory (GiB), one row per ensemble / single fit
    cost = cost_table(runs, full_runs)
    cost
    return


@app.cell
def _(full_runs, heldout, per_component, runs, subpatch_summary):
    summary_table = subpatch_summary(runs, full_runs, heldout, per_component)
    summary_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Q4: why is the error at 100% samples higher than at 75%?

    1. **Convergence**: do bigger patches stop at `max_iter`?
    2. **Paired change**: same probe patches at every ratio (same `random_state`)
       and the same held-out blocks for every W. Is the 75% → 100% change bigger
       than the noise?
    3. **Weights**: rounding and `min_per_group` keep small studies at a higher
       share below 100%. How much does the weight per study shift?
    """)
    return


@app.cell
def _(pd, runs):
    # Q4.1 convergence per ratio (means over ensembles)
    convergence = pd.DataFrame([
        {"ratio": _r, "share_converged": _o["patches"]["converged"].mean(),
         "n_iter_median": _o["patches"]["n_iter"].median(),
         "n_iter_max": _o["patches"]["n_iter"].max()}
        for (_r, _s), _o in runs.items()]).groupby("ratio").mean()
    convergence
    return


@app.cell
def _(ERR, heldout, np, pd, wilcoxon):
    # Q4.2 paired change from one ratio to another (positive = worse at r_to),
    # matched by ensemble and held-out block
    def _held_ratio(r):
        return heldout[heldout["kind"] == r][["ensemble", "block", ERR]]

    def _change(r_from, r_to):
        m = _held_ratio(r_from).merge(_held_ratio(r_to), on=["ensemble", "block"],
                                      suffixes=("_a", "_b"))
        diff = (m[f"{ERR}_b"] - m[f"{ERR}_a"]).to_numpy()
        p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else np.nan
        return {"change": f"{r_from:g} -> {r_to:g}", "mean_change": diff.mean(),
                "se": diff.std(ddof=1) / np.sqrt(len(diff)),
                "n_worse": int((diff > 0).sum()), "n_pairs": len(diff), "wilcoxon_p": p}

    paired = pd.DataFrame([_change(0.75, 1.0), _change(0.5, 1.0), _change(0.5, 0.75)])
    paired.set_index("change")
    return


@app.cell
def _(meta, pd, runs, train, w):
    # Q4.3 weight shift per study: patch vs all training samples
    #   tv_distance: half the sum over studies of |share in patch - share in all|
    #   small_study_share: total weight share of studies with <= 5 training samples
    _study = meta.loc[train, "Dataset"]
    _wt = w.loc[train]

    def _shares(names):
        s = _wt.loc[names].groupby(_study.loc[names]).sum()
        return s / s.sum()

    _all = _shares(train)
    _sizes = _study.value_counts()
    _small = set(_sizes.index[_sizes <= 5])
    _rows = []
    for (_r, _s), _o in runs.items():
        for _names in _o["patch_rows"].values():
            _sh = _shares(_names).reindex(_all.index, fill_value=0.0)
            _rows.append({"ratio": _r, "tv_distance": 0.5 * (_sh - _all).abs().sum(),
                          "small_study_share": _sh[_sh.index.isin(_small)].sum()})
    weight_shift = pd.DataFrame(_rows).groupby("ratio").mean()
    weight_shift["small_study_share_all_training"] = _all[_all.index.isin(_small)].sum()
    weight_shift
    return


if __name__ == "__main__":
    app.run()
