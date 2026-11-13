import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pickle
    from collections import defaultdict, deque
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    return (
        Path,
        defaultdict,
        deque,
        multipletests,
        np,
        pd,
        pickle,
    )


@app.cell
def _():
    from summit import SUMMITEnsemble
    from analysis_utils import compute_reference
    from ontology_weights import organ_system_weights, read_ontology, weight_tables

    return (
        SUMMITEnsemble,
        compute_reference,
        organ_system_weights,
        read_ontology,
        weight_tables,
    )


@app.cell
def _(Path):
    # Input files (change to your own paths)
    META = "meta_train.csv"                  # one row per sample: Sample, training.ID, training.Name, Dataset
    MVAL = "train.parquet"                   # samples x probes
    ONTOLOGY = "full_ontology.edgelist"      # tissue ontology
    ID_NAME = "id_name_mapping.csv"          # tissue ID -> name
    OUT = Path("signatures")                 # all outputs go here
    OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    return DEVICE, ID_NAME, META, MVAL, ONTOLOGY, OUT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## load data
    """)
    return


@app.cell
def _(META, MVAL, ONTOLOGY, np, organ_system_weights, pd, read_ontology, weight_tables):
    meta = pd.read_csv(META)
    mval = pd.read_parquet(MVAL).loc[meta["Sample"], :]

    G = read_ontology(ONTOLOGY, header=False)
    id_to_name = dict(zip(meta["training.ID"], meta["training.Name"]))
    tissue_table, system_table = weight_tables(meta["training.ID"], G, names=id_to_name, cap=5)

    # sample and organ system weights
    gw = tissue_table.set_index("tissue")["weight_share"]          # capped share per tissue
    w = organ_system_weights(meta["training.ID"], G, cap=5)
    w = pd.Series(np.asarray(w, dtype=float), index=mval.index)    # one weight per sample
    return gw, meta, mval, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ensemble training
    """)
    return


@app.cell
def _(DEVICE, OUT, SUMMITEnsemble, compute_reference, gw, meta, mval, np, pickle, w):
    # Fit once and save; later runs load the saved model.
    # mu (weighted median of tissue medians, all samples) is computed here and
    # saved with the model, so new data can be centered the same way.
    MODEL = OUT / "summit_ensemble_k50.pkl"
    if MODEL.exists():
        with open(MODEL, "rb") as _f:
            ens, mu = pickle.load(_f)
    else:
        mu = compute_reference(mval, meta["training.Name"], gw)
        ens = SUMMITEnsemble(
            n_components=50,
            n_patches=10, col_ratio=0.01, row_ratio=0.5,
            random_state=0,
            init="soft_kmeans", init_responsibility=0.8, max_iter=5000, device=DEVICE,
        )
        ens.fit(mval - mu.astype(np.float32), row_weight=w, row_groups=meta["Dataset"])
        with open(MODEL, "wb") as _f:
            pickle.dump((ens, mu), _f)
    ens.consensus_["summary"]
    return (ens,)


@app.cell(hide_code=True)
def _(multipletests, np, pd):
    def permutation_tissue_enrichment(
        W,
        tissue_labels,
        study_labels,
        n_permutations=1000,
        random_state=42,
    ):
        """
        Calculate equal-study tissue x signature means and global-row permutation
        p-values.

        W must be samples x signatures. Labels must align with W's rows.

        For the observed and each permuted matrix:
          1. Calculate the mean W score for each tissue within each study.
          2. Average those study means equally for each tissue.

        Each permutation shuffles complete W rows across all samples, while keeping
        tissue and study labels fixed.
        """
        if isinstance(W, pd.DataFrame):
            scores = W.to_numpy(dtype=float)
            signature_names = list(W.columns)
        elif hasattr(W, "toarray"):  # scipy sparse matrix
            scores = W.toarray().astype(float)
            signature_names = None
        else:
            scores = np.asarray(W, dtype=float)
            signature_names = None

        if scores.ndim != 2:
            raise ValueError("W must be a 2D samples-by-signatures matrix.")

        if signature_names is None:
            signature_names = [f"signature_{i}" for i in range(scores.shape[1])]

        tissues = np.asarray(tissue_labels)
        studies = np.asarray(study_labels)

        if len(tissues) != scores.shape[0] or len(studies) != scores.shape[0]:
            raise ValueError(
                "W, tissue_labels, and study_labels must have the same number "
                "of samples and be in the same row order."
            )
        if pd.isna(tissues).any() or pd.isna(studies).any():
            raise ValueError("Tissue and study labels cannot contain missing values.")
        if not np.isfinite(scores).all():
            raise ValueError("W contains NaN or infinite values.")
        if (scores < 0).any():
            raise ValueError("W must contain nonnegative scores.")
        if n_permutations < 1:
            raise ValueError("n_permutations must be at least 1.")

        tissue_codes, tissue_names = pd.factorize(tissues, sort=False)
        study_codes, study_names = pd.factorize(studies, sort=False)

        n_tissues = len(tissue_names)
        n_signatures = scores.shape[1]
        rng = np.random.default_rng(random_state)

        # The study and tissue labels stay fixed for every permutation.
        study_sample_indices = [
            np.flatnonzero(study_codes == study_id)
            for study_id in range(len(study_names))
        ]

        def calculate_equal_study_means(score_matrix):
            study_mean_sum = np.zeros((n_tissues, n_signatures), dtype=float)
            n_studies_per_tissue = np.zeros(n_tissues, dtype=int)

            for sample_idx in study_sample_indices:
                tissue_code = tissue_codes[sample_idx]
                study_scores = score_matrix[sample_idx]

                counts = np.bincount(tissue_code, minlength=n_tissues)
                sums = np.zeros((n_tissues, n_signatures), dtype=float)
                np.add.at(sums, tissue_code, study_scores)

                study_means = np.divide(
                    sums,
                    counts[:, None],
                    out=np.zeros_like(sums),
                    where=counts[:, None] > 0,
                )

                present = counts > 0
                study_mean_sum[present] += study_means[present]
                n_studies_per_tissue[present] += 1

            equal_study_mean = np.divide(
                study_mean_sum,
                n_studies_per_tissue[:, None],
                out=np.full_like(study_mean_sum, np.nan),
                where=n_studies_per_tissue[:, None] > 0,
            )

            return equal_study_mean, n_studies_per_tissue

        # Observed statistic from the original W.
        observed, n_studies_per_tissue = calculate_equal_study_means(scores)

        # Global permutation: shuffle W rows across all samples.
        exceedances = np.zeros((n_tissues, n_signatures), dtype=int)

        for _ in range(n_permutations):
            permuted_scores = scores[rng.permutation(scores.shape[0])]
            permuted_statistic, _ = calculate_equal_study_means(permuted_scores)

            # One-sided test: is observed W unusually high?
            exceedances += permuted_statistic >= observed

        p_values = (exceedances + 1) / (n_permutations + 1)

        q_values = multipletests(
            p_values.ravel(),
            method="fdr_bh",
        )[1].reshape(p_values.shape)

        tissue_index = pd.Index(tissue_names, name="tissue")
        signature_columns = pd.Index(signature_names, name="signature")

        # Which datasets each tissue comes from
        tissue_studies = (
            pd.Series(studies).groupby(tissues).unique()
            .apply(list)
            .reindex(tissue_names)
        )
        tissue_studies.index = tissue_index
        tissue_studies.name = "studies"

        return {
            "mean_W": pd.DataFrame(observed, index=tissue_index, columns=signature_columns),
            "p_value": pd.DataFrame(p_values, index=tissue_index, columns=signature_columns),
            "q_value": pd.DataFrame(q_values, index=tissue_index, columns=signature_columns),
            "enriched": pd.DataFrame(q_values < 0.05, index=tissue_index, columns=signature_columns),
            "n_studies": pd.Series(n_studies_per_tissue, index=tissue_index, name="n_studies"),
            "n_samples": pd.Series(
                np.bincount(tissue_codes, minlength=n_tissues), index=tissue_index, name="n_samples"
            ),
            "tissue_studies": tissue_studies,
        }

    return (permutation_tissue_enrichment,)


@app.cell
def _(np):
    def normalize_factors(W, H, A, how="max", q=0.99):
        """Move the scale from W into H and A; W @ H and H = A @ Xc are unchanged.

        W: samples x k, H: k x probes, A: k x samples (numpy).
        how='max': each W column max = 1.  how='quantile': its q-quantile = 1.
        Returns (W_n, H_n, A_n, scale), scale = one number per component.
        """
        W = np.asarray(W, dtype=np.float64)
        scale = W.max(axis=0) if how == "max" else np.quantile(W, q, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        return W / scale, np.asarray(H) * scale[:, None], np.asarray(A) * scale[:, None], scale

    return (normalize_factors,)


@app.cell
def _(ens, normalize_factors):
    # ens.H is already the least-squares H on all probes (W fixed)
    W_n, H_n, A_n, w_scale = normalize_factors(ens.W, ens.H, ens.A, how="max")
    return (H_n, W_n)


@app.cell
def _(W_n, meta, permutation_tissue_enrichment):
    result = permutation_tissue_enrichment(
        W=W_n,
        tissue_labels=meta["training.Name"],
        study_labels=meta["Dataset"],
        n_permutations=10000,
        random_state=42,
    )
    return (result,)


@app.function
def get_enriched_tissues_perm(result):
    sig_enriched_tissues_perm = {}
    enriched_results = result["enriched"]

    for i in range(enriched_results.shape[1]):
        
        enriched_tissues = enriched_results.loc[enriched_results[f"signature_{i}"]].index
        not_enriched_tissues = enriched_results.loc[~enriched_results[f"signature_{i}"]].index
        sig_enriched_tissues_perm[i] = {
            "enriched": list(enriched_tissues),
            "not_enriched": list(not_enriched_tissues)
        }

    return sig_enriched_tissues_perm


@app.cell
def _(OUT, pickle, result):
    enriched_tissue_perm = get_enriched_tissues_perm(result)
    with open(OUT / "enriched_tissues_permutation.pkl", "wb") as of:
        pickle.dump((result, enriched_tissue_perm), of)
    return


@app.cell
def _(multipletests, np):
    from scipy.stats import norm

    def fdr_from_H(H):
        """H: signatures x probes. Each probe is compared to the other probes in the same signature."""
        H = np.asarray(H, float)
        med = np.median(H, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(H - med), axis=1, keepdims=True)
        z = (H - med) / mad
        p = 2 * norm.sf(np.abs(z))
        return np.vstack([multipletests(row, method="fdr_bh")[1] for row in p])

    return (fdr_from_H,)


@app.cell
def _(H_n, fdr_from_H):
    H_fdr = fdr_from_H(H_n)
    return (H_fdr,)


@app.cell
def _(H_fdr, H_n, ens, pd):
    df_H = pd.DataFrame(H_n, columns=ens.col_names_)
    df_H_fdr = pd.DataFrame(H_fdr, columns=ens.col_names_)
    return df_H, df_H_fdr


@app.cell
def _(OUT, df_H, df_H_fdr, np, pd):
    # H, H_fdr: DataFrames, signatures x probes (same order)
    fdr_cut = 0.05
    Hv, Fv = df_H.values, df_H_fdr.values

    # keep only significant, non-zero entries
    r, c = np.nonzero((Fv <= fdr_cut) & (Hv != 0))
    sig = pd.DataFrame({
        "signature": df_H.index[r],
        "probe": df_H.columns[c],
        "H": Hv[r, c],
        "fdr": Fv[r, c],
    })
    sig["direction"] = np.where(sig["H"] > 0, "hyper", "hypo")

    # rank within each signature + direction, strongest |H| = 1
    sig["rank"] = (sig["H"].abs()
                   .groupby([sig["signature"], sig["direction"]])
                   .rank(ascending=False, method="first")
                   .astype(int))

    sig.to_csv(OUT / "sig_probes_fdr.csv.gz", index=False)
    pd.Series(df_H.columns, name="probe").to_csv(
        OUT / "background_fdr.csv.gz", index=False)
    return


@app.cell
def _(ID_NAME, ONTOLOGY, Path, defaultdict, deque, pd):
    ID_NAME_PATH = ID_NAME
    EDGELIST_PATH = ONTOLOGY
    ROOT = "root"


    def _clean(s):
        """Remove hidden characters: BOM, surrounding spaces, Windows line endings."""
        return s.astype(str).str.replace("\ufeff", "", regex=False).str.strip()


    # Step 2: tissue name -> ID
    def load_id_name_mapping(path=ID_NAME_PATH):
        m = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        m["ID"] = _clean(m["ID"])
        m["Name"] = _clean(m["Name"])
        return m.drop_duplicates().reset_index(drop=True)


    def names_to_ids(tissue_names, mapping, edges, name_overrides=None):
        """
        For each tissue name, find its ID.
        If a name has several IDs (e.g. melanocyte), use the one that is in the edgelist.
        """
        name_overrides = name_overrides or {}
        nodes = set(edges["parent"]) | set(edges["child"])
        name2ids = mapping.groupby("Name")["ID"].apply(list).to_dict()

        name2id, not_in_mapping, not_in_edgelist, ambiguous = {}, [], {}, {}
        for name in tissue_names:
            if name in name_overrides:
                name2id[name] = name_overrides[name]
                continue
            cands = name2ids.get(name, [])
            in_graph = [c for c in cands if c in nodes]
            if not cands:
                not_in_mapping.append(name)
            elif len(in_graph) == 1:
                name2id[name] = in_graph[0]
            elif len(in_graph) == 0:
                not_in_edgelist[name] = cands
                name2id[name] = cands[0]          # keep it; it just won't have a system
            else:
                ambiguous[name] = in_graph

        if not_in_mapping:
            raise ValueError(f"Tissue names not found in the ID-name mapping:\n  {not_in_mapping}")
        if ambiguous:
            raise ValueError("Names with more than one ID in the edgelist - "
                             "choose one with name_overrides={name: ID}:\n"
                             + "\n".join(f"  {k}: {v}" for k, v in ambiguous.items()))
        if not_in_edgelist:
            print("WARNING - these tissues' IDs are not in the edgelist (no system):")
            for k, v in not_in_edgelist.items():
                print(f"  {k}: {v}")
        return name2id


    # Step 3: ID -> systems
    def load_edgelist(path=EDGELIST_PATH):
        edges = pd.read_csv(path, sep=r"\s+", header=None, names=["parent", "child"],
                            dtype=str, engine="python", encoding="utf-8-sig")
        edges["parent"] = _clean(edges["parent"])
        edges["child"] = _clean(edges["child"])
        edges = edges[edges["parent"] != edges["child"]]  # drop self-loops, if any
        return edges.reset_index(drop=True)


    def find_systems(tissue_ids, edges, root=ROOT):
        """
        Systems = direct children of root.
        For each tissue ID, walk up through its parents and collect every system reached.
        Returns ({tissue_id: set of system IDs}, list of system IDs in file order).
        """
        system_ids = list(dict.fromkeys(edges.loc[edges["parent"] == root, "child"]))
        system_set = set(system_ids)
        parents = defaultdict(set)
        for p, c in zip(edges["parent"], edges["child"]):
            parents[c].add(p)

        def systems_of(node):
            found, seen, queue = set(), set(), deque([node])
            while queue:
                n = queue.popleft()
                if n in seen:
                    continue
                seen.add(n)
                if n in system_set:
                    found.add(n)
                    continue
                queue.extend(parents.get(n, ()))
            return found

        return {tid: systems_of(tid) for tid in tissue_ids}, system_ids


    def build_membership(tissue_names, name2id, edges, mapping, root=ROOT):
        """tissue name x system name table, 1 = tissue sits under that system."""
        id2name = dict(zip(mapping["ID"], mapping["Name"]))
        tid2sys, system_ids = find_systems([name2id[t] for t in tissue_names], edges, root)

        rows = [{sid: 1 for sid in tid2sys[name2id[t]]} for t in tissue_names]
        member = pd.DataFrame(rows, columns=system_ids).fillna(0).astype(int)
        member.index = pd.Index(tissue_names, name="tissue")
        member.columns = [id2name.get(s, s) for s in member.columns]
        member = member.loc[:, member.sum(axis=0) > 0]   # drop systems with no tissues

        no_system = member.index[member.sum(axis=1) == 0].tolist()
        if no_system:
            print("WARNING - tissues without a system:", no_system)
        return member


    def export_for_R(res, out_dir="for_R", q_cutoff=0.05,
                     id_name_path=ID_NAME_PATH, edgelist_path=EDGELIST_PATH,
                     name_overrides=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Step 1: tissue names from the results
        tissue_names = list(res["mean_W"].index)
        sigs = res["mean_W"].columns

        # Step 2: name -> ID
        mapping = load_id_name_mapping(id_name_path)
        edges = load_edgelist(edgelist_path)
        name2id = names_to_ids(tissue_names, mapping, edges, name_overrides)

        # Step 3: ID -> systems
        member = build_membership(tissue_names, name2id, edges, mapping)

        # Row info
        row_anno = pd.DataFrame({
            "tissue_id": [name2id[t] for t in tissue_names],
            "n_studies": res["n_studies"].reindex(tissue_names).to_numpy(),
            "n_samples": res["n_samples"].reindex(tissue_names).to_numpy(),
            "n_systems": member.sum(axis=1).to_numpy(),
            "systems": [";".join(member.columns[member.loc[t] == 1]) for t in tissue_names],
        }, index=pd.Index(tissue_names, name="tissue"))

        # Matrices
        mean_W = res["mean_W"].loc[tissue_names, sigs]
        q_val = res["q_value"].loc[tissue_names, sigs]
        enriched = q_val < q_cutoff

        # Column info
        tissue_studies = res["tissue_studies"]

        def n_datasets(sig):
            """Number of different datasets among the tissues enriched for this signature."""
            enriched_tissues = enriched.index[enriched[sig]]
            return len(set().union(*[set(tissue_studies[t]) for t in enriched_tissues]))

        col_anno = pd.DataFrame({
            "n_enriched_tissues": enriched.sum(axis=0),
            "n_enriched_datasets": [n_datasets(s) for s in sigs],
        }, index=sigs)
        col_anno.index.name = "signature"

        # Save
        tables = {
            "mean_W.tsv": mean_W,
            "mean_W_maxscaled.tsv": mean_W / mean_W.max(axis=0),  # each signature's top tissue = 1
            "p_value.tsv": res["p_value"].loc[tissue_names, sigs],
            "q_value.tsv": q_val,
            "enriched.tsv": enriched.astype(int),
            "row_annotation.tsv": row_anno,
            "col_annotation.tsv": col_anno,
            "system_membership.tsv": member,
        }
        for name, df in tables.items():
            df.to_csv(out / name, sep="\t")
        pd.Series({"q_cutoff": q_cutoff}).to_csv(out / "settings.tsv", sep="\t", header=False)
        print(f"Saved {len(tables) + 1} files to {out.resolve()}")


    return (export_for_R,)


@app.cell
def _(OUT, export_for_R, result):
    export_for_R(
        result,
        out_dir=OUT / "for_R"
    )
    return


if __name__ == "__main__":
    app.run()
