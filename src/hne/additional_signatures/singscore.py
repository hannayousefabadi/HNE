"""src/hne/additional_signatures/singscore.py

singscore (Foroutan et al., BMC Bioinformatics 2018), the rank-based single-sample score built for signatures with
up and down genes. No maintained Python package exists (the authors' PySingscore, 2019, needs pandas < 2, and its
down score re-ranks genes in descending order, which differs from the R package when expression ties, as zeros do
in every Visium spot). This follows the Bioconductor package (R/rankAndScoring.R: rankExpr, calcBounds, calcScores,
calcScoresUpDn), with its defaults tiesMethod = "min", centerScore = TRUE, knownDirection = TRUE:
  - in every spot, all genes of the h5ad are ranked by expression, lowest = 1, ties taking the lowest rank;
  - score of a set = (mean rank of its genes - (n + 1) / 2) / ((2N - n + 1) / 2 - (n + 1) / 2) - 0.5, in [-0.5, 0.5]
    (n = genes of the set present, N = genes ranked);
  - for an up / down signature, UpScore = score(up set), DownScore = -score(down set), TotalScore = UpScore + DownScore,
    i.e. score(up set) - score(down set).
Ranks are the same on raw counts, CPM and log1p CPM, so the layer only has to be monotone in the counts.
"""
import numpy as np
import pandas as pd


def singscores(vis, gene_sets: dict, layer: str = "log_norm_count"):
    """singscore of every set in `gene_sets` (one column each), on the genes present in the h5ad, plus the gene
    coverage of each set. The total score of an up / down signature is the up column minus the down column."""
    present = {name: [g for g in genes if g in vis.var_names] for name, genes in gene_sets.items()}
    empty = [name for name, genes in present.items() if not genes]
    if empty:
        raise ValueError(f"no gene of {empty} is in the h5ad")
    values = vis.layers[layer]
    values = np.asarray(values.toarray() if hasattr(values, "toarray") else values, dtype=np.float64)
    n_genes = values.shape[1]
    ordered = np.sort(values, axis=1)
    column = {g: i for i, g in enumerate(vis.var_names)}
    scores = {}
    for name, genes in present.items():
        chosen = values[:, [column[g] for g in genes]]
        # "min" rank = 1 + number of genes in the spot with lower expression
        ranks = np.array([np.searchsorted(row, v, side="left") + 1 for row, v in zip(ordered, chosen)])
        n = len(genes)
        low, high = (n + 1) / 2, (2 * n_genes - n + 1) / 2
        scores[name] = (ranks.mean(axis=1) - low) / (high - low) - 0.5
    genes_used = {name: f"{len(genes)}/{len(gene_sets[name])}" for name, genes in present.items()}
    return pd.DataFrame(scores, index=vis.obs_names)[list(gene_sets)], genes_used
