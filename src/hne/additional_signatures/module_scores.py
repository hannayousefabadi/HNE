"""src/hne/additional_signatures/module_scores.py

Seurat-style module score per spot (scanpy.tl.score_genes, the AddModuleScore method), on the log1p-CPM layer:
mean expression of the set's genes minus the mean of control genes drawn from the same expression bins.
The controls cancel most of the spot-depth effect that every ssGSEA score carries (genes detected per spot),
which is why the FMRP up / down sets are also scored this way. Gene bins come from the spots passed in,
so callers pass the in-tissue spots of one patient.
"""
import anndata as ad
import pandas as pd
import scanpy as sc


def module_scores(vis, gene_sets: dict, n_bins: int, ctrl_size: int, seed: int):
    """One column per set in `gene_sets`, scored on the genes present in the h5ad, plus the gene coverage of each set."""
    present = {name: [g for g in genes if g in vis.var_names] for name, genes in gene_sets.items()}
    empty = [name for name, genes in present.items() if not genes]
    if empty:
        raise ValueError(f"no gene of {empty} is in the h5ad")
    expr = ad.AnnData(X=vis.layers["log_norm_count"], obs=pd.DataFrame(index=vis.obs_names),
                      var=pd.DataFrame(index=vis.var_names))
    for name, genes in present.items():
        sc.tl.score_genes(expr, genes, score_name=name, n_bins=n_bins, ctrl_size=ctrl_size,
                          random_state=seed, use_raw=False)
    genes_used = {name: f"{len(genes)}/{len(gene_sets[name])}" for name, genes in present.items()}
    return expr.obs[list(gene_sets)].astype(float), genes_used
