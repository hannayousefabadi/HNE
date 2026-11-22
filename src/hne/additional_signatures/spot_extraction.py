"""src/hne/additional_signatures/spot_extraction.py

Per-spot FMRP, APM4, APM12 scores, tumor fraction and immune count of one patient.

Scores are ssGSEA on the log1p-CPM layer, as in hne.preprocessing.spot_signatures.
Immune count of a spot = sum of the cell2location weights (absolute cell abundance) of the immune
cell types over all in-tissue spots within `radius_um`. Distances come from the Visium array grid:
neighbouring spots are 100 um apart, so a step of one array column is 50 um and one array row is
50 * sqrt(3) um.
"""
import gseapy as gp  # noqa
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hne.additional_signatures.gene_sets import GENE_SETS, IMMUNE_CELL_TYPES
from hne.core.data_io import load_visium
from hne.core.paths import PATIENTS

WEIGHTS_KEY = "deconv_sample_level_custom1_weights"
FRACTIONS_KEY = "deconv_sample_level_custom1_frac"
HALF_SPOT_PITCH_UM = 50.0       # Visium centre-to-centre distance is 100 um


def grid_coordinates_um(array_row: np.ndarray, array_col: np.ndarray) -> np.ndarray:
    return np.column_stack([array_col * HALF_SPOT_PITCH_UM, array_row * HALF_SPOT_PITCH_UM * np.sqrt(3)])


def immune_count_within_radius(array_row, array_col, immune_weight, radius_um: float, include_self: bool):
    """Immune cells within radius_um of each spot, and how many spots that radius covered."""
    xy = grid_coordinates_um(array_row, array_col)
    tree = cKDTree(xy)
    neighbours = tree.query_ball_point(xy, r=radius_um + 1e-6)
    counts, n_spots = np.empty(len(xy)), np.empty(len(xy), dtype=int)
    for i, idx in enumerate(neighbours):
        if not include_self:
            idx = [j for j in idx if j != i]
        counts[i] = immune_weight[idx].sum()
        n_spots[i] = len(idx)
    return counts, n_spots


def score_gene_sets(vis, processes: int, min_size: int):
    """ssGSEA score per spot for every set in GENE_SETS, scored on the genes present in the h5ad."""
    present = {name: [g for g in genes if g in vis.var_names] for name, genes in GENE_SETS.items()}
    empty = [name for name, genes in present.items() if not genes]
    if empty:
        raise ValueError(f"no gene of {empty} is in the h5ad")
    layer = vis.layers["log_norm_count"]
    expr = pd.DataFrame(layer.toarray() if hasattr(layer, "toarray") else layer,
                        index=vis.obs_names, columns=vis.var_names).T
    res = gp.ssgsea(data=expr, gene_sets=present, outdir=None, permutation_num=0,
                    no_plot=True, processes=processes, min_size=min_size).res2d
    scores = res.pivot(index="Name", columns="Term", values="NES").astype(float)
    scores = scores.reindex(index=vis.obs_names)[list(GENE_SETS)]
    genes_used = {name: f"{len(genes)}/{len(GENE_SETS[name])}" for name, genes in present.items()}
    return scores, genes_used


def extract_patient_spots(patient_id: str, radius_um: float, include_self: bool,
                          ssgsea_processes: int, ssgsea_min_size: int):
    """Table with one row per in-tissue spot, plus the gene coverage of each set."""
    vis = load_visium(PATIENTS[patient_id], keep_layers=["log_norm_count"])
    vis = vis[vis.obs["in_tissue"].astype(int) == 1].copy()

    for key in (WEIGHTS_KEY, FRACTIONS_KEY):
        if key not in vis.obsm:
            raise KeyError(f"{patient_id}: obsm has no {key!r}")
    weights, fractions = vis.obsm[WEIGHTS_KEY], vis.obsm[FRACTIONS_KEY]
    missing = [c for c in IMMUNE_CELL_TYPES if c not in weights.columns]
    if missing:
        raise KeyError(f"{patient_id}: no deconvolution column for {missing}; "
                       f"available: {list(weights.columns)}")
    tumor_columns = [c for c in fractions.columns if c.startswith("Tu_")]
    if not tumor_columns:
        raise KeyError(f"{patient_id}: no Tu_ column in {FRACTIONS_KEY}")

    array_row = vis.obs["array_row"].astype(int).to_numpy()
    array_col = vis.obs["array_col"].astype(int).to_numpy()
    immune_weight = weights[IMMUNE_CELL_TYPES].sum(axis=1).to_numpy(dtype=float)
    immune_count, n_spots = immune_count_within_radius(array_row, array_col, immune_weight, radius_um, include_self)
    scores, genes_used = score_gene_sets(vis, ssgsea_processes, ssgsea_min_size)

    table = pd.DataFrame({
        "patient_id": patient_id,
        "barcode": vis.obs_names,
        "array_row": array_row,
        "array_col": array_col,
        "tumor_fraction": fractions[tumor_columns].sum(axis=1).to_numpy(dtype=float),
        "immune_count": immune_count,
        "n_spots_in_radius": n_spots,
    })
    for name in GENE_SETS:
        table[f"{name}_score"] = scores[name].to_numpy()
    return table, genes_used
