import gseapy as gp
import pandas as pd
from hne.utils import get_logger
from hne.preprocessing.preprocessing_config import PREPROCESSING_CONFIG

logger = get_logger()

def compute_signatures(vis, 
                       final_df, 
                       patient_id=None, 
                       qc_tracker=None, 
                       cfg=PREPROCESSING_CONFIG):
    """
    Compute pathway signatures per spot using ssGSEA.
    Scores represent rank-based enrichment per spot, removing depth bias
    and making raw scores directly comparable across patients.
    """
    signatures = {
    "FMRP_signature": ["MARCKSL1", "S100A16", "DDAH1", "MYCL", "SHANK2", "ITIH2", "PIK3AP1", 
            "LHFPL6", "FRMD5", "CLDN6", "ATP11A", "SLC25A21", "B4GALNT3", "WNT10A", 
            "KCTD17", "BCAM", "CCL14", "CCL15", "CCL23", "DLG4", "SPTSSB", "SOGA1", 
            "MAP9", "CCDC149", "CMBL", "PTPRN", "WTIP", "FXR1", "ARHGEF26", "PROS1", 
            "PARP8", "OSR1", "TFF2", "UCHL1", "PRSS35", "KCNK5", "AEBP1", "SP8", "CFTR", 
            "CYSLTR1", "FSCN1", "IL33", "ELFN1", "AFAP1L1", "LPAR4", "CASD1", "HS6ST2",
            "CD109", "MAL2", "PHF19"], # 50
    "Cell_cycle_signature": ["MCM4", "MCM3", "MCM2", "MCM6", "POLA1", "LIG1", "MCM5", 
                "PCNA", "CLSPN", "PCLAF", "CHAF1B", "SLFN11", "DUT", "FAM111A", 
                "UHRF1", "TYMS", "HELLS", "DHFRP1", "SIVA1", "MAP7D2"], # 20
    "YAP_signature": ["YAP1","TAZ", "TEAD4", "TEAD2", "TEAD3","TEAD1"], # 6
"WNT_signature": ["WNT2B","WNT5A", "WNT3A", "FZD2", "FZD3", "FZD4", "FZD8", 
                    "FZD9", "FZD10", "LRP5", "LRP6", "DVL1", "DVL3", "AXIN1", 
                    "AXIN2", "CSNK1A1", "CTNNB1"], # 17
    "EMT_signature": ["VIM","SNAI2","ZEB2","FN1", "MMP2", "AGER"] # 6
    }

    genes_in_data = set(vis.var_names)

    # filter signatures to genes actually detected in this slide
    active_signatures = {
        sig: [g for g in genes if g in genes_in_data]
        for sig, genes in signatures.items()
    }
    missing_signatures = [sig for sig, genes in active_signatures.items() if len(genes) == 0]
    valid_signatures = {sig: genes for sig, genes in active_signatures.items() if len(genes) > 0}
    all_signature_names = list(signatures.keys())

    metadata = {
        "genes_per_signature": sorted([
            f"{sig}: {len(v)}/{len(signatures[sig])} genes" 
            for sig, v in active_signatures.items()
        ]),
        "n_missing_signatures": len(missing_signatures)
    }

    if len(valid_signatures) == 0:
        logger.error(f"No genes detected for ANY signature in patient {patient_id}")
        if qc_tracker and patient_id:
            qc_tracker.add_record(patient_id, "signature_qc", "EXCLUDE",
                                  "Failed to compute ANY signatures", metadata)
        return [], active_signatures, final_df, metadata
    
    # expression matrix (genes x spots)
    # log_norm_count = log1p(CPM) -> total count normalized per-spot, then log-transfered
    # corrects for: sequencing depth differences between spots (removes within-sample depth artifacts)
    # raw counts -> CPM (per-spot)
    expr_df = pd.DataFrame(
        vis.layers["log_norm_count"].toarray() if hasattr(vis.layers["log_norm_count"], "toarray") else vis.layers["log_norm_count"],
        index=vis.obs_names,
        columns=vis.var_names
    ).T

    # run ssGSEA
    res = gp.ssgsea(
        data=expr_df,
        gene_sets=valid_signatures,
        outdir=None,
        permutation_num=0,
        no_plot=True,
        processes=cfg.ssgsea_processes,
        min_size=cfg.ssgsea_min_size
    )

    # res.res2d layout: index = (gene_set), columns = (sample / barcode) 
    raw_df = res.res2d.copy() 
    if raw_df.shape[1] == len(vis.obs_names):
        # transpose shape (n_signatures, n_spots) to (n_spots, n_sigantures)
        ssgsea_df = raw_df.T
    else:
        # already (n_spots, n_sigantures)
        ssgsea_df = raw_df

    # align index with the exact spots barcode
    ssgsea_df.index = vis.obs_names  
    ssgsea_df = ssgsea_df.reindex(columns=all_signature_names, fill_value=0.0)

    # format results: index = spot barcode, cols = f"{sig}_score"
    sig_cols = [f"{sig}_score" for sig in ssgsea_df.columns]
    ssgsea_df.columns = sig_cols
    ssgsea_df = ssgsea_df.rename_axis("barcode").reset_index()

    # merge into spots DataFrame
    spots_df = final_df.merge(ssgsea_df, on="barcode", how="inner")

    if qc_tracker and patient_id:
        if len(missing_signatures) > 0:
            failed_sigs = ", ".join(missing_signatures)
            qc_tracker.add_record(patient_id, "signature_qc", "FLAG",
                                  f"Missing genes for signatures: {failed_sigs}", metadata)
        else:
            qc_tracker.add_record(patient_id, "signature_qc", "OK",
                                  "All 5 ssGSEA signatures computed successfully", metadata)    
    
    return sig_cols, active_signatures, spots_df, metadata

