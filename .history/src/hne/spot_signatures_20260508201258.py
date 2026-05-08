import scanpy as sc

def compute_signatures(vis, df, logger=None):
    """
    Compute pathway signatures per spot
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

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
    "WNT_signature": ["WNT2B","WNT5A", "ANT3A", "FZD2", "FZD3", "FZD4", "FZD8", 
                    "FZD9", "FZD10", "LRP5", "LRP6", "DVL1", "DVL3", "AXIN1", 
                    "AXIN2", "CSNK1A1", "CTNNB1"], # 17
    "EMT_signature": ["VIM","SNAI2","ZEB2","FN1", "MMP2", "AGER"] # 6
    }

    genes_in_data = set(vis.var_names)

    signature_genes = {
        key: [g for g in genes if g in genes_in_data]
        for key, genes in signatures.items()
    }

    # log genes kept per signature
    logger.info("Genes kept per signature:")
    for k, v in signature_genes.items():
        logger.info(f"{k}: {len(v)}/{len(signatures[k])} genes")

    # compute signature scores
    for sig, genes in signatures.items():
        genes_present = [g for g in genes if g in vis.var_names]

        if len(genes_present) == 0:
            logger.warning(f"No genes found for {sig} - skipping")
            continue

        sc.tl.score_genes(
            vis,
            gene_list=genes_present,
            score_name=f"{sig}_score",
            layer="log_norm_count"
        )


    sig_cols = [
    "FMRP_signature_score",
    "Cell_cycle_signature_score",
    "YAP_signature_score",
    "WNT_signature_score",
    "EMT_signature_score"
    ]

    # extract signatures to df
    obs_sig = vis.obs[sig_cols].copy()
    obs_sig = obs_sig.rename_axis("barcode")
    # merge with spots df
    spots_df = df.merge(obs_sig, on="barcode", how="inner")

    # metadata
    metadata = {
        "genes_per_signatures": {f"{k}: {len(v)}/{len(signatures[k])} genes" for k, v in signature_genes.items()}
    }


    return sig_cols, signature_genes, spots_df



