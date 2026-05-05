# 2. Compute/attach tumor fraction per spot

import pandas as pd
import numpy as np

def attach_tumor_fraction(spots, vis):
    """
    Compute tumor fraction per spot. Attach spots and their metadata.
    """
    in_tissue_spots = spots[spots["in_tissue"] == 1].copy()
    in_tissue_spots["barcode"] = in_tissue_spots["barcode"].astype(str)

    grp = adata.obsm["deconv_sample_level_custom1_frac"]
    tumor_fraction = grp["Tu_CH_L_282a_c01"] + grp["Tu_CH_L_282a_nos"]






# 3. Merge spots metadata and tumor fractions

merge_spot_metadata()


# 4. Derive tile coordinates from spot positions


add_tile_coordinates()


# 5. Compute Bayesian tile purity
compute_tile_purity()



# 6. Filter tumor tiles (purity + min_spots)
filter_tumor_tiles()
