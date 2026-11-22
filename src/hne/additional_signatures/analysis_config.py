"""src/hne/additional_signatures/analysis_config.py

Settings every plotting script of the analysis shares, so they cannot drift apart.
"""

# tumor_fraction >= this keeps a spot: tumor outweighs all other cell types together.
# Do not change it after looking at the association plots to improve a result.
CANCER_DOMINANT_THRESHOLD = 0.5

# selected patients whose extraction failed, and why. The plots run on the others.
EXCLUDED_PATIENTS = {
    "CH_L_378a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_394a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_401a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_377a": "h5ad has no deconv_sample_level_custom1_weights",
}
