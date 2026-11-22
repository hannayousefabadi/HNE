"""scripts/additional_signatures/select_naive_patients.py

Step 1. Reads the Clinical/ forms, decides which manifest blocks are treatment-naive (rules in
hne.additional_signatures.naive_cohort), and draws the subset the analysis will run on.
Writes results/additional_signatures/naive_patients.csv (all blocks, with the evidence)
and prints how many blocks fall in each category.
"""
import json

from hne.additional_signatures.naive_cohort import build_naive_table
from hne.additional_signatures.paths import ADDITIONAL_SIGNATURES, NAIVE_PATIENTS_CSV
from hne.core.paths import PATIENT_IDS

CONFIG = {
    "n_patients": 20,        # the run uses this many naive patients, or all of them if fewer exist
    "random_seed": 0,        # the subset is a seeded random draw, not chosen by looking at any score
}

table = build_naive_table(PATIENT_IDS)
naive = table[table["naive"]]
chosen = naive.sample(n=min(CONFIG["n_patients"], len(naive)), random_state=CONFIG["random_seed"])
table["selected"] = table["patient_id"].isin(chosen["patient_id"])
table.to_csv(NAIVE_PATIENTS_CSV, index=False)
(ADDITIONAL_SIGNATURES / "naive_selection_config.json").write_text(json.dumps(CONFIG, indent=2))

print(f"{len(table)} manifest blocks, {len(naive)} naive, {len(chosen)} selected (seed {CONFIG['random_seed']})")
print("\nsample_chronology:\n", table["sample_chronology"].value_counts().to_string())
print("\nnot naive, by first reason:\n", table.loc[~table["naive"], "reasons"].str.split("; ").str[0]
      .str.replace(r"\d{4}-\d{2}-\d{2}.*", "<date>", regex=True).value_counts().to_string())
print("\nselected:", sorted(chosen["patient_id"]))
print("written:", NAIVE_PATIENTS_CSV)
