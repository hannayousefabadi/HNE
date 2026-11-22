"""scripts/additional_signatures/inspect_clinical.py

Read-only look at the Clinical/ forms, to find which columns say whether a patient was treated
before the Visium sample was taken, and which column holds the subject id.

Prints column names, dtypes, null counts, value counts of low-cardinality columns and a few
example values of id-like columns. It never prints one row per patient.
"""
from hne.core.config import RAW_DATA_BUCKET, RAW_DATA_PREFIX
from hne.core.data_io import get_s3_loader
from hne.core.paths import PATIENT_IDS

CONFIG = {
    "clinical_forms": ["DM", "CANTRT", "TU_TRT", "CM", "SAMPBLK", "ONCE", "TU_CLN", "IE"],
    "max_levels_shown": 15,         # a column with at most this many distinct values gets its counts printed
    "id_like_markers": ["ID", "SUBJ", "PAT"],
    "n_example_ids": 4,
}

loader = get_s3_loader()
print("manifest ids, for comparison:", PATIENT_IDS[:CONFIG["n_example_ids"]], f"({len(PATIENT_IDS)} patients)")

for form in CONFIG["clinical_forms"]:
    df = loader.read_csv(f"s3://{RAW_DATA_BUCKET}/{RAW_DATA_PREFIX}/Clinical/{form}.csv")
    print(f"\n--- {form}.csv: {df.shape[0]} rows x {df.shape[1]} columns")
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        print(f"  {col:45s} {str(df[col].dtype):10s} n_unique={n_unique:<5d} n_null={int(df[col].isna().sum())}")
        if 0 < n_unique <= CONFIG["max_levels_shown"]:
            print("      ", df[col].value_counts(dropna=False).to_dict())
        elif any(marker in col.upper() for marker in CONFIG["id_like_markers"]):
            print("       examples:", df[col].dropna().astype(str).unique()[:CONFIG["n_example_ids"]].tolist())
