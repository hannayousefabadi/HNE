"""src/hne/additional_signatures/naive_cohort.py

Which Visium blocks come from treatment-naive patients.

A block is naive when ALL of these hold:
  1. SAMPBLK says the sample was collected at "Baseline" (not after neoadjuvant therapy, not at recurrence)
  2. TU_TRT does not say the tumour was resected after neoadjuvant therapy, nor that neoadjuvant
     immunotherapy was given
  3. CANTRT has no chemotherapy, immunotherapy, targeted therapy, radiotherapy or other treatment
     whose start could be on or before the sample harvest date. A start date that is missing or only
     partly known (e.g. "2019-12-NK") counts at its earliest possible day, so doubt excludes.
Surgery rows are ignored: the resection is how the sample was obtained.
"""
import pandas as pd

from hne.core.config import RAW_DATA_BUCKET, RAW_DATA_PREFIX
from hne.core.data_io import get_s3_loader

BLOCK_ID_COLUMN = "Tumour block ID for MOSAIC"
SUBJECT_COLUMN = "Subject Id"

# CANTRT: treatment type -> the column holding its start date
START_DATE_COLUMN = {
    "Chemotherapy": "Chemotherapy start date",
    "Immunotherapy": "Immunotherapy start date",
    "Targeted therapy": "Targeted therapy start date",
    "Radiotherapy": "Radiotherapy start date",
    "Other": "Other treatment start date",
}


def read_clinical_form(form: str) -> pd.DataFrame:
    """One Clinical/ CSV. The export repeats the variable names as its first data row; it is dropped."""
    df = get_s3_loader().read_csv(f"s3://{RAW_DATA_BUCKET}/{RAW_DATA_PREFIX}/Clinical/{form}.csv", dtype=str)
    if df.iloc[0][SUBJECT_COLUMN] != "SubjectId":
        raise ValueError(f"{form}.csv: first data row is not the variable-name row, export format changed")
    return df.iloc[1:].reset_index(drop=True)


def earliest_date(text) -> pd.Timestamp:
    """'2019-12-NK' -> 2019-12-01, '2019-NK-NK' -> 2019-01-01. NaT if there is no year."""
    if pd.isna(text):
        return pd.NaT
    year, *rest = str(text).strip().split("-")
    if not (year.isdigit() and len(year) == 4):
        return pd.NaT
    month, day = (rest + ["01", "01"])[:2]
    month = month if month.isdigit() else "01"
    day = day if day.isdigit() else "01"
    return pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")


def build_naive_table(patient_ids: list[str]) -> pd.DataFrame:
    """One row per manifest block: the evidence for each rule and the final `naive` verdict."""
    sampblk = read_clinical_form("SAMPBLK")
    cantrt = read_clinical_form("CANTRT")
    tu_trt = read_clinical_form("TU_TRT")

    sampblk = sampblk.set_index(BLOCK_ID_COLUMN)
    tu_trt = tu_trt.set_index(SUBJECT_COLUMN)

    rows = []
    for block in patient_ids:
        if block not in sampblk.index:
            raise KeyError(f"{block} is in the manifest but not in SAMPBLK.csv")
        sample = sampblk.loc[block]
        subject = block[:-1]                       # CH_L_275a -> CH_L_275
        harvest = earliest_date(sample["Date of sample harvest"])
        if pd.isna(harvest):
            raise ValueError(f"{block}: cannot read Date of sample harvest {sample['Date of sample harvest']!r}")

        reasons = []
        if sample["Sample collection chronology"] != "Baseline":
            reasons.append(f"sample collected '{sample['Sample collection chronology']}'")

        if subject not in tu_trt.index:
            raise KeyError(f"{subject} is not in TU_TRT.csv")
        tumour = tu_trt.loc[subject]
        if tumour["NS - tumour resection chronology"] == "Surgery after neoadjuvant therapy":
            reasons.append("resection after neoadjuvant therapy")
        if tumour["Neoadjuvant immunotherapy"] == "Yes":
            reasons.append("neoadjuvant immunotherapy")

        treatments = cantrt[(cantrt[SUBJECT_COLUMN] == subject) & cantrt["Treatment type"].isin(START_DATE_COLUMN)]
        for _, treatment in treatments.iterrows():
            start = earliest_date(treatment[START_DATE_COLUMN[treatment["Treatment type"]]])
            if pd.isna(start):
                reasons.append(f"{treatment['Treatment type']} start date unknown")
            elif start <= harvest:
                reasons.append(f"{treatment['Treatment type']} started {start.date()} <= harvest {harvest.date()}")

        rows.append({"patient_id": block, "subject_id": subject,
                     "sample_chronology": sample["Sample collection chronology"],
                     "sample_source": sample["Sample source"],
                     "harvest_date": harvest.date(),
                     "n_non_surgical_treatments": len(treatments),
                     "naive": len(reasons) == 0,
                     "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows)
