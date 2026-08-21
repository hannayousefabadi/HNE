"""
diagnose_tif_cohort.py

Loops over the whole cohort and, for each patient with a TIF_MAP entry,
checks:
  1. Does the S3 object exist / how big is it (rules out bad key / empty download)
  2. Can tifffile read it (rules out a genuinely corrupt/truncated TIFF)
  3. Can openslide.OpenSlide read it (the thing that's actually crashing)

"""

import io
import traceback
import boto3
import pandas as pd
import tifffile
import openslide

from hne.core.paths import PatientS3Paths, TIF_MAP
from hne.core.config import RESULTS

s3 = boto3.client("s3")


def check_patient(patient_id: str, filename: str):
    row = {
        "patient_id": patient_id,
        "filename": filename,
        "s3_key": None,
        "exists": None,
        "size_bytes": None,
        "tifffile_ok": None,
        "tifffile_error": None,
        "openslide_ok": None,
        "openslide_error": None,
    }

    try:
        patient_paths = PatientS3Paths(patient_id)
        tif_path = f"{patient_paths.raw_image_prefix}/{filename}"
        row["s3_key"] = tif_path

        # parse s3://bucket/key
        path = tif_path[5:] if tif_path.startswith("s3://") else tif_path
        bucket, key = path.split("/", 1)

        # 1. does it exist / how big
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            row["exists"] = True
            row["size_bytes"] = head["ContentLength"]
        except Exception as e:
            row["exists"] = False
            row["openslide_error"] = f"HEAD failed: {e}"
            return row

        # download once, reuse bytes for both checks
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()

        # 2. tifffile
        try:
            with io.BytesIO(data) as bio:
                arr = tifffile.imread(bio)
            row["tifffile_ok"] = True
        except Exception as e:
            row["tifffile_ok"] = False
            row["tifffile_error"] = str(e)

        # 3. openslide (write to a real temp file, same as production code)
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            slide = openslide.OpenSlide(tmp.name)
            row["openslide_ok"] = True
            slide.close()
        except Exception as e:
            row["openslide_ok"] = False
            row["openslide_error"] = str(e)
        finally:
            os.unlink(tmp.name)

    except Exception as e:
        row["openslide_error"] = f"unexpected: {e}\n{traceback.format_exc()}"

    return row


def main():
    print(f"Checking {len(TIF_MAP)} patients...\n")
    results = []
    for i, (patient_id, filename) in enumerate(TIF_MAP.items(), 1):
        print(f"[{i}/{len(TIF_MAP)}] {patient_id} ...", end=" ")
        row = check_patient(patient_id, filename)
        results.append(row)
        if row["openslide_ok"] is True:
            print("OK")
        else:
            print(f"FAIL -> {row['openslide_error'] or row['tifffile_error']}")

    df = pd.DataFrame(results)
    out_path = f"{RESULTS}/tif_diagnostic_report.csv"
    df.to_csv(out_path, index=False)

    print("\n--- Summary ---")
    print(f"Total patients checked: {len(df)}")
    print(f"Missing/HEAD failed:    {(df['exists'] == False).sum()}")
    print(f"tifffile failed:        {(df['tifffile_ok'] == False).sum()}")
    print(f"openslide failed:       {(df['openslide_ok'] == False).sum()}")
    print(f"\nFull report written to: {out_path}")

    failing = df[df["openslide_ok"] == False]
    if len(failing):
        print("\nPatients failing OpenSlide specifically:")
        print(failing[["patient_id", "filename", "size_bytes", "tifffile_ok", "openslide_error"]].to_string(index=False))


if __name__ == "__main__":
    main()