"""scripts/audit/feature_diagnostics.py

Registration audit, part A2 (local, no images needed): look for signs of bad crops in the
extracted tile embeddings themselves.

  - near-duplicate tile embeddings within a patient (crops clamped to the same slide edge)
  - patients whose tiles are nearly identical to each other (mean pairwise cosine near 1)
"""
import numpy as np
import pandas as pd

from hne.core.paths import PATIENT_IDS, REGISTRATION_AUDIT
from hne.models.data import FEATURE_REGISTRY, get_available_patient_ids, load_features_and_targets

CONFIG = {
    "model_name": "phikon_v2",
    "output_dir": REGISTRATION_AUDIT,
    "duplicate_cosine": 0.999,   # raw cosine similarity above this counts as a near-duplicate
    "uniform_cosine": 0.98,      # mean pairwise cosine above this: the patient's tiles are nearly identical
}


def run():
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = FEATURE_REGISTRY[CONFIG["model_name"]]

    patients = get_available_patient_ids(PATIENT_IDS, spec)
    X, _, tile_meta = load_features_and_targets(
        features_dir=spec["dir"], patient_ids=patients,
        target_cols=["tile_purity"], filename_suffix=spec["suffix"],
    )
    groups = np.array([pid for pid, _ in tile_meta])
    tile_ids = np.array([tid for _, tid in tile_meta])

    X_unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    cohort_centroid = X.mean(axis=0)
    # typical distance of a tile from the cohort centroid: the scale to compare spreads against
    cohort_spread = float(np.linalg.norm(X - cohort_centroid, axis=1).mean())

    patient_rows, duplicate_rows = [], []
    for pid in np.unique(groups):
        idx = np.flatnonzero(groups == pid)
        n = len(idx)
        centroid = X[idx].mean(axis=0)
        within_spread = float(np.linalg.norm(X[idx] - centroid, axis=1).mean())

        row = {
            "patient_id": pid,
            "n_tiles": n,
            "within_patient_spread": within_spread,
            "spread_ratio": within_spread / cohort_spread,
            "centroid_distance_to_cohort": float(np.linalg.norm(centroid - cohort_centroid)),
            "mean_pairwise_cosine": np.nan,
            "max_pairwise_cosine": np.nan,
            "n_duplicate_pairs": 0,
            "n_tiles_in_duplicates": 0,
        }
        if n >= 2:
            sim = X_unit[idx] @ X_unit[idx].T
            iu = np.triu_indices(n, k=1)
            pair_sim = sim[iu]
            dup = pair_sim > CONFIG["duplicate_cosine"]
            row["mean_pairwise_cosine"] = float(pair_sim.mean())
            row["max_pairwise_cosine"] = float(pair_sim.max())
            row["n_duplicate_pairs"] = int(dup.sum())
            row["n_tiles_in_duplicates"] = int(len(set(iu[0][dup]) | set(iu[1][dup])))
            for a, b, s in zip(iu[0][dup], iu[1][dup], pair_sim[dup]):
                duplicate_rows.append({"patient_id": pid, "tile_a": tile_ids[idx[a]],
                                       "tile_b": tile_ids[idx[b]], "cosine": float(s)})
        patient_rows.append(row)

    df = pd.DataFrame(patient_rows)
    ratio = df["spread_ratio"]
    df["flag_duplicates"] = df["n_duplicate_pairs"] > 0
    df["flag_low_spread"] = df["mean_pairwise_cosine"] > CONFIG["uniform_cosine"]
    df["flagged"] = df["flag_duplicates"] | df["flag_low_spread"]

    df.sort_values("spread_ratio").to_csv(output_dir / "feature_diagnostics.csv", index=False)
    pd.DataFrame(duplicate_rows, columns=["patient_id", "tile_a", "tile_b", "cosine"]).to_csv(
        output_dir / "feature_duplicate_pairs.csv", index=False)

    print(f"\n{len(df)} patients, {len(X)} tiles ({CONFIG['model_name']})")
    print(f"Spread ratio (within-patient / cohort): median {ratio.median():.3f}, "
          f"range {ratio.min():.3f} to {ratio.max():.3f}")
    print(f"Mean pairwise cosine within patient: median {df['mean_pairwise_cosine'].median():.4f}, "
          f"max over patients {df['mean_pairwise_cosine'].max():.4f}")
    print(f"Patients with near-duplicate tiles (cosine > {CONFIG['duplicate_cosine']}): {int(df['flag_duplicates'].sum())} "
          f"({int(df['n_duplicate_pairs'].sum())} pairs)")
    print(f"Tiles that are a near-duplicate of another tile of the same patient: "
          f"{int(df['n_tiles_in_duplicates'].sum())} of {int(df['n_tiles'].sum())}")
    print(f"Patients with nearly identical tiles (mean pairwise cosine > {CONFIG['uniform_cosine']}): "
          f"{int(df['flag_low_spread'].sum())}")
    flagged = df[df["flagged"]].sort_values("spread_ratio")
    if not flagged.empty:
        print("\nFlagged patients:")
        print(flagged[["patient_id", "n_tiles", "spread_ratio", "mean_pairwise_cosine",
                       "n_duplicate_pairs", "flag_duplicates", "flag_low_spread"]].to_string(index=False))
    print(f"\nWritten to {output_dir}")


if __name__ == "__main__":
    run()
