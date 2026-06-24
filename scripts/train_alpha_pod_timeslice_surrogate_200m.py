from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIELD_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "field_surrogate"

INFO_JSON = FIELD_DIR / "alpha_field_dataset_info.json"
SNAPSHOT_METADATA_CSV = FIELD_DIR / "alpha_snapshot_metadata.csv"
SNAPSHOT_MEMMAP = FIELD_DIR / "alpha_snapshots_float32.dat"

MODEL_DIR = FIELD_DIR / "pod_timeslice_models_200m"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PCA_MODEL = MODEL_DIR / "alpha_timeslice_incremental_pca.joblib"
TIMESLICE_MODEL = MODEL_DIR / "alpha_timeslice_pod_surrogate.joblib"
SUMMARY_TXT = MODEL_DIR / "alpha_timeslice_pod_surrogate_summary.txt"
VALIDATION_ERRORS_CSV = MODEL_DIR / "alpha_timeslice_validation_errors.csv"

N_MODES = 200
BATCH_SIZE = 512
N_NEIGHBOURS = 6


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def batch_indices(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def flatten_batch(snapshots: np.memmap, indices: np.ndarray) -> np.ndarray:
    batch = np.asarray(snapshots[indices, :, :], dtype=np.float64)
    return batch.reshape(batch.shape[0], -1)


def geometry_input(row: dict[str, str]) -> list[float]:
    return [
        float(row["water_height_m"]),
        float(row["obstacle_height_m"]),
        float(row["obstacle_front_x_m"]),
    ]


def time_key(row: dict[str, str]) -> str:
    return f"{float(row['time_s']):.6f}"


def nearest_time_key(value: float, available_keys: list[str]) -> str:
    return min(available_keys, key=lambda key: abs(float(key) - value))


def main() -> None:
    info = json.loads(INFO_JSON.read_text())
    rows = read_csv(SNAPSHOT_METADATA_CSV)

    n_snapshots = int(info["n_snapshots"])
    ny = int(info["ny"])
    nx = int(info["nx"])

    snapshots = np.memmap(
        SNAPSHOT_MEMMAP,
        dtype="float32",
        mode="r",
        shape=(n_snapshots, ny, nx),
    )

    train_rows = [row for row in rows if row["dataset_split"] == "training_grid"]
    valid_rows = [row for row in rows if row["dataset_split"] == "validation_lhs"]

    train_indices = np.asarray([int(row["snapshot_index"]) for row in train_rows], dtype=int)
    valid_indices = np.asarray([int(row["snapshot_index"]) for row in valid_rows], dtype=int)

    print(f"Training snapshots: {len(train_rows)}")
    print(f"Validation snapshots: {len(valid_rows)}")
    print(f"POD modes: {N_MODES}")
    print(f"Time-sliced KNN neighbours: {N_NEIGHBOURS}")

    ipca = IncrementalPCA(n_components=N_MODES, batch_size=BATCH_SIZE)

    print()
    print("Fitting higher-rank Incremental PCA...")

    for batch_number, indices in enumerate(batch_indices(train_indices, BATCH_SIZE), start=1):
        X_batch = flatten_batch(snapshots, indices)
        ipca.partial_fit(X_batch)
        print(f"  PCA batch {batch_number}")

    print()
    print("Transforming training fields to POD coefficients...")

    train_coeffs = np.zeros((len(train_rows), N_MODES), dtype=np.float64)

    row_start = 0
    for indices in batch_indices(train_indices, BATCH_SIZE):
        X_batch = flatten_batch(snapshots, indices)
        coeffs = ipca.transform(X_batch)

        row_end = row_start + coeffs.shape[0]
        train_coeffs[row_start:row_end, :] = coeffs
        row_start = row_end

    grouped_indices: dict[str, list[int]] = defaultdict(list)

    for local_index, row in enumerate(train_rows):
        grouped_indices[time_key(row)].append(local_index)

    print()
    print(f"Training one surrogate per saved time: {len(grouped_indices)} time slices")

    timeslice_models: dict[str, object] = {}

    for key in sorted(grouped_indices.keys(), key=float):
        local_indices = grouped_indices[key]

        X = np.asarray([geometry_input(train_rows[i]) for i in local_indices], dtype=np.float64)
        y = train_coeffs[local_indices, :]

        n_neighbors = min(N_NEIGHBOURS, len(local_indices))

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")),
            ]
        )

        model.fit(X, y)
        timeslice_models[key] = model

    print()
    print("Evaluating validation reconstruction...")

    available_time_keys = sorted(timeslice_models.keys(), key=float)

    error_rows: list[dict[str, object]] = []
    rmse_values: list[float] = []
    mae_values: list[float] = []

    for batch_number, local_batch in enumerate(batch_indices(np.arange(len(valid_rows)), BATCH_SIZE), start=1):
        snapshot_indices = valid_indices[local_batch]
        true_flat = flatten_batch(snapshots, snapshot_indices)

        pred_flat_rows: list[np.ndarray] = []

        for local_index in local_batch:
            row = valid_rows[int(local_index)]
            key = nearest_time_key(float(row["time_s"]), available_time_keys)

            X = np.asarray([geometry_input(row)], dtype=np.float64)
            pred_coeffs = timeslice_models[key].predict(X)
            pred_flat = ipca.inverse_transform(pred_coeffs)[0]
            pred_flat = np.clip(pred_flat, 0.0, 1.0)

            pred_flat_rows.append(pred_flat)

        pred_flat_batch = np.asarray(pred_flat_rows, dtype=np.float64)

        errors = true_flat - pred_flat_batch

        batch_rmse = np.sqrt(np.mean(errors ** 2, axis=1))
        batch_mae = np.mean(np.abs(errors), axis=1)

        rmse_values.extend(batch_rmse.tolist())
        mae_values.extend(batch_mae.tolist())

        for row_i, rmse_value, mae_value in zip(local_batch, batch_rmse, batch_mae):
            row = valid_rows[int(row_i)]
            error_rows.append(
                {
                    "snapshot_index": row["snapshot_index"],
                    "case_name": row["case_name"],
                    "time_s": row["time_s"],
                    "water_height_m": row["water_height_m"],
                    "obstacle_height_m": row["obstacle_height_m"],
                    "obstacle_front_x_m": row["obstacle_front_x_m"],
                    "rmse_alpha": float(rmse_value),
                    "mae_alpha": float(mae_value),
                }
            )

        print(f"  validation batch {batch_number}")

    cumulative_explained = np.cumsum(ipca.explained_variance_ratio_)

    joblib.dump(ipca, PCA_MODEL)
    joblib.dump(
        {
            "models_by_time": timeslice_models,
            "available_time_keys": available_time_keys,
            "n_modes": N_MODES,
            "n_neighbours": N_NEIGHBOURS,
            "input_columns": [
                "water_height_m",
                "obstacle_height_m",
                "obstacle_front_x_m",
            ],
        },
        TIMESLICE_MODEL,
    )

    with VALIDATION_ERRORS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(error_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_rows)

    mean_rmse = float(np.mean(rmse_values))
    median_rmse = float(np.median(rmse_values))
    p95_rmse = float(np.percentile(rmse_values, 95.0))
    mean_mae = float(np.mean(mae_values))
    median_mae = float(np.median(mae_values))

    lines = [
        "Time-sliced alpha.water POD field surrogate summary",
        "===================================================",
        "",
        f"Grid: {ny} x {nx}",
        f"Training snapshots: {len(train_rows)}",
        f"Validation snapshots: {len(valid_rows)}",
        f"POD modes: {N_MODES}",
        f"Time slices: {len(timeslice_models)}",
        f"KNN neighbours per time slice: {N_NEIGHBOURS}",
        "",
        "POD energy:",
        f"  cumulative explained variance, 40 modes: {cumulative_explained[min(39, N_MODES - 1)]:.6f}",
        f"  cumulative explained variance, 80 modes: {cumulative_explained[min(79, N_MODES - 1)]:.6f}",
        f"  cumulative explained variance, 120 modes: {cumulative_explained[min(119, N_MODES - 1)]:.6f}",
        f"  cumulative explained variance, 160 modes: {cumulative_explained[min(159, N_MODES - 1)]:.6f}",
        f"  cumulative explained variance, 200 modes: {cumulative_explained[min(199, N_MODES - 1)]:.6f}",
        "",
        "Validation reconstruction error:",
        f"  mean alpha RMSE: {mean_rmse:.6f}",
        f"  median alpha RMSE: {median_rmse:.6f}",
        f"  95th percentile alpha RMSE: {p95_rmse:.6f}",
        f"  mean alpha MAE: {mean_mae:.6f}",
        f"  median alpha MAE: {median_mae:.6f}",
        "",
        "Model files:",
        f"  {PCA_MODEL}",
        f"  {TIMESLICE_MODEL}",
        f"  {VALIDATION_ERRORS_CSV}",
    ]

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Training complete.")
    print(SUMMARY_TXT.read_text())


if __name__ == "__main__":
    main()
