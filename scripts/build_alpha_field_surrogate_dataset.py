from __future__ import annotations

import csv
import gzip
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.tri as mtri
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_ROOT = PROJECT_ROOT / "parametric_study" / "surrogate_database_cases"

METRICS_CSV = (
    PROJECT_ROOT
    / "results"
    / "surrogate_database"
    / "robust_metrics"
    / "surrogate_database_robust_metrics.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "field_surrogate"

INFO_JSON = OUTPUT_DIR / "alpha_field_dataset_info.json"
SNAPSHOT_METADATA_CSV = OUTPUT_DIR / "alpha_snapshot_metadata.csv"
SNAPSHOT_MEMMAP = OUTPUT_DIR / "alpha_snapshots_float32.dat"

DOMAIN_LENGTH = 0.584
DOMAIN_HEIGHT = 0.584

NX = 96
NY = 96

# Use every saved field time. Set to 2 if you need a faster/lighter test.
TIME_STRIDE = 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return handle.read()

    return path.read_text(encoding="utf-8", errors="ignore")


def foam_file(path: Path) -> Path:
    if path.exists():
        return path

    gz_path = Path(str(path) + ".gz")

    if gz_path.exists():
        return gz_path

    raise FileNotFoundError(f"Missing OpenFOAM field file: {path}")


def parse_scalar_internal_field(path: Path) -> np.ndarray:
    text = read_text(foam_file(path))

    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(f"Could not parse nonuniform scalar internalField in {path}")

    n_values = int(match.group(1))
    values = np.fromstring(match.group(2), sep=" ", dtype=np.float64)

    if values.size != n_values:
        raise ValueError(f"Expected {n_values} scalar values, got {values.size} in {path}")

    return values


def parse_vector_internal_field(path: Path) -> np.ndarray:
    text = read_text(foam_file(path))

    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(f"Could not parse nonuniform vector internalField in {path}")

    n_values = int(match.group(1))

    vector_matches = re.findall(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
        match.group(2),
    )

    if len(vector_matches) != n_values:
        raise ValueError(f"Expected {n_values} vectors, got {len(vector_matches)} in {path}")

    return np.asarray(vector_matches, dtype=np.float64)


def ensure_cell_centres(case_dir: Path) -> Path:
    existing = sorted(case_dir.glob("*/C"))

    if existing:
        return existing[0]

    existing_gz = sorted(case_dir.glob("*/C.gz"))

    if existing_gz:
        return existing_gz[0]

    command = (
        "source /usr/lib/openfoam/openfoam2312/etc/bashrc && "
        f"cd '{case_dir}' && "
        "postProcess -func writeCellCentres -time 0 > log.writeCellCentres 2>&1"
    )

    subprocess.run(["bash", "-lc", command], check=False)

    existing = sorted(case_dir.glob("*/C"))

    if existing:
        return existing[0]

    existing_gz = sorted(case_dir.glob("*/C.gz"))

    if existing_gz:
        return existing_gz[0]

    raise FileNotFoundError(f"Could not locate or generate C field for {case_dir}")


def numeric_time_dirs(case_dir: Path) -> list[Path]:
    output: list[tuple[float, Path]] = []

    for item in case_dir.iterdir():
        if not item.is_dir():
            continue

        try:
            time_value = float(item.name)
        except ValueError:
            continue

        alpha_path = item / "alpha.water"

        if alpha_path.exists() or Path(str(alpha_path) + ".gz").exists():
            output.append((time_value, item))

    output.sort(key=lambda pair: pair[0])

    return [path for _, path in output][::TIME_STRIDE]


def build_common_grid() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, DOMAIN_LENGTH, NX)
    y = np.linspace(0.0, DOMAIN_HEIGHT, NY)

    return np.meshgrid(x, y)


def resample_alpha_to_grid(
    centres: np.ndarray,
    triangulation: mtri.Triangulation,
    alpha_values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    obstacle_x: float,
    obstacle_h: float,
    obstacle_w: float,
) -> np.ndarray:
    interpolator = mtri.LinearTriInterpolator(triangulation, alpha_values)
    field = interpolator(grid_x, grid_y)

    if np.ma.is_masked(field):
        field = field.filled(0.0)

    field = np.asarray(field, dtype=np.float32)
    field = np.clip(field, 0.0, 1.0)

    # Mask the solid obstacle region so the surrogate learns a consistent visual field.
    obstacle_mask = (
        (grid_x >= obstacle_x)
        & (grid_x <= obstacle_x + obstacle_w)
        & (grid_y >= 0.0)
        & (grid_y <= obstacle_h)
    )

    field[obstacle_mask] = 0.0

    return field.astype(np.float32)


def count_snapshots(rows: list[dict[str, str]]) -> int:
    total = 0

    for row in rows:
        case_dir = CASE_ROOT / row["case_name"]
        total += len(numeric_time_dirs(case_dir))

    return total


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(METRICS_CSV)
    grid_x, grid_y = build_common_grid()

    n_snapshots = count_snapshots(rows)

    print(f"Cases: {len(rows)}")
    print(f"Snapshots: {n_snapshots}")
    print(f"Common grid: {NY} x {NX}")
    print(f"Memmap: {SNAPSHOT_MEMMAP}")

    snapshots = np.memmap(
        SNAPSHOT_MEMMAP,
        dtype="float32",
        mode="w+",
        shape=(n_snapshots, NY, NX),
    )

    metadata_rows: list[dict[str, object]] = []

    snapshot_index = 0

    for case_index, row in enumerate(rows, start=1):
        case_name = row["case_name"]
        case_dir = CASE_ROOT / case_name

        print(f"[{case_index:03d}/{len(rows):03d}] Processing {case_name}")

        centre_file = ensure_cell_centres(case_dir)
        centres = parse_vector_internal_field(centre_file)
        triangulation = mtri.Triangulation(centres[:, 0], centres[:, 1])

        obstacle_x = float(row["obstacle_front_x_m"])
        obstacle_h = float(row["obstacle_height_m"])
        obstacle_w = float(row["obstacle_width_m"])

        time_dirs = numeric_time_dirs(case_dir)

        for time_dir in time_dirs:
            time_value = float(time_dir.name)
            alpha_values = parse_scalar_internal_field(time_dir / "alpha.water")

            if alpha_values.size != centres.shape[0]:
                raise ValueError(
                    f"alpha size {alpha_values.size} does not match centre count "
                    f"{centres.shape[0]} for {case_name}, t={time_value}"
                )

            field = resample_alpha_to_grid(
                centres=centres,
                triangulation=triangulation,
                alpha_values=alpha_values,
                grid_x=grid_x,
                grid_y=grid_y,
                obstacle_x=obstacle_x,
                obstacle_h=obstacle_h,
                obstacle_w=obstacle_w,
            )

            snapshots[snapshot_index, :, :] = field

            metadata_rows.append(
                {
                    "snapshot_index": snapshot_index,
                    "case_name": case_name,
                    "dataset_split": row["dataset_split"],
                    "time_s": time_value,
                    "water_height_m": row["water_height_m"],
                    "obstacle_height_m": row["obstacle_height_m"],
                    "obstacle_front_x_m": row["obstacle_front_x_m"],
                    "obstacle_width_m": row["obstacle_width_m"],
                }
            )

            snapshot_index += 1

    snapshots.flush()

    with SNAPSHOT_METADATA_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    info = {
        "n_cases": len(rows),
        "n_snapshots": n_snapshots,
        "nx": NX,
        "ny": NY,
        "domain_length_m": DOMAIN_LENGTH,
        "domain_height_m": DOMAIN_HEIGHT,
        "time_stride": TIME_STRIDE,
        "snapshot_memmap": str(SNAPSHOT_MEMMAP),
        "snapshot_metadata_csv": str(SNAPSHOT_METADATA_CSV),
    }

    INFO_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print()
    print("Alpha field dataset complete.")
    print(f"Wrote: {SNAPSHOT_MEMMAP}")
    print(f"Wrote: {SNAPSHOT_METADATA_CSV}")
    print(f"Wrote: {INFO_JSON}")


if __name__ == "__main__":
    main()
