from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_ROOT = PROJECT_ROOT / "parametric_study" / "surrogate_database_cases"
STATUS_CSV = PROJECT_ROOT / "results" / "surrogate_database" / "surrogate_database_run_status.csv"
METADATA_CSV = CASE_ROOT / "surrogate_case_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "metrics"
OUTPUT_CSV = OUTPUT_DIR / "surrogate_database_metrics.csv"
SUMMARY_TXT = OUTPUT_DIR / "surrogate_database_metric_summary.txt"
FAILED_CSV = OUTPUT_DIR / "surrogate_database_extraction_failures.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_surface_file(path: Path) -> list[tuple[float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing surfaceFieldValue file: {path}")

    rows: list[tuple[float, float, float]] = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 3:
            continue

        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))

    if not rows:
        raise ValueError(f"No numerical rows found in {path}")

    return rows


def first_time_above(rows: list[tuple[float, float, float]], column_index: int, threshold: float) -> float | None:
    for row in rows:
        if row[column_index] >= threshold:
            return row[0]

    return None


def last_time_above(rows: list[tuple[float, float, float]], column_index: int, threshold: float) -> float | None:
    for row in reversed(rows):
        if row[column_index] >= threshold:
            return row[0]

    return None


def duration_above(rows: list[tuple[float, float, float]], column_index: int, threshold: float) -> float:
    first = first_time_above(rows, column_index, threshold)
    last = last_time_above(rows, column_index, threshold)

    if first is None or last is None:
        return 0.0

    return max(0.0, last - first)


def positive_trapezoidal_integral(rows: list[tuple[float, float, float]], column_index: int) -> float:
    total = 0.0

    for left, right in zip(rows[:-1], rows[1:]):
        t0 = left[0]
        t1 = right[0]

        y0 = max(left[column_index], 0.0)
        y1 = max(right[column_index], 0.0)

        total += 0.5 * (y0 + y1) * (t1 - t0)

    return total


def summarise_case(metadata_row: dict[str, str], status_by_case: dict[str, dict[str, str]]) -> dict[str, object]:
    case_name = metadata_row["case_name"]
    case_dir = CASE_ROOT / case_name

    average_file = case_dir / "postProcessing" / "obstaclePressureAverage" / "0" / "surfaceFieldValue.dat"
    maximum_file = case_dir / "postProcessing" / "obstaclePressureMaximum" / "0" / "surfaceFieldValue.dat"

    average_rows = read_surface_file(average_file)
    maximum_rows = read_surface_file(maximum_file)

    peak_average_pressure = max(average_rows, key=lambda row: row[1])
    peak_maximum_pressure = max(maximum_rows, key=lambda row: row[1])

    peak_average_alpha = max(average_rows, key=lambda row: row[2])
    peak_maximum_alpha = max(maximum_rows, key=lambda row: row[2])

    first_local_wetting = first_time_above(maximum_rows, column_index=2, threshold=0.5)
    first_distributed_wetting = first_time_above(average_rows, column_index=2, threshold=0.5)

    local_wetting_duration = duration_above(maximum_rows, column_index=2, threshold=0.5)
    distributed_wetting_duration = duration_above(average_rows, column_index=2, threshold=0.5)

    average_pressure_impulse = positive_trapezoidal_integral(average_rows, column_index=1)
    maximum_pressure_impulse = positive_trapezoidal_integral(maximum_rows, column_index=1)

    status = status_by_case.get(case_name, {})

    return {
        "case_name": case_name,
        "dataset_split": metadata_row["dataset_split"],
        "water_height_m": float(metadata_row["water_height_m"]),
        "obstacle_height_m": float(metadata_row["obstacle_height_m"]),
        "obstacle_front_x_m": float(metadata_row["obstacle_front_x_m"]),
        "obstacle_width_m": float(metadata_row["obstacle_width_m"]),
        "end_time_s": float(metadata_row["end_time_s"]),
        "field_write_interval_s": float(metadata_row["field_write_interval_s"]),
        "metric_write_interval_s": float(metadata_row["metric_write_interval_s"]),

        "run_status": status.get("status", ""),
        "final_time_s": status.get("final_time", ""),
        "mesh_ok": status.get("mesh_ok", ""),
        "n_pressure_samples_status": status.get("n_pressure_samples", ""),

        "n_area_average_samples": len(average_rows),
        "n_maximum_samples": len(maximum_rows),

        "peak_area_average_p_rgh_Pa": peak_average_pressure[1],
        "time_peak_area_average_p_rgh_s": peak_average_pressure[0],

        "peak_maximum_p_rgh_Pa": peak_maximum_pressure[1],
        "time_peak_maximum_p_rgh_s": peak_maximum_pressure[0],

        "peak_area_average_alpha_water": peak_average_alpha[2],
        "time_peak_area_average_alpha_water_s": peak_average_alpha[0],

        "peak_maximum_alpha_water": peak_maximum_alpha[2],
        "time_peak_maximum_alpha_water_s": peak_maximum_alpha[0],

        "first_local_wetting_time_s": first_local_wetting,
        "first_distributed_wetting_time_s": first_distributed_wetting,

        "local_wetting_duration_s": local_wetting_duration,
        "distributed_wetting_duration_s": distributed_wetting_duration,

        "area_average_pressure_impulse_Pa_s": average_pressure_impulse,
        "maximum_pressure_impulse_Pa_s": maximum_pressure_impulse,

        "final_area_average_p_rgh_Pa": average_rows[-1][1],
        "final_maximum_p_rgh_Pa": maximum_rows[-1][1],
        "final_area_average_alpha_water": average_rows[-1][2],
        "final_maximum_alpha_water": maximum_rows[-1][2],
    }


def write_metric_summary(rows: list[dict[str, object]], failures: list[dict[str, str]]) -> None:
    metrics = [
        "peak_area_average_p_rgh_Pa",
        "peak_maximum_p_rgh_Pa",
        "area_average_pressure_impulse_Pa_s",
        "maximum_pressure_impulse_Pa_s",
        "first_distributed_wetting_time_s",
    ]

    lines: list[str] = []

    lines.append("Surrogate database metric extraction summary")
    lines.append("===========================================")
    lines.append("")
    lines.append(f"Successful cases: {len(rows)}")
    lines.append(f"Failed extractions: {len(failures)}")
    lines.append("")

    split_counts: dict[str, int] = {}

    for row in rows:
        split = str(row["dataset_split"])
        split_counts[split] = split_counts.get(split, 0) + 1

    lines.append("Successful cases by split:")

    for split, count in sorted(split_counts.items()):
        lines.append(f"  {split}: {count}")

    lines.append("")

    for metric in metrics:
        valid_rows = [row for row in rows if row[metric] not in ("", None)]
        values = [float(row[metric]) for row in valid_rows]

        if not values:
            continue

        max_row = max(valid_rows, key=lambda row: float(row[metric]))
        min_row = min(valid_rows, key=lambda row: float(row[metric]))

        lines.append(metric)
        lines.append("-" * len(metric))
        lines.append(f"  min: {min(values):.6g}")
        lines.append(f"  max: {max(values):.6g}")
        lines.append(f"  max case: {max_row['case_name']}")
        lines.append(
            f"    H={float(max_row['water_height_m']):.6g}, "
            f"h={float(max_row['obstacle_height_m']):.6g}, "
            f"x={float(max_row['obstacle_front_x_m']):.6g}"
        )
        lines.append(f"  min case: {min_row['case_name']}")
        lines.append(
            f"    H={float(min_row['water_height_m']):.6g}, "
            f"h={float(min_row['obstacle_height_m']):.6g}, "
            f"x={float(min_row['obstacle_front_x_m']):.6g}"
        )
        lines.append("")

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = read_csv(METADATA_CSV)
    status_rows = read_csv(STATUS_CSV)

    status_by_case = {row["case_name"]: row for row in status_rows}

    extracted_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, metadata_row in enumerate(metadata_rows, start=1):
        case_name = metadata_row["case_name"]

        try:
            row = summarise_case(metadata_row, status_by_case)
            extracted_rows.append(row)

            if index % 25 == 0 or index == len(metadata_rows):
                print(f"Extracted {index}/{len(metadata_rows)} cases")

        except Exception as exc:
            failures.append(
                {
                    "case_name": case_name,
                    "error": str(exc),
                }
            )
            print(f"FAILED extraction for {case_name}: {exc}")

    if extracted_rows:
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(extracted_rows[0].keys()))
            writer.writeheader()
            writer.writerows(extracted_rows)

    with FAILED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_name", "error"])
        writer.writeheader()
        writer.writerows(failures)

    write_metric_summary(extracted_rows, failures)

    print()
    print("Extraction complete.")
    print(f"Successful cases: {len(extracted_rows)}")
    print(f"Failed cases:     {len(failures)}")
    print()
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {FAILED_CSV}")
    print(f"Wrote: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
