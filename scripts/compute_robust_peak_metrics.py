from __future__ import annotations

import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_ROOT = PROJECT_ROOT / "parametric_study" / "surrogate_database_cases"

BASE_METRICS_CSV = (
    PROJECT_ROOT
    / "results"
    / "surrogate_database"
    / "metrics"
    / "surrogate_database_metrics.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "robust_metrics"
OUTPUT_CSV = OUTPUT_DIR / "surrogate_database_robust_metrics.csv"
SUMMARY_TXT = OUTPUT_DIR / "robust_peak_metric_summary.txt"


TOP_FRACTIONS = [0.01, 0.02, 0.05, 0.10]
ROLLING_WINDOWS = [0.025, 0.050, 0.100]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

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
        raise ValueError(f"No numerical data found in {path}")

    return rows


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")

    return sum(values) / len(values)


def top_fraction_mean(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")

    n_values = len(values)
    n_top = max(1, int(math.ceil(fraction * n_values)))

    sorted_values = sorted(values, reverse=True)

    return mean(sorted_values[:n_top])


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return float("nan")

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile_value / 100.0

    lower_index = int(math.floor(index))
    upper_index = int(math.ceil(index))

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_weight = upper_index - index
    upper_weight = index - lower_index

    return lower_weight * sorted_values[lower_index] + upper_weight * sorted_values[upper_index]


def median_time_step(rows: list[tuple[float, float, float]]) -> float:
    if len(rows) < 2:
        return float("nan")

    differences = [
        rows[index + 1][0] - rows[index][0]
        for index in range(len(rows) - 1)
    ]

    sorted_differences = sorted(differences)
    middle = len(sorted_differences) // 2

    if len(sorted_differences) % 2 == 1:
        return sorted_differences[middle]

    return 0.5 * (sorted_differences[middle - 1] + sorted_differences[middle])


def rolling_average_peak(
    rows: list[tuple[float, float, float]],
    value_index: int,
    window_seconds: float,
) -> tuple[float, float]:
    dt = median_time_step(rows)

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Invalid time step in pressure history.")

    window_samples = max(1, int(round(window_seconds / dt)) + 1)

    values = [row[value_index] for row in rows]
    times = [row[0] for row in rows]

    if window_samples >= len(values):
        return mean(values), times[len(times) // 2]

    best_value = -float("inf")
    best_time = times[0]

    running_sum = sum(values[:window_samples])

    for start_index in range(0, len(values) - window_samples + 1):
        if start_index > 0:
            running_sum -= values[start_index - 1]
            running_sum += values[start_index + window_samples - 1]

        current_average = running_sum / window_samples

        if current_average > best_value:
            best_value = current_average
            centre_index = start_index + window_samples // 2
            best_time = times[centre_index]

    return best_value, best_time


def summarise_history(
    prefix: str,
    rows: list[tuple[float, float, float]],
) -> dict[str, float]:
    pressure_values = [row[1] for row in rows]
    alpha_values = [row[2] for row in rows]

    output: dict[str, float] = {}

    output[f"{prefix}_p_rgh_percentile_90_Pa"] = percentile(pressure_values, 90.0)
    output[f"{prefix}_p_rgh_percentile_95_Pa"] = percentile(pressure_values, 95.0)
    output[f"{prefix}_p_rgh_percentile_99_Pa"] = percentile(pressure_values, 99.0)

    for fraction in TOP_FRACTIONS:
        tag = str(fraction).replace(".", "p")
        output[f"{prefix}_p_rgh_top_{tag}_mean_Pa"] = top_fraction_mean(pressure_values, fraction)

    for window_seconds in ROLLING_WINDOWS:
        tag = str(window_seconds).replace(".", "p")
        rolling_peak, rolling_peak_time = rolling_average_peak(
            rows,
            value_index=1,
            window_seconds=window_seconds,
        )

        output[f"{prefix}_p_rgh_rolling_{tag}s_peak_mean_Pa"] = rolling_peak
        output[f"{prefix}_p_rgh_rolling_{tag}s_peak_time_s"] = rolling_peak_time

    output[f"{prefix}_alpha_percentile_95"] = percentile(alpha_values, 95.0)

    return output


def process_case(row: dict[str, str]) -> dict[str, object]:
    case_name = row["case_name"]
    case_dir = CASE_ROOT / case_name

    average_rows = read_surface_file(
        case_dir
        / "postProcessing"
        / "obstaclePressureAverage"
        / "0"
        / "surfaceFieldValue.dat"
    )

    maximum_rows = read_surface_file(
        case_dir
        / "postProcessing"
        / "obstaclePressureMaximum"
        / "0"
        / "surfaceFieldValue.dat"
    )

    output: dict[str, object] = dict(row)

    output.update(summarise_history("area_average", average_rows))
    output.update(summarise_history("maximum", maximum_rows))

    return output


def write_summary(rows: list[dict[str, object]]) -> None:
    metrics = [
        "area_average_p_rgh_top_0p01_mean_Pa",
        "area_average_p_rgh_top_0p05_mean_Pa",
        "area_average_p_rgh_rolling_0p025s_peak_mean_Pa",
        "area_average_p_rgh_rolling_0p05s_peak_mean_Pa",
        "maximum_p_rgh_top_0p01_mean_Pa",
        "maximum_p_rgh_top_0p05_mean_Pa",
        "maximum_p_rgh_rolling_0p025s_peak_mean_Pa",
        "maximum_p_rgh_rolling_0p05s_peak_mean_Pa",
    ]

    lines: list[str] = []

    lines.append("Robust peak metric summary")
    lines.append("==========================")
    lines.append("")
    lines.append(f"Cases processed: {len(rows)}")
    lines.append("")

    for metric in metrics:
        values = [float(row[metric]) for row in rows]

        max_row = max(rows, key=lambda row: float(row[metric]))
        min_row = min(rows, key=lambda row: float(row[metric]))

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

    base_rows = read_csv(BASE_METRICS_CSV)

    processed_rows: list[dict[str, object]] = []

    for index, row in enumerate(base_rows, start=1):
        processed_rows.append(process_case(row))

        if index % 25 == 0 or index == len(base_rows):
            print(f"Processed {index}/{len(base_rows)} cases")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(processed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(processed_rows)

    write_summary(processed_rows)

    print()
    print("Robust peak metric extraction complete.")
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
