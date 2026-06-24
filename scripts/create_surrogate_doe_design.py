from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "design"

EXISTING_METRICS_CSV = (
    PROJECT_ROOT
    / "results"
    / "parametric_matrix_high_time_1p5"
    / "high_time_1p5_matrix_impact_metrics.csv"
)

TRAINING_CSV = OUTPUT_DIR / "surrogate_training_grid_343_cases.csv"
VALIDATION_CSV = OUTPUT_DIR / "surrogate_validation_lhs_60_cases.csv"
ALL_CASES_CSV = OUTPUT_DIR / "surrogate_all_403_cases.csv"

RANDOM_SEED = 20260622

# Training design:
# 7 x 7 x 7 full-factorial grid.
WATER_HEIGHT_LEVELS = [
    0.240,
    0.250,
    0.275,
    0.292,
    0.315,
    0.340,
    0.365,
]

OBSTACLE_HEIGHT_LEVELS = [
    0.030,
    0.036,
    0.042,
    0.048,
    0.060,
    0.066,
    0.072,
]

OBSTACLE_FRONT_X_LEVELS = [
    0.220,
    0.250,
    0.275,
    0.292,
    0.320,
    0.350,
    0.380,
]

# Off-grid validation ranges.
WATER_HEIGHT_RANGE = (0.240, 0.365)
OBSTACLE_HEIGHT_RANGE = (0.030, 0.072)
OBSTACLE_FRONT_X_RANGE = (0.220, 0.380)

N_VALIDATION_CASES = 60

DEFAULT_OBSTACLE_WIDTH_M = 0.024


def read_existing_obstacle_width() -> float:
    if not EXISTING_METRICS_CSV.exists():
        return DEFAULT_OBSTACLE_WIDTH_M

    with EXISTING_METRICS_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)

    if first_row is None:
        return DEFAULT_OBSTACLE_WIDTH_M

    value = first_row.get("obstacle_width_m")

    if value is None or value == "":
        return DEFAULT_OBSTACLE_WIDTH_M

    return float(value)


def mm_tag(value_m: float) -> str:
    value_mm = int(round(value_m * 1000.0))
    return f"{value_mm:04d}"


def case_name(prefix: str, water_height: float, obstacle_height: float, obstacle_front_x: float) -> str:
    return (
        f"{prefix}_"
        f"H{mm_tag(water_height)}_"
        f"obsH{mm_tag(obstacle_height)}_"
        f"x{mm_tag(obstacle_front_x)}"
    )


def make_case_row(
    case_name_value: str,
    split: str,
    water_height: float,
    obstacle_height: float,
    obstacle_front_x: float,
    obstacle_width: float,
) -> dict[str, object]:
    return {
        "case_name": case_name_value,
        "dataset_split": split,
        "water_height_m": water_height,
        "obstacle_height_m": obstacle_height,
        "obstacle_front_x_m": obstacle_front_x,
        "obstacle_width_m": obstacle_width,
        "end_time_s": 1.5,
        "field_write_interval_s": 0.025,
        "metric_write_interval_s": 0.005,
        "notes": "",
    }


def create_training_rows(obstacle_width: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for water_height in WATER_HEIGHT_LEVELS:
        for obstacle_height in OBSTACLE_HEIGHT_LEVELS:
            for obstacle_front_x in OBSTACLE_FRONT_X_LEVELS:
                rows.append(
                    make_case_row(
                        case_name_value=case_name(
                            "surrGrid",
                            water_height,
                            obstacle_height,
                            obstacle_front_x,
                        ),
                        split="training_grid",
                        water_height=water_height,
                        obstacle_height=obstacle_height,
                        obstacle_front_x=obstacle_front_x,
                        obstacle_width=obstacle_width,
                    )
                )

    return rows


def latin_hypercube_values(
    lower: float,
    upper: float,
    n_values: int,
    rng: random.Random,
) -> list[float]:
    interval_width = (upper - lower) / n_values

    values = []

    for index in range(n_values):
        interval_lower = lower + index * interval_width
        interval_upper = interval_lower + interval_width
        values.append(rng.uniform(interval_lower, interval_upper))

    rng.shuffle(values)

    return values


def is_too_close_to_training_grid(
    water_height: float,
    obstacle_height: float,
    obstacle_front_x: float,
) -> bool:
    water_margin = 0.0025
    obstacle_height_margin = 0.0010
    obstacle_x_margin = 0.0025

    close_to_water_level = any(
        abs(water_height - level) < water_margin
        for level in WATER_HEIGHT_LEVELS
    )

    close_to_obstacle_height_level = any(
        abs(obstacle_height - level) < obstacle_height_margin
        for level in OBSTACLE_HEIGHT_LEVELS
    )

    close_to_x_level = any(
        abs(obstacle_front_x - level) < obstacle_x_margin
        for level in OBSTACLE_FRONT_X_LEVELS
    )

    return close_to_water_level and close_to_obstacle_height_level and close_to_x_level


def create_validation_rows(obstacle_width: float) -> list[dict[str, object]]:
    rng = random.Random(RANDOM_SEED)

    water_values = latin_hypercube_values(
        WATER_HEIGHT_RANGE[0],
        WATER_HEIGHT_RANGE[1],
        N_VALIDATION_CASES,
        rng,
    )

    obstacle_height_values = latin_hypercube_values(
        OBSTACLE_HEIGHT_RANGE[0],
        OBSTACLE_HEIGHT_RANGE[1],
        N_VALIDATION_CASES,
        rng,
    )

    obstacle_x_values = latin_hypercube_values(
        OBSTACLE_FRONT_X_RANGE[0],
        OBSTACLE_FRONT_X_RANGE[1],
        N_VALIDATION_CASES,
        rng,
    )

    rows: list[dict[str, object]] = []

    for index, (water_height, obstacle_height, obstacle_front_x) in enumerate(
        zip(water_values, obstacle_height_values, obstacle_x_values),
        start=1,
    ):
        # Very small deterministic perturbation if a validation point lands too
        # close to a training grid point.
        if is_too_close_to_training_grid(water_height, obstacle_height, obstacle_front_x):
            obstacle_front_x = min(
                OBSTACLE_FRONT_X_RANGE[1],
                obstacle_front_x + 0.003,
            )

        rows.append(
            make_case_row(
                case_name_value=(
                    f"surrVal_{index:03d}_"
                    f"H{mm_tag(water_height)}_"
                    f"obsH{mm_tag(obstacle_height)}_"
                    f"x{mm_tag(obstacle_front_x)}"
                ),
                split="validation_lhs",
                water_height=round(water_height, 6),
                obstacle_height=round(obstacle_height, 6),
                obstacle_front_x=round(obstacle_front_x, 6),
                obstacle_width=obstacle_width,
            )
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_projection(
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    split_styles = {
        "training_grid": {"marker": "o", "label": "Training grid", "s": 28, "alpha": 0.55},
        "validation_lhs": {"marker": "x", "label": "Validation LHS", "s": 42, "alpha": 0.9},
    }

    for split, style in split_styles.items():
        split_rows = [row for row in rows if row["dataset_split"] == split]

        ax.scatter(
            [float(row[x_key]) for row in split_rows],
            [float(row[y_key]) for row in split_rows],
            marker=style["marker"],
            label=style["label"],
            s=style["s"],
            alpha=style["alpha"],
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def write_design_summary(
    training_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
) -> None:
    summary_path = OUTPUT_DIR / "surrogate_design_summary.txt"

    lines = [
        "Surrogate CFD database design",
        "=============================",
        "",
        f"Training cases:   {len(training_rows)}",
        f"Validation cases: {len(validation_rows)}",
        f"Total cases:      {len(all_rows)}",
        "",
        "Active input variables:",
        f"  Initial water height H: {min(WATER_HEIGHT_LEVELS):.3f} to {max(WATER_HEIGHT_LEVELS):.3f} m",
        f"  Obstacle height h:      {min(OBSTACLE_HEIGHT_LEVELS):.3f} to {max(OBSTACLE_HEIGHT_LEVELS):.3f} m",
        f"  Obstacle front x:       {min(OBSTACLE_FRONT_X_LEVELS):.3f} to {max(OBSTACLE_FRONT_X_LEVELS):.3f} m",
        "",
        "Training design:",
        "  7 x 7 x 7 structured full-factorial grid = 343 cases.",
        "",
        "Validation design:",
        "  60 off-grid Latin-hypercube-style validation cases.",
        "",
        "Simulation settings:",
        "  endTime = 1.5 s",
        "  full-field writeInterval = 0.025 s",
        "  obstacle metric writeInterval = 0.005 s",
        "",
        "Purpose:",
        "  Build a sufficiently dense CFD database for Gaussian process, radial-basis,",
        "  support-vector, tree-based, and ensemble surrogate models.",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    obstacle_width = read_existing_obstacle_width()

    training_rows = create_training_rows(obstacle_width)
    validation_rows = create_validation_rows(obstacle_width)
    all_rows = training_rows + validation_rows

    write_csv(TRAINING_CSV, training_rows)
    write_csv(VALIDATION_CSV, validation_rows)
    write_csv(ALL_CASES_CSV, all_rows)

    plot_projection(
        all_rows,
        x_key="water_height_m",
        y_key="obstacle_height_m",
        filename="surrogate_design_projection_H_vs_h.png",
        title="Surrogate design projection: water height versus obstacle height",
        x_label="Initial water height, H [m]",
        y_label="Obstacle height, h [m]",
    )

    plot_projection(
        all_rows,
        x_key="water_height_m",
        y_key="obstacle_front_x_m",
        filename="surrogate_design_projection_H_vs_x.png",
        title="Surrogate design projection: water height versus obstacle position",
        x_label="Initial water height, H [m]",
        y_label="Obstacle front position, x_obs [m]",
    )

    plot_projection(
        all_rows,
        x_key="obstacle_height_m",
        y_key="obstacle_front_x_m",
        filename="surrogate_design_projection_h_vs_x.png",
        title="Surrogate design projection: obstacle height versus obstacle position",
        x_label="Obstacle height, h [m]",
        y_label="Obstacle front position, x_obs [m]",
    )

    write_design_summary(training_rows, validation_rows, all_rows)

    print("Created surrogate DOE design:")
    print(f"  {TRAINING_CSV}")
    print(f"  {VALIDATION_CSV}")
    print(f"  {ALL_CASES_CSV}")
    print()
    print("Case counts:")
    print(f"  training grid:   {len(training_rows)}")
    print(f"  validation LHS:  {len(validation_rows)}")
    print(f"  total:           {len(all_rows)}")
    print()
    print("Design projections:")
    for path in sorted(OUTPUT_DIR.glob("surrogate_design_projection_*.png")):
        print(f"  {path}")
    print()
    print(f"Summary: {OUTPUT_DIR / 'surrogate_design_summary.txt'}")


if __name__ == "__main__":
    main()
