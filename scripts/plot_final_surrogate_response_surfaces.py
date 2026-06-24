from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from train_robust_surrogate_models import make_models, INPUT_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBUST_METRICS_CSV = (
    PROJECT_ROOT
    / "results"
    / "surrogate_database"
    / "robust_metrics"
    / "surrogate_database_robust_metrics.csv"
)

SELECTED_MODELS_CSV = (
    PROJECT_ROOT
    / "results"
    / "surrogate_database"
    / "robust_surrogate_models"
    / "selected_robust_surrogate_models.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "final_surrogate_plots"

FINAL_TARGETS = [
    "area_average_p_rgh_rolling_0p05s_peak_mean_Pa",
    "maximum_p_rgh_rolling_0p025s_peak_mean_Pa",
    "maximum_p_rgh_rolling_0p05s_peak_mean_Pa",
    "area_average_pressure_impulse_Pa_s",
    "maximum_pressure_impulse_Pa_s",
    "first_distributed_wetting_time_s",
]

X_SLICES = [0.25, 0.292, 0.35]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_filename(value: str) -> str:
    return (
        value.replace("_", "-")
        .replace(" ", "-")
        .replace("/", "-")
        .replace("[", "")
        .replace("]", "")
    )


def fit_selected_model(
    rows_train: list[dict[str, str]],
    target: str,
    model_name: str,
    target_transform: str,
):
    models = make_models()

    if model_name not in models:
        raise KeyError(f"Unknown selected model: {model_name}")

    model = models[model_name]

    X_train = np.asarray(
        [[float(row[column]) for column in INPUT_COLUMNS] for row in rows_train],
        dtype=float,
    )

    y_train = np.asarray([float(row[target]) for row in rows_train], dtype=float)

    if target_transform == "log1p":
        model.fit(X_train, np.log1p(y_train))
    elif target_transform == "raw":
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown target transform: {target_transform}")

    return model


def predict_selected_model(model, X: np.ndarray, target_transform: str) -> np.ndarray:
    prediction = model.predict(X)

    if target_transform == "log1p":
        prediction = np.expm1(prediction)

    return prediction


def make_performance_plot(selected_rows: list[dict[str, str]]) -> None:
    final_rows = [row for row in selected_rows if row["target"] in FINAL_TARGETS]

    labels = [row["target_label"] for row in final_rows]
    nrmse_percent = [100.0 * float(row["validation_normalised_rmse"]) for row in final_rows]
    r2_values = [float(row["validation_r2"]) for row in final_rows]

    y_positions = np.arange(len(final_rows))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ax.barh(y_positions, nrmse_percent)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Validation normalised RMSE [%]")
    ax.set_title("Selected final surrogate validation error")
    ax.grid(True, axis="x", alpha=0.3)

    nrmse_max = max(nrmse_percent)
    ax.set_xlim(0.0, nrmse_max * 1.22)

    for index, value in enumerate(nrmse_percent):
        ax.text(value + nrmse_max * 0.015, index, f"{value:.2f}%", va="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_selected_surrogate_validation_nrmse.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ax.barh(y_positions, r2_values)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Validation R²")
    ax.set_xlim(0.0, 1.12)
    ax.set_title("Selected final surrogate validation R²")
    ax.grid(True, axis="x", alpha=0.3)

    for index, value in enumerate(r2_values):
        ax.text(value + 0.005, index, f"{value:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_selected_surrogate_validation_r2.png", dpi=300)
    plt.close(fig)


def make_response_surface_plot(
    model,
    target: str,
    target_label: str,
    model_name: str,
    target_transform: str,
    H_values: np.ndarray,
    h_values: np.ndarray,
    x_slice: float,
) -> None:
    H_grid, h_grid = np.meshgrid(H_values, h_values)

    X_predict = np.column_stack(
        [
            H_grid.ravel(),
            h_grid.ravel(),
            np.full(H_grid.size, x_slice),
        ]
    )

    prediction = predict_selected_model(
        model=model,
        X=X_predict,
        target_transform=target_transform,
    ).reshape(H_grid.shape)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    contour = ax.contourf(H_grid, h_grid, prediction, levels=24)
    line_contour = ax.contour(H_grid, h_grid, prediction, levels=8, linewidths=0.7)
    ax.clabel(line_contour, inline=True, fontsize=8)

    ax.set_title(
        f"{target_label}\n"
        f"x_obs = {x_slice:.3f} m | {model_name}_{target_transform}"
    )
    ax.set_xlabel("Initial water height, H [m]")
    ax.set_ylabel("Obstacle height, h [m]")
    ax.grid(True, alpha=0.25)

    colourbar = fig.colorbar(contour, ax=ax)
    colourbar.set_label(target_label)

    fig.tight_layout()

    output_name = (
        f"response_surface_{safe_filename(target)}"
        f"_x{str(x_slice).replace('.', 'p')}.png"
    )

    fig.savefig(OUTPUT_DIR / output_name, dpi=300)
    plt.close(fig)


def make_recommendation_summary(selected_rows: list[dict[str, str]]) -> None:
    lines: list[str] = []

    lines.append("Final surrogate result interpretation")
    lines.append("=====================================")
    lines.append("")

    lines.append("Recommended final surrogate targets:")
    lines.append("")

    for row in selected_rows:
        target = row["target"]

        if target not in FINAL_TARGETS:
            continue

        lines.append(f"- {row['target_label']}")
        lines.append(f"  Model: {row['model_name']} with {row['target_transform']} target")
        lines.append(f"  Validation R2: {float(row['validation_r2']):.6g}")
        lines.append(f"  Validation normalised RMSE: {100.0 * float(row['validation_normalised_rmse']):.3f}%")
        lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "  The raw instantaneous peak-pressure quantities were difficult to learn "
        "because they are controlled by short-duration transient events. The more "
        "physically robust outputs, especially rolling peak pressures, pressure "
        "impulses, and distributed wetting time, produced substantially better "
        "validation performance on the 60 off-grid validation cases."
    )

    summary_path = OUTPUT_DIR / "final_surrogate_recommendation_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(ROBUST_METRICS_CSV)
    selected_rows = read_csv(SELECTED_MODELS_CSV)

    rows_train = [row for row in rows if row["dataset_split"] == "training_grid"]

    selected_by_target = {row["target"]: row for row in selected_rows}

    H_min = min(float(row["water_height_m"]) for row in rows)
    H_max = max(float(row["water_height_m"]) for row in rows)

    h_min = min(float(row["obstacle_height_m"]) for row in rows)
    h_max = max(float(row["obstacle_height_m"]) for row in rows)

    H_values = np.linspace(H_min, H_max, 90)
    h_values = np.linspace(h_min, h_max, 90)

    make_performance_plot(selected_rows)

    for target in FINAL_TARGETS:
        selected = selected_by_target[target]

        model_name = selected["model_name"]
        target_transform = selected["target_transform"]
        target_label = selected["target_label"]

        print(f"Plotting {target_label}")
        print(f"  model = {model_name}")
        print(f"  transform = {target_transform}")

        model = fit_selected_model(
            rows_train=rows_train,
            target=target,
            model_name=model_name,
            target_transform=target_transform,
        )

        for x_slice in X_SLICES:
            make_response_surface_plot(
                model=model,
                target=target,
                target_label=target_label,
                model_name=model_name,
                target_transform=target_transform,
                H_values=H_values,
                h_values=h_values,
                x_slice=x_slice,
            )

    make_recommendation_summary(selected_rows)

    print()
    print("Final surrogate plotting complete.")
    print(f"Wrote plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
