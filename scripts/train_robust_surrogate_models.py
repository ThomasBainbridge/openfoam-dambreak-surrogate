from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "surrogate_database"
    / "robust_metrics"
    / "surrogate_database_robust_metrics.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "robust_surrogate_models"

INPUT_COLUMNS = [
    "water_height_m",
    "obstacle_height_m",
    "obstacle_front_x_m",
]

TARGETS = [
    ("area_average_p_rgh_top_0p01_mean_Pa", "Area-average top 1% pressure [Pa]"),
    ("area_average_p_rgh_top_0p05_mean_Pa", "Area-average top 5% pressure [Pa]"),
    ("area_average_p_rgh_rolling_0p025s_peak_mean_Pa", "Area-average 0.025 s rolling peak pressure [Pa]"),
    ("area_average_p_rgh_rolling_0p05s_peak_mean_Pa", "Area-average 0.050 s rolling peak pressure [Pa]"),

    ("maximum_p_rgh_top_0p01_mean_Pa", "Local maximum top 1% pressure [Pa]"),
    ("maximum_p_rgh_top_0p05_mean_Pa", "Local maximum top 5% pressure [Pa]"),
    ("maximum_p_rgh_rolling_0p025s_peak_mean_Pa", "Local maximum 0.025 s rolling peak pressure [Pa]"),
    ("maximum_p_rgh_rolling_0p05s_peak_mean_Pa", "Local maximum 0.050 s rolling peak pressure [Pa]"),

    ("area_average_pressure_impulse_Pa_s", "Area-average pressure impulse [Pa s]"),
    ("maximum_pressure_impulse_Pa_s", "Maximum-pressure impulse [Pa s]"),
    ("first_distributed_wetting_time_s", "First distributed wetting time [s]"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def normalised_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value_range = float(np.max(y_true) - np.min(y_true))

    if value_range <= 0.0:
        return float("nan")

    return rmse(y_true, y_pred) / value_range


def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def make_models() -> dict[str, object]:
    return {
        "poly2_ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=1.0e-6)),
            ]
        ),
        "poly3_ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("model", Ridge(alpha=1.0e-5)),
            ]
        ),
        "rbf_svr": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=40.0, gamma="scale", epsilon=0.02)),
            ]
        ),
        "kernel_ridge_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KernelRidge(alpha=1.0e-3, kernel="rbf", gamma=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=700,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=900,
            max_features=1.0,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
        "knn_distance": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=8, weights="distance")),
            ]
        ),
        "gaussian_process_matern": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=(
                            ConstantKernel(1.0, (1.0e-2, 1.0e3))
                            * Matern(length_scale=[1.0, 1.0, 1.0], nu=2.5)
                            + WhiteKernel(noise_level=1.0e-5)
                        ),
                        alpha=1.0e-8,
                        normalize_y=True,
                        random_state=42,
                        n_restarts_optimizer=3,
                    ),
                ),
            ]
        ),
        "gaussian_process_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=(
                            ConstantKernel(1.0, (1.0e-2, 1.0e3))
                            * RBF(length_scale=[1.0, 1.0, 1.0])
                            + WhiteKernel(noise_level=1.0e-5)
                        ),
                        alpha=1.0e-8,
                        normalize_y=True,
                        random_state=42,
                        n_restarts_optimizer=3,
                    ),
                ),
            ]
        ),
    }


def fit_predict_model(
    model_template: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    use_log_target: bool,
) -> tuple[object, np.ndarray, np.ndarray]:
    model = clone(model_template)

    if use_log_target:
        y_fit = np.log1p(y_train)
        model.fit(X_train, y_fit)

        y_train_pred = np.expm1(model.predict(X_train))
        y_valid_pred = np.expm1(model.predict(X_valid))
    else:
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)

    return model, y_train_pred, y_valid_pred


def metric_row(
    target_name: str,
    target_label: str,
    model_name: str,
    target_transform: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_valid: np.ndarray,
    y_valid_pred: np.ndarray,
) -> dict[str, object]:
    return {
        "target": target_name,
        "target_label": target_label,
        "model_name": model_name,
        "target_transform": target_transform,

        "train_rmse": rmse(y_train, y_train_pred),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "train_normalised_rmse": normalised_rmse(y_train, y_train_pred),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "train_max_abs_error": max_abs_error(y_train, y_train_pred),

        "validation_rmse": rmse(y_valid, y_valid_pred),
        "validation_mae": float(mean_absolute_error(y_valid, y_valid_pred)),
        "validation_normalised_rmse": normalised_rmse(y_valid, y_valid_pred),
        "validation_r2": float(r2_score(y_valid, y_valid_pred)),
        "validation_max_abs_error": max_abs_error(y_valid, y_valid_pred),
    }


def safe_filename(value: str) -> str:
    return value.replace("_", "-").replace(" ", "-").replace("/", "-")


def save_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_label: str,
    model_name: str,
    filename: str,
) -> None:
    lower = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper = max(float(np.max(y_true)), float(np.max(y_pred)))
    margin = 0.06 * (upper - lower) if upper > lower else 1.0

    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    ax.scatter(y_true, y_pred, s=60, edgecolors="black", linewidths=0.7)
    ax.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        linewidth=1.5,
    )

    ax.set_title(f"Validation parity: {target_label}\n{model_name}")
    ax.set_xlabel("CFD value")
    ax.set_ylabel("Surrogate prediction")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def save_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_label: str,
    model_name: str,
    filename: str,
) -> None:
    residual = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    ax.scatter(y_true, residual, s=60, edgecolors="black", linewidths=0.7)
    ax.axhline(0.0, linewidth=1.5)

    ax.set_title(f"Validation residuals: {target_label}\n{model_name}")
    ax.set_xlabel("CFD value")
    ax.set_ylabel("CFD - surrogate")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def save_prediction_csv(
    rows_valid: list[dict[str, str]],
    target_name: str,
    target_label: str,
    model_name: str,
    y_valid: np.ndarray,
    y_valid_pred: np.ndarray,
) -> None:
    output_path = OUTPUT_DIR / f"validation_predictions_{safe_filename(target_name)}_{safe_filename(model_name)}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case_name",
            "water_height_m",
            "obstacle_height_m",
            "obstacle_front_x_m",
            "target",
            "target_label",
            "model_name",
            "cfd_value",
            "surrogate_prediction",
            "error_cfd_minus_prediction",
            "absolute_error",
            "relative_error_percent",
        ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row, cfd_value, prediction in zip(rows_valid, y_valid, y_valid_pred):
            error = float(cfd_value - prediction)
            relative_error = 100.0 * error / float(cfd_value) if cfd_value != 0.0 else float("nan")

            writer.writerow(
                {
                    "case_name": row["case_name"],
                    "water_height_m": row["water_height_m"],
                    "obstacle_height_m": row["obstacle_height_m"],
                    "obstacle_front_x_m": row["obstacle_front_x_m"],
                    "target": target_name,
                    "target_label": target_label,
                    "model_name": model_name,
                    "cfd_value": float(cfd_value),
                    "surrogate_prediction": float(prediction),
                    "error_cfd_minus_prediction": error,
                    "absolute_error": abs(error),
                    "relative_error_percent": relative_error,
                }
            )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows(INPUT_CSV)

    train_rows = [row for row in rows if row["dataset_split"] == "training_grid"]
    valid_rows = [row for row in rows if row["dataset_split"] == "validation_lhs"]

    if len(train_rows) == 0 or len(valid_rows) == 0:
        raise ValueError("Expected both training_grid and validation_lhs rows.")

    X_train = np.asarray(
        [[float(row[column]) for column in INPUT_COLUMNS] for row in train_rows],
        dtype=float,
    )

    X_valid = np.asarray(
        [[float(row[column]) for column in INPUT_COLUMNS] for row in valid_rows],
        dtype=float,
    )

    models = make_models()

    all_model_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for target_name, target_label in TARGETS:
        print()
        print("=" * 100)
        print(f"Training robust surrogate models for: {target_label}")
        print("=" * 100)

        y_train = np.asarray([float(row[target_name]) for row in train_rows], dtype=float)
        y_valid = np.asarray([float(row[target_name]) for row in valid_rows], dtype=float)

        target_results: list[tuple[dict[str, object], np.ndarray]] = []

        for model_name, model_template in models.items():
            for use_log_target in [False, True]:
                transform_name = "log1p" if use_log_target else "raw"
                full_model_name = f"{model_name}_{transform_name}"

                print(f"  Fitting {full_model_name}")

                try:
                    _, y_train_pred, y_valid_pred = fit_predict_model(
                        model_template=model_template,
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        use_log_target=use_log_target,
                    )

                    row = metric_row(
                        target_name=target_name,
                        target_label=target_label,
                        model_name=model_name,
                        target_transform=transform_name,
                        y_train=y_train,
                        y_train_pred=y_train_pred,
                        y_valid=y_valid,
                        y_valid_pred=y_valid_pred,
                    )

                    all_model_rows.append(row)
                    target_results.append((row, y_valid_pred))

                except Exception as exc:
                    print(f"    FAILED: {exc}")

        selected_row, selected_prediction = min(
            target_results,
            key=lambda item: float(item[0]["validation_normalised_rmse"]),
        )

        selected_rows.append(selected_row)

        selected_name = f"{selected_row['model_name']}_{selected_row['target_transform']}"

        print()
        print(f"  SELECTED: {selected_name}")
        print(f"  validation RMSE:            {float(selected_row['validation_rmse']):.6g}")
        print(f"  validation normalised RMSE: {100.0 * float(selected_row['validation_normalised_rmse']):.3f}%")
        print(f"  validation R2:              {float(selected_row['validation_r2']):.6g}")

        target_safe = safe_filename(target_name)

        save_parity_plot(
            y_true=y_valid,
            y_pred=selected_prediction,
            target_label=target_label,
            model_name=selected_name,
            filename=f"selected_validation_parity_{target_safe}.png",
        )

        save_residual_plot(
            y_true=y_valid,
            y_pred=selected_prediction,
            target_label=target_label,
            model_name=selected_name,
            filename=f"selected_validation_residuals_{target_safe}.png",
        )

        save_prediction_csv(
            rows_valid=valid_rows,
            target_name=target_name,
            target_label=target_label,
            model_name=selected_name,
            y_valid=y_valid,
            y_valid_pred=selected_prediction,
        )

    write_csv(OUTPUT_DIR / "robust_surrogate_model_comparison.csv", all_model_rows)
    write_csv(OUTPUT_DIR / "selected_robust_surrogate_models.csv", selected_rows)

    summary_path = OUTPUT_DIR / "selected_robust_surrogate_summary.txt"

    lines = [
        "Selected robust surrogate models",
        "================================",
        "",
        f"Training cases: {len(train_rows)}",
        f"Validation cases: {len(valid_rows)}",
        "",
    ]

    for row in selected_rows:
        lines.extend(
            [
                str(row["target_label"]),
                "-" * len(str(row["target_label"])),
                f"Model: {row['model_name']} with {row['target_transform']} target",
                f"Validation RMSE: {float(row['validation_rmse']):.6g}",
                f"Validation MAE: {float(row['validation_mae']):.6g}",
                f"Validation normalised RMSE: {100.0 * float(row['validation_normalised_rmse']):.3f}%",
                f"Validation R2: {float(row['validation_r2']):.6g}",
                f"Validation max absolute error: {float(row['validation_max_abs_error']):.6g}",
                "",
            ]
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 100)
    print("Robust surrogate model training complete.")
    print("=" * 100)
    print(f"Wrote: {OUTPUT_DIR / 'robust_surrogate_model_comparison.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'selected_robust_surrogate_models.csv'}")
    print(f"Wrote: {summary_path}")
    print()
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
