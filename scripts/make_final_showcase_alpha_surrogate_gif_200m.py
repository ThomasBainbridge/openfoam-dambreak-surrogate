from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIELD_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "field_surrogate"
MODEL_DIR = FIELD_DIR / "pod_timeslice_models_200m"

INFO_JSON = FIELD_DIR / "alpha_field_dataset_info.json"
SNAPSHOT_METADATA_CSV = FIELD_DIR / "alpha_snapshot_metadata.csv"
SNAPSHOT_MEMMAP = FIELD_DIR / "alpha_snapshots_float32.dat"

PCA_MODEL = MODEL_DIR / "alpha_timeslice_incremental_pca.joblib"
TIMESLICE_MODEL = MODEL_DIR / "alpha_timeslice_pod_surrogate.joblib"

OUTPUT_GIF_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "portfolio_media" / "gifs"
OUTPUT_FRAME_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "portfolio_media" / "frames" / "final_showcase_alpha_surrogate_200m"

OUTPUT_GIF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FRAME_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CASE = "surrVal_006_H0342_obsH0062_x0366"

N_FRAMES = 44
ERROR_VMAX = 0.50


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def nearest_time_key(value: float, available_keys: list[str]) -> str:
    return min(available_keys, key=lambda key: abs(float(key) - value))


def geometry_input(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [[
            float(row["water_height_m"]),
            float(row["obstacle_height_m"]),
            float(row["obstacle_front_x_m"]),
        ]],
        dtype=np.float64,
    )


def select_frame_rows(case_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(case_rows, key=lambda row: float(row["time_s"]))

    if len(ordered) <= N_FRAMES:
        return ordered

    indices = np.linspace(0, len(ordered) - 1, N_FRAMES).round().astype(int)
    return [ordered[i] for i in sorted(set(int(i) for i in indices))]


def add_obstacle(ax, row: dict[str, str]) -> None:
    x_obs = float(row["obstacle_front_x_m"])
    h_obs = float(row["obstacle_height_m"])
    w_obs = float(row["obstacle_width_m"])

    ax.add_patch(
        Rectangle(
            (x_obs, 0.0),
            w_obs,
            h_obs,
            facecolor="0.55",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )
    )


def clean_axis(ax, row: dict[str, str], domain_length: float, domain_height: float, title: str) -> None:
    ax.set_xlim(0.0, domain_length)
    ax.set_ylim(0.0, domain_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    add_obstacle(ax, row)


def render_frame(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    row: dict[str, str],
    domain_length: float,
    domain_height: float,
    frame_path: Path,
) -> None:
    t = float(row["time_s"])

    error_field = np.abs(true_field - pred_field)

    x = np.linspace(0.0, domain_length, true_field.shape[1])
    y = np.linspace(0.0, domain_height, true_field.shape[0])

    fig = plt.figure(figsize=(16, 9), facecolor="white")

    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.045],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.035,
        right=0.965,
        top=0.82,
        bottom=0.10,
        wspace=0.08,
        hspace=0.11,
    )

    ax_cfd = fig.add_subplot(grid[0, 0])
    ax_surrogate = fig.add_subplot(grid[0, 1])
    ax_error = fig.add_subplot(grid[0, 2])

    cax_alpha = fig.add_subplot(grid[1, 0:2])
    cax_error = fig.add_subplot(grid[1, 2])

    image_cfd = ax_cfd.imshow(
        true_field,
        origin="lower",
        extent=[0.0, domain_length, 0.0, domain_height],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    image_surrogate = ax_surrogate.imshow(
        pred_field,
        origin="lower",
        extent=[0.0, domain_length, 0.0, domain_height],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    image_error = ax_error.imshow(
        error_field,
        origin="lower",
        extent=[0.0, domain_length, 0.0, domain_height],
        cmap="magma",
        vmin=0.0,
        vmax=ERROR_VMAX,
        interpolation="nearest",
    )

    for ax, field in [(ax_cfd, true_field), (ax_surrogate, pred_field)]:
        try:
            ax.contour(
                x,
                y,
                field,
                levels=[0.5],
                colors="black",
                linewidths=1.0,
            )
        except Exception:
            pass

    clean_axis(ax_cfd, row, domain_length, domain_height, "Actual CFD")
    clean_axis(ax_surrogate, row, domain_length, domain_height, "POD surrogate")
    clean_axis(ax_error, row, domain_length, domain_height, "Absolute error")

    ax_cfd.text(
        0.025,
        0.955,
        f"t = {t:.3f} s",
        transform=ax_cfd.transAxes,
        fontsize=14,
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.0,
            "pad": 5,
        },
    )

    cbar_alpha = fig.colorbar(
        image_surrogate,
        cax=cax_alpha,
        orientation="horizontal",
    )
    cbar_alpha.set_label(r"$\alpha_{water}$", fontsize=13)
    cbar_alpha.ax.tick_params(labelsize=10)

    cbar_error = fig.colorbar(
        image_error,
        cax=cax_error,
        orientation="horizontal",
    )
    cbar_error.set_label(r"$|\alpha_{CFD}-\alpha_{surrogate}|$", fontsize=13)
    cbar_error.ax.tick_params(labelsize=10)

    fig.text(
        0.5,
        0.945,
        "CFD vs POD surrogate free-surface reconstruction",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.895,
        "Unseen validation case  |  200-mode time-sliced POD",
        ha="center",
        va="center",
        fontsize=15,
        color="0.30",
    )

    fig.savefig(frame_path, dpi=110)
    plt.close(fig)


def main() -> None:
    required = [
        INFO_JSON,
        SNAPSHOT_METADATA_CSV,
        SNAPSHOT_MEMMAP,
        PCA_MODEL,
        TIMESLICE_MODEL,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    info = json.loads(INFO_JSON.read_text())

    n_snapshots = int(info["n_snapshots"])
    ny = int(info["ny"])
    nx = int(info["nx"])
    domain_length = float(info["domain_length_m"])
    domain_height = float(info["domain_height_m"])

    snapshots = np.memmap(
        SNAPSHOT_MEMMAP,
        dtype="float32",
        mode="r",
        shape=(n_snapshots, ny, nx),
    )

    rows = read_csv(SNAPSHOT_METADATA_CSV)
    case_rows = [row for row in rows if row["case_name"] == SELECTED_CASE]

    if not case_rows:
        raise RuntimeError(f"Could not find selected case: {SELECTED_CASE}")

    frame_rows = select_frame_rows(case_rows)

    ipca = joblib.load(PCA_MODEL)
    package = joblib.load(TIMESLICE_MODEL)

    models_by_time = package["models_by_time"]
    available_time_keys = package["available_time_keys"]

    frame_dir = OUTPUT_FRAME_DIR / SELECTED_CASE
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []

    print(f"Selected case: {SELECTED_CASE}")
    print(f"Frames: {len(frame_rows)}")

    for frame_number, row in enumerate(frame_rows, start=1):
        snapshot_index = int(row["snapshot_index"])

        true_field = np.asarray(snapshots[snapshot_index, :, :], dtype=np.float64)

        key = nearest_time_key(float(row["time_s"]), available_time_keys)
        model = models_by_time[key]

        coeffs = model.predict(geometry_input(row))
        pred_flat = ipca.inverse_transform(coeffs)[0]
        pred_field = np.clip(pred_flat.reshape(ny, nx), 0.0, 1.0)

        frame_path = frame_dir / f"frame_{frame_number:03d}.png"

        render_frame(
            true_field=true_field,
            pred_field=pred_field,
            row=row,
            domain_length=domain_length,
            domain_height=domain_height,
            frame_path=frame_path,
        )

        frame_paths.append(frame_path)

        if frame_number % 5 == 0 or frame_number == len(frame_rows):
            print(f"  rendered {frame_number}/{len(frame_rows)}")

    images = [Image.open(path).convert("RGB") for path in frame_paths]

    gif_path = OUTPUT_GIF_DIR / "FINAL_cfd_vs_pod_surrogate_alpha_field_200m.gif"

    durations = [110] * len(images)
    durations[-1] = 1500

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        format="GIF",
    )

    for image in images:
        image.close()

    print()
    print(f"Wrote final GIF: {gif_path}")
    print(f"Wrote frames: {frame_dir}")


if __name__ == "__main__":
    main()
