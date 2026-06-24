from __future__ import annotations

import csv
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLOT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "final_surrogate_plots"
DATA_CSV = PROJECT_ROOT / "results" / "surrogate_database" / "robust_metrics" / "surrogate_database_robust_metrics.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "portfolio_media" / "gifs"
FRAME_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "portfolio_media" / "frames" / "best_design_explorer"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

X_SLICES = [
    ("x0p25", 0.250, "Obstacle front location x = 0.250 m"),
    ("x0p292", 0.292, "Obstacle front location x = 0.292 m"),
    ("x0p35", 0.350, "Obstacle front location x = 0.350 m"),
]

METRICS = [
    {
        "metric": "area_average_p_rgh_rolling_0p05s_peak_mean_Pa",
        "title": "Area-average 0.050 s rolling peak pressure",
        "short_name": "area_average_rolling_peak_0p05s",
        "plot_prefix": "response_surface_area-average-p-rgh-rolling-0p05s-peak-mean-Pa",
        "goal": "min",
        "units": "Pa",
        "nice_goal": "Minimise",
    },
    {
        "metric": "maximum_p_rgh_rolling_0p025s_peak_mean_Pa",
        "title": "Local maximum 0.025 s rolling peak pressure",
        "short_name": "maximum_rolling_peak_0p025s",
        "plot_prefix": "response_surface_maximum-p-rgh-rolling-0p025s-peak-mean-Pa",
        "goal": "min",
        "units": "Pa",
        "nice_goal": "Minimise",
    },
    {
        "metric": "maximum_p_rgh_rolling_0p05s_peak_mean_Pa",
        "title": "Local maximum 0.050 s rolling peak pressure",
        "short_name": "maximum_rolling_peak_0p05s",
        "plot_prefix": "response_surface_maximum-p-rgh-rolling-0p05s-peak-mean-Pa",
        "goal": "min",
        "units": "Pa",
        "nice_goal": "Minimise",
    },
    {
        "metric": "area_average_pressure_impulse_Pa_s",
        "title": "Area-average pressure impulse",
        "short_name": "area_average_impulse",
        "plot_prefix": "response_surface_area-average-pressure-impulse-Pa-s",
        "goal": "min",
        "units": "Pa s",
        "nice_goal": "Minimise",
    },
    {
        "metric": "maximum_pressure_impulse_Pa_s",
        "title": "Maximum-pressure impulse",
        "short_name": "maximum_impulse",
        "plot_prefix": "response_surface_maximum-pressure-impulse-Pa-s",
        "goal": "min",
        "units": "Pa s",
        "nice_goal": "Minimise",
    },
    {
        "metric": "first_distributed_wetting_time_s",
        "title": "First distributed wetting time",
        "short_name": "distributed_wetting_time",
        "plot_prefix": "response_surface_first-distributed-wetting-time-s",
        "goal": "max",
        "units": "s",
        "nice_goal": "Maximise",
    },
]


def get_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"]
    else:
        candidates = ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass

    return ImageFont.load_default()


TITLE_FONT = get_font(30, bold=True)
HEADER_FONT = get_font(24, bold=True)
BODY_FONT = get_font(20, bold=False)
SMALL_FONT = get_font(17, bold=False)
SMALL_BOLD_FONT = get_font(17, bold=True)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def select_best_row(rows: list[dict[str, str]], x_target: float, metric: str, goal: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    candidates = [
        row
        for row in rows
        if row["dataset_split"] == "training_grid"
        and abs(as_float(row, "obstacle_front_x_m") - x_target) < 1e-9
    ]

    if not candidates:
        raise RuntimeError(f"No candidates found for x = {x_target:.3f}")

    if goal == "min":
        best_row = min(candidates, key=lambda row: as_float(row, metric))
    else:
        best_row = max(candidates, key=lambda row: as_float(row, metric))

    return best_row, candidates


def format_value(value: float, units: str) -> str:
    if units == "Pa":
        return f"{value:,.1f} {units}"
    if units == "Pa s":
        return f"{value:,.2f} {units}"
    if units == "s":
        return f"{value:.3f} {units}"
    return f"{value:.3f} {units}"


def wrap_text(text: str, width: int) -> list[str]:
    return wrap(text, width=width)


def draw_multiline(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[str], font, fill="black", gap: int = 6) -> int:
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, current_y), line, font=font)
        current_y = bbox[3] + gap
    return current_y


def resize_preserve_width(img: Image.Image, target_width: int) -> Image.Image:
    w, h = img.size
    if w == target_width:
        return img
    scale = target_width / w
    return img.resize((target_width, int(round(h * scale))), Image.LANCZOS)


def build_frame(metric_info: dict, x_tag: str, x_value: float, x_label: str, best_row: dict[str, str], n_candidates: int) -> Image.Image:
    plot_path = PLOT_DIR / f"{metric_info['plot_prefix']}_{x_tag}.png"
    if not plot_path.exists():
        raise FileNotFoundError(f"Missing plot: {plot_path}")

    plot_img = Image.open(plot_path).convert("RGB")
    plot_img = resize_preserve_width(plot_img, 980)

    left_margin = 24
    right_panel_w = 520
    top_margin = 90
    bottom_margin = 24
    gap = 24

    canvas_w = left_margin + plot_img.size[0] + gap + right_panel_w + left_margin
    canvas_h = max(top_margin + plot_img.size[1] + bottom_margin, 980)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(plot_img, (left_margin, top_margin))

    draw = ImageDraw.Draw(canvas)

    draw.text((24, 18), "Surrogate-guided best-design explorer", font=TITLE_FONT, fill="black")
    draw.text((24, 52), metric_info["title"], font=HEADER_FONT, fill="black")

    panel_x0 = left_margin + plot_img.size[0] + gap
    panel_y0 = top_margin
    panel_x1 = panel_x0 + right_panel_w
    panel_y1 = canvas_h - 30

    draw.rounded_rectangle((panel_x0, panel_y0, panel_x1, panel_y1), radius=18, outline="black", width=2, fill=(248, 248, 248))

    x_text = x_label
    goal_text = f"Design goal: {metric_info['nice_goal']} this objective"
    value = as_float(best_row, metric_info["metric"])
    objective_value = format_value(value, metric_info["units"])

    info_lines = [
        f"Best training-grid design on this x-slice",
        "",
        x_text,
        f"Candidate designs on this slice: {n_candidates}",
        "",
        goal_text,
        "",
        "Recommended design",
        f"Initial water height H = {as_float(best_row, 'water_height_m'):.3f} m",
        f"Obstacle height h = {as_float(best_row, 'obstacle_height_m'):.3f} m",
        f"Obstacle front x = {as_float(best_row, 'obstacle_front_x_m'):.3f} m",
        f"Obstacle width = {as_float(best_row, 'obstacle_width_m'):.3f} m",
        "",
        "Objective value",
        f"{metric_info['title']}: {objective_value}",
        "",
        "Additional response metrics",
        f"Area-average impulse: {as_float(best_row, 'area_average_pressure_impulse_Pa_s'):.2f} Pa s",
        f"Maximum impulse: {as_float(best_row, 'maximum_pressure_impulse_Pa_s'):.2f} Pa s",
        f"Distributed wetting time: {as_float(best_row, 'first_distributed_wetting_time_s'):.3f} s",
        "",
        "Interpretation",
    ]

    paragraph = (
        "Left: surrogate response surface. Right: the best CFD database design on the same x-slice, "
        "selected from the 343-point training grid."
    )

    y = panel_y0 + 18

    for line in info_lines:
        if line == "":
            y += 10
            continue

        font = BODY_FONT
        if line in {"Best training-grid design on this x-slice", "Recommended design", "Objective value", "Additional response metrics", "Interpretation"}:
            font = SMALL_BOLD_FONT

        draw.text((panel_x0 + 18, y), line, font=font, fill="black")
        bbox = draw.textbbox((panel_x0 + 18, y), line, font=font)
        y = bbox[3] + 8

    wrapped = wrap_text(paragraph, width=42)
    draw_multiline(draw, panel_x0 + 18, y, wrapped, SMALL_FONT, fill="black", gap=4)

    return canvas


def save_gif(frames: list[Image.Image], output_path: Path, end_pause_ms: int = 2200) -> None:
    durations = [1300] * len(frames)
    if durations:
        durations[-1] = end_pause_ms

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        format="GIF",
    )


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing data CSV: {DATA_CSV}")

    rows = load_rows(DATA_CSV)

    master_frames: list[Image.Image] = []
    summary_lines = [
        "Best-design explorer summary",
        "============================",
        "",
        "This media set combines:",
        "- surrogate response surfaces",
        "- best CFD database designs on each x-slice",
        "",
    ]

    print("Building best-design explorer GIFs...")

    for metric_info in METRICS:
        metric_frames: list[Image.Image] = []

        summary_lines.append(metric_info["title"])
        summary_lines.append("-" * len(metric_info["title"]))

        for x_tag, x_value, x_label in X_SLICES:
            best_row, candidates = select_best_row(rows, x_value, metric_info["metric"], metric_info["goal"])

            frame = build_frame(metric_info, x_tag, x_value, x_label, best_row, len(candidates))
            metric_frames.append(frame)
            master_frames.append(frame.copy())

            frame_path = FRAME_DIR / f"{metric_info['short_name']}_{x_tag}.png"
            frame.save(frame_path)

            objective_value = format_value(as_float(best_row, metric_info["metric"]), metric_info["units"])
            summary_lines.append(
                f"x = {x_value:.3f} m -> H = {as_float(best_row, 'water_height_m'):.3f} m, "
                f"h = {as_float(best_row, 'obstacle_height_m'):.3f} m, value = {objective_value}"
            )

        summary_lines.append("")

        metric_gif_path = OUTPUT_DIR / f"best_design_{metric_info['short_name']}.gif"
        save_gif(metric_frames, metric_gif_path)

        print(f"  wrote {metric_gif_path}")

    master_gif_path = OUTPUT_DIR / "surrogate_best_design_explorer.gif"
    save_gif(master_frames, master_gif_path, end_pause_ms=2600)

    print(f"  wrote {master_gif_path}")

    summary_path = OUTPUT_DIR / "best_design_explorer_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print("Done.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
