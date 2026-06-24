from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "final_surrogate_plots"
OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate_database" / "portfolio_media" / "gifs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_SLICES = [
    ("x0p25", "Obstacle front location x = 0.250 m"),
    ("x0p292", "Obstacle front location x = 0.292 m"),
    ("x0p35", "Obstacle front location x = 0.350 m"),
]

METRICS = [
    (
        "area-average-p-rgh-rolling-0p05s-peak-mean-Pa",
        "Area-average 0.050 s rolling peak pressure",
        "response_surface_area-average-p-rgh-rolling-0p05s-peak-mean-Pa",
    ),
    (
        "area-average-pressure-impulse-Pa-s",
        "Area-average pressure impulse",
        "response_surface_area-average-pressure-impulse-Pa-s",
    ),
    (
        "first-distributed-wetting-time-s",
        "First distributed wetting time",
        "response_surface_first-distributed-wetting-time-s",
    ),
    (
        "maximum-p-rgh-rolling-0p025s-peak-mean-Pa",
        "Local maximum 0.025 s rolling peak pressure",
        "response_surface_maximum-p-rgh-rolling-0p025s-peak-mean-Pa",
    ),
    (
        "maximum-p-rgh-rolling-0p05s-peak-mean-Pa",
        "Local maximum 0.050 s rolling peak pressure",
        "response_surface_maximum-p-rgh-rolling-0p05s-peak-mean-Pa",
    ),
    (
        "maximum-pressure-impulse-Pa-s",
        "Maximum-pressure impulse",
        "response_surface_maximum-pressure-impulse-Pa-s",
    ),
]


def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


TITLE_FONT = get_font(28)
SUBTITLE_FONT = get_font(22)
SMALL_FONT = get_font(18)


def add_banner(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size

    top_pad = 95
    bottom_pad = 4

    canvas = Image.new("RGB", (width, height + top_pad + bottom_pad), "white")
    canvas.paste(image, (0, top_pad))

    draw = ImageDraw.Draw(canvas)

    draw.text((20, 15), title, font=TITLE_FONT, fill="black")
    draw.text((20, 52), subtitle, font=SUBTITLE_FONT, fill="black")

    return canvas


def resize_to_width(image: Image.Image, target_width: int) -> Image.Image:
    width, height = image.size
    if width == target_width:
        return image
    scale = target_width / width
    target_height = int(round(height * scale))
    return image.resize((target_width, target_height), Image.LANCZOS)


def build_metric_frames(metric_title: str, file_prefix: str) -> list[Image.Image]:
    frames: list[Image.Image] = []

    for x_tag, x_label in X_SLICES:
        path = INPUT_DIR / f"{file_prefix}_{x_tag}.png"

        if not path.exists():
            raise FileNotFoundError(f"Missing input plot: {path}")

        img = Image.open(path)
        img = resize_to_width(img, 1000)
        img = add_banner(img, metric_title, x_label)
        frames.append(img)

    return frames


def save_gif(frames: list[Image.Image], output_path: Path, final_pause_ms: int = 1800) -> None:
    durations = [1200] * len(frames)
    durations[-1] = final_pause_ms

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        format="GIF",
    )


def build_dashboard_frame(frame_index: int, metric_frames: dict[str, list[Image.Image]]) -> Image.Image:
    ordered_frames = []
    for metric_key, _, _ in METRICS:
        frame = metric_frames[metric_key][frame_index]
        ordered_frames.append(frame)

    tile_width = 560

    processed = []
    for frame in ordered_frames:
        inner = frame.convert("RGB")
        inner = resize_to_width(inner, tile_width)
        processed.append(inner)

    single_width = processed[0].size[0]
    single_height = processed[0].size[1]

    cols = 2
    rows = 3
    margin = 20
    header_h = 95

    canvas_w = cols * single_width + (cols + 1) * margin
    canvas_h = header_h + rows * single_height + (rows + 1) * margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    _, x_label = X_SLICES[frame_index]
    draw.text((20, 18), "Surrogate response-surface dashboard", font=TITLE_FONT, fill="black")
    draw.text((20, 54), x_label, font=SUBTITLE_FONT, fill="black")

    for idx, img in enumerate(processed):
        row = idx // cols
        col = idx % cols

        x0 = margin + col * (single_width + margin)
        y0 = header_h + margin + row * (single_height + margin)

        canvas.paste(img, (x0, y0))

    return canvas


def main() -> None:
    metric_frames: dict[str, list[Image.Image]] = {}

    print("Building individual response-surface GIFs...")

    for metric_key, metric_title, file_prefix in METRICS:
        frames = build_metric_frames(metric_title, file_prefix)
        metric_frames[metric_key] = frames

        output_path = OUTPUT_DIR / f"{metric_key}.gif"
        save_gif(frames, output_path)

        print(f"  wrote {output_path}")

    print()
    print("Building dashboard GIF...")

    dashboard_frames = [
        build_dashboard_frame(i, metric_frames)
        for i in range(len(X_SLICES))
    ]

    dashboard_path = OUTPUT_DIR / "surrogate_response_surface_dashboard.gif"
    save_gif(dashboard_frames, dashboard_path, final_pause_ms=2200)

    print(f"  wrote {dashboard_path}")

    summary_path = OUTPUT_DIR / "gif_generation_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Generated surrogate response-surface GIFs",
                "========================================",
                "",
                "Individual GIFs:",
                *[
                    f"- {metric_key}.gif"
                    for metric_key, _, _ in METRICS
                ],
                "",
                "Dashboard GIF:",
                "- surrogate_response_surface_dashboard.gif",
            ]
        ),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
