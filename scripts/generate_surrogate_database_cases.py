from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DESIGN_CSV = (
    PROJECT_ROOT
    / "results"
    / "surrogate_database"
    / "design"
    / "surrogate_all_403_cases.csv"
)

PARAMETRIC_ROOT = PROJECT_ROOT / "parametric_study"

# Use the already-corrected high-time full-duration setup as the template.
BASE_SETUP = PARAMETRIC_ROOT / "matrix_high_time_1p5" / "highTime1p5_H0292_obsH0048_x0292"

OUTPUT_ROOT = PARAMETRIC_ROOT / "surrogate_database_cases"

DOMAIN_LENGTH = 0.584
DOMAIN_HEIGHT = 0.584
DOMAIN_DEPTH = 0.0146

DX_LEFT_TARGET = 0.00635
DX_OBS_TARGET = 0.00300
DX_RIGHT_TARGET = 0.00705
DY_LOWER_TARGET = 0.00300
DY_UPPER_TARGET = 0.00638

WATER_WIDTH = 0.1461


@dataclass(frozen=True)
class SurrogateCase:
    name: str
    dataset_split: str
    water_height: float
    obstacle_height: float
    obstacle_front_x: float
    obstacle_width: float
    end_time: float
    field_write_interval: float
    metric_write_interval: float


def read_design_csv(path: Path) -> list[SurrogateCase]:
    if not path.exists():
        raise FileNotFoundError(f"Design CSV not found: {path}")

    cases: list[SurrogateCase] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required_columns = {
            "case_name",
            "dataset_split",
            "water_height_m",
            "obstacle_height_m",
            "obstacle_front_x_m",
            "obstacle_width_m",
            "end_time_s",
            "field_write_interval_s",
            "metric_write_interval_s",
        }

        missing_columns = required_columns.difference(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(f"Design CSV is missing columns: {sorted(missing_columns)}")

        for row in reader:
            cases.append(
                SurrogateCase(
                    name=row["case_name"],
                    dataset_split=row["dataset_split"],
                    water_height=float(row["water_height_m"]),
                    obstacle_height=float(row["obstacle_height_m"]),
                    obstacle_front_x=float(row["obstacle_front_x_m"]),
                    obstacle_width=float(row["obstacle_width_m"]),
                    end_time=float(row["end_time_s"]),
                    field_write_interval=float(row["field_write_interval_s"]),
                    metric_write_interval=float(row["metric_write_interval_s"]),
                )
            )

    if not cases:
        raise ValueError(f"No cases found in design CSV: {path}")

    names = [case.name for case in cases]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate case names found in surrogate design CSV.")

    return cases


def cell_count(length: float, target_size: float, minimum: int) -> int:
    return max(minimum, int(round(length / target_size)))


def copy_base_setup(case_dir: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)

    case_dir.mkdir(parents=True)

    for item_name in ["0.orig", "constant", "system"]:
        source = BASE_SETUP / item_name
        destination = case_dir / item_name

        if not source.exists():
            raise FileNotFoundError(f"Required base setup item missing: {source}")

        shutil.copytree(source, destination)

    poly_mesh = case_dir / "constant" / "polyMesh"

    if poly_mesh.exists():
        shutil.rmtree(poly_mesh)

    dynamic_mesh = case_dir / "constant" / "dynamicMeshDict"

    if dynamic_mesh.exists():
        dynamic_mesh.unlink()

    zero_dir = case_dir / "0"

    if zero_dir.exists():
        shutil.rmtree(zero_dir)

    shutil.copytree(case_dir / "0.orig", zero_dir)


def write_block_mesh_dict(case_dir: Path, case: SurrogateCase) -> None:
    x0 = 0.0
    x1 = case.obstacle_front_x
    x2 = case.obstacle_front_x + case.obstacle_width
    x3 = DOMAIN_LENGTH

    y0 = 0.0
    y1 = case.obstacle_height
    y2 = DOMAIN_HEIGHT

    z0 = 0.0
    z1 = DOMAIN_DEPTH

    if not (0.05 < x1 < x2 < DOMAIN_LENGTH - 0.05):
        raise ValueError(
            f"Invalid obstacle x-location for {case.name}: "
            f"x1={x1}, x2={x2}, domain length={DOMAIN_LENGTH}"
        )

    if not (0.01 < y1 < DOMAIN_HEIGHT - 0.05):
        raise ValueError(
            f"Invalid obstacle height for {case.name}: "
            f"h={y1}, domain height={DOMAIN_HEIGHT}"
        )

    nx_left = cell_count(x1 - x0, DX_LEFT_TARGET, 8)
    nx_obs = cell_count(x2 - x1, DX_OBS_TARGET, 4)
    nx_right = cell_count(x3 - x2, DX_RIGHT_TARGET, 8)

    ny_lower = cell_count(y1 - y0, DY_LOWER_TARGET, 4)
    ny_upper = cell_count(y2 - y1, DY_UPPER_TARGET, 12)

    block_mesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;

vertices
(
    ({x0:.9f} {y0:.9f} {z0:.9f})
    ({x1:.9f} {y0:.9f} {z0:.9f})
    ({x2:.9f} {y0:.9f} {z0:.9f})
    ({x3:.9f} {y0:.9f} {z0:.9f})

    ({x0:.9f} {y1:.9f} {z0:.9f})
    ({x1:.9f} {y1:.9f} {z0:.9f})
    ({x2:.9f} {y1:.9f} {z0:.9f})
    ({x3:.9f} {y1:.9f} {z0:.9f})

    ({x0:.9f} {y2:.9f} {z0:.9f})
    ({x1:.9f} {y2:.9f} {z0:.9f})
    ({x2:.9f} {y2:.9f} {z0:.9f})
    ({x3:.9f} {y2:.9f} {z0:.9f})

    ({x0:.9f} {y0:.9f} {z1:.9f})
    ({x1:.9f} {y0:.9f} {z1:.9f})
    ({x2:.9f} {y0:.9f} {z1:.9f})
    ({x3:.9f} {y0:.9f} {z1:.9f})

    ({x0:.9f} {y1:.9f} {z1:.9f})
    ({x1:.9f} {y1:.9f} {z1:.9f})
    ({x2:.9f} {y1:.9f} {z1:.9f})
    ({x3:.9f} {y1:.9f} {z1:.9f})

    ({x0:.9f} {y2:.9f} {z1:.9f})
    ({x1:.9f} {y2:.9f} {z1:.9f})
    ({x2:.9f} {y2:.9f} {z1:.9f})
    ({x3:.9f} {y2:.9f} {z1:.9f})
);

blocks
(
    hex (0 1 5 4 12 13 17 16) ({nx_left} {ny_lower} 1) simpleGrading (1 1 1)
    hex (2 3 7 6 14 15 19 18) ({nx_right} {ny_lower} 1) simpleGrading (1 1 1)
    hex (4 5 9 8 16 17 21 20) ({nx_left} {ny_upper} 1) simpleGrading (1 1 1)
    hex (5 6 10 9 17 18 22 21) ({nx_obs} {ny_upper} 1) simpleGrading (1 1 1)
    hex (6 7 11 10 18 19 23 22) ({nx_right} {ny_upper} 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    leftWall
    {{
        type wall;
        faces
        (
            (0 12 16 4)
            (4 16 20 8)
        );
    }}

    rightWall
    {{
        type wall;
        faces
        (
            (7 19 15 3)
            (11 23 19 7)
        );
    }}

    lowerWall
    {{
        type wall;
        faces
        (
            (0 1 13 12)
            (2 3 15 14)
        );
    }}

    obstacle
    {{
        type wall;
        faces
        (
            (1 5 17 13)
            (5 6 18 17)
            (2 14 18 6)
        );
    }}

    atmosphere
    {{
        type patch;
        faces
        (
            (8 20 21 9)
            (9 21 22 10)
            (10 22 23 11)
        );
    }}
);

defaultPatch
{{
    name defaultFaces;
    type empty;
}}

mergePatchPairs
(
);

// ************************************************************************* //
"""

    (case_dir / "system" / "blockMeshDict").write_text(block_mesh)


def write_set_fields_dict(case_dir: Path, case: SurrogateCase) -> None:
    set_fields = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}}

defaultFieldValues
(
    volScalarFieldValue alpha.water 0
);

regions
(
    boxToCell
    {{
        box (0 0 -1) ({WATER_WIDTH:.9f} {case.water_height:.9f} 1);

        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}
);

// ************************************************************************* //
"""

    (case_dir / "system" / "setFieldsDict").write_text(set_fields)


def update_control_dict(case_dir: Path, case: SurrogateCase) -> None:
    control_dict = case_dir / "system" / "controlDict"

    if not control_dict.exists():
        raise FileNotFoundError(f"Missing controlDict: {control_dict}")

    text = control_dict.read_text()

    # Enforce full-duration high-time setup.
    text = replace_dictionary_entry(text, "endTime", f"{case.end_time:g}")
    text = replace_dictionary_entry(text, "writeInterval", f"{case.field_write_interval:g}", first_only=True)

    # The high-time template should already contain these function objects.
    if "obstaclePressureAverage" not in text or "obstaclePressureMaximum" not in text:
        raise ValueError(
            f"Expected obstaclePressureAverage and obstaclePressureMaximum "
            f"function objects in {control_dict}"
        )

    text = enforce_function_object_interval(
        text,
        object_name="obstaclePressureAverage",
        interval=case.metric_write_interval,
    )

    text = enforce_function_object_interval(
        text,
        object_name="obstaclePressureMaximum",
        interval=case.metric_write_interval,
    )

    control_dict.write_text(text)


def replace_dictionary_entry(
    text: str,
    keyword: str,
    value: str,
    first_only: bool = False,
) -> str:
    lines = text.splitlines()
    replaced = False

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith(keyword) and stripped.endswith(";"):
            indentation = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indentation}{keyword:<16}{value};"
            replaced = True

            if first_only:
                break

    if not replaced:
        raise ValueError(f"Could not find dictionary entry: {keyword}")

    return "\n".join(lines) + "\n"


def enforce_function_object_interval(text: str, object_name: str, interval: float) -> str:
    lines = text.splitlines()

    inside_object = False
    brace_depth = 0
    object_seen = False
    write_control_seen = False
    write_interval_seen = False

    output_lines: list[str] = []

    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped == object_name:
            inside_object = True
            object_seen = True
            brace_depth = 0
            write_control_seen = False
            write_interval_seen = False
            output_lines.append(line)
            index += 1
            continue

        if inside_object:
            brace_depth += line.count("{")
            brace_depth -= line.count("}")

            if stripped.startswith("writeControl"):
                indentation = line[: len(line) - len(line.lstrip())]
                output_lines.append(f"{indentation}writeControl    adjustableRunTime;")
                write_control_seen = True
                index += 1
                continue

            if stripped.startswith("writeInterval"):
                indentation = line[: len(line) - len(line.lstrip())]
                output_lines.append(f"{indentation}writeInterval   {interval:g};")
                write_interval_seen = True
                index += 1
                continue

            if stripped.startswith("writeFields") and not write_interval_seen:
                indentation = line[: len(line) - len(line.lstrip())]

                if not write_control_seen:
                    output_lines.append(f"{indentation}writeControl    adjustableRunTime;")

                output_lines.append(f"{indentation}writeInterval   {interval:g};")
                output_lines.append(line)
                write_interval_seen = True
                index += 1
                continue

            output_lines.append(line)

            if brace_depth <= 0 and stripped == "}":
                inside_object = False

            index += 1
            continue

        output_lines.append(line)
        index += 1

    if not object_seen:
        raise ValueError(f"Could not find function object: {object_name}")

    return "\n".join(output_lines) + "\n"


def write_case_metadata(case_dir: Path, case: SurrogateCase) -> None:
    metadata_path = case_dir / "surrogate_case_metadata.csv"

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerow(["case_name", case.name])
        writer.writerow(["dataset_split", case.dataset_split])
        writer.writerow(["water_height_m", case.water_height])
        writer.writerow(["obstacle_height_m", case.obstacle_height])
        writer.writerow(["obstacle_front_x_m", case.obstacle_front_x])
        writer.writerow(["obstacle_width_m", case.obstacle_width])
        writer.writerow(["end_time_s", case.end_time])
        writer.writerow(["field_write_interval_s", case.field_write_interval])
        writer.writerow(["metric_write_interval_s", case.metric_write_interval])


def write_global_metadata(cases: list[SurrogateCase]) -> None:
    metadata_path = OUTPUT_ROOT / "surrogate_case_metadata.csv"

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "case_name",
                "dataset_split",
                "water_height_m",
                "obstacle_height_m",
                "obstacle_front_x_m",
                "obstacle_width_m",
                "end_time_s",
                "field_write_interval_s",
                "metric_write_interval_s",
            ]
        )

        for case in cases:
            writer.writerow(
                [
                    case.name,
                    case.dataset_split,
                    case.water_height,
                    case.obstacle_height,
                    case.obstacle_front_x,
                    case.obstacle_width,
                    case.end_time,
                    case.field_write_interval,
                    case.metric_write_interval,
                ]
            )


def main() -> None:
    if not BASE_SETUP.exists():
        raise FileNotFoundError(f"High-time base setup not found: {BASE_SETUP}")

    cases = read_design_csv(DESIGN_CSV)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        case_dir = OUTPUT_ROOT / case.name

        print(f"[{index:03d}/{len(cases):03d}] Generating {case.name}")

        copy_base_setup(case_dir)
        write_block_mesh_dict(case_dir, case)
        write_set_fields_dict(case_dir, case)
        update_control_dict(case_dir, case)
        write_case_metadata(case_dir, case)

    write_global_metadata(cases)

    print()
    print(f"Generated {len(cases)} surrogate CFD cases in:")
    print(f"  {OUTPUT_ROOT}")
    print()
    print("Global metadata written to:")
    print(f"  {OUTPUT_ROOT / 'surrogate_case_metadata.csv'}")


if __name__ == "__main__":
    main()
