#!/usr/bin/env bash

set -e

PROJECT_ROOT="/mnt/c/Users/thoma/Desktop/damBreak_interFoam_laminar"
cd "$PROJECT_ROOT"

delete_if_exists() {
    target="$1"

    if [ -e "$target" ]; then
        echo "DELETING: $target"
        rm -rf "$target"
    fi
}

keep_notice() {
    target="$1"

    if [ -e "$target" ]; then
        echo "KEEPING:  $target"
    fi
}

echo
echo "PERMANENT cleanup of non-final project material"
echo "Project root: $PROJECT_ROOT"
echo

# ---------------------------------------------------------------------
# 0. Create clean final output folder before deleting anything else
# ---------------------------------------------------------------------

mkdir -p results/final_project_outputs/gifs
mkdir -p results/final_project_outputs/figures
mkdir -p results/final_project_outputs/data
mkdir -p results/final_project_outputs/summaries

if [ -f "results/surrogate_database/portfolio_media/gifs/FINAL_cfd_vs_pod_surrogate_alpha_field_200m.gif" ]; then
    cp "results/surrogate_database/portfolio_media/gifs/FINAL_cfd_vs_pod_surrogate_alpha_field_200m.gif" \
       "results/final_project_outputs/gifs/FINAL_cfd_vs_pod_surrogate_alpha_field.gif"
fi

if [ -f "results/surrogate_database/portfolio_media/gifs/surrogate_response_surface_dashboard.gif" ]; then
    cp "results/surrogate_database/portfolio_media/gifs/surrogate_response_surface_dashboard.gif" \
       "results/final_project_outputs/gifs/"
fi

if [ -f "results/surrogate_database/portfolio_media/gifs/surrogate_best_design_explorer.gif" ]; then
    cp "results/surrogate_database/portfolio_media/gifs/surrogate_best_design_explorer.gif" \
       "results/final_project_outputs/gifs/"
fi

if [ -f "results/surrogate_database/final_surrogate_plots/final_selected_surrogate_validation_r2.png" ]; then
    cp "results/surrogate_database/final_surrogate_plots/final_selected_surrogate_validation_r2.png" \
       "results/final_project_outputs/figures/"
fi

if [ -f "results/surrogate_database/final_surrogate_plots/final_selected_surrogate_validation_nrmse.png" ]; then
    cp "results/surrogate_database/final_surrogate_plots/final_selected_surrogate_validation_nrmse.png" \
       "results/final_project_outputs/figures/"
fi

if [ -f "results/surrogate_database/field_surrogate/pod_timeslice_models_200m/alpha_timeslice_pod_surrogate_summary.txt" ]; then
    cp "results/surrogate_database/field_surrogate/pod_timeslice_models_200m/alpha_timeslice_pod_surrogate_summary.txt" \
       "results/final_project_outputs/summaries/"
fi

if [ -f "results/surrogate_database/robust_surrogate_models/selected_robust_surrogate_summary.txt" ]; then
    cp "results/surrogate_database/robust_surrogate_models/selected_robust_surrogate_summary.txt" \
       "results/final_project_outputs/summaries/"
fi

if [ -f "results/surrogate_database/robust_surrogate_models/selected_robust_surrogate_models.csv" ]; then
    cp "results/surrogate_database/robust_surrogate_models/selected_robust_surrogate_models.csv" \
       "results/final_project_outputs/data/"
fi

# ---------------------------------------------------------------------
# 1. Delete early/tutorial/non-final OpenFOAM folders
# ---------------------------------------------------------------------

delete_if_exists "damBreak"
delete_if_exists "baseline_damBreak_tutorial"
delete_if_exists "damBreak_custom_v1"
delete_if_exists "damBreak_custom_v2_higher_column"
delete_if_exists "mesh_study"
delete_if_exists "time_step_study"
delete_if_exists "impact_study"

# ---------------------------------------------------------------------
# 2. Clean parametric_study
# Keep only the final 403-case surrogate database
# ---------------------------------------------------------------------

delete_if_exists "parametric_study/parametric_case_metadata.csv"
delete_if_exists "parametric_study/surrogate_adaptive_peak_cases"
delete_if_exists "parametric_study/pilot_H0292_obsH0048_x0292"
delete_if_exists "parametric_study/pilot_H0365_obsH0048_x0292"
delete_if_exists "parametric_study/pilot_H0292_obsH0072_x0292"
delete_if_exists "parametric_study/diagnostic_high_time_H0292"
delete_if_exists "parametric_study/matrix_fixed_x0292"
delete_if_exists "parametric_study/matrix_high_time_1p5"

keep_notice "parametric_study/surrogate_database_cases"

# ---------------------------------------------------------------------
# 3. Delete old result folders outside final surrogate_database workflow
# ---------------------------------------------------------------------

delete_if_exists "results/figures"
delete_if_exists "results/final_figures"
delete_if_exists "results/front_position"
delete_if_exists "results/impact_pressure"
delete_if_exists "results/mesh_convergence"
delete_if_exists "results/paraview_screenshots"
delete_if_exists "results/paraview_screenshots_clean"
delete_if_exists "results/paraview_screenshots_final"
delete_if_exists "results/time_step_sensitivity"

delete_if_exists "results/anomaly_diagnostics"
delete_if_exists "results/parametric_impact"
delete_if_exists "results/parametric_matrix"
delete_if_exists "results/parametric_matrix_high_time_1p5"
delete_if_exists "results/response_surface"

# ---------------------------------------------------------------------
# 4. Clean final surrogate_database folder
# ---------------------------------------------------------------------

delete_if_exists "results/surrogate_database/adaptive_peak_design"
delete_if_exists "results/surrogate_database/case_status"
delete_if_exists "results/surrogate_database/run_logs"
delete_if_exists "results/surrogate_database/overnight_runner.pid"
delete_if_exists "results/surrogate_database/overnight_runner_stdout.log"
delete_if_exists "results/surrogate_database/surrogate_database_progress.log"

delete_if_exists "results/surrogate_database/models"
delete_if_exists "results/surrogate_database/peak_pressure_risk"

# Old field surrogate attempts. Keep only final 200-mode time-sliced POD model.
delete_if_exists "results/surrogate_database/field_surrogate/pod_models"
delete_if_exists "results/surrogate_database/field_surrogate/pod_timeslice_models"

keep_notice "results/surrogate_database/design"
keep_notice "results/surrogate_database/metrics"
keep_notice "results/surrogate_database/robust_metrics"
keep_notice "results/surrogate_database/robust_surrogate_models"
keep_notice "results/surrogate_database/final_surrogate_plots"
keep_notice "results/surrogate_database/field_surrogate/pod_timeslice_models_200m"

# Delete generated animation frames and old zipped media.
delete_if_exists "results/surrogate_database/portfolio_media/frames"
delete_if_exists "results/surrogate_database/portfolio_media.zip"

# ---------------------------------------------------------------------
# 5. Keep only final portfolio GIFs
# ---------------------------------------------------------------------

GIF_DIR="results/surrogate_database/portfolio_media/gifs"

if [ -d "$GIF_DIR" ]; then
    for file in "$GIF_DIR"/*; do
        [ -e "$file" ] || continue

        base="$(basename "$file")"

        case "$base" in
            FINAL_cfd_vs_pod_surrogate_alpha_field_200m.gif|\
            surrogate_response_surface_dashboard.gif|\
            surrogate_best_design_explorer.gif|\
            gif_generation_summary.txt|\
            best_design_explorer_summary.txt)
                echo "KEEPING:  $file"
                ;;
            *)
                delete_if_exists "$file"
                ;;
        esac
    done
fi

# ---------------------------------------------------------------------
# 6. Delete obsolete scripts
# ---------------------------------------------------------------------

KEEP_SCRIPTS="
final_delete_non_project_material.sh
create_surrogate_doe_design.py
generate_surrogate_database_cases.py
run_surrogate_database_all.sh
extract_surrogate_database_metrics.py
compute_robust_peak_metrics.py
train_robust_surrogate_models.py
plot_final_surrogate_response_surfaces.py
make_response_surface_gifs.py
make_best_design_explorer_gifs.py
build_alpha_field_surrogate_dataset.py
train_alpha_pod_timeslice_surrogate_200m.py
make_final_showcase_alpha_surrogate_gif_200m.py
"

if [ -d "scripts" ]; then
    for script in scripts/*; do
        [ -f "$script" ] || continue

        base="$(basename "$script")"
        keep="no"

        for keep_script in $KEEP_SCRIPTS; do
            if [ "$base" = "$keep_script" ]; then
                keep="yes"
                break
            fi
        done

        if [ "$keep" = "yes" ]; then
            echo "KEEPING:  $script"
        else
            delete_if_exists "$script"
        fi
    done
fi

# ---------------------------------------------------------------------
# 7. Delete temporary inventory files
# ---------------------------------------------------------------------

delete_if_exists "project_folder_structure.txt"
delete_if_exists "project_folder_structure_no403.txt"
delete_if_exists "project_folder_structure_no403_v2.txt"
delete_if_exists "project_parametric_structure.txt"
delete_if_exists "project_parametric_structure_no403.txt"
delete_if_exists "project_parametric_structure_no403_v2.txt"
delete_if_exists "project_parametric_sizes.txt"
delete_if_exists "project_results_files.txt"
delete_if_exists "project_results_sizes.txt"
delete_if_exists "project_scripts_list.txt"
delete_if_exists "project_top_level_sizes.txt"

# ---------------------------------------------------------------------
# 8. Final report
# ---------------------------------------------------------------------

echo
echo "Cleanup complete."
echo
echo "Final project outputs:"
find results/final_project_outputs -type f | sort

echo
echo "Remaining top-level folders:"
find . -maxdepth 1 -type d | sort

echo
echo "Remaining scripts:"
ls scripts

echo
echo "Remaining result folders:"
find results -maxdepth 3 -type d | sort

echo
echo "Remaining parametric folders:"
find parametric_study -maxdepth 2 -type d | sort
