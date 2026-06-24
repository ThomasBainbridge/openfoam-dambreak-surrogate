#!/usr/bin/env bash

PROJECT_ROOT="/mnt/c/Users/thoma/Desktop/damBreak_interFoam_laminar"
CASE_ROOT="$PROJECT_ROOT/parametric_study/surrogate_database_cases"
METADATA_FILE="$CASE_ROOT/surrogate_case_metadata.csv"

RESULTS_ROOT="$PROJECT_ROOT/results/surrogate_database"
STATUS_DIR="$RESULTS_ROOT/case_status"
RUN_LOG_DIR="$RESULTS_ROOT/run_logs"

MASTER_STATUS="$RESULTS_ROOT/surrogate_database_run_status.csv"
PROGRESS_LOG="$RESULTS_ROOT/surrogate_database_progress.log"

source /usr/lib/openfoam/openfoam2312/etc/bashrc

mkdir -p "$RESULTS_ROOT"
mkdir -p "$STATUS_DIR"
mkdir -p "$RUN_LOG_DIR"

echo "Surrogate database run started: $(date)" | tee -a "$PROGRESS_LOG"
echo "Project root: $PROJECT_ROOT" | tee -a "$PROGRESS_LOG"
echo "Case root:    $CASE_ROOT" | tee -a "$PROGRESS_LOG"
echo "Metadata:     $METADATA_FILE" | tee -a "$PROGRESS_LOG"
echo | tee -a "$PROGRESS_LOG"

if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: metadata file not found: $METADATA_FILE" | tee -a "$PROGRESS_LOG"
    exit 1
fi

total_cases=$(($(wc -l < "$METADATA_FILE") - 1))
case_counter=0

tail -n +2 "$METADATA_FILE" | while IFS=, read -r case_name dataset_split water_height obstacle_height obstacle_front_x obstacle_width end_time field_write_interval metric_write_interval
do
    case_counter=$((case_counter + 1))
    case_dir="$CASE_ROOT/$case_name"
    status_file="$STATUS_DIR/$case_name.status"

    if [ -f "$status_file" ] && grep -q ",completed," "$status_file"; then
        echo "[$case_counter/$total_cases] SKIPPING completed case: $case_name" | tee -a "$PROGRESS_LOG"
        continue
    fi

    echo | tee -a "$PROGRESS_LOG"
    echo "============================================================" | tee -a "$PROGRESS_LOG"
    echo "[$case_counter/$total_cases] Running case: $case_name" | tee -a "$PROGRESS_LOG"
    echo "Split: $dataset_split | H=$water_height | h=$obstacle_height | x=$obstacle_front_x" | tee -a "$PROGRESS_LOG"
    echo "Started: $(date)" | tee -a "$PROGRESS_LOG"
    echo "============================================================" | tee -a "$PROGRESS_LOG"

    if [ ! -d "$case_dir" ]; then
        echo "$case_name,missing_case_directory,NA,no,0,$dataset_split,$water_height,$obstacle_height,$obstacle_front_x" > "$status_file"
        echo "ERROR: missing case directory: $case_dir" | tee -a "$PROGRESS_LOG"
        continue
    fi

    cd "$case_dir" || {
        echo "$case_name,failed_cd,NA,no,0,$dataset_split,$water_height,$obstacle_height,$obstacle_front_x" > "$status_file"
        continue
    }

    rm -rf postProcessing
    rm -rf [1-9]* 0.[0-9]*
    rm -rf 0
    cp -r 0.orig 0

    block_status=0
    check_status=0
    setfields_status=0
    interfoam_status=0

    blockMesh > log.blockMesh 2>&1 || block_status=$?
    checkMesh > log.checkMesh 2>&1 || check_status=$?
    setFields > log.setFields 2>&1 || setfields_status=$?
    interFoam > log.interFoam 2>&1 || interfoam_status=$?

    final_time=$(grep "^Time =" log.interFoam 2>/dev/null | tail -1 | awk '{print $3}')
    if [ -z "$final_time" ]; then
        final_time="NA"
    fi

    pressure_file="$case_dir/postProcessing/obstaclePressureAverage/0/surfaceFieldValue.dat"

    if [ -f "$pressure_file" ]; then
        n_pressure_samples=$(grep -v "^#" "$pressure_file" | sed '/^[[:space:]]*$/d' | wc -l)
    else
        n_pressure_samples=0
    fi

    mesh_ok="no"
    if grep -q "Mesh OK" log.checkMesh 2>/dev/null; then
        mesh_ok="yes"
    fi

    ended="no"
    if grep -q "^End" log.interFoam 2>/dev/null; then
        ended="yes"
    fi

    if [ "$block_status" -eq 0 ] && [ "$check_status" -eq 0 ] && [ "$setfields_status" -eq 0 ] && [ "$interfoam_status" -eq 0 ] && [ "$mesh_ok" = "yes" ] && [ "$ended" = "yes" ]; then
        status="completed"
    else
        status="failed_or_incomplete"
    fi

    echo "$case_name,$status,$final_time,$mesh_ok,$n_pressure_samples,$dataset_split,$water_height,$obstacle_height,$obstacle_front_x" > "$status_file"

    echo "Finished: $(date)" | tee -a "$PROGRESS_LOG"
    echo "Status: $status | final_time=$final_time | mesh_ok=$mesh_ok | pressure_samples=$n_pressure_samples" | tee -a "$PROGRESS_LOG"

    {
        echo "---- blockMesh tail ----"
        tail -5 log.blockMesh 2>/dev/null
        echo
        echo "---- checkMesh tail ----"
        tail -8 log.checkMesh 2>/dev/null
        echo
        echo "---- interFoam tail ----"
        tail -12 log.interFoam 2>/dev/null
    } > "$RUN_LOG_DIR/$case_name.summary.log"
done

echo "case_name,status,final_time,mesh_ok,n_pressure_samples,dataset_split,water_height_m,obstacle_height_m,obstacle_front_x_m" > "$MASTER_STATUS"

find "$STATUS_DIR" -name "*.status" -type f -print0 | sort -z | xargs -0 cat >> "$MASTER_STATUS"

echo | tee -a "$PROGRESS_LOG"
echo "Surrogate database run finished: $(date)" | tee -a "$PROGRESS_LOG"
echo "Master status file: $MASTER_STATUS" | tee -a "$PROGRESS_LOG"

completed_count=$(grep -c ",completed," "$MASTER_STATUS" || true)
failed_count=$(grep -c ",failed_or_incomplete," "$MASTER_STATUS" || true)

echo "Completed cases: $completed_count / $total_cases" | tee -a "$PROGRESS_LOG"
echo "Failed/incomplete cases: $failed_count" | tee -a "$PROGRESS_LOG"
