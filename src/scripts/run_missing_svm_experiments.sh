#!/bin/bash
################################################################################
# Run Missing SVM Transfer Learning Experiments
#
# Runs only the 12 missing SVM experiments (svm_linear and svm_rbf):
# - externo→interno: 6 experiments (3 embeddings × 2 SVM types)
# - interno→externo: 6 experiments (3 embeddings × 2 SVM types)
################################################################################

set -e  # Exit on error

# Create logs directory
LOGS_DIR="transfer_learning_logs"
mkdir -p "$LOGS_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOGS_DIR}/missing_svm_experiments_${TIMESTAMP}.log"

# Configuration - only missing experiments
EMBEDDINGS=("hubert_base_ls960" "wav2vec2_large_960h_lv60_self" "wav2vec2_large_xlsr_53_portuguese")
ML_MODELS=("svm_linear" "svm_rbf")

# Dataset configurations
DATASET_CONFIGS=(
    "externo:interno:externo_to_interno"
    "interno:externo:interno_to_externo"
)

# Fixed hyperparameters
K_FOLDS=3

################################################################################
# Helper functions
################################################################################

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$MAIN_LOG"
}

run_ml_transfer_experiment() {
    local train_dataset=$1
    local test_dataset=$2
    local embedding=$3
    local model_type=$4
    local config_name=$5

    local exp_name="ML_${config_name}_${embedding}_${model_type}"
    local log_file="${LOGS_DIR}/${exp_name}_${TIMESTAMP}.log"

    log "Starting Classic ML transfer learning experiment: $exp_name"
    log "  Train: $train_dataset | Test: $test_dataset"

    python src/training/train_classic_ml.py \
        --model_name "$embedding" \
        --model_type "$model_type" \
        --train_dataset "$train_dataset" \
        --test_dataset "$test_dataset" \
        --k_folds "$K_FOLDS" \
        --optimize \
        --random_state 42 2>&1 | tee "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✓ Completed: $exp_name"
        return 0
    else
        log "✗ Failed: $exp_name"
        return 1
    fi
}

################################################################################
# Main execution
################################################################################

log "========================================================================"
log "Running Missing SVM Transfer Learning Experiments"
log "========================================================================"
log ""
log "Configuration:"
log "  - Embeddings: ${EMBEDDINGS[*]}"
log "  - SVM Models: ${ML_MODELS[*]}"
log "  - K-Folds: $K_FOLDS"
log ""
log "Dataset Configurations:"
for config in "${DATASET_CONFIGS[@]}"; do
    IFS=':' read -r train_ds test_ds desc <<< "$config"
    log "  - $desc: Train on $train_ds, Test on $test_ds"
done
log ""
log "Total experiments to run: $((${#DATASET_CONFIGS[@]} * ${#EMBEDDINGS[@]} * ${#ML_MODELS[@]}))"
log "Main log: $MAIN_LOG"
log ""

# Counters
ml_total=$((${#DATASET_CONFIGS[@]} * ${#EMBEDDINGS[@]} * ${#ML_MODELS[@]}))
current_exp=0
total_success=0
total_failed=0

# Summary counters
externo_to_interno_success=0
externo_to_interno_failed=0
interno_to_externo_success=0
interno_to_externo_failed=0

################################################################################
# Run all experiments
################################################################################

for config in "${DATASET_CONFIGS[@]}"; do
    IFS=':' read -r train_dataset test_dataset config_name <<< "$config"

    log "========================================================================"
    log "Dataset Configuration: $config_name"
    log "Train: $train_dataset | Test: $test_dataset"
    log "========================================================================"
    log ""

    for embedding in "${EMBEDDINGS[@]}"; do
        for model_type in "${ML_MODELS[@]}"; do
            current_exp=$((current_exp + 1))

            log ""
            log "--------------------------------------------------------------------"
            log "Experiment $current_exp/$ml_total"
            log "--------------------------------------------------------------------"

            if run_ml_transfer_experiment "$train_dataset" "$test_dataset" "$embedding" "$model_type" "$config_name"; then
                total_success=$((total_success + 1))
                if [ "$config_name" = "externo_to_interno" ]; then
                    externo_to_interno_success=$((externo_to_interno_success + 1))
                else
                    interno_to_externo_success=$((interno_to_externo_success + 1))
                fi
            else
                total_failed=$((total_failed + 1))
                if [ "$config_name" = "externo_to_interno" ]; then
                    externo_to_interno_failed=$((externo_to_interno_failed + 1))
                else
                    interno_to_externo_failed=$((interno_to_externo_failed + 1))
                fi
            fi
        done
    done

    log ""
    log "========================================================================"
    log "Summary for $config_name"
    log "========================================================================"
    if [ "$config_name" = "externo_to_interno" ]; then
        log "  Success: $externo_to_interno_success, Failed: $externo_to_interno_failed"
    else
        log "  Success: $interno_to_externo_success, Failed: $interno_to_externo_failed"
    fi
    log ""
done

################################################################################
# Final Summary
################################################################################

log "========================================================================"
log "MISSING SVM EXPERIMENTS COMPLETED"
log "========================================================================"
log ""
log "Summary:"
log "  Total experiments: $ml_total"
log "  Successful: $total_success"
log "  Failed: $total_failed"
log ""
log "Per-Configuration Summary:"
log "  Externo → Interno: $externo_to_interno_success success, $externo_to_interno_failed failed"
log "  Interno → Externo: $interno_to_externo_success success, $interno_to_externo_failed failed"
log ""
log "Results Location:"
log "  - JSON/PNG: results/transfer_learning/"
log "  - Logs: $LOGS_DIR/"
log ""

if [ $total_failed -eq 0 ]; then
    log "✓ All SVM experiments completed successfully!"
    exit 0
else
    log "✗ Some experiments failed. Check logs for details."
    exit 1
fi
