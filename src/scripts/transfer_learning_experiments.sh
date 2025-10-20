#!/bin/bash
################################################################################
# Transfer Learning Experiments Script
#
# Runs cross-dataset transfer learning experiments for both RNN and Classic ML models:
# - Transfer: Train on externo, test on interno
# - Transfer: Train on interno, test on externo
#
# RNN Models - For each configuration, tests all combinations of:
# - 3 embeddings (HuBERT, Wav2Vec2-large, Wav2Vec2-XLSR-PT)
# - 2 RNN types (LSTM, GRU)
# - 2 strategies (instant/unidirectional, full_context/bidirectional)
# Total RNN experiments: 2 transfer configs x 3 embeddings x 2 RNN types x 2 strategies = 24 experiments
#
# Classic ML Models - For each configuration, tests all combinations of:
# - 3 embeddings (HuBERT, Wav2Vec2-large, Wav2Vec2-XLSR-PT)
# - 8 ML models (LightGBM, XGBoost, RF, Extra Trees, Gradient Boosting, SVM-Linear, SVM-RBF, Logistic Regression)
# Total Classic ML experiments: 2 transfer configs x 3 embeddings x 8 models = 48 experiments
#
# TOTAL: 72 experiments
################################################################################

set -e  # Exit on error

# Create logs directory
LOGS_DIR="transfer_learning_logs"
mkdir -p "$LOGS_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOGS_DIR}/transfer_experiments_${TIMESTAMP}.log"

# Configuration
EMBEDDINGS=("hubert_base_ls960" "wav2vec2_large_960h_lv60_self" "wav2vec2_large_xlsr_53_portuguese")
RNN_TYPES=("lstm" "gru")
STRATEGIES=("instant" "full_context")
ML_MODELS=("lightgbm" "xgboost" "random_forest" "extra_trees" "gradient_boosting" "svm_linear" "svm_rbf" "logistic_regression")

# Dataset configurations: [train_dataset, test_dataset, description]
# Only cross-dataset transfer learning experiments
DATASET_CONFIGS=(
    "externo:interno:externo_to_interno"
    "interno:externo:interno_to_externo"
)

# Fixed hyperparameters
EPOCHS=20
BATCH_SIZE=8
LEARNING_RATE=0.003
HIDDEN_SIZE=256
NUM_LAYERS=2
K_FOLDS=3

################################################################################
# Helper functions
################################################################################

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$MAIN_LOG"
}

run_rnn_transfer_experiment() {
    local train_dataset=$1
    local test_dataset=$2
    local embedding=$3
    local rnn_type=$4
    local strategy=$5
    local config_name=$6

    local exp_name="RNN_${config_name}_${embedding}_${rnn_type}_${strategy}"
    local log_file="${LOGS_DIR}/${exp_name}_${TIMESTAMP}.log"

    log "Starting RNN transfer learning experiment: $exp_name"
    log "  Train: $train_dataset | Test: $test_dataset"

    python src/training/train_rnn.py \
        --model_name "$embedding" \
        --rnn_type "$rnn_type" \
        --strategy "$strategy" \
        --train_dataset "$train_dataset" \
        --test_dataset "$test_dataset" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --hidden_size "$HIDDEN_SIZE" \
        --num_layers "$NUM_LAYERS" \
        --k_folds "$K_FOLDS" \
        2>&1 | tee "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✓ Completed: $exp_name"
        return 0
    else
        log "✗ Failed: $exp_name"
        return 1
    fi
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
log "Starting Transfer Learning Experiments Suite (RNN + Classic ML)"
log "========================================================================"
log ""
log "Configuration:"
log "  - Embeddings: ${EMBEDDINGS[*]}"
log "  - RNN Types: ${RNN_TYPES[*]}"
log "  - RNN Strategies: ${STRATEGIES[*]}"
log "  - Classic ML Models: ${ML_MODELS[*]}"
log "  - K-Folds: $K_FOLDS"
log ""
log "RNN Hyperparameters:"
log "  - Epochs: $EPOCHS"
log "  - Batch Size: $BATCH_SIZE"
log "  - Learning Rate: $LEARNING_RATE"
log "  - Hidden Size: $HIDDEN_SIZE"
log "  - Num Layers: $NUM_LAYERS"
log ""
log "Dataset Configurations:"
for config in "${DATASET_CONFIGS[@]}"; do
    IFS=':' read -r train_ds test_ds desc <<< "$config"
    log "  - $desc: Train on $train_ds, Test on $test_ds"
done
log ""
log "Main log: $MAIN_LOG"
log ""

# Counters
rnn_total=$((${#DATASET_CONFIGS[@]} * ${#EMBEDDINGS[@]} * ${#RNN_TYPES[@]} * ${#STRATEGIES[@]}))
ml_total=$((${#DATASET_CONFIGS[@]} * ${#EMBEDDINGS[@]} * ${#ML_MODELS[@]}))
total_experiments=$((rnn_total + ml_total))
current_exp=0
total_success=0
total_failed=0

# Summary counters (using simple variables instead of associative arrays)
rnn_externo_to_interno_success=0
rnn_externo_to_interno_failed=0
rnn_interno_to_externo_success=0
rnn_interno_to_externo_failed=0
ml_externo_to_interno_success=0
ml_externo_to_interno_failed=0
ml_interno_to_externo_success=0
ml_interno_to_externo_failed=0

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

    # Run RNN experiments
    log "========================================================================"
    log "RNN EXPERIMENTS"
    log "========================================================================"
    log ""

    for embedding in "${EMBEDDINGS[@]}"; do
        for rnn_type in "${RNN_TYPES[@]}"; do
            for strategy in "${STRATEGIES[@]}"; do
                current_exp=$((current_exp + 1))

                log ""
                log "--------------------------------------------------------------------"
                log "RNN Experiment $current_exp/$total_experiments"
                log "--------------------------------------------------------------------"

                if run_rnn_transfer_experiment "$train_dataset" "$test_dataset" "$embedding" "$rnn_type" "$strategy" "$config_name"; then
                    total_success=$((total_success + 1))
                    if [ "$config_name" = "externo_to_interno" ]; then
                        rnn_externo_to_interno_success=$((rnn_externo_to_interno_success + 1))
                    else
                        rnn_interno_to_externo_success=$((rnn_interno_to_externo_success + 1))
                    fi
                else
                    total_failed=$((total_failed + 1))
                    if [ "$config_name" = "externo_to_interno" ]; then
                        rnn_externo_to_interno_failed=$((rnn_externo_to_interno_failed + 1))
                    else
                        rnn_interno_to_externo_failed=$((rnn_interno_to_externo_failed + 1))
                    fi
                fi
            done
        done
    done

    # Run Classic ML experiments
    log ""
    log "========================================================================"
    log "CLASSIC ML EXPERIMENTS"
    log "========================================================================"
    log ""

    for embedding in "${EMBEDDINGS[@]}"; do
        for model_type in "${ML_MODELS[@]}"; do
            current_exp=$((current_exp + 1))

            log ""
            log "--------------------------------------------------------------------"
            log "Classic ML Experiment $current_exp/$total_experiments"
            log "--------------------------------------------------------------------"

            if run_ml_transfer_experiment "$train_dataset" "$test_dataset" "$embedding" "$model_type" "$config_name"; then
                total_success=$((total_success + 1))
                if [ "$config_name" = "externo_to_interno" ]; then
                    ml_externo_to_interno_success=$((ml_externo_to_interno_success + 1))
                else
                    ml_interno_to_externo_success=$((ml_interno_to_externo_success + 1))
                fi
            else
                total_failed=$((total_failed + 1))
                if [ "$config_name" = "externo_to_interno" ]; then
                    ml_externo_to_interno_failed=$((ml_externo_to_interno_failed + 1))
                else
                    ml_interno_to_externo_failed=$((ml_interno_to_externo_failed + 1))
                fi
            fi
        done
    done

    log ""
    log "========================================================================"
    log "Summary for $config_name"
    log "========================================================================"
    if [ "$config_name" = "externo_to_interno" ]; then
        log "  RNN Success: $rnn_externo_to_interno_success, Failed: $rnn_externo_to_interno_failed"
        log "  Classic ML Success: $ml_externo_to_interno_success, Failed: $ml_externo_to_interno_failed"
    else
        log "  RNN Success: $rnn_interno_to_externo_success, Failed: $rnn_interno_to_externo_failed"
        log "  Classic ML Success: $ml_interno_to_externo_success, Failed: $ml_interno_to_externo_failed"
    fi
    log ""
done

################################################################################
# Final Summary
################################################################################

log "========================================================================"
log "TRANSFER LEARNING EXPERIMENTS COMPLETED (RNN + Classic ML)"
log "========================================================================"
log ""
log "Overall Summary:"
log "  Total Experiments: $total_experiments (RNN: $rnn_total, Classic ML: $ml_total)"
log "  Successful: $total_success"
log "  Failed: $total_failed"
log ""
log "Per-Configuration Summary:"
log ""
log "  Externo → Interno:"
log "    RNN: $rnn_externo_to_interno_success success, $rnn_externo_to_interno_failed failed"
log "    Classic ML: $ml_externo_to_interno_success success, $ml_externo_to_interno_failed failed"
log ""
log "  Interno → Externo:"
log "    RNN: $rnn_interno_to_externo_success success, $rnn_interno_to_externo_failed failed"
log "    Classic ML: $ml_interno_to_externo_success success, $ml_interno_to_externo_failed failed"
log ""
log "Results Location:"
log "  - JSON/PNG: results/transfer_learning/"
log "  - Checkpoints: checkpoints/"
log "  - Logs: $LOGS_DIR/"
log ""

if [ $total_failed -eq 0 ]; then
    log "✓ All experiments completed successfully!"
    exit 0
else
    log "✗ Some experiments failed. Check logs for details."
    exit 1
fi
