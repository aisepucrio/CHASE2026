#!/bin/bash
################################################################################
# Comprehensive Experiments Script
#
# Runs all experiments for both LSTM and Classic ML pipelines with:
# - LSTM: All combinations of embeddings, RNN types, and strategies
# - Classic ML: All models with all embeddings, optimized with Optuna
#
# Total experiments:
#   LSTM: 3 embeddings x 2 RNN types x 2 strategies = 12 experiments
#   Classic ML: 8 models x 3 embeddings = 24 experiments
#   Total: 36 experiments
################################################################################

set -e  # Exit on error

# Create logs directory
LOGS_DIR="experiment_logs"
mkdir -p "$LOGS_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOGS_DIR}/experiments_${TIMESTAMP}.log"

# Configuration
EMBEDDINGS=("hubert_base_ls960" "wav2vec2_large_960h_lv60_self" "wav2vec2_large_xlsr_53_portuguese")
RNN_TYPES=("lstm" "gru")
STRATEGIES=("instant" "full_context")
CLASSIC_MODELS=("lightgbm" "xgboost" "random_forest" "extra_trees" "gradient_boosting" "svm_linear" "svm_rbf" "logistic_regression")

# Fixed hyperparameters
LSTM_EPOCHS=20
LSTM_BATCH_SIZE=8
LSTM_LEARNING_RATE=0.003
LSTM_HIDDEN_SIZE=256
LSTM_NUM_LAYERS=2
K_FOLDS=3

# Optuna configuration for Classic ML
OPTUNA_TRIALS=50

################################################################################
# Helper functions
################################################################################

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$MAIN_LOG"
}

run_lstm_experiment() {
    local embedding=$1
    local rnn_type=$2
    local strategy=$3

    local exp_name="lstm_${embedding}_${rnn_type}_${strategy}"
    local log_file="${LOGS_DIR}/${exp_name}_${TIMESTAMP}.log"

    log "Starting LSTM experiment: $exp_name"

    python src/training/train_rnn.py \
        --model_name "$embedding" \
        --rnn_type "$rnn_type" \
        --strategy "$strategy" \
        --epochs "$LSTM_EPOCHS" \
        --batch_size "$LSTM_BATCH_SIZE" \
        --learning_rate "$LSTM_LEARNING_RATE" \
        --hidden_size "$LSTM_HIDDEN_SIZE" \
        --num_layers "$LSTM_NUM_LAYERS" \
        --k_folds "$K_FOLDS" \
        2>&1 | tee "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "Completed LSTM experiment: $exp_name"
    else
        log "Failed LSTM experiment: $exp_name"
        return 1
    fi
}

run_classic_ml_experiment() {
    local embedding=$1
    local model=$2

    local exp_name="classic_${model}_${embedding}"
    local log_file="${LOGS_DIR}/${exp_name}_${TIMESTAMP}.log"

    log "Starting Classic ML experiment: $exp_name"

    python src/training/train_classic_ml.py \
        --model_name "$embedding" \
        --model_type "$model" \
        --k_folds "$K_FOLDS" \
        --optimize \
        --n_trials "$OPTUNA_TRIALS" \
        2>&1 | tee "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "Completed Classic ML experiment: $exp_name"
    else
        log "Failed Classic ML experiment: $exp_name"
        return 1
    fi
}

################################################################################
# Main execution
################################################################################

log "========================================================================"
log "Starting comprehensive experiments suite"
log "========================================================================"
log ""
log "Configuration:"
log "  - Embeddings: ${EMBEDDINGS[*]}"
log "  - K-Folds: $K_FOLDS"
log "  - Main log: $MAIN_LOG"
log ""
log "LSTM Configuration:"
log "  - RNN Types: ${RNN_TYPES[*]}"
log "  - Strategies: ${STRATEGIES[*]}"
log "  - Epochs: $LSTM_EPOCHS"
log "  - Batch Size: $LSTM_BATCH_SIZE"
log "  - Learning Rate: $LSTM_LEARNING_RATE"
log "  - Hidden Size: $LSTM_HIDDEN_SIZE"
log "  - Num Layers: $LSTM_NUM_LAYERS"
log ""
log "Classic ML Configuration:"
log "  - Models: ${CLASSIC_MODELS[*]}"
log "  - Optuna Optimization: Enabled"
log "  - Optuna Trials: $OPTUNA_TRIALS"
log ""

# Counters
lstm_success=0
lstm_failed=0
classic_success=0
classic_failed=0

################################################################################
# Part 1: LSTM Experiments
################################################################################

log "========================================================================"
log "PART 1/2: LSTM/RNN EXPERIMENTS"
log "========================================================================"
log ""

lstm_total=$((${#EMBEDDINGS[@]} * ${#RNN_TYPES[@]} * ${#STRATEGIES[@]}))
lstm_current=0

for embedding in "${EMBEDDINGS[@]}"; do
    for rnn_type in "${RNN_TYPES[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            lstm_current=$((lstm_current + 1))
            log ""
            log "--------------------------------------------------------------------"
            log "LSTM Experiment $lstm_current/$lstm_total"
            log "--------------------------------------------------------------------"

            if run_lstm_experiment "$embedding" "$rnn_type" "$strategy"; then
                lstm_success=$((lstm_success + 1))
            else
                lstm_failed=$((lstm_failed + 1))
            fi
        done
    done
done

log ""
log "========================================================================"
log "LSTM Experiments Summary"
log "========================================================================"
log "  Total: $lstm_total"
log "  Success: $lstm_success"
log "  Failed: $lstm_failed"
log ""

################################################################################
# Part 2: Classic ML Experiments
################################################################################

log "========================================================================"
log "PART 2/2: CLASSIC ML EXPERIMENTS"
log "========================================================================"
log ""

classic_total=$((${#EMBEDDINGS[@]} * ${#CLASSIC_MODELS[@]}))
classic_current=0

for embedding in "${EMBEDDINGS[@]}"; do
    for model in "${CLASSIC_MODELS[@]}"; do
        classic_current=$((classic_current + 1))
        log ""
        log "--------------------------------------------------------------------"
        log "Classic ML Experiment $classic_current/$classic_total"
        log "--------------------------------------------------------------------"

        if run_classic_ml_experiment "$embedding" "$model"; then
            classic_success=$((classic_success + 1))
        else
            classic_failed=$((classic_failed + 1))
        fi
    done
done

log ""
log "========================================================================"
log "Classic ML Experiments Summary"
log "========================================================================"
log "  Total: $classic_total"
log "  Success: $classic_success"
log "  Failed: $classic_failed"
log ""

################################################################################
# Final Summary
################################################################################

total_experiments=$((lstm_total + classic_total))
total_success=$((lstm_success + classic_success))
total_failed=$((lstm_failed + classic_failed))

log "========================================================================"
log "ALL EXPERIMENTS COMPLETED"
log "========================================================================"
log ""
log "Overall Summary:"
log "  Total Experiments: $total_experiments"
log "  Successful: $total_success"
log "  Failed: $total_failed"
log ""
log "Results Location:"
log "  - JSON/PNG: results/"
log "  - Checkpoints: checkpoints/"
log "  - Logs: $LOGS_DIR/"
log ""

if [ $total_failed -eq 0 ]; then
    log "All experiments completed successfully!"
    exit 0
else
    log "Some experiments failed. Check logs for details."
    exit 1
fi
