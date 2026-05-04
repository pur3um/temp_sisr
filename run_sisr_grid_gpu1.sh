#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# GPU 1 shard for full SISR grid search
# Run together with run_sisr_grid_gpu0.sh
# IMAGE_START=1 IMAGE_END=1 FAIL_FAST=1 bash run_sisr_grid_gpu1.sh
# =========================================================
export CUDA_VISIBLE_DEVICES=1
GPU_TAG="gpu1"
SHARD_ID=1
NUM_SHARDS=2

PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-fit_sisr_sched.py}"
IMG_DIR="${IMG_DIR:-DIV2K_train_HR}"
ROOT="${ROOT:-logs/sisr_grid}"

IMAGE_START="${IMAGE_START:-1}"
IMAGE_END="${IMAGE_END:-30}"

MODELS=(relu_mlp siren_mlp real_wire finer_mlp)
EPOCHS_LIST=(2000 3000 4000 5000)
RANK_STARTS=(100 150 200)

# LR defaults. Change from environment if needed, e.g. ADAM_LR=3e-4 bash run_sisr_grid_gpu1.sh
ADAM_LR="${ADAM_LR:-1e-3}"
AUX_LR="${AUX_LR:-3e-2}"
MUON_LR="${MUON_LR:-3e-3}"
MUON_NS_STEPS="${MUON_NS_STEPS:-10}"

# auto_cos_inc_rank rank schedule
RANK_FLOOR="${RANK_FLOOR:-100}"
RANK_END="${RANK_END:-250}"
RANK_OVERSAMPLE="${RANK_OVERSAMPLE:-4}"

# rank_wsd schedule. 2/3 matches your example: epochs=3000 -> decay_start=2000.
RANK_WSD_DECAY_NUM="${RANK_WSD_DECAY_NUM:-2}"
RANK_WSD_DECAY_DEN="${RANK_WSD_DECAY_DEN:-3}"
RANK_WSD_MIN_LR_RATIO="${RANK_WSD_MIN_LR_RATIO:-0.1}"
RANK_WSD_WARMUP_STEPS="${RANK_WSD_WARMUP_STEPS:-0}"

SCALE_FACTOR="${SCALE_FACTOR:-4}"
LOG_N_EPOCHS="${LOG_N_EPOCHS:-500}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

# If your Python code uses a different name, override this:
#   LRSIGN10_OPT=lr-sign10-rsclF bash run_sisr_grid_gpu1.sh
LRSIGN10_OPT="${LRSIGN10_OPT:-lr_sign10_rsclF}"

mkdir -p "$ROOT"
FAIL_LOG="${ROOT}/failed_${GPU_TAG}.txt"
: > "$FAIL_LOG"

optimizer_output_dir_name() {
    local opt="$1"
    case "$opt" in
        auto_cos_inc|auto_cos_inc_rank|auto-cos-inc|auto-cos-inc-rank)
            echo "auto_cos_inc"
            ;;
        *)
            echo "$opt"
            ;;
    esac
}

sanitize_tag() {
    echo "$1" | sed 's#[/,:]#_#g; s#[+]#p#g'
}

rank_wsd_decay_start() {
    local epochs="$1"
    echo $(( epochs * RANK_WSD_DECAY_NUM / RANK_WSD_DECAY_DEN ))
}

select_lr() {
    local opt="$1"
    case "$opt" in
        adam)
            echo "$ADAM_LR"
            ;;
        *)
            echo "$AUX_LR"
            ;;
    esac
}

make_setting_name() {
    local opt="$1"
    local scheduler="$2"
    local epochs="$3"
    local rank_start="${4:-}"
    local lr="$5"
    local decay_start
    decay_start="$(rank_wsd_decay_start "$epochs")"

    local setting="${opt}_${scheduler}_ep${epochs}"
    if [[ "$scheduler" == "cosine" ]]; then
        setting+="_T${epochs}"
    elif [[ "$scheduler" == "rank_wsd" ]]; then
        setting+="_decay${decay_start}_min${RANK_WSD_MIN_LR_RATIO}"
    fi

    if [[ "$opt" == "auto_cos_inc_rank" || "$opt" == "auto_cos_inc" || "$opt" == "auto-cos-inc-rank" ]]; then
        setting+="_r${RANK_FLOOR}_${rank_start}to${RANK_END}_ns${MUON_NS_STEPS}_lr${lr}_mlr${MUON_LR}"
    elif [[ "$opt" == "muon" || "$opt" == "lr-sign" || "$opt" == "$LRSIGN10_OPT" ]]; then
        setting+="_ns${MUON_NS_STEPS}_lr${lr}_mlr${MUON_LR}"
    else
        setting+="_lr${lr}"
    fi

    sanitize_tag "$setting"
}

JOB_INDEX=0
should_run_this_shard() {
    local idx="$JOB_INDEX"
    JOB_INDEX=$((JOB_INDEX + 1))
    (( idx % NUM_SHARDS == SHARD_ID ))
}

run_one() {
    local img_i="$1"
    local model="$2"
    local opt="$3"
    local scheduler="$4"
    local epochs="$5"
    local rank_start="${6:-}"

    should_run_this_shard || return 0

    local img_id outer_id img_path lr decay_start setting folder_name opt_dir readme_path
    img_id="$(printf "%04d" "$img_i")"
    outer_id="$(printf "DIV2K%02d" "$img_i")"
    img_path="${IMG_DIR}/${img_id}.png"

    if [[ ! -f "$img_path" ]]; then
        echo "[${GPU_TAG}] SKIP missing image: ${img_path}"
        return 0
    fi

    lr="$(select_lr "$opt")"
    decay_start="$(rank_wsd_decay_start "$epochs")"
    setting="$(make_setting_name "$opt" "$scheduler" "$epochs" "$rank_start" "$lr")"
    folder_name="${ROOT}/${setting}/${outer_id}"
    opt_dir="$(optimizer_output_dir_name "$opt")"
    readme_path="${folder_name}/${model}/${opt_dir}/Readme.txt"

    if [[ "$FORCE" != "1" && -f "$readme_path" ]]; then
        echo "[${GPU_TAG}] SKIP existing: ${readme_path}"
        return 0
    fi

    local cmd=(
        "$PYTHON" "$SCRIPT"
        --task super_resolution
        --image "$img_path"
        --model "$model"
        --optimizer "$opt"
        --scheduler "$scheduler"
        --epochs "$epochs"
        --log_n_epochs "$LOG_N_EPOCHS"
        --seed "$SEED"
        --scale_factor "$SCALE_FACTOR"
        --lr "$lr"
        --muon_lr "$MUON_LR"
        --muon_ns_steps "$MUON_NS_STEPS"
        --folder_name "$folder_name"
        --skip_lpips
    )

    if [[ "$scheduler" == "cosine" ]]; then
        cmd+=(--T_max "$epochs")
    elif [[ "$scheduler" == "rank_wsd" ]]; then
        cmd+=(
            --rank_wsd_warmup_steps "$RANK_WSD_WARMUP_STEPS"
            --rank_wsd_decay_start_step "$decay_start"
            --rank_wsd_min_lr_ratio "$RANK_WSD_MIN_LR_RATIO"
        )
    fi

    if [[ "$opt" == "auto_cos_inc_rank" || "$opt" == "auto_cos_inc" || "$opt" == "auto-cos-inc-rank" ]]; then
        local rank_warmup_steps
        if [[ "$scheduler" == "rank_wsd" ]]; then
            rank_warmup_steps="$decay_start"
        else
            rank_warmup_steps="$epochs"
        fi

        cmd+=(
            --rank "$RANK_FLOOR"
            --rank_start "$rank_start"
            --rank_end "$RANK_END"
            --rank_warmup_steps "$rank_warmup_steps"
            --rank_oversample "$RANK_OVERSAMPLE"
        )
    fi

    echo "============================================================"
    echo "[${GPU_TAG}] image=${outer_id} model=${model} opt=${opt} scheduler=${scheduler} epochs=${epochs} rank_start=${rank_start:-NA}"
    echo "[${GPU_TAG}] output=${folder_name}/${model}/${opt_dir}"
    printf '[%s] CMD:' "$GPU_TAG"
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if "${cmd[@]}"; then
        echo "[${GPU_TAG}] DONE ${outer_id} ${model} ${opt} ${scheduler} ep${epochs} r${rank_start:-NA}"
    else
        local status=$?
        echo "[${GPU_TAG}] FAILED status=${status}: ${outer_id} ${model} ${opt} ${scheduler} ep${epochs} r${rank_start:-NA}" | tee -a "$FAIL_LOG"
        printf '%q ' "${cmd[@]}" >> "$FAIL_LOG"
        printf '\n\n' >> "$FAIL_LOG"
        if [[ "$FAIL_FAST" == "1" ]]; then
            exit "$status"
        fi
    fi
}

for i in $(seq "$IMAGE_START" "$IMAGE_END"); do
    for model in "${MODELS[@]}"; do
        for epochs in "${EPOCHS_LIST[@]}"; do
            # cosine-only optimizers
            run_one "$i" "$model" adam cosine "$epochs"
            run_one "$i" "$model" muon cosine "$epochs"
            run_one "$i" "$model" "$LRSIGN10_OPT" cosine "$epochs"

            # lr-sign: cosine + rank_wsd
            run_one "$i" "$model" lr-sign cosine "$epochs"
            run_one "$i" "$model" lr-sign rank_wsd "$epochs"

            # auto_cos_inc_rank: cosine + rank_wsd, rank_start sweep 100/150/200
            for rank_start in "${RANK_STARTS[@]}"; do
                run_one "$i" "$model" auto_cos_inc_rank cosine "$epochs" "$rank_start"
                run_one "$i" "$model" auto_cos_inc_rank rank_wsd "$epochs" "$rank_start"
            done
        done
    done
done

echo "[${GPU_TAG}] all assigned jobs finished. Failure log: ${FAIL_LOG}"
