#@ Ours
# 
#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1
PYTHON=python
SCRIPT=fit_sisr_sched.py 
IMG_DIR=DIV2K_train_HR
ROOT=logs/sisr_3k

for i in $(seq 1 30); do
    img_id=$(printf "%04d" "$i")      # 0001 ~ 0030
    outer_id=$(printf "DIV2K%02d" "$i")  # DIV2K01 ~ DIV2K30

    img_path="${IMG_DIR}/${img_id}.png"

    # 1) WIRE complex + Adam
    # 3) WIRE real + Adam
    $PYTHON $SCRIPT \
        --task super_resolution \
        --image "$img_path" \
        --model real_wire \
        --optimizer auto_cos_inc \
        --scheduler rank_wsd \
        --rank 200 \
        --rank_start 200 \
        --rank_end 250 \
        --rank_warmup_steps 2000 \
        --rank_wsd_decay_start_step 2000 \
        --rank_wsd_min_lr_ratio 0.1 \
        --muon_ns_steps 10 \
        --epochs 3000 \
        --lr 0.03 \
        --muon_lr 3e-3 \
        --folder_name "${ROOT}/wire_real_ns10_200to250_lr3e-2_mlr3e-3/${outer_id}"
done
