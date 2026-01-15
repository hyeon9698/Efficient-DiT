#!/bin/bash
# SDTM*-a Reproduction Benchmark
# Target: Paper Table 1 - SDTM*-a (MACs 4.13T, Speed 1.33x)
# Using dynamic compression (cosine ratio scheduling)

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/mambaforge/etc/profile.d/conda.sh
conda activate sdtm_pixart

echo "========================================"
echo "SDTM*-a Reproduction Benchmark"
echo "Target: MACs 4.13T, Speed 1.33x"
echo "Testing ratio=0.4, 0.5, 0.6 (deviation=0.2 fixed)"
echo "========================================"

OUTPUT_DIR="./benchmark_outputs_sdtm"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Run in parallel on GPUs 4,5,6,7
# Baseline on GPU 4
CUDA_VISIBLE_DEVICES=4 python benchmark_metrics.py \
    --method baseline \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_0=$!

# SDTM ratio=0.4 on GPU 5
CUDA_VISIBLE_DEVICES=5 python benchmark_metrics.py \
    --method sdtm \
    --ratio 0.4 \
    --deviation 0.2 \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_1=$!

# SDTM ratio=0.5 on GPU 6
CUDA_VISIBLE_DEVICES=6 python benchmark_metrics.py \
    --method sdtm \
    --ratio 0.5 \
    --deviation 0.2 \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_2=$!

# SDTM ratio=0.6 on GPU 7
CUDA_VISIBLE_DEVICES=7 python benchmark_metrics.py \
    --method sdtm \
    --ratio 0.6 \
    --deviation 0.2 \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_3=$!

echo "Running in parallel..."
echo "  Baseline (GPU 4)"
echo "  SDTM r=0.4 (GPU 5)"
echo "  SDTM r=0.5 (GPU 6)"
echo "  SDTM r=0.6 (GPU 7)"

wait $PID_0 && echo "  Baseline done."
wait $PID_1 && echo "  r=0.4 done."
wait $PID_2 && echo "  r=0.5 done."
wait $PID_3 && echo "  r=0.6 done."

# Aggregate results
echo ""
python benchmark_metrics.py --aggregate --output_dir $OUTPUT_DIR

echo "========================================"
echo "Complete! Check images in $OUTPUT_DIR"
echo "========================================"
