#!/bin/bash
# Final Benchmark: Baseline vs ToMe vs SDTM
# ToMe: ratio=0.8, SDTM: ratio=0.6, deviation=0.2

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/mambaforge/etc/profile.d/conda.sh
conda activate sdtm_pixart

echo "========================================"
echo "Final Benchmark: Baseline vs ToMe vs SDTM"
echo "========================================"
echo "  Baseline: No acceleration"
echo "  ToMe: ratio=0.8"
echo "  SDTM: ratio=0.6, deviation=0.2"
echo "========================================"

OUTPUT_DIR="./benchmark_outputs"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Backup current TR_SDTM.py
cp TR_SDTM.py versions/TR_SDTM_$(date +%Y%m%d_%H%M%S)_final.py

# Run in parallel on GPUs 4,5,6
# Baseline on GPU 4
CUDA_VISIBLE_DEVICES=4 python benchmark_metrics.py \
    --method baseline \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_0=$!

# ToMe ratio=0.8 on GPU 5
CUDA_VISIBLE_DEVICES=5 python benchmark_metrics.py \
    --method tome \
    --ratio 0.8 \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_1=$!

# SDTM ratio=0.6 on GPU 6
CUDA_VISIBLE_DEVICES=6 python benchmark_metrics.py \
    --method sdtm \
    --ratio 0.6 \
    --deviation 0.2 \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_2=$!

echo "Running in parallel..."
echo "  Baseline (GPU 4)"
echo "  ToMe r=0.8 (GPU 5)"
echo "  SDTM r=0.6 (GPU 6)"

wait $PID_0 && echo "  Baseline done."
wait $PID_1 && echo "  ToMe done."
wait $PID_2 && echo "  SDTM done."

# Aggregate results
echo ""
python benchmark_metrics.py --aggregate --output_dir $OUTPUT_DIR

echo "========================================"
echo "Complete! Check images in $OUTPUT_DIR"
echo "========================================"
