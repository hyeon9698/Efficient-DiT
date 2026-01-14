#!/bin/bash
# ToMeSD Benchmark - Higher Ratio Test
# Target: Paper ToMeSD-a (MACs 4.27T, Speed 1.30x)

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/mambaforge/etc/profile.d/conda.sh
conda activate sdtm_pixart

echo "========================================"
echo "ToMeSD Benchmark - Higher Ratio Test"
echo "Testing ratio=0.85, 0.9"
echo "========================================"

OUTPUT_DIR="./benchmark_outputs_tome"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Run in parallel on GPUs 4,5,6
# Baseline
CUDA_VISIBLE_DEVICES=4 python benchmark_metrics.py --method baseline --num_runs 3 --output_dir $OUTPUT_DIR &
PID_0=$!

# ToMe ratio=0.85
CUDA_VISIBLE_DEVICES=5 python benchmark_metrics.py --method tome --ratio 0.85 --num_runs 3 --output_dir $OUTPUT_DIR &
PID_1=$!

# ToMe ratio=0.9
CUDA_VISIBLE_DEVICES=6 python benchmark_metrics.py --method tome --ratio 0.9 --num_runs 3 --output_dir $OUTPUT_DIR &
PID_2=$!

echo "Running in parallel..."
echo "  Baseline (GPU 4)"
echo "  ToMe r=0.85 (GPU 5)"
echo "  ToMe r=0.9 (GPU 6)"

wait $PID_0 && echo "  Baseline done."
wait $PID_1 && echo "  r=0.85 done."
wait $PID_2 && echo "  r=0.9 done."

# Aggregate results
echo ""
python benchmark_metrics.py --aggregate --output_dir $OUTPUT_DIR

echo "========================================"
echo "Complete!"
echo "========================================"
