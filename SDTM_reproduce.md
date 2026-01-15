# SDTM*-a Reproduction Plan

## Target: Paper Table 1 - SDTM*-a Configuration
| Metric | Target Value |
|--------|-------------|
| MACs | 4.13T |
| Latency | 8.02s (on 4×A100) |
| Speed | 1.33× |

**Note**: The `*` indicates dynamic compression (cosine ratio scheduling from `ratio+deviation` to `ratio-deviation`)

## Quick Start
```bash
cd /home/server39/donghyeon_workspace/SDTM
./run_benchmark_sdtm.sh
```

---

## 1. Problem Analysis

### 1.1 GitHub Issues Summary
Users report **~1.18× speed instead of 1.33×**:
- [Issue #4](https://github.com/ICTMCG/SDTM/issues/4): RTX 3090, 50 steps → 17.8s baseline, 15.1s SDTM (1.18×)
- [Issue #3](https://github.com/ICTMCG/SDTM/issues/3): A800, 50 steps → 8.6s baseline, 8.4s SDTM (minimal improvement)

### 1.2 Root Causes Identified
1. **Missing `cache_each_step=True`**: Official sample.py uses this but benchmark doesn't
2. **Wrong merge flags**: `merge_attn=False, merge_mlp=False` in `apply_SDTM` defaults
3. **Missing `protect_steps_frequency=3`**: Skips merging every 3rd step for quality
4. **Try/except blocks**: Silently swallow errors, masking bugs

### 1.3 Paper Algorithm Summary

**Two-Phase Approach**:
1. **Structure Phase (steps 0→switch_step)**: Use **SSM** (Similarity-prioritized Structure Merging)
   - Window-based merging using cosine similarity
   - Merges both MHSA (attention) and MLP blocks
   - Higher ratio (more aggressive merging)
   - Uses `a_s` for frequency priority: `P_w = P_sim + a_s * P_fre`

2. **Detail Phase (steps switch_step→end)**: Use **IDM** (Inattentive-prioritized Detail Merging)
   - Paper: "ceased accelerating the MHSA module" - **MLP only**
   - Uses `a_d` for frequency priority: `P_x = P_ina + a_d * P_fre`
   - Lower ratio (preserve details)
   - **Note**: `merge_attn=False` in this phase is CORRECT per paper

**Dynamic Compression (the `*`)**: Cosine decay from `(ratio + deviation)` to `(ratio - deviation)`

---

## 2. Official Hyperparameters (from sample.py)

```python
apply_SDTM(
    pipe,
    ratio=0.3,                      # Base compression ratio
    deviation=0.2,                  # Dynamic range: 0.5 → 0.1 over steps
    switch_step=20,                 # Switch from SSM to FIDM at step 20
    use_rand=True,                  # Random token selection in windows
    sx=2, sy=2,                     # Window size 2×2
    a_s=0.05,                       # Frequency priority weight for SSM
    a_d=0.05,                       # Frequency priority weight for IDM (FIXED!)
    a_p=2,                          # (unused in current code)
    pseudo_merge=False,             # Actual merge, not pseudo
    mcw=0.1,                        # Merge cache weight (merge=w, cache=1-w)
    protect_steps_frequency=3,      # Skip merging every 3rd step
    protect_layers_frequency=-1,    # No layer protection
    # CRITICAL MISSING PARAMS:
    merge_attn=True,                # Must enable!
    merge_mlp=True,                 # Must enable!
    cache_each_step=True,           # Must enable for quality!
)
```

---

## 3. Code Changes Made

### 3.1 TR_SDTM.py Modifications (COMPLETED)

#### ✅ Task 3.1.1: Removed try/except blocks
- SSM `unmerge()` function - converted to explicit if-check
- FIDM `unmerge()` function - converted to explicit if-check
- `_store_feature()` helper - removed try/except entirely
- `use_dual_attention` setting - converted to hasattr check

#### ✅ Task 3.1.2: Fixed `cache_each_step` parameter
- Added `cache_each_step: bool = True` to `apply_SDTM()` signature
- Now properly passes to `_tore_info["args"]["cache_each_step"]`

#### ✅ Task 3.1.3: Fixed default values in `apply_SDTM()`
```python
# NEW DEFAULTS (matching official sample.py):
ratio: float = 0.3,              # was 0.5
mcw: float = 0.1,                # was 0.2
protect_steps_frequency: int = 3, # was None
protect_layers_frequency: int = -1, # was None
merge_attn: bool = True,          # was False ← CRITICAL FIX
merge_mlp: bool = True,           # was False ← CRITICAL FIX
cache_each_step: bool = True,     # NEW PARAMETER
```

### 3.2 benchmark_metrics.py Modifications (COMPLETED)

#### ✅ Updated SDTM section with all official parameters:
```python
apply_SDTM(
    pipe,
    ratio=args.ratio,
    deviation=args.deviation,
    switch_step=switch_step,
    a_s=0.05,
    mcw=0.1,
    protect_steps_frequency=3,
    protect_layers_frequency=-1,
    merge_attn=True,
    merge_mlp=True,
    cache_each_step=True,
)
```

### 3.3 Created run_benchmark_sdtm.sh (NEW)
- Tests baseline + SDTM with ratio=0.3, 0.35, 0.4
- Uses GPUs 4,5,6,7 in parallel
- Outputs to `./benchmark_outputs_sdtm/`

---

## 4. Implementation Steps

### Phase 1: Code Cleanup (COMPLETED)
- [x] Modified TR_SDTM.py directly (no separate file needed)
- [x] Removed all try/except blocks (4 locations)
- [x] Added `cache_each_step` parameter to `apply_SDTM`
- [x] Fixed default merge flags: `merge_attn=True, merge_mlp=True`

### Phase 2: Benchmark Script Update (COMPLETED)
- [x] Created `run_benchmark_sdtm.sh`
- [x] Updated `benchmark_metrics.py` with all official params
- [x] Tests ratio=0.3, 0.35, 0.4 with deviation=0.2 fixed

### Phase 3: Iterative Tuning (READY TO RUN)
```bash
./run_benchmark_sdtm.sh
```

**Tuning Strategy:**
- If MACs > 4.13T: increase ratio (try 0.35, 0.4)
- If image quality bad: decrease ratio (try 0.25)
- Target speed: 1.33× (compare with baseline in same run)

---

## 5. Verification Checklist

### 5.1 Speed Verification
```
Expected: baseline ~17.8s → SDTM ~13.4s (1.33×) on RTX 3090
```

### 5.2 MACs Verification
```
Expected: baseline 6.01T → SDTM 4.13T (31% reduction)
```

### 5.3 Quality Verification
- Visual inspection: No artifacts, coherent structure
- Compare with ToMe output at same ratio

---

## 6. Benchmark Script Template

```bash
#!/bin/bash
# SDTM*-a Reproduction Benchmark
# Target: MACs 4.13T, Speed 1.33×

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/mambaforge/etc/profile.d/conda.sh
conda activate sdtm_pixart

OUTPUT_DIR="./benchmark_outputs_sdtm_reproduce"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Baseline on GPU 4
CUDA_VISIBLE_DEVICES=4 python benchmark_metrics.py \
    --method baseline --num_runs 3 --output_dir $OUTPUT_DIR &
PID_0=$!

# SDTM*-a official params on GPU 5
CUDA_VISIBLE_DEVICES=5 python benchmark_metrics.py \
    --method sdtm \
    --ratio 0.3 \
    --deviation 0.2 \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_1=$!

# SDTM*-a tuned (if needed) on GPU 6
CUDA_VISIBLE_DEVICES=6 python benchmark_metrics.py \
    --method sdtm \
    --ratio 0.35 \
    --deviation 0.2 \
    --num_runs 3 \
    --output_dir $OUTPUT_DIR &
PID_2=$!

wait $PID_0 $PID_1 $PID_2

python benchmark_metrics.py --aggregate --output_dir $OUTPUT_DIR
```

---

## 7. Key Formulas from Paper

### 7.1 Cosine Ratio Scheduling (Dynamic Compression `*`)
```python
# Progress: 0 → 1 over steps
progress = step_current / (step_count - 1)

# Cosine decay: starts at (ratio+deviation), ends at (ratio-deviation)
alpha = cos(progress * pi / 2)  # 1 → 0
ratio_current = (ratio - deviation) + 2 * deviation * alpha

# Example with ratio=0.3, deviation=0.2:
# Step 0:  ratio_current = 0.1 + 0.4 * 1.0 = 0.5
# Step 25: ratio_current = 0.1 + 0.4 * 0.707 = 0.38
# Step 49: ratio_current = 0.1 + 0.4 * 0.0 = 0.1
```

### 7.2 SSM Score (Structure Phase)
```python
# Window similarity score
P_sim = sum(cosine_similarity(xi, xj)) / (m² * (m² - 1))

# Frequency priority score
P_fre = T_x / mean(T)  # T_x = steps since last independent

# Combined score
P_w = P_sim + a_s * P_fre
```

### 7.3 IDM Score (Detail Phase) - FIXED
```python
# Inattentiveness score (from cosine similarity in bipartite matching)
# node_max = max similarity score to dst tokens
node_max, node_idx = scores.max(dim=-1)

# Frequency priority: P_fre = T_x / mean(T)
# T_x = steps since token was last independent (not merged)
P_fre = last_independent / mean(last_independent)

# Combined score per paper: P_x = P_ina + a_d * P_fre
node_max = node_max + a_d * P_fre  # FIXED: a_d was unused before!
```

**CRITICAL FIX**: The `a_d` parameter was defined but never used in the original code.
Now implemented at TR_SDTM.py lines 376-390.

---

## 8. Debugging Tips

### 8.1 Verify Merging is Active
Add to `make_SDTM_block`:
```python
print(f"Step {step_current}, Layer {layer_current}: "
      f"merge_attn={self._tore_info['states']['merge_attn']}, "
      f"merge_mlp={self._tore_info['states']['merge_mlp']}, "
      f"ratio={self._tore_info['states'].get('ratio_current', 'N/A')}")
```

### 8.2 Verify Token Count Reduction
Add after merge:
```python
print(f"Before merge: {norm_hidden_states.shape[1]} tokens")
norm_hidden_states = m_a(norm_hidden_states)
print(f"After merge: {norm_hidden_states.shape[1]} tokens")
```

### 8.3 Common Failure Modes
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No speedup | merge_attn=False, merge_mlp=False | Enable both |
| Artifacts | ratio too high | Reduce ratio, enable protect_steps |
| Only 1.1× speed | cache_each_step=False | Enable cache |
| Worse than ToMe | Wrong switch_step | Verify switch_step=20 |

---

## 9. File Structure

```
SDTM/
├── TR_SDTM.py              # Modified with fixes (no try/except, correct defaults)
├── TR_ToMe.py              # Working ToMe baseline for comparison
├── benchmark_metrics.py    # Updated with official SDTM parameters
├── run_benchmark_sdtm.sh   # NEW: Benchmark script (GPU 4,5,6,7)
├── run_benchmark_tome.sh   # Existing ToMe benchmark
├── SDTM_reproduce.md       # This document
└── benchmark_outputs_sdtm/ # Created after running benchmark
    ├── baseline.png
    ├── sdtm_r0.3_d0.2.png
    ├── sdtm_r0.35_d0.2.png
    ├── sdtm_r0.4_d0.2.png
    └── *.json results
```

---

## 10. Success Criteria

| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| Speed | 1.33× | 1.28× - 1.38× |
| MACs | 4.13T | 4.0T - 4.3T |
| Image Quality | Clean, no artifacts | Visual inspection |
| FID (if measured) | ~28.57 | < 30 |

---

## References

- Paper: [arXiv:2505.11707](https://arxiv.org/abs/2505.11707)
- Code: [ICTMCG/SDTM](https://github.com/ICTMCG/SDTM)
- Issues: [GitHub Issues](https://github.com/ICTMCG/SDTM/issues)
