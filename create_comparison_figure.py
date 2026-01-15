#!/usr/bin/env python3
"""Create comparison figure for paper: Baseline vs ToMe vs SDTM"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import json
import os

# Load results
with open("benchmark_outputs/all_results.json", "r") as f:
    data = json.load(f)

results = {r["method"]: r for r in data["results"]}

# Load images
baseline_img = mpimg.imread("benchmark_outputs/baseline.png")
tome_img = mpimg.imread("benchmark_outputs/tome_r0.8.png")
sdtm_img = mpimg.imread("benchmark_outputs/sdtm_r0.6_d0.2.png")

# Create figure
fig = plt.figure(figsize=(15, 7))

# Top row: images
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(baseline_img)
ax1.set_title("Baseline", fontsize=14, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(tome_img)
ax2.set_title("ToMe (r=0.8)", fontsize=14, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(sdtm_img)
ax3.set_title("SDTM (r=0.6)", fontsize=14, fontweight='bold')
ax3.axis('off')

# Bottom row: table
ax_table = fig.add_subplot(2, 1, 2)
ax_table.axis('off')

# Prepare table data
baseline = results["baseline"]
tome = results["tome"]
sdtm = results["sdtm"]

table_data = [
    ["Baseline", f"{baseline['macs_t']:.2f}", f"{baseline['latency_s']:.2f}", "1.00x"],
    ["ToMe (r=0.8)", f"{tome['macs_t']:.2f}", f"{tome['latency_s']:.2f}", f"{tome['speed']:.2f}x"],
    ["SDTM (r=0.6)", f"{sdtm['macs_t']:.2f}", f"{sdtm['latency_s']:.2f}", f"{sdtm['speed']:.2f}x"],
]

col_labels = ["Method", "MACs (T)", "Latency (s)", "Speedup"]

# Create table
table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colWidths=[0.25, 0.18, 0.18, 0.18]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.0)

# Header style
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Row colors
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        if i == 3:  # SDTM row - highlight
            table[(i, j)].set_facecolor('#E2EFDA')
        elif i % 2 == 0:
            table[(i, j)].set_facecolor('#F2F2F2')

# Add note
fig.text(0.5, 0.02,
         "Prompt: 'A photo of a cat sitting on a windowsill' | Steps: 50 | Seed: 42 | GPU: RTX 3090",
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, hspace=0.15)

# Save
output_path = "benchmark_outputs/comparison_figure.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

plt.close()
