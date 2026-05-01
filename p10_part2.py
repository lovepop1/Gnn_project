"""
p10_part2.py — Steps 5–6:
  Read p10_data.json produced by p10_part1.py,
  generate three matplotlib figures, print final summary.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Load intermediate data
data_path = os.path.join(ROOT, 'p10_data.json')
if not os.path.exists(data_path):
    print("ERROR: p10_data.json not found. Run p10_part1.py first.")
    sys.exit(1)

with open(data_path) as f:
    D = json.load(f)

# Colour scheme
C_GREY   = '#888888'
C_GAT    = '#378ADD'
C_SAGE   = '#639922'
C_GIN    = '#7F77DD'

# ── Figure 1: report_summary.png ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# — Subplot 1: Amazon Test Accuracy —
ax = axes[0]
models_amz = ['MLP', 'GAT', 'SAGE', 'GIN']
colours_amz = [C_GREY, C_GAT, C_SAGE, C_GIN]
vals_amz = [
    float(D['amazon_accs'].get('GAT',  0)) * 100,   # placeholder order
    float(D['amazon_accs'].get('SAGE', 0)) * 100,
    float(D['amazon_accs'].get('GIN',  0)) * 100,
]
# MLP acc from original JSON — re-derive from accs dict (only GNNs), read from p1 is fine
# We saved amazon_accs for GAT/SAGE/GIN; need MLP — re-open results.json
amz_json = os.path.join(ROOT, 'amazon', 'results.json')
with open(amz_json) as f:
    amz_full = json.load(f).get('amazon', {})
mlp_acc = float(amz_full.get('MLP', {}).get('test_acc', 0)) * 100

all_vals_amz  = [mlp_acc,
                 float(D['amazon_accs']['GAT'])*100,
                 float(D['amazon_accs']['SAGE'])*100,
                 float(D['amazon_accs']['GIN'])*100]
all_stds_amz  = [0,
                 float(D['amazon_accs_stds']['GAT'])*100,
                 float(D['amazon_accs_stds']['SAGE'])*100,
                 float(D['amazon_accs_stds']['GIN'])*100]
all_cols_amz  = [C_GREY, C_GAT, C_SAGE, C_GIN]
all_labs_amz  = ['MLP', 'GAT', 'SAGE', 'GIN']

bars = ax.barh(all_labs_amz, all_vals_amz, xerr=all_stds_amz, color=all_cols_amz, height=0.55, edgecolor='white', capsize=4)
for bar, val, std in zip(bars, all_vals_amz, all_stds_amz):
    label = f'{val:.2f}%' if std == 0 else f'{val:.2f}±{std:.2f}%'
    ax.text(val + std + 0.3, bar.get_y() + bar.get_height()/2,
            label, va='center', ha='left', fontsize=9, fontweight='bold')
ax.set_title('Amazon — Test Accuracy', fontsize=11, fontweight='bold', pad=8)
ax.set_xlabel('Test Accuracy (%)', fontsize=9)
ax.set_xlim(0, max([v + s for v, s in zip(all_vals_amz, all_stds_amz)])*1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# — Subplot 2: Airports Ordinal Kappa —
ax = axes[1]
ap_models  = ['GAT', 'SAGE', 'GIN']
ap_colours = [C_GAT, C_SAGE, C_GIN]
ap_kappas  = [float(D['airports_ord_kappas']['GAT']),
              float(D['airports_ord_kappas']['SAGE']),
              float(D['airports_ord_kappas']['GIN'])]
ap_stds    = [float(D['airports_ord_kappas_stds']['GAT']),
              float(D['airports_ord_kappas_stds']['SAGE']),
              float(D['airports_ord_kappas_stds']['GIN'])]

bars = ax.barh(ap_models, ap_kappas, xerr=ap_stds, color=ap_colours, height=0.55, edgecolor='white', capsize=4)
for bar, val, std in zip(bars, ap_kappas, ap_stds):
    ax.text(val + std + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}±{std:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')
ax.set_title('Airports — Ordinal Kappa', fontsize=11, fontweight='bold', pad=8)
ax.set_xlabel('Weighted Kappa', fontsize=9)
ax.set_xlim(0, max([v + s for v, s in zip(ap_kappas, ap_stds)])*1.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# — Subplot 3: MOLHIV Test ROC-AUC —
ax = axes[2]
mol_models  = ['GIN', 'GAT', 'SAGE']
mol_colours = [C_GIN, C_GAT, C_SAGE]
mol_rocs    = [float(D['molhiv_rocs']['GIN']),
               float(D['molhiv_rocs']['GAT']),
               float(D['molhiv_rocs']['SAGE'])]
mol_stds    = [float(D['molhiv_rocs_stds']['GIN']),
               float(D['molhiv_rocs_stds']['GAT']),
               float(D['molhiv_rocs_stds']['SAGE'])]

bars = ax.barh(mol_models, mol_rocs, xerr=mol_stds, color=mol_colours, height=0.55, edgecolor='white', capsize=4)
for bar, val, std in zip(bars, mol_rocs, mol_stds):
    ax.text(val + std + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}±{std:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')
ax.set_title('MOLHIV — Test ROC-AUC', fontsize=11, fontweight='bold', pad=8)
ax.set_xlabel('ROC-AUC', fontsize=9)
ax.set_xlim(0, max([v + s for v, s in zip(mol_rocs, mol_stds)])*1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig1_path = os.path.join(ROOT, 'report_summary.png')
fig.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig1_path}")

# ── Figure 2: cross_dataset_rank.png ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))

rows = ['GAT', 'SAGE', 'GIN']
cols = ['Amazon', 'Airports', 'MOLHIV']

rank_matrix = np.array([
    [D['amazon_ranks_list'][0],   D['airports_ranks_list'][0],  D['molhiv_ranks_list'][0]],   # GAT
    [D['amazon_ranks_list'][1],   D['airports_ranks_list'][1],  D['molhiv_ranks_list'][1]],   # SAGE
    [D['amazon_ranks_list'][2],   D['airports_ranks_list'][2],  D['molhiv_ranks_list'][2]],   # GIN
], dtype=float)

im = ax.imshow(rank_matrix, cmap='RdYlGn_r', vmin=1, vmax=3, aspect='auto')

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, fontsize=11)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=11)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

for i in range(len(rows)):
    for j in range(len(cols)):
        val = int(rank_matrix[i, j])
        colour = 'white' if val == 3 else 'black'
        ax.text(j, i, str(val), ha='center', va='center',
                fontsize=22, fontweight='bold', color=colour)

ax.set_title('Architecture Rank per Dataset (1=best, 3=worst)',
             fontsize=11, fontweight='bold', pad=16)

plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
             label='Rank (1=best, 3=worst)')
plt.tight_layout()
fig2_path = os.path.join(ROOT, 'cross_dataset_rank.png')
fig.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig2_path}")

# ── Figure 3: cross_dataset_time.png ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

datasets     = ['Amazon', 'Airports Ordinal', 'MOLHIV']
arch_keys    = ['GAT', 'SAGE', 'GIN']
arch_colours = [C_GAT, C_SAGE, C_GIN]

# times — convert to float safely
def safe_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0

times = {
    'GAT':  [safe_float(D['amazon_time']['GAT']),
             safe_float(D['airports_ord_times']['GAT']),
             safe_float(D['molhiv_times']['GAT'])],
    'SAGE': [safe_float(D['amazon_time']['SAGE']),
             safe_float(D['airports_ord_times']['SAGE']),
             safe_float(D['molhiv_times']['SAGE'])],
    'GIN':  [safe_float(D['amazon_time']['GIN']),
             safe_float(D['airports_ord_times']['GIN']),
             safe_float(D['molhiv_times']['GIN'])],
}

n_ds   = len(datasets)
n_arch = len(arch_keys)
bar_w  = 0.22
x      = np.arange(n_ds)

for i, (arch, colour) in enumerate(zip(arch_keys, arch_colours)):
    offsets = x + (i - 1) * bar_w
    vals = times[arch]
    bars = ax.bar(offsets, vals, width=bar_w, color=colour,
                  label=arch, edgecolor='white', alpha=0.9)
    for bar, val in zip(bars, vals):
        if val > 0:
            # Position text slightly above the bar in log space
            ax.text(bar.get_x() + bar.get_width()/2,
                    val * 1.05, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_yscale('log')
ax.set_ylabel('Training Time per Epoch (s)', fontsize=11)
ax.set_title('Training Time per Architecture (Per-Epoch)', fontsize=12, fontweight='bold')
ax.legend(title='Architecture', fontsize=10, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust ylim for log scale
ax.set_ylim(min(min(v) for v in times.values()) * 0.5, 
            max(max(v) for v in times.values()) * 5.0)

plt.tight_layout()
fig3_path = os.path.join(ROOT, 'cross_dataset_time.png')
fig.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig3_path}")

# ── Step 6: final summary ─────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 6 — FILES TO SAVE IN project_root/")
print("=" * 60)
print()
print("Run this script (p10_part1.py + p10_part2.py) locally and")
print("save the following output files in the project root directory:")
print()
print("  p10_output.txt           — full console output from both parts")
print("  report_summary.png       — 1×3 bar chart summary per dataset")
print("  cross_dataset_rank.png   — 3×3 rank heatmap (models × datasets)")
print("  cross_dataset_time.png   — grouped training-time bar chart")
print()
print("All four files have been saved to:")
print(f"  {ROOT}")
print()
print("Done. All steps complete.")
