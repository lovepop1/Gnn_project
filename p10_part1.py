"""
p10_part1.py — Steps 0–4:
  Validate files, load results, extract values from txt files,
  compute derived stats, print structured P10_OUTPUT block,
  save intermediate data to p10_data.json for part2.
"""

import matplotlib
matplotlib.use('Agg')
import json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── helpers ────────────────────────────────────────────────────────────────

def path(p):
    return os.path.join(ROOT, p)

def fmt(v, decimals=4):
    if v == 'N/A' or v is None:
        return 'N/A'
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)

def pct(v, decimals=2):
    if v == 'N/A' or v is None:
        return 'N/A'
    try:
        return f"{float(v)*100:.{decimals}f}%"
    except Exception:
        return str(v)

def extract_float(text, *labels, fallback='N/A'):
    """Return first float found after any of the given label substrings."""
    for label in labels:
        esc = re.escape(label)
        m = re.search(esc + r'\s*[=:\s]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return fallback

# ── Step 0: file validation ────────────────────────────────────────────────

def resolve(p):
    """Accept .png or .jpeg/.jpg variants."""
    full = path(p)
    if os.path.exists(full):
        return full
    base, ext = os.path.splitext(full)
    for alt in ['.jpeg', '.jpg', '.png']:
        if alt != ext and os.path.exists(base + alt):
            return base + alt
    return full   # return original so missing-file message is clear

REQUIRED = [
    "amazon/results.json", "amazon/p1_output.txt", "amazon/p2_output.txt", "amazon/p3_output.txt",
    "amazon/amazon_all_curves.png", "amazon/amazon_oversmoothing.png", "amazon/amazon_tsne.png",
    "airports/results.json", "airports/p1_output.txt", "airports/p2_output.txt", "airports/p3_output.txt",
    "airports/airports_ordinal_curves.png", "airports/airports_cls_curves.png",
    "airports/airports_oversmoothing_both.png", "airports/airports_tsne.png", "airports/airports_error_scatter.png",
    "molhiv/results.json", "molhiv/p1_output.txt", "molhiv/p2_output.txt", "molhiv/p3_output.txt",
    "molhiv/molhiv_all_curves.png", "molhiv/molhiv_depth_all_models.png",
    "molhiv/molhiv_tsne_all_models.png", "molhiv/molhiv_efficiency.png",
]

missing = []
for rel in REQUIRED:
    resolved = resolve(rel)
    if not os.path.exists(resolved):
        missing.append(rel)

if missing:
    print("MISSING FILES — cannot continue:")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)

print("Step 0: All required files found.\n")

# ── Step 1: load results ───────────────────────────────────────────────────

with open(path("amazon/results.json"))   as f: amazon_r   = json.load(f).get("amazon",   {})
with open(path("airports/results.json")) as f: airports_r = json.load(f).get("airports", {})
with open(path("molhiv/results.json"))   as f: molhiv_r   = json.load(f).get("molhiv",   {})

# ── Step 2: load txt files ─────────────────────────────────────────────────

def read(p):
    with open(path(p), encoding='utf-8', errors='replace') as f:
        return f.read()

a_p1 = read("amazon/p1_output.txt")
a_p2 = read("amazon/p2_output.txt")
a_p3 = read("amazon/p3_output.txt")
ap_p1 = read("airports/p1_output.txt")
ap_p2 = read("airports/p2_output.txt")
ap_p3 = read("airports/p3_output.txt")
m_p1 = read("molhiv/p1_output.txt")
m_p2 = read("molhiv/p2_output.txt")
m_p3 = read("molhiv/p3_output.txt")

# Amazon extractions
feat_sparsity = extract_float(a_p1, 'Sparsity', 'sparsity')
# peak depth from p3: "Peak at depth X"
peak_depth_m = re.search(r'[Pp]eak\s+at\s+depth\s+(\d+)', a_p3)
peak_depth = int(peak_depth_m.group(1)) if peak_depth_m else 'N/A'

# Airports extractions
spearman_corr = extract_float(ap_p1, 'corr=', 'corr')
spearman_pval = extract_float(ap_p1, 'pval=', 'pval')
# catastrophic errors from p3
cat_cls_m  = re.search(r'[Cc]lassification[:\s]+(\d+\.?\d*)%', ap_p3)
cat_ord_m  = re.search(r'[Oo]rdinal[:\s]+(\d+\.?\d*)%', ap_p3)
cat_red_m  = re.search(r'[Rr]eduction[:\s]+(\d+\.?\d*)\s*pp', ap_p3)
cat_cls  = float(cat_cls_m.group(1))  if cat_cls_m  else 'N/A'
cat_ord  = float(cat_ord_m.group(1))  if cat_ord_m  else 'N/A'
cat_red  = float(cat_red_m.group(1))  if cat_red_m  else 'N/A'

# MOLHIV extractions
# pos_weight line: "pos_weight = num_negatives / num_positives = 25.71"
# grab the LAST float on that line
pw_m = re.search(r'pos_weight[^\n]*=\s*([\d.]+)\s*\n', m_p1)
pos_weight = float(pw_m.group(1)) if pw_m else extract_float(m_p1, 'pos_weight')

# Ablation table from p3 — rows: GIN, GAT, GraphSAGE
abl = {}
for model in ['GIN', 'GAT', 'GraphSAGE']:
    pat = re.compile(
        re.escape(model) +
        r'\s*\|\s*([\d.]+)(?:±[\d.]+)?\s*\|\s*([\d.]+)(?:±[\d.]+)?\s*\|\s*([\d.]+)(?:±[\d.]+)?'
    )
    m2 = pat.search(m_p3)
    if m2:
        abl[model] = {
            'full':   float(m2.group(1)),
            'atoms':  float(m2.group(2)),
            'struct': float(m2.group(3)),
        }
    else:
        abl[model] = {'full': 'N/A', 'atoms': 'N/A', 'struct': 'N/A'}

# Depth sweep — best depth per model: pick depth with max val ROC per model column
# Table format: |  2  | 0.8225 | 0.7784 | 0.8008 | (or with ±)
depth_rows = re.findall(r'\|\s*(\d+)\s*\|\s*([\d.]+)(?:±[\d.]+)?\s*\|\s*([\d.]+)(?:±[\d.]+)?\s*\|\s*([\d.]+)(?:±[\d.]+)?\s*\|', m_p3)
best_depths = {'GIN': 'N/A', 'GAT': 'N/A', 'GraphSAGE': 'N/A'}
if depth_rows:
    cols = {'GIN': 1, 'GAT': 2, 'GraphSAGE': 3}
    for mod, col in cols.items():
        best_val = -1
        for row in depth_rows:
            try:
                v = float(row[col])
                dep = int(row[0])
                if v > best_val:
                    best_val = v
                    best_depths[mod] = dep
            except Exception:
                pass

# Misclassification rates
mis = {}
for model, label in [('GAT', 'GAT'), ('GraphSAGE', 'GraphSAGE'), ('GIN', 'GIN')]:
    pat = re.compile(
        re.escape(label) +
        r'\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%'
    )
    m3 = pat.search(m_p3)
    if m3:
        mis[model] = {
            'active_correct': float(m3.group(1)),
            'active_mis':     float(m3.group(2)),
            'inactive_mis':   float(m3.group(3)),
        }
    else:
        mis[model] = {'active_correct': 'N/A', 'active_mis': 'N/A', 'inactive_mis': 'N/A'}

# Trainable params — efficiency table: "| GIN | 1445701 | 0.7503 | 11.82 |"
params = {}
for model, label in [('GIN', 'GIN'), ('GAT', 'GAT'), ('GraphSAGE', 'GraphSAGE')]:
    # match label then first column of digits (must be >= 5 digits to be a param count)
    pat = re.compile(r'\|\s*' + re.escape(label) + r'\s*\|\s*(\d{5,})\s*\|')
    m4 = pat.search(m_p3)
    params[model] = int(m4.group(1)) if m4 else 'N/A'

# t-SNE sample size — look for balanced or sample in p3 (not present in files → N/A is fine)
tsne_m = re.search(r'(?:balanced\s+sample\s+size|t-SNE.*?sample)[^\d]*(\d{3,6})', m_p3, re.IGNORECASE)
if not tsne_m:
    tsne_m = re.search(r'sample.*?(\d{3,6})', m_p3, re.IGNORECASE)
tsne_sample = int(tsne_m.group(1)) if tsne_m else 'N/A'

# ── Step 3: compute derived values ────────────────────────────────────────

# Airports kappa deltas (ordinal - cls)
# results.json keys: GAT_ordinal, GAT_cls, GraphSAGE_ordinal, GraphSAGE_cls, GIN_ordinal, GIN_cls
def get_kappa(key):
    return airports_r.get(key, {}).get('kappa', 'N/A')

GAT_ord_k  = get_kappa('GAT_ordinal')
GAT_cls_k  = get_kappa('GAT_cls')
SAGE_ord_k = get_kappa('GraphSAGE_ordinal')
SAGE_cls_k = get_kappa('GraphSAGE_cls')
GIN_ord_k  = get_kappa('GIN_ordinal')
GIN_cls_k  = get_kappa('GIN_cls')

def delta(a, b):
    if a == 'N/A' or b == 'N/A':
        return 'N/A'
    return round(float(a) - float(b), 8)

GAT_kd  = delta(GAT_ord_k,  GAT_cls_k)
SAGE_kd = delta(SAGE_ord_k, SAGE_cls_k)
GIN_kd  = delta(GIN_ord_k,  GIN_cls_k)

valid_deltas = [d for d in [GAT_kd, SAGE_kd, GIN_kd] if d != 'N/A']
mean_kd = round(float(np.mean(valid_deltas)), 8) if valid_deltas else 'N/A'

# Amazon: best model by test_acc among GAT, SAGE, GIN
amazon_accs = {
    'GAT':  amazon_r.get('GAT',  {}).get('test_acc', 0),
    'SAGE': amazon_r.get('SAGE', {}).get('test_acc', 0),
    'GIN':  amazon_r.get('GIN',  {}).get('test_acc', 0),
}
amazon_best = max(amazon_accs, key=amazon_accs.get)
amazon_best_acc = amazon_accs[amazon_best]

# Amazon ranks (1=best)
sorted_amazon = sorted(amazon_accs, key=amazon_accs.get, reverse=True)
amazon_rank = {m: sorted_amazon.index(m)+1 for m in sorted_amazon}

# GNN gain over MLP
mlp_acc = amazon_r.get('MLP', {}).get('test_acc', 0)
best_gnn_acc = amazon_best_acc
gnn_gain = round((best_gnn_acc - mlp_acc) * 100, 4)

# Airports: best ordinal model by kappa
airports_ord_kappas = {
    'GAT':  float(GAT_ord_k)  if GAT_ord_k  != 'N/A' else 0,
    'SAGE': float(SAGE_ord_k) if SAGE_ord_k != 'N/A' else 0,
    'GIN':  float(GIN_ord_k)  if GIN_ord_k  != 'N/A' else 0,
}
airports_best = max(airports_ord_kappas, key=airports_ord_kappas.get)
airports_best_k = airports_ord_kappas[airports_best]
sorted_airports = sorted(airports_ord_kappas, key=airports_ord_kappas.get, reverse=True)
airports_rank = {m: sorted_airports.index(m)+1 for m in sorted_airports}

# MOLHIV: best model by test_roc (keys: GIN, GAT, GraphSAGE)
molhiv_rocs = {
    'GIN':  molhiv_r.get('GIN',        {}).get('test_roc', 0),
    'GAT':  molhiv_r.get('GAT',        {}).get('test_roc', 0),
    'SAGE': molhiv_r.get('GraphSAGE',  {}).get('test_roc', 0),
}
molhiv_best = max(molhiv_rocs, key=molhiv_rocs.get)
molhiv_best_roc = molhiv_rocs[molhiv_best]
sorted_molhiv = sorted(molhiv_rocs, key=molhiv_rocs.get, reverse=True)
molhiv_rank = {m: sorted_molhiv.index(m)+1 for m in sorted_molhiv}

# SAGE bond gap: best_roc - SAGE_roc
sage_bond_gap = round(molhiv_best_roc - molhiv_rocs['SAGE'], 6)

# Cross-dataset rank sums
rank_sum = {}
for m in ['GAT', 'SAGE', 'GIN']:
    rank_sum[m] = amazon_rank[m] + airports_rank[m] + molhiv_rank[m]
best_overall = min(rank_sum, key=rank_sum.get)

# ── Step 4: print structured output ───────────────────────────────────────

def sgn(v):
    if v == 'N/A':
        return 'N/A'
    return f"{float(v):+.4f}"

def _cfg(d, key):
    c = d.get(key, {}).get('best_cfg', 'N/A')
    if isinstance(c, dict):
        return ','.join(f"{k}={v}" for k, v in c.items())
    return str(c)

def _t(d, key):
    t = d.get(key, {}).get('time', 'N/A')
    if t == 'N/A':
        return 'N/A'
    return f"{float(t):.1f}s"

print("P10_OUTPUT_START")
print()

# AMAZON
amz = amazon_r
print(f"[AMAZON] feature_sparsity: {fmt(feat_sparsity, 2) if feat_sparsity != 'N/A' else 'N/A'}"
      + (f" ({float(feat_sparsity)*100:.2f}%)" if feat_sparsity != 'N/A' else ""))
print(f"[AMAZON] MLP_test_acc: {float(amz.get('MLP',{}).get('test_acc',0))*100:.2f}%")
for m in ['GAT', 'SAGE', 'GIN']:
    d = amz.get(m, {})
    print(f"[AMAZON] {m}_test_acc: {float(d.get('test_acc',0))*100:.2f}±{float(d.get('test_acc_std',0))*100:.2f}%  "
          f"{m}_f1: {fmt(d.get('f1','N/A'), 3)}±{fmt(d.get('f1_std',0), 3)}  "
          f"{m}_time: {_t(amz, m)}±{fmt(d.get('time_std',0), 1)}s  "
          f"{m}_cfg: {_cfg(amz, m)}")
print(f"[AMAZON] best_model: {amazon_best}  best_acc: {amazon_best_acc*100:.2f}%")
print(f"[AMAZON] gnn_gain_over_mlp: {gnn_gain:.2f} pp")
print(f"[AMAZON] peak_oversmoothing_depth: {peak_depth}")
print(f"[AMAZON] amazon_rank_GAT: {amazon_rank['GAT']}  amazon_rank_SAGE: {amazon_rank['SAGE']}  amazon_rank_GIN: {amazon_rank['GIN']}")
print()

# AIRPORTS
ap = airports_r
dh = ap.get('degree_heuristic', {})
print(f"[AIRPORTS] spearman_corr: {fmt(spearman_corr, 4)}  spearman_pval: {spearman_pval}")
print(f"[AIRPORTS] degree_heuristic_kappa: {fmt(dh.get('kappa','N/A'), 4)}")
print(f"[AIRPORTS] GAT_ordinal_kappa: {fmt(GAT_ord_k,4)}  GAT_cls_kappa: {fmt(GAT_cls_k,4)}  GAT_kappa_delta: {sgn(GAT_kd)}")
print(f"[AIRPORTS] SAGE_ordinal_kappa: {fmt(SAGE_ord_k,4)}  SAGE_cls_kappa: {fmt(SAGE_cls_k,4)}  SAGE_kappa_delta: {sgn(SAGE_kd)}")
print(f"[AIRPORTS] GIN_ordinal_kappa: {fmt(GIN_ord_k,4)}  GIN_cls_kappa: {fmt(GIN_cls_k,4)}  GIN_kappa_delta: {sgn(GIN_kd)}")
print(f"[AIRPORTS] mean_kappa_delta: {sgn(mean_kd)}")
print(f"[AIRPORTS] catastrophic_err_cls: {cat_cls}%  catastrophic_err_ord: {cat_ord}%  catastrophic_reduction: {cat_red} pp")

for mk, rk in [('GAT','GAT_ordinal'), ('SAGE','GraphSAGE_ordinal'), ('GIN','GIN_ordinal')]:
    d = ap.get(rk, {})
    print(f"[AIRPORTS] {mk}_ordinal_full: mse={fmt(d.get('mse','N/A'),3)}±{fmt(d.get('mse_std',0),3)} mae={fmt(d.get('mae','N/A'),3)}±{fmt(d.get('mae_std',0),3)} "
          f"acc={fmt(d.get('acc','N/A'),3)}±{fmt(d.get('acc_std',0),3)} kappa={fmt(d.get('kappa','N/A'),4)}±{fmt(d.get('kappa_std',0),4)} f1={fmt(d.get('f1','N/A'),3)}±{fmt(d.get('f1_std',0),3)} "
          f"time={round(float(d.get('time',0)))}s")

for mk, ck in [('GAT','GAT_cls'), ('SAGE','GraphSAGE_cls'), ('GIN','GIN_cls')]:
    d = ap.get(ck, {})
    print(f"[AIRPORTS] {mk}_cls_full: mse={fmt(d.get('mse','N/A'),3)}±{fmt(d.get('mse_std',0),3)} mae={fmt(d.get('mae','N/A'),3)}±{fmt(d.get('mae_std',0),3)} "
          f"acc={fmt(d.get('acc','N/A'),3)}±{fmt(d.get('acc_std',0),3)} kappa={fmt(d.get('kappa','N/A'),4)}±{fmt(d.get('kappa_std',0),4)} f1={fmt(d.get('f1','N/A'),3)}±{fmt(d.get('f1_std',0),3)}")

print(f"[AIRPORTS] best_ordinal_model: {airports_best}  best_kappa: {airports_best_k:.4f}")
print(f"[AIRPORTS] airports_rank_GAT: {airports_rank['GAT']}  airports_rank_SAGE: {airports_rank['SAGE']}  airports_rank_GIN: {airports_rank['GIN']}")
print()

# MOLHIV
mv = molhiv_r
print(f"[MOLHIV] pos_weight: {fmt(pos_weight, 2)}")
for mk, rk in [('GIN','GIN'), ('GAT','GAT'), ('SAGE','GraphSAGE')]:
    d = mv.get(rk, {})
    cfg = d.get('best_cfg', {})
    if isinstance(cfg, dict):
        cfg_s = ','.join(f"{k}={v}" for k, v in cfg.items())
    else:
        cfg_s = str(cfg)
    print(f"[MOLHIV] {mk}_test_roc: {fmt(d.get('test_roc','N/A'),4)}±{fmt(d.get('test_roc_std',0),4)}  "
          f"{mk}_prec: {fmt(d.get('precision','N/A'),3)}±{fmt(d.get('precision_std',0),3)}  "
          f"{mk}_rec: {fmt(d.get('recall','N/A'),3)}±{fmt(d.get('recall_std',0),3)}  "
          f"{mk}_f1: {fmt(d.get('f1','N/A'),3)}±{fmt(d.get('f1_std',0),3)}  "
          f"{mk}_time: {round(float(d.get('time',0)))}s  "
          f"{mk}_cfg: {cfg_s}")

print(f"[MOLHIV] best_model: {molhiv_best}  best_roc: {molhiv_best_roc:.4f}")
print(f"[MOLHIV] sage_bond_gap: {sage_bond_gap:.4f}")

for mk, ak in [('GIN','GIN'), ('GAT','GAT'), ('SAGE','GraphSAGE')]:
    ab = abl.get(ak, {})
    print(f"[MOLHIV] {mk}_ablation: full={fmt(ab.get('full','N/A'),4)} "
          f"atoms_only={fmt(ab.get('atoms','N/A'),4)} "
          f"struct_only={fmt(ab.get('struct','N/A'),4)} "
          f"best_depth={best_depths.get(ak,'N/A')}")

for mk, ak in [('GIN','GIN'), ('GAT','GAT'), ('SAGE','GraphSAGE')]:
    ms = mis.get(ak, {})
    ac = ms.get('active_correct', 'N/A')
    am = ms.get('active_mis', 'N/A')
    im = ms.get('inactive_mis', 'N/A')
    print(f"[MOLHIV] {mk}_misclass: active_correct={ac}%  active_mis={am}%  inactive_mis={im}%")

gin_p  = params.get('GIN',        'N/A')
gat_p  = params.get('GAT',        'N/A')
sage_p = params.get('GraphSAGE',  'N/A')
print(f"[MOLHIV] GIN_params: {gin_p}  GAT_params: {gat_p}  SAGE_params: {sage_p}")
print(f"[MOLHIV] tsne_sample_size: {tsne_sample}")
print(f"[MOLHIV] molhiv_rank_GIN: {molhiv_rank['GIN']}  molhiv_rank_GAT: {molhiv_rank['GAT']}  molhiv_rank_SAGE: {molhiv_rank['SAGE']}")
print()

# CROSS
print(f"[CROSS] GAT_rank_sum: {rank_sum['GAT']}  SAGE_rank_sum: {rank_sum['SAGE']}  GIN_rank_sum: {rank_sum['GIN']}")
print(f"[CROSS] best_overall_model: {best_overall}  best_overall_rank_sum: {rank_sum[best_overall]}")
print()
print("P10_OUTPUT_END")

# ── Save intermediate data for part2 ──────────────────────────────────────

p10_data = {
    # Amazon
    'amazon_accs': amazon_accs,
    'amazon_accs_stds': {m: float(amazon_r.get(m, {}).get('test_acc_std', 0)) for m in ['GAT','SAGE','GIN']},
    'amazon_f1':   {m: amazon_r.get(m,{}).get('f1','N/A') for m in ['GAT','SAGE','GIN']},
    'amazon_time': {m: amazon_r.get(m,{}).get('time','N/A') for m in ['GAT','SAGE','GIN']},
    'amazon_rank': amazon_rank,
    # Airports ordinal kappas
    'airports_ord_kappas': airports_ord_kappas,
    'airports_ord_kappas_stds': {
        'GAT':  float(airports_r.get('GAT_ordinal', {}).get('kappa_std', 0)),
        'SAGE': float(airports_r.get('GraphSAGE_ordinal', {}).get('kappa_std', 0)),
        'GIN':  float(airports_r.get('GIN_ordinal', {}).get('kappa_std', 0)),
    },
    'airports_rank': airports_rank,
    'airports_ord_times': {
        'GAT':  airports_r.get('GAT_ordinal',{}).get('time','N/A'),
        'SAGE': airports_r.get('GraphSAGE_ordinal',{}).get('time','N/A'),
        'GIN':  airports_r.get('GIN_ordinal',{}).get('time','N/A'),
    },
    # MOLHIV
    'molhiv_rocs': molhiv_rocs,
    'molhiv_rocs_stds': {
        'GIN':  float(molhiv_r.get('GIN', {}).get('test_roc_std', 0)),
        'GAT':  float(molhiv_r.get('GAT', {}).get('test_roc_std', 0)),
        'SAGE': float(molhiv_r.get('GraphSAGE', {}).get('test_roc_std', 0)),
    },
    'molhiv_rank': molhiv_rank,
    'molhiv_times': {
        'GIN':  molhiv_r.get('GIN',{}).get('time','N/A'),
        'GAT':  molhiv_r.get('GAT',{}).get('time','N/A'),
        'SAGE': molhiv_r.get('GraphSAGE',{}).get('time','N/A'),
    },
    # Cross
    'rank_sum': rank_sum,
    'amazon_ranks_list':   [amazon_rank['GAT'],  amazon_rank['SAGE'],  amazon_rank['GIN']],
    'airports_ranks_list': [airports_rank['GAT'], airports_rank['SAGE'], airports_rank['GIN']],
    'molhiv_ranks_list':   [molhiv_rank['GAT'],  molhiv_rank['SAGE'],  molhiv_rank['GIN']],
}

out_path = os.path.join(ROOT, 'p10_data.json')
with open(out_path, 'w') as f:
    json.dump(p10_data, f, indent=2, default=str)
print(f"\nIntermediate data saved to: {out_path}")
print("Run p10_part2.py next to generate figures.")
