"""Generate sample comparison data for testing the Compare tab."""
import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path

np.random.seed(42)

Path("data").mkdir(exist_ok=True)

pairs = []
variant_data = {}

# Trait 1: Type 2 Diabetes - 5 PRS, mix of correlations
t2d_scores = {
    "PGS000001": 150000,
    "PGS000045": 180000,
    "PGS000089": 50000,
    "PGS000120": 1200000,
    "PGS000200": 800,
}
t2d_pairs = [
    ("PGS000001", "PGS000045", 125000, 0.96, 94.2),  # redundant pair (r > 0.95)
    ("PGS000001", "PGS000089", 35000, 0.72, 88.5),    # strong
    ("PGS000001", "PGS000120", 148000, 0.45, 76.1),   # moderate
    ("PGS000001", "PGS000200", 0, 0.0, 0.0),          # zero overlap
    ("PGS000045", "PGS000089", 28000, 0.68, 85.3),    # moderate-strong
    ("PGS000045", "PGS000120", 170000, 0.41, 74.8),   # moderate
    ("PGS000045", "PGS000200", 500, 0.12, 62.0),      # weak
    ("PGS000089", "PGS000120", 49000, 0.55, 80.2),    # moderate
    ("PGS000089", "PGS000200", 200, 0.08, 58.0),      # weak
    ("PGS000120", "PGS000200", 780, -0.15, 45.0),     # negative correlation
]

for p1, p2, n_shared, r, conc in t2d_pairs:
    n1 = t2d_scores[p1]
    n2 = t2d_scores[p2]
    o1 = (n_shared / n1 * 100) if n1 > 0 else 0
    o2 = (n_shared / n2 * 100) if n2 > 0 else 0
    pairs.append({
        "pgs_id_1": p1, "pgs_id_2": p2,
        "trait_efo": "EFO_0001360", "trait_label": "Type 2 diabetes",
        "n_variants_1": n1, "n_variants_2": n2,
        "n_shared": n_shared, "pct_overlap_1": round(o1, 1), "pct_overlap_2": round(o2, 1),
        "pearson_r": r, "pearson_p": max(1e-300, 10 ** (-abs(r) * 50)),
        "pct_concordant_sign": conc, "n_sampled": min(n_shared, 10000),
    })

# Trait 2: Coronary Artery Disease - 4 PRS
cad_scores = {
    "PGS000002": 200000,
    "PGS000003": 300000,
    "PGS000050": 6000,
    "PGS000078": 95000,
}
cad_pairs = [
    ("PGS000002", "PGS000003", 180000, 0.87, 91.0),  # strong
    ("PGS000002", "PGS000050", 5500, 0.35, 72.0),     # moderate
    ("PGS000002", "PGS000078", 88000, 0.78, 89.5),    # strong
    ("PGS000003", "PGS000050", 5800, 0.28, 68.0),     # weak
    ("PGS000003", "PGS000078", 90000, 0.82, 90.1),    # strong
    ("PGS000050", "PGS000078", 4200, 0.22, 65.0),     # weak
]

for p1, p2, n_shared, r, conc in cad_pairs:
    n1 = cad_scores[p1]
    n2 = cad_scores[p2]
    o1 = (n_shared / n1 * 100) if n1 > 0 else 0
    o2 = (n_shared / n2 * 100) if n2 > 0 else 0
    pairs.append({
        "pgs_id_1": p1, "pgs_id_2": p2,
        "trait_efo": "EFO_0001645", "trait_label": "Coronary artery disease",
        "n_variants_1": n1, "n_variants_2": n2,
        "n_shared": n_shared, "pct_overlap_1": round(o1, 1), "pct_overlap_2": round(o2, 1),
        "pearson_r": r, "pearson_p": max(1e-300, 10 ** (-abs(r) * 50)),
        "pct_concordant_sign": conc, "n_sampled": min(n_shared, 10000),
    })

# Trait 3: Breast Cancer - 3 PRS
bc_scores = {
    "PGS000004": 70000,
    "PGS000015": 120000,
    "PGS000033": 45000,
}
bc_pairs = [
    ("PGS000004", "PGS000015", 55000, 0.62, 82.0),  # moderate-strong
    ("PGS000004", "PGS000033", 30000, 0.48, 77.5),   # moderate
    ("PGS000015", "PGS000033", 40000, 0.71, 86.8),   # strong
]

for p1, p2, n_shared, r, conc in bc_pairs:
    n1 = bc_scores[p1]
    n2 = bc_scores[p2]
    o1 = (n_shared / n1 * 100) if n1 > 0 else 0
    o2 = (n_shared / n2 * 100) if n2 > 0 else 0
    pairs.append({
        "pgs_id_1": p1, "pgs_id_2": p2,
        "trait_efo": "EFO_0000305", "trait_label": "Breast cancer",
        "n_variants_1": n1, "n_variants_2": n2,
        "n_shared": n_shared, "pct_overlap_1": round(o1, 1), "pct_overlap_2": round(o2, 1),
        "pearson_r": r, "pearson_p": max(1e-300, 10 ** (-abs(r) * 50)),
        "pct_concordant_sign": conc, "n_sampled": min(n_shared, 10000),
    })

stats_df = pd.DataFrame(pairs)
stats_df.to_parquet("data/pgs_pairwise_stats.parquet", index=False)
print(f"Created pgs_pairwise_stats.parquet with {len(stats_df)} pairs")

# Generate variant-level data for scatterplots
def generate_variants(n_shared, r, n_sample=None):
    if n_shared == 0:
        return []
    n = min(n_shared, n_sample or 5000)
    w1 = np.random.normal(0, 0.05, n)
    noise = np.random.normal(0, 0.05 * np.sqrt(max(0.01, 1 - r**2)), n)
    w2 = r * w1 + noise
    variants = []
    for i in range(n):
        variants.append({
            "variant_id": f"rs{np.random.randint(1000, 99999999)}",
            "weight_1": round(float(w1[i]), 6),
            "weight_2": round(float(w2[i]), 6),
        })
    return variants

all_variant_data = {}
for pair in pairs:
    if pair["n_shared"] > 0:
        key = f"{pair['pgs_id_1']}_{pair['pgs_id_2']}"
        n_sample = min(pair["n_shared"], 5000)
        all_variant_data[key] = generate_variants(
            pair["n_shared"], pair["pearson_r"], n_sample
        )

with gzip.open("data/pgs_pairwise_variants.json.gz", "wt") as f:
    json.dump(all_variant_data, f)
print(f"Created pgs_pairwise_variants.json.gz with {len(all_variant_data)} pairs")

metadata = {
    "generated_date": "2026-02-20",
    "genome_build": "GRCh38",
    "pipeline_version": "0.1.0",
    "n_scores_compared": 12,
    "n_traits": 3,
    "n_pairs": len(pairs),
    "sampling_note": "Variant data sampled to max 5,000 per pair for performance",
}
with open("data/pipeline_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Created pipeline_metadata.json")
print("Sample data generation complete!")
