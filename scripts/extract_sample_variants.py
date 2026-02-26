#!/usr/bin/env python3
"""Extract sample variant pairs using streaming (memory-safe)."""

import gzip
import json
from decimal import Decimal
import ijson
import pandas as pd
from pathlib import Path


def convert_decimals(obj):
    """Recursively convert Decimal objects to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    return obj


def select_representative_pairs(stats_path: Path, n_pairs: int = 50) -> list[str]:
    """Select diverse pairs: high/moderate/low correlation across traits."""
    stats = pd.read_parquet(stats_path)

    pairs = []
    # High correlation (r > 0.9)
    high = stats[stats["pearson_r"] > 0.9].nlargest(15, "n_shared")
    pairs.extend(f"{r['pgs_id_1']}_{r['pgs_id_2']}" for _, r in high.iterrows())

    # Moderate correlation (0.3 < r < 0.7)
    mod = stats[(stats["pearson_r"] > 0.3) & (stats["pearson_r"] < 0.7)].nlargest(15, "n_shared")
    pairs.extend(f"{r['pgs_id_1']}_{r['pgs_id_2']}" for _, r in mod.iterrows())

    # Low/negative correlation (r < 0.3)
    low = stats[stats["pearson_r"] < 0.3].nlargest(20, "n_shared")
    pairs.extend(f"{r['pgs_id_1']}_{r['pgs_id_2']}" for _, r in low.iterrows())

    return pairs[:n_pairs]


def extract_pairs(input_path: Path, output_path: Path, pair_keys: list[str], max_variants: int = 1000):
    """Stream through 4.6GB file, extract only needed pairs."""
    sample = {}
    needed = set(pair_keys)

    print(f"Extracting {len(needed)} pairs from {input_path.name}...")
    print(f"This may take 10-20 minutes (streaming through 4.6GB file)...")

    with gzip.open(input_path, "rb") as f:
        for key, variants in ijson.kvitems(f, ""):
            if key in needed:
                sample[key] = variants[:max_variants]  # Cap variants
                print(f"  {key}: {len(sample[key])} variants")
                needed.remove(key)
                if not needed:
                    break

    # Convert Decimal objects to float for JSON serialization
    sample = convert_decimals(sample)

    with gzip.open(output_path, "wt") as f:
        json.dump(sample, f)

    print(f"\nSaved {len(sample)} pairs to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    stats_path = Path("data/pgs_comparison_data/output/pgs_pairwise_stats.parquet")
    variants_path = Path("data/pgs_comparison_data/output/pgs_pairwise_variants.json.gz")
    output_path = Path("data/pgs_pairwise_variants_sample.json.gz")

    pair_keys = select_representative_pairs(stats_path, n_pairs=50)
    extract_pairs(variants_path, output_path, pair_keys)
