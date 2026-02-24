#!/usr/bin/env python3
"""Convert variant JSON to SQLite for memory-efficient queries.

The original pgs_pairwise_variants.json.gz file (4.4 GB compressed, 10-20+ GB
decompressed) crashes servers when loaded entirely into memory. This script
converts it to a SQLite database that enables efficient single-pair lookups.

Usage:
    python scripts/convert_to_sqlite.py

Requires ~20 GB RAM to run (loads full JSON once during conversion).
Output database will be ~10-15 GB.
"""

import json
import gzip
import sqlite3
import sys
from pathlib import Path


def convert_to_sqlite(input_path: Path, output_path: Path) -> None:
    """Convert nested JSON to SQLite database.

    Args:
        input_path: Path to gzipped JSON file with variant data
        output_path: Path for output SQLite database
    """
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading {input_path}...")
    print("(This may take several minutes and requires ~20 GB RAM)")
    with gzip.open(input_path, "rt") as f:
        all_variants = json.load(f)

    print(f"Loaded {len(all_variants):,} pairs")

    # Remove existing database if present
    if output_path.exists():
        output_path.unlink()

    print(f"Creating {output_path}...")
    conn = sqlite3.connect(output_path)

    # Create table
    conn.execute("""
        CREATE TABLE variants (
            pair_key TEXT PRIMARY KEY,
            data TEXT
        )
    """)

    # Optimize for bulk insert
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print(f"Inserting {len(all_variants):,} pairs...")
    conn.executemany(
        "INSERT INTO variants (pair_key, data) VALUES (?, ?)",
        ((k, json.dumps(v)) for k, v in all_variants.items())
    )

    conn.commit()

    # Create index after bulk insert (faster than indexing during insert)
    print("Creating index...")
    conn.execute("CREATE INDEX idx_pair ON variants(pair_key)")
    conn.commit()

    # Compact the database
    print("Optimizing database...")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.close()

    size_gb = output_path.stat().st_size / 1e9
    print(f"Done! Database size: {size_gb:.2f} GB")


if __name__ == "__main__":
    # Default paths relative to project root
    project_root = Path(__file__).parent.parent

    input_path = project_root / "data" / "pgs_comparison_data" / "output" / "pgs_pairwise_variants.json.gz"
    output_path = project_root / "data" / "pgs_pairwise_variants.db"

    # Check alternate input location (if file was placed directly in data/)
    if not input_path.exists():
        alt_input = project_root / "data" / "pgs_pairwise_variants.json.gz"
        if alt_input.exists():
            input_path = alt_input

    convert_to_sqlite(input_path, output_path)
