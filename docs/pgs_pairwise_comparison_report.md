# PGS Pairwise Comparison Analysis

**For ARK Team / Data Scientists**
*Generated: February 2026*

---

## Motivation

When multiple polygenic scores (PGS) exist for the same trait, researchers face a critical selection decision: Which PGS should I use? This pipeline addresses that need by computing variant-level comparisons between all PGS pairs targeting the same trait in the PGS Catalog.

The approach responds to the challenge raised by Gwênlyn Glusman: understanding how to choose between alternative PGS for the same condition by examining the relationship between their effect weights for shared variants.

---

## Methodology

For each pair of PGS targeting the same trait:

1. **Identify shared variants** — Match variants by rsID or chr:pos coordinate (GRCh38)
2. **Compute Pearson correlation** — Assess linear relationship of effect weights
3. **Calculate sign concordance** — Percentage of shared variants where both scores agree on effect direction (same sign)
4. **Measure overlap percentages** — Jaccard-like metrics from each score's perspective

**Visualization approach:** Scatterplot with effect weights on x/y axes, points colored by sign concordance (green = concordant, red = discordant). A diagonal line (y=x) shows perfect agreement.

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| PGS Catalog scores analyzed | 5,042 |
| Traits with ≥2 PGS | 432 |
| Pairwise comparisons | 157,775 |
| Pairs with ≥1 shared variant | 129,474 (82.1%) |
| Genome build | GRCh38 |

---

## Correlation Distribution

Analysis of the 124,814 pairs with valid correlation values (at least one shared variant and computable Pearson r):

| Category | Count | % of Total | Interpretation |
|----------|------:|:----------:|----------------|
| Highly correlated (r > 0.9) | 6,461 | 4.1% | Effectively equivalent |
| Moderately correlated (0.5 < r ≤ 0.9) | 28,406 | 18.0% | Similar but distinct signals |
| Weakly correlated (0 < r ≤ 0.5) | 46,141 | 29.2% | Different approaches |
| Negatively correlated (r < 0) | 43,806 | 27.8% | Possible allele coding issues |
| Zero overlap (no shared variants) | 28,301 | 17.9% | Completely different variant sets |

The high proportion of negative correlations (27.8%) warrants investigation—see QC Findings below.

---

## Key Insights for PGS Selection

### When to prefer one PGS over another

For **highly correlated pairs (r > 0.9)**:
- Scores are effectively interchangeable for variant weighting
- Choose based on: sample size in original GWAS, ancestry match to your cohort, publication recency, or validation performance

For **moderately correlated pairs (0.5 < r ≤ 0.9)**:
- Scores capture similar but not identical genetic signals
- May complement each other in ensemble approaches
- Consider using both if computational resources allow

For **weakly/negatively correlated pairs**:
- Investigate whether both are valid or if one has data quality issues
- Check for effect allele inconsistencies (common cause of negative correlation)
- May represent genuinely different biological signals for complex traits

### Red flags requiring investigation

| Warning Sign | Count | Recommended Action |
|--------------|------:|-------------------|
| **Negative correlation** (r < 0) | 43,806 | Check effect/reference allele alignment |
| **Strong negative** (r < -0.1) | 35,445 | Likely allele coding inconsistency |
| **Low concordance** (<40% with ≥100 shared variants) | 8,415 | Verify strand orientation and allele coding |

### Recommended selection criteria

1. **Check correlation with alternatives** — r > 0.7 indicates safe substitutes; choose based on secondary criteria
2. **Verify sign concordance** — >70% preferred; lower values suggest data quality concerns
3. **Consider overlap percentage** — Higher overlap means more directly comparable; lower overlap may capture complementary signals
4. **Match to your genotyping platform** — Check what percentage of PGS variants are present in your array

---

## QC Findings

### Potential redundancy
**4,971 pairs** show r > 0.95, indicating potential catalog redundancy. These scores are essentially equivalent and researchers need only evaluate one from each cluster.

### Allele coding concerns
**35,445 pairs** show r < -0.1, suggesting systematic allele coding inconsistencies between scores. This affects approximately 23% of all computable correlations and represents a significant data quality issue in the catalog.

**Root causes may include:**
- Effect allele vs. reference allele swapped between studies
- Different strand conventions (forward vs. reverse)
- Different reference genome builds in source GWAS

### Directional disagreement
**8,415 pairs** have <40% sign concordance despite having ≥100 shared variants. When most shared variants have opposite effect directions, the biological validity of one or both scores should be questioned.

---

## Using the Compare Tab

The PGS Explorer web application includes a **Compare** tab providing:

1. **Filterable statistics table** — Browse all 157,775 pairs with trait, overlap, and correlation filters
2. **Pairwise scatterplot** — Select any two PGS IDs to visualize their effect weight relationship
3. **Network view** — Graph visualization showing PGS relationships within a trait (nodes = scores, edges = correlations above threshold)
4. **CSV export** — Download filtered comparison data for offline analysis

### Sample workflow

1. Filter to your trait of interest
2. Sort by correlation to identify highly similar PGS clusters
3. Select pairs for detailed scatterplot inspection
4. Use network view to understand the overall structure of PGS for that trait

---

## Files & Access

| File | Size | Description |
|------|-----:|-------------|
| `pgs_pairwise_stats.parquet` | 4.8 MB | All pairwise statistics (157,775 pairs) |
| `pgs_pairwise_variants_sample.json.gz` | 847 KB | Variant data for 50 representative pairs |
| `pipeline_metadata.json` | 648 B | Pipeline configuration and provenance |
| Full variant data (on request) | 4.6 GB | Complete variant-level data for all pairs |

### Data dictionary

| Column | Description |
|--------|-------------|
| `pgs_id_1`, `pgs_id_2` | PGS Catalog identifiers for the pair |
| `n_variants_1`, `n_variants_2` | Total variants in each score |
| `n_shared` | Number of overlapping variants |
| `pct_overlap_1`, `pct_overlap_2` | Overlap as percentage of each score |
| `pearson_r`, `pearson_p` | Pearson correlation and p-value |
| `pct_concordant_sign` | Percentage of shared variants with same effect direction |
| `trait_efo`, `trait_label` | EFO ontology ID and human-readable trait name |

---

## Technical Notes

- **Matching strategy:** Variants matched by rsID where available; chr:pos:ref:alt otherwise
- **Coordinate system:** GRCh38 (lifted over where necessary)
- **Sampling:** For pairs with >10,000 shared variants, correlation computed on full set but visualization samples 10,000
- **Reproducibility:** Pipeline code available in `data/pgs_pairwise_comparison_pipeline_optimized.ipynb`

---

## Contact

For questions about this analysis or access to the full variant dataset, contact the Phenome Health bioinformatics team.
