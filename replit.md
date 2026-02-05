# PGS Catalog Explorer

## Overview
A Streamlit web application for exploring the Polygenic Score (PGS) Catalog database. Allows researchers to browse, filter, and analyze polygenic scores, traits, publications, and performance metrics with ancestry context. Includes quality tier classification, ontology mapping breakdown (EFO/MONDO/HP), and Kraken knowledge graph ingest estimation for research planning.

## Project Architecture

### File Structure
```
├── app.py              # Main Streamlit application
├── data_layer.py       # Abstract data source interface and API implementation
├── utils.py            # Utility functions for filtering, classification, export
├── README.md           # Documentation with production upgrade path
├── .streamlit/
│   └── config.toml     # Streamlit server configuration
└── attached_assets/    # Reference materials
```

### Key Components

**data_layer.py**
- `PGSDataSource`: Abstract base class defining the data interface
- `APIDataSource`: REST API implementation with smart caching (30-day TTL, auto-refresh)
- `get_data_source()`: Factory function for backend swapping
- `get_evaluation_summary()`: Fetches evaluation counts and ancestry coverage per score
- `get_api_counts()`: Quick count check to detect new data (limit=1 requests)
- `get_performance_metrics()`: Returns individual metric columns (AUC, R², OR, HR, Beta)

**utils.py**
- Method classification (LD-aware vs C+T approaches)
- Quality tier computation (Gold/Silver/Bronze/Unrated)
- Trait and publication tier statistics aggregation
- Filtering functions for scores, traits, publications (with quality tier + ontology support)
- Kraken ingest statistics computation with ontology breakdown
- CSV export utilities (standard and Kraken ingest plan)
- Color schemes for visualizations

**app.py**
- Main Streamlit UI with 4 tabs: Scores, Traits, Publications, Performance
- Preset filter buttons: ARK-Ready, Kraken-Ready, All High-Quality
- Ontology mapping filter: EFO only, MONDO only, HP only, Multiple, No mapping
- Quality tier filtering and distribution visualization in all tabs
- Variant Count Distribution section with summary stats and histogram (log/linear toggle)
- Kraken Ingest Estimator with tiered gene estimates based on actual variant counts
- Clickable DOI links for publications
- Sidebar with quality tier info, method classification, and ancestry categories

### Data Flow
```
PGS Catalog REST API
        ↓
   APIDataSource (with @st.cache_data)
        ↓
   Filter/Transform (utils.py)
        ↓
   Streamlit UI (app.py)
```

## Technical Notes

### API Configuration
- Base URL: `https://www.pgscatalog.org/rest/`
- Rate limit: 100 queries/minute
- Pagination: 100 results per page, follows 'next' until null (loads all ~5,200 scores)
- Cache TTL: 30 days with smart validation
- Progress indicator: Shows "Page X of Y · X of Y items" during actual API fetches

### Smart Caching System
- On each page load, pings API with `limit=1` to get current counts (fast, minimal data transfer)
- Compares API counts with cached data counts
- If counts differ → new data available → cache is invalidated and refreshed
- If cache is >30 days old → forced refresh regardless of counts
- Sidebar shows: cached counts, last checked timestamp, freshness status (✓ Up to date)

### Method Classification
High (LD-aware): PRS-CS, PRS-CSx, LDpred, LDpred2, lassosum, SBayesR, MegaPRS
Moderate (C+T): C+T, P+T, PRSice, PRSice2
Uses case-insensitive matching on method_name field.

### Quality Tier Classification
- **Gold (ARK-ready)**: LD-aware method + ≥2 evaluations + multi-ancestry (≥2 groups)
- **Silver (Research-grade)**: LD-aware method + ≥1 evaluation
- **Bronze**: Moderate (C+T) method + ≥1 evaluation
- **Unrated**: Missing evaluations or Other/Unknown method

### Ontology Mapping
The PGS Catalog's `trait_efo` field contains trait IDs with various prefixes:
- **EFO_**: Experimental Factor Ontology (primary mapping)
- **MONDO_**: Mondo Disease Ontology
- **HP_**: Human Phenotype Ontology

Kraken-eligible requires any one of EFO, MONDO, or HP mapping (not all required).

### Preset Filters
- **ARK-Ready**: Gold tier + any ontology mapping + GRCh38 available
- **Kraken-Ready**: Any rated tier (Gold/Silver/Bronze) + any ontology mapping + harmonized files
- **All High-Quality**: Gold + Silver tiers

### Kraken-Eligible Criteria
A score is Kraken-eligible if ALL of:
1. Has any ontology mapping (EFO, MONDO, or HP)
2. Has harmonized file (GRCh37 or GRCh38)
3. Quality tier is Gold, Silver, or Bronze (excludes Unrated)

### Kraken Ingest CSV Columns
pgs_id, name, trait_efo, trait_reported, method_name, method_class, quality_tier, n_variants, n_evaluations, ancestry_dev, ancestry_eval, efo_mapped, mondo_mapped, hp_mapped, grch37_available, grch38_available, publication_doi

### Production Upgrade Path
See README.md for DuckDB backend implementation using bulk metadata dumps from:
https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

### Gene Edge Heuristics (Kraken Estimator)
Tiered gene estimate based on variant count per score:
- **Low variant scores (<1,000)**: ~50 genes per PRS
- **Medium variant scores (1,000–100,000)**: ~200 genes per PRS
- **High variant scores (>100,000)**: ~500 genes per PRS

This accounts for many variants mapping to the same genes in high-variant scores.

## User Preferences
- Scientific interface with proper ancestry context for metrics
- FTP links for scoring files (not direct serving)
- CSV export with key columns
- Method classification distinguishing LD-aware from simpler approaches
- Quality tier system for score assessment (ARK/Kraken readiness)
- Kraken ingest estimation for graph database planning with tiered gene estimates
- Variant count distribution analysis with log-scale histograms
- Ontology mapping breakdown (EFO/MONDO/HP) to assess cross-mapping needs
- Clickable DOI links for publication references
