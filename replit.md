# PGS Catalog Explorer

## Overview
A Streamlit web application for exploring the Polygenic Score (PGS) Catalog database. Allows researchers to browse, filter, and analyze polygenic scores, traits, publications, and performance metrics with ancestry context. Includes quality tier classification and Kraken ingest estimation for graph database planning.

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
- `APIDataSource`: REST API implementation with 24hr caching
- `get_data_source()`: Factory function for backend swapping
- `get_evaluation_summary()`: Fetches evaluation counts and ancestry coverage per score

**utils.py**
- Method classification (LD-aware vs C+T approaches)
- Quality tier computation (Gold/Silver/Bronze/Unrated)
- Filtering functions for scores, traits, publications (with quality tier support)
- Kraken ingest statistics computation
- CSV export utilities (standard and Kraken ingest plan)
- Color schemes for visualizations

**app.py**
- Main Streamlit UI with 4 tabs: Scores, Traits, Publications, Performance
- Preset filter buttons: ARK-Ready, Kraken-Ready, All High-Quality
- Quality tier filtering and distribution visualization
- Kraken Ingest Estimator panel with graph impact estimates
- Sidebar with quality tier info, method classification, and ancestry categories
- Interactive filtering and data exploration

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
- Pagination: Max 250 results per page, capped at 20 pages for evaluation summary
- Cache TTL: 24 hours

### Method Classification
High (LD-aware): PRS-CS, PRS-CSx, LDpred, LDpred2, lassosum, SBayesR, MegaPRS
Moderate (C+T): C+T, P+T, PRSice, PRSice2
Uses case-insensitive matching on method_name field.

### Quality Tier Classification
- **Gold (ARK-ready)**: LD-aware method + ≥2 evaluations + multi-ancestry (≥2 groups)
- **Silver (Research-grade)**: LD-aware method + ≥1 evaluation
- **Bronze**: Moderate (C+T) method + ≥1 evaluation
- **Unrated**: Missing evaluations or Other/Unknown method

### Kraken-Eligible Criteria
A score is Kraken-eligible if ALL of:
1. Has EFO/MONDO mapping (hard requirement)
2. Has harmonized file (GRCh37 or GRCh38)
3. Quality tier is Gold, Silver, or Bronze (not Unrated)

### Kraken Ingest CSV Columns
pgs_id, name, trait_efo, trait_reported, method_name, method_class, quality_tier, n_variants, n_evaluations, ancestry_dev, ancestry_eval, grch37_available, grch38_available, publication_doi

### Production Upgrade Path
See README.md for DuckDB backend implementation using bulk metadata dumps from:
https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## User Preferences
- Scientific interface with proper ancestry context for metrics
- FTP links for scoring files (not direct serving)
- CSV export with key columns
- Method classification distinguishing LD-aware from simpler approaches
- Quality tier system for score assessment (ARK/Kraken readiness)
- Kraken ingest estimation for graph database planning
