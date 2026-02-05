# PGS Catalog Explorer

## Overview
A Streamlit web application for exploring the Polygenic Score (PGS) Catalog database. Allows researchers to browse, filter, and analyze polygenic scores, traits, publications, and performance metrics with ancestry context.

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

**utils.py**
- Method classification (LD-aware vs C+T approaches)
- Filtering functions for scores, traits, publications
- CSV export utilities
- Color schemes for visualizations

**app.py**
- Main Streamlit UI with 4 tabs: Scores, Traits, Publications, Performance
- Sidebar with method classification info and ancestry categories
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
- Pagination: Max 250 results per page
- Cache TTL: 24 hours

### Method Classification
High (LD-aware): PRS-CS, PRS-CSx, LDpred, LDpred2, lassosum, SBayesR, MegaPRS
Moderate (C+T): C+T, P+T, PRSice, PRSice2
Uses case-insensitive matching on method_name field.

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
