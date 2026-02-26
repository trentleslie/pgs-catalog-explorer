# PGS Catalog Explorer

A Streamlit web application for exploring the [Polygenic Score (PGS) Catalog](https://www.pgscatalog.org/) database with interactive search, method filtering, and ancestry-aware performance metrics.

## Features

### Score Browser
- Search and filter polygenic scores by PGS ID, trait, or publication
- Filter by construction method classification:
  - **High (LD-aware)**: PRS-CS, PRS-CSx, LDpred, LDpred2, lassosum, SBayesR, MegaPRS
  - **Moderate (C+T)**: C+T, P+T, PRSice, PRSice2
- Filter by EFO/MONDO trait mapping status
- View harmonized file availability (GRCh37/GRCh38) and variant counts
- Export filtered results as CSV

### Trait Explorer
- Browse all traits with hierarchical categories
- Filter by category and minimum associated scores
- Visualize top traits by score count

### Publication Browser
- Search publications by author, title, or journal
- Filter by year and development/evaluation status
- View publication trends over time

### Performance Metrics
- View ancestry-aware performance metrics for any PGS
- Displays evaluation ancestry, sample size, cohorts, and metrics (AUC, OR, etc.)
- Proper context for interpreting metrics across populations

### Compare Tab (Pairwise PGS Analysis)

Compare polygenic scores for the same trait to assess redundancy and effect direction concordance.

**Features:**
- Select any trait with multiple PGS to see all pairwise comparisons
- View effect weight correlations (Pearson r) and variant overlap statistics
- Interactive scatterplots showing shared variant weights
- Quality flags for highly correlated (r > 0.95) or negatively correlated pairs

**Pipeline:** The comparison data is pre-computed using a memory-safe batch processing pipeline. See the [pipeline notebook](data/pgs_pairwise_comparison_pipeline_optimized.ipynb) for methodology.

**Statistics (February 2026):**
| Metric | Value |
|--------|-------|
| Total PGS analyzed | 5,042 |
| Traits with ≥2 PGS | 432 |
| Pairwise comparisons | 157,775 |
| Highly correlated pairs (r > 0.95) | 4,971 |
| Negatively correlated (r < -0.1) | 35,445 |
| Zero variant overlap | 28,301 |

**Key Findings:**
- Median Pearson r = 0.155 (high variability between PGS for same trait)
- ~5K pairs are essentially redundant (r > 0.95)
- ~35K pairs have opposite effect directions, suggesting effect allele coding differences

## Architecture

The application uses an abstracted data layer that enables easy backend swapping:

```python
# data_layer.py
class PGSDataSource(ABC):
    """Abstract interface for PGS data access"""
    def get_scores(self, filters: dict) -> pd.DataFrame: ...
    def get_score_details(self, pgs_id: str) -> dict: ...
    def get_traits(self) -> pd.DataFrame: ...
    # ... more methods

class APIDataSource(PGSDataSource):
    """REST API implementation (current)"""
    ...

# In app.py - UI never knows where data comes from
data_source = get_data_source()  # Returns APIDataSource
scores = data_source.get_scores(filters)
```

## Current Implementation: REST API

The current implementation uses the PGS Catalog REST API:
- Base URL: `https://www.pgscatalog.org/rest/`
- Endpoints: `/score/all`, `/trait/all`, `/publication/all`, `/performance/search`, etc.
- Caching: 24-hour TTL using `@st.cache_data`
- Pagination: Automatic handling via `next` field

## Production Upgrade: DuckDB Backend

For production deployment, the REST API can be replaced with a DuckDB backend using bulk metadata dumps. This provides significant performance improvements.

### Why DuckDB?

| Aspect | REST API | DuckDB |
|--------|----------|--------|
| Full catalog load | 1-5 minutes | 1-5 seconds |
| Rate limiting | 100 queries/min | None |
| Offline support | No | Yes |
| Complex queries | Multiple API calls | Single SQL query |

### Bulk Metadata Files

The PGS Catalog provides weekly bulk metadata dumps at:
https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/

| File | Contents |
|------|----------|
| `pgs_all_metadata_scores.csv` | All score-level metadata |
| `pgs_all_metadata_evals.csv` | All performance evaluations |
| `pgs_all_metadata_samples.csv` | Sample/cohort information |
| `pgs_all_metadata_cohorts.csv` | Cohort definitions |

### DuckDB Implementation Example

```python
import duckdb
import pandas as pd

class DuckDBDataSource(PGSDataSource):
    """Bulk file implementation for production"""
    
    def __init__(self, metadata_dir: str = "./metadata"):
        self.conn = duckdb.connect(':memory:')
        
        # Load bulk metadata (one-time or weekly refresh)
        self.conn.execute(f"""
            CREATE TABLE scores AS 
            SELECT * FROM '{metadata_dir}/pgs_all_metadata_scores.csv'
        """)
        self.conn.execute(f"""
            CREATE TABLE evals AS 
            SELECT * FROM '{metadata_dir}/pgs_all_metadata_evals.csv'
        """)
        self.conn.execute(f"""
            CREATE TABLE samples AS 
            SELECT * FROM '{metadata_dir}/pgs_all_metadata_samples.csv'
        """)
    
    def get_scores(self, filters: dict = None) -> pd.DataFrame:
        query = "SELECT * FROM scores"
        conditions = []
        
        if filters:
            if filters.get('method_name'):
                conditions.append(f"method_name ILIKE '%{filters['method_name']}%'")
            if filters.get('has_efo'):
                conditions.append("trait_efo IS NOT NULL")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        return self.conn.execute(query).df()
    
    def get_performance_metrics(self, pgs_id: str = None) -> pd.DataFrame:
        query = """
            SELECT e.*, s.ancestry_broad, s.sample_number
            FROM evals e
            LEFT JOIN samples s ON e.sample_id = s.id
        """
        if pgs_id:
            query += f" WHERE e.pgs_id = '{pgs_id}'"
        return self.conn.execute(query).df()
    
    # ... implement other methods
```

### Switching to DuckDB

To switch backends, update `get_data_source()` in `data_layer.py`:

```python
def get_data_source() -> PGSDataSource:
    # Download bulk files first (weekly cron job recommended)
    # wget https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/pgs_all_metadata_scores.csv
    
    # For production:
    # return DuckDBDataSource(metadata_dir="./metadata")
    
    # For development:
    return APIDataSource()
```

## Running the App

```bash
streamlit run app.py --server.port 5000
```

## Dependencies

- streamlit
- pandas
- plotly
- requests

## Links

- [PGS Catalog](https://www.pgscatalog.org/)
- [PGS Catalog REST API](https://www.pgscatalog.org/rest/)
- [pgscatalog-utils](https://pypi.org/project/pgscatalog-utils/)
- [Bulk Metadata Downloads](https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/)
