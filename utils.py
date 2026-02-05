import pandas as pd
from typing import Optional
import io

LD_AWARE_METHODS = [
    'prs-cs', 'prs-csx', 'ldpred', 'ldpred2', 'lassosum', 
    'sbayesr', 'megaprs', 'ldpred-inf', 'ldpred-funct',
    'prscs', 'prs cs', 'ld pred', 'lasso sum', 'sbayesc'
]

MODERATE_METHODS = [
    'c+t', 'p+t', 'prsice', 'prsice2', 'prsice-2',
    'clumping', 'thresholding', 'c + t', 'p + t'
]


def classify_method(method_name: Optional[str]) -> str:
    """Classify a PGS construction method.
    
    Returns:
        - 'High (LD-aware)': For sophisticated LD-aware methods
        - 'Moderate (C+T)': For clumping and thresholding approaches
        - 'Other': For unclassified methods
        - 'Unknown': If method_name is empty or None
    """
    if not method_name:
        return 'Unknown'
    
    method_lower = method_name.lower()
    
    for ld_method in LD_AWARE_METHODS:
        if ld_method in method_lower:
            return 'High (LD-aware)'
    
    for mod_method in MODERATE_METHODS:
        if mod_method in method_lower:
            return 'Moderate (C+T)'
    
    return 'Other'


def add_method_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Add method classification column to scores dataframe."""
    if 'method_name' not in df.columns:
        df['method_class'] = 'Unknown'
    else:
        df['method_class'] = df['method_name'].apply(classify_method)
    return df


def filter_scores(
    df: pd.DataFrame,
    search_query: Optional[str] = None,
    method_classes: Optional[list] = None,
    has_efo_only: bool = False,
    has_grch37: bool = False,
    has_grch38: bool = False,
    min_variants: Optional[int] = None,
    max_variants: Optional[int] = None,
) -> pd.DataFrame:
    """Apply filters to scores dataframe."""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if search_query:
        query_lower = search_query.lower()
        mask = (
            filtered['pgs_id'].str.lower().str.contains(query_lower, na=False) |
            filtered['name'].str.lower().str.contains(query_lower, na=False) |
            filtered['trait_names'].str.lower().str.contains(query_lower, na=False) |
            filtered['trait_ids'].str.lower().str.contains(query_lower, na=False) |
            filtered['first_author'].str.lower().str.contains(query_lower, na=False)
        )
        filtered = filtered[mask]
    
    if method_classes:
        filtered = filtered[filtered['method_class'].isin(method_classes)]
    
    if has_efo_only:
        filtered = filtered[filtered['has_efo_mapping'] == True]
    
    if has_grch37:
        filtered = filtered[filtered['grch37_available'] == True]
    
    if has_grch38:
        filtered = filtered[filtered['grch38_available'] == True]
    
    if min_variants is not None:
        filtered = filtered[filtered['n_variants'] >= min_variants]
    
    if max_variants is not None:
        filtered = filtered[filtered['n_variants'] <= max_variants]
    
    return filtered


def filter_traits(
    df: pd.DataFrame,
    search_query: Optional[str] = None,
    category: Optional[str] = None,
    min_scores: Optional[int] = None,
) -> pd.DataFrame:
    """Apply filters to traits dataframe."""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if search_query:
        query_lower = search_query.lower()
        mask = (
            filtered['trait_id'].str.lower().str.contains(query_lower, na=False) |
            filtered['label'].str.lower().str.contains(query_lower, na=False) |
            filtered['description'].str.lower().str.contains(query_lower, na=False)
        )
        filtered = filtered[mask]
    
    if category:
        filtered = filtered[filtered['categories'].str.contains(category, case=False, na=False)]
    
    if min_scores is not None:
        filtered = filtered[filtered['n_scores'] >= min_scores]
    
    return filtered


def filter_publications(
    df: pd.DataFrame,
    search_query: Optional[str] = None,
    year: Optional[int] = None,
    has_development: bool = False,
    has_evaluation: bool = False,
) -> pd.DataFrame:
    """Apply filters to publications dataframe."""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if search_query:
        query_lower = search_query.lower()
        mask = (
            filtered['pgp_id'].str.lower().str.contains(query_lower, na=False) |
            filtered['title'].str.lower().str.contains(query_lower, na=False) |
            filtered['first_author'].str.lower().str.contains(query_lower, na=False) |
            filtered['journal'].str.lower().str.contains(query_lower, na=False)
        )
        filtered = filtered[mask]
    
    if year is not None:
        filtered = filtered[filtered['date_publication'].str.startswith(str(year), na=False)]
    
    if has_development:
        filtered = filtered[filtered['n_development'] > 0]
    
    if has_evaluation:
        filtered = filtered[filtered['n_evaluation'] > 0]
    
    return filtered


def export_scores_csv(df: pd.DataFrame) -> str:
    """Export filtered scores to CSV string with key columns."""
    export_cols = [
        'pgs_id', 'name', 'trait_names', 'method_name', 'method_class',
        'n_variants', 'has_efo_mapping', 'grch37_available', 'grch38_available',
        'dev_ancestry', 'eval_ancestry', 'first_author', 'publication_date',
        'ftp_scoring_file', 'grch37_url', 'grch38_url'
    ]
    
    available_cols = [c for c in export_cols if c in df.columns]
    export_df = df[available_cols].copy()
    
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    return output.getvalue()


def export_traits_csv(df: pd.DataFrame) -> str:
    """Export filtered traits to CSV string."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def export_publications_csv(df: pd.DataFrame) -> str:
    """Export filtered publications to CSV string."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def format_ancestry_data(ancestry_dist: dict) -> dict:
    """Format ancestry distribution data for visualization."""
    formatted = {
        'Development/GWAS': {},
        'Evaluation': {}
    }
    
    if not ancestry_dist:
        return formatted
    
    for stage, data in ancestry_dist.items():
        if isinstance(data, dict):
            if 'dev' in stage.lower() or 'gwas' in stage.lower():
                for ancestry, count in data.items():
                    if ancestry in formatted['Development/GWAS']:
                        formatted['Development/GWAS'][ancestry] += count
                    else:
                        formatted['Development/GWAS'][ancestry] = count
            elif 'eval' in stage.lower():
                for ancestry, count in data.items():
                    if ancestry in formatted['Evaluation']:
                        formatted['Evaluation'][ancestry] += count
                    else:
                        formatted['Evaluation'][ancestry] = count
    
    return formatted


def get_method_class_colors() -> dict:
    """Get colors for method classification visualization."""
    return {
        'High (LD-aware)': '#2ecc71',
        'Moderate (C+T)': '#f39c12',
        'Other': '#95a5a6',
        'Unknown': '#bdc3c7'
    }


def get_ancestry_colors() -> dict:
    """Get colors for ancestry visualization."""
    return {
        'European': '#3498db',
        'East Asian': '#e74c3c',
        'African': '#2ecc71',
        'South Asian': '#9b59b6',
        'Hispanic/Latino': '#f39c12',
        'Mixed': '#1abc9c',
        'Other': '#95a5a6',
        'Multi-ancestry': '#34495e',
    }
