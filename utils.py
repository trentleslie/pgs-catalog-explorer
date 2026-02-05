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
        'pgs_id', 'name', 'trait_names', 'method_name', 'method_class', 'quality_tier',
        'n_variants', 'n_evaluations', 'has_efo', 'has_mondo', 'has_hp', 'has_any_ontology',
        'grch37_available', 'grch38_available', 'dev_ancestry', 'eval_ancestry', 
        'first_author', 'publication_date', 'ftp_scoring_file', 'grch37_url', 'grch38_url'
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


def compute_quality_tier(
    method_class: str,
    n_evaluations: int,
    n_ancestry_groups: int
) -> str:
    """Compute quality tier for a PGS.
    
    Tier Definitions (checked in order of priority):
    - Gold: High (LD-aware) method + ≥2 evaluations + multi-ancestry (≥2 groups)
    - Silver: High (LD-aware) method + ≥1 evaluation
    - Bronze: Has evaluation + Moderate (C+T) method (not LD-aware)
    - Unrated: Missing evaluations or Other/Unknown method
    """
    is_ld_aware = method_class == 'High (LD-aware)'
    is_moderate = method_class == 'Moderate (C+T)'
    has_evaluations = n_evaluations >= 1
    has_multi_eval = n_evaluations >= 2
    is_multi_ancestry = n_ancestry_groups >= 2
    
    if is_ld_aware and has_multi_eval and is_multi_ancestry:
        return 'Gold'
    elif is_ld_aware and has_evaluations:
        return 'Silver'
    elif is_moderate and has_evaluations:
        return 'Bronze'
    else:
        return 'Unrated'


def add_quality_tiers(scores_df: pd.DataFrame, eval_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add quality tier classification to scores dataframe.
    
    Args:
        scores_df: Scores dataframe with method_class column
        eval_summary_df: Evaluation summary with n_evaluations and n_ancestry_groups per pgs_id
    
    Returns:
        Scores dataframe with quality_tier, n_evaluations, n_ancestry_groups columns
    """
    if scores_df.empty:
        return scores_df
    
    result = scores_df.copy()
    
    if eval_summary_df.empty:
        result['n_evaluations'] = 0
        result['n_ancestry_groups'] = 0
        result['eval_ancestry_groups'] = ''
    else:
        result = result.merge(
            eval_summary_df[['pgs_id', 'n_evaluations', 'n_ancestry_groups', 'ancestry_groups']],
            on='pgs_id',
            how='left'
        )
        result['n_evaluations'] = result['n_evaluations'].fillna(0).astype(int)
        result['n_ancestry_groups'] = result['n_ancestry_groups'].fillna(0).astype(int)
        result['eval_ancestry_groups'] = result['ancestry_groups'].fillna('')
        result = result.drop(columns=['ancestry_groups'], errors='ignore')
    
    result['quality_tier'] = result.apply(
        lambda row: compute_quality_tier(
            row.get('method_class', 'Unknown'),
            row.get('n_evaluations', 0),
            row.get('n_ancestry_groups', 0)
        ),
        axis=1
    )
    
    return result


def get_quality_tier_colors() -> dict:
    """Get colors for quality tier visualization."""
    return {
        'Gold': '#FFD700',
        'Silver': '#C0C0C0',
        'Bronze': '#CD7F32',
        'Unrated': '#808080'
    }


def compute_trait_tier_stats(scores_df: pd.DataFrame) -> dict:
    """Compute quality tier statistics per trait.
    
    Returns dict mapping trait_id to {best_tier, breakdown, counts}.
    """
    if scores_df.empty or 'trait_ids' not in scores_df.columns or 'quality_tier' not in scores_df.columns:
        return {}
    
    tier_order = {'Gold': 1, 'Silver': 2, 'Bronze': 3, 'Unrated': 4}
    result = {}
    
    for _, row in scores_df.iterrows():
        trait_ids_str = row.get('trait_ids', '')
        if not trait_ids_str:
            continue
        
        trait_ids = [t.strip() for t in trait_ids_str.split(';') if t.strip()]
        tier = row.get('quality_tier', 'Unrated')
        
        for trait_id in trait_ids:
            if trait_id not in result:
                result[trait_id] = {'counts': {'Gold': 0, 'Silver': 0, 'Bronze': 0, 'Unrated': 0}}
            result[trait_id]['counts'][tier] = result[trait_id]['counts'].get(tier, 0) + 1
    
    for trait_id, data in result.items():
        counts = data['counts']
        best_tier = 'Unrated'
        for tier in ['Gold', 'Silver', 'Bronze', 'Unrated']:
            if counts.get(tier, 0) > 0:
                best_tier = tier
                break
        data['best_tier'] = best_tier
        
        parts = []
        for tier in ['Gold', 'Silver', 'Bronze', 'Unrated']:
            if counts.get(tier, 0) > 0:
                parts.append(f"{counts[tier]} {tier}")
        data['breakdown'] = ', '.join(parts)
    
    return result


def compute_publication_tier_stats(scores_df: pd.DataFrame) -> dict:
    """Compute quality tier statistics per publication.
    
    Returns dict mapping pgp_id to {best_tier, counts}.
    """
    if scores_df.empty or 'pgp_id' not in scores_df.columns or 'quality_tier' not in scores_df.columns:
        return {}
    
    result = {}
    
    for _, row in scores_df.iterrows():
        pgp_id = row.get('pgp_id', '')
        if not pgp_id:
            continue
        
        tier = row.get('quality_tier', 'Unrated')
        
        if pgp_id not in result:
            result[pgp_id] = {'counts': {'Gold': 0, 'Silver': 0, 'Bronze': 0, 'Unrated': 0}}
        result[pgp_id]['counts'][tier] = result[pgp_id]['counts'].get(tier, 0) + 1
    
    for pgp_id, data in result.items():
        counts = data['counts']
        best_tier = 'Unrated'
        for tier in ['Gold', 'Silver', 'Bronze', 'Unrated']:
            if counts.get(tier, 0) > 0:
                best_tier = tier
                break
        data['best_tier'] = best_tier
    
    return result


def filter_scores(
    df: pd.DataFrame,
    search_query: Optional[str] = None,
    method_classes: Optional[list] = None,
    quality_tiers: Optional[list] = None,
    ontology_filters: Optional[list] = None,
    has_grch37: bool = False,
    has_grch38: bool = False,
    min_variants: Optional[int] = None,
    max_variants: Optional[int] = None,
) -> pd.DataFrame:
    """Apply filters to scores dataframe.
    
    ontology_filters: List of ontology filter options:
        - 'EFO only': Has EFO but not MONDO or HP
        - 'MONDO only': Has MONDO but not EFO or HP
        - 'HP only': Has HP but not EFO or MONDO
        - 'Multiple': Has more than one ontology type
        - 'No mapping': Has no ontology mappings
    """
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
    
    if quality_tiers and 'quality_tier' in filtered.columns:
        filtered = filtered[filtered['quality_tier'].isin(quality_tiers)]
    
    if ontology_filters and 'has_efo' in filtered.columns:
        ontology_mask = pd.Series([False] * len(filtered), index=filtered.index)
        
        for ont_filter in ontology_filters:
            if ont_filter == 'EFO only':
                ontology_mask |= (filtered['has_efo'] & ~filtered['has_mondo'] & ~filtered['has_hp'])
            elif ont_filter == 'MONDO only':
                ontology_mask |= (~filtered['has_efo'] & filtered['has_mondo'] & ~filtered['has_hp'])
            elif ont_filter == 'HP only':
                ontology_mask |= (~filtered['has_efo'] & ~filtered['has_mondo'] & filtered['has_hp'])
            elif ont_filter == 'Multiple':
                count = filtered['has_efo'].astype(int) + filtered['has_mondo'].astype(int) + filtered['has_hp'].astype(int)
                ontology_mask |= (count >= 2)
            elif ont_filter == 'No mapping':
                ontology_mask |= (~filtered['has_efo'] & ~filtered['has_mondo'] & ~filtered['has_hp'])
        
        filtered = filtered[ontology_mask]
    
    if has_grch37:
        filtered = filtered[filtered['grch37_available'] == True]
    
    if has_grch38:
        filtered = filtered[filtered['grch38_available'] == True]
    
    if min_variants is not None:
        filtered = filtered[filtered['n_variants'] >= min_variants]
    
    if max_variants is not None:
        filtered = filtered[filtered['n_variants'] <= max_variants]
    
    return filtered


def compute_kraken_stats(df: pd.DataFrame) -> dict:
    """Compute Kraken ingest statistics from filtered scores.
    
    Returns dict with ontology breakdown, harmonized files, and variant stats.
    Kraken-eligible = has any ontology (EFO/MONDO/HP) + harmonized files + rated tier.
    """
    if df.empty:
        return {
            'total_scores': 0,
            'efo_only': 0,
            'mondo_only': 0,
            'hp_only': 0,
            'multiple_ontologies': 0,
            'no_mapping': 0,
            'total_mappable': 0,
            'with_harmonized': 0,
            'kraken_eligible': 0,
            'total_variants': 0,
            'avg_variants': 0,
            'max_variants': 0,
            'estimated_gene_edges': 0
        }
    
    total = len(df)
    
    has_efo = df.get('has_efo', pd.Series([False] * total))
    has_mondo = df.get('has_mondo', pd.Series([False] * total))
    has_hp = df.get('has_hp', pd.Series([False] * total))
    has_any = df.get('has_any_ontology', has_efo | has_mondo | has_hp)
    
    ontology_count = has_efo.astype(int) + has_mondo.astype(int) + has_hp.astype(int)
    
    efo_only = len(df[has_efo & ~has_mondo & ~has_hp])
    mondo_only = len(df[~has_efo & has_mondo & ~has_hp])
    hp_only = len(df[~has_efo & ~has_mondo & has_hp])
    multiple_ontologies = len(df[ontology_count >= 2])
    no_mapping = len(df[~has_any])
    total_mappable = len(df[has_any])
    
    with_harmonized = len(df[(df['grch37_available'] == True) | (df['grch38_available'] == True)])
    
    eligible_mask = (
        has_any &
        ((df['grch37_available'] == True) | (df['grch38_available'] == True))
    )
    if 'quality_tier' in df.columns:
        eligible_mask = eligible_mask & (df['quality_tier'].isin(['Gold', 'Silver', 'Bronze']))
    
    kraken_eligible = len(df[eligible_mask])
    
    total_variants = int(df['n_variants'].sum())
    avg_variants = int(df['n_variants'].mean()) if total > 0 else 0
    max_variants = int(df['n_variants'].max()) if total > 0 else 0
    
    estimated_gene_edges = kraken_eligible * 50
    
    return {
        'total_scores': total,
        'efo_only': efo_only,
        'mondo_only': mondo_only,
        'hp_only': hp_only,
        'multiple_ontologies': multiple_ontologies,
        'no_mapping': no_mapping,
        'total_mappable': total_mappable,
        'with_harmonized': with_harmonized,
        'kraken_eligible': kraken_eligible,
        'total_variants': total_variants,
        'avg_variants': avg_variants,
        'max_variants': max_variants,
        'estimated_gene_edges': estimated_gene_edges
    }


def export_kraken_ingest_csv(df: pd.DataFrame) -> str:
    """Export Kraken ingest plan CSV with specified columns."""
    export_cols = [
        'pgs_id', 'name', 'trait_ids', 'trait_names', 'method_name', 'method_class',
        'quality_tier', 'n_variants', 'n_evaluations', 'dev_ancestry', 'eval_ancestry',
        'has_efo', 'has_mondo', 'has_hp', 'grch37_available', 'grch38_available', 'doi'
    ]
    
    available_cols = [c for c in export_cols if c in df.columns]
    export_df = df[available_cols].copy()
    
    export_df = export_df.rename(columns={
        'trait_ids': 'trait_efo',
        'trait_names': 'trait_reported',
        'dev_ancestry': 'ancestry_dev',
        'eval_ancestry': 'ancestry_eval',
        'has_efo': 'efo_mapped',
        'has_mondo': 'mondo_mapped',
        'has_hp': 'hp_mapped',
        'doi': 'publication_doi'
    })
    
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    return output.getvalue()
