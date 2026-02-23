import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_layer import get_data_source
from utils import (
    add_method_classification, add_quality_tiers, filter_scores, filter_traits, filter_publications,
    export_scores_csv, export_traits_csv, export_publications_csv, export_kraken_ingest_csv,
    get_method_class_colors, get_ancestry_colors, get_quality_tier_colors, 
    classify_method, compute_quality_tier, compute_kraken_stats, compute_trait_tier_stats, compute_publication_tier_stats,
    translate_ancestry_codes
)
from compare import (
    load_comparison_data, load_variant_data, filter_comparison_data,
    get_interpretation, create_scatterplot, build_network, plot_network,
    get_network_stats, export_comparison_csv
)

st.set_page_config(
    page_title="PGS Catalog Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

data_source = get_data_source()


from datetime import datetime

CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
CACHE_VERSION = 5

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_enriched_scores_cached(_version=CACHE_VERSION):
    """Get scores dataframe enriched with method classification and quality tiers.
    
    Returns:
        Tuple of (scores_df, eval_summary_df, timestamp, is_complete)
    """
    scores_df, is_complete = data_source.get_scores()
    eval_summary_df = data_source.get_evaluation_summary()
    
    if scores_df.empty:
        return scores_df, eval_summary_df, datetime.now().isoformat(), is_complete
    
    scores_df = add_method_classification(scores_df)
    scores_df = add_quality_tiers(scores_df, eval_summary_df)
    return scores_df, eval_summary_df, datetime.now().isoformat(), is_complete


def load_data_with_smart_cache():
    """Load data with smart cache validation and progress display."""
    if 'cache_metadata' not in st.session_state:
        st.session_state.cache_metadata = {
            'scores_count': 0,
            'evals_count': 0,
            'last_checked': None,
            'last_loaded': None,
            'is_fresh': None,
            'is_complete': True,
            'expected_scores': 0
        }
    
    if st.session_state.get('force_refresh'):
        get_enriched_scores_cached.clear()
        data_source.get_scores.clear()
        data_source.get_evaluation_summary.clear()
        st.session_state.force_refresh = False
        st.session_state.cache_metadata['scores_count'] = 0
    
    status_container = st.empty()
    
    status_container.markdown("**Checking PGS Catalog for updates...**")
    api_scores, api_evals = data_source.get_api_counts()
    
    cached_scores = st.session_state.cache_metadata.get('scores_count', 0)
    cached_evals = st.session_state.cache_metadata.get('evals_count', 0)
    last_loaded = st.session_state.cache_metadata.get('last_loaded')
    
    cache_stale = False
    stale_reason = None
    
    if api_scores > 0 and (api_scores != cached_scores or api_evals != cached_evals):
        cache_stale = True
        stale_reason = "new data"
    
    if last_loaded:
        try:
            loaded_dt = datetime.fromisoformat(last_loaded)
            age_days = (datetime.now() - loaded_dt).days
            if age_days >= 30:
                cache_stale = True
                stale_reason = f"cache is {age_days} days old"
        except:
            pass
    
    if cache_stale and cached_scores > 0:
        reason_msg = f" ({stale_reason})" if stale_reason else ""
        status_container.markdown(f"**Refreshing data{reason_msg}...** API: {api_scores:,} scores, {api_evals:,} evaluations")
        get_enriched_scores_cached.clear()
        data_source.get_scores.clear()
        data_source.get_evaluation_summary.clear()
    elif cached_scores == 0:
        status_container.markdown("**Loading PGS Catalog data...** This may take a few minutes on first load.")
    
    scores_df, eval_summary_df, load_timestamp, is_complete = get_enriched_scores_cached()
    
    if not scores_df.empty:
        st.session_state.cache_metadata = {
            'scores_count': api_scores if api_scores > 0 else len(scores_df),
            'evals_count': api_evals if api_evals > 0 else 0,
            'scores_loaded': len(scores_df),
            'evals_summary_count': len(eval_summary_df) if not eval_summary_df.empty else 0,
            'last_checked': datetime.now().isoformat(),
            'last_loaded': load_timestamp,
            'is_fresh': not cache_stale,
            'is_complete': is_complete,
            'expected_scores': api_scores if api_scores > 0 else len(scores_df)
        }
    
    status_container.empty()
    return scores_df, eval_summary_df


def main():
    st.markdown('<p class="main-header">PGS Catalog Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Browse and explore polygenic scores, traits, and publications from the PGS Catalog</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Scores", "Traits", "Publications", "Performance Metrics", "Compare", "Supplemental Info"])
    
    with tab3:
        render_publications_tab_independent()
    
    with tab4:
        render_performance_tab_independent()
    
    with tab5:
        render_compare_tab()
    
    scores_df, eval_summary_df = load_data_with_smart_cache()
    
    data_loaded = not scores_df.empty
    
    with tab1:
        if data_loaded:
            render_scores_tab(scores_df, eval_summary_df)
        else:
            st.info("Scores data is loading. Please wait...")
    
    with tab2:
        if data_loaded:
            render_traits_tab(scores_df)
        else:
            st.info("Traits data is loading. Please wait...")
    
    with tab6:
        render_supplemental_tab()
    
    with st.sidebar:
        render_sidebar_info(eval_summary_df)


def render_scores_tab(scores_df, eval_summary_df):
    st.header("Polygenic Scores Browser")
    
    if not eval_summary_df.empty:
        eval_coverage = len(eval_summary_df)
        if eval_coverage < len(scores_df) * 0.5:
            st.info(f"Quality tier data based on {eval_coverage:,} scores with evaluations. Tiers shown may be partial.")
    
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    if 'preset_filter' not in st.session_state:
        st.session_state.preset_filter = None
    
    with preset_col1:
        if st.button("ARK-Ready", help="Gold tier + EFO mapping + GRCh38"):
            st.session_state.preset_filter = 'ark'
    with preset_col2:
        if st.button("Kraken-Ready", help="Any rated tier (Gold/Silver/Bronze) + EFO mapping + harmonized files"):
            st.session_state.preset_filter = 'kraken'
    with preset_col3:
        if st.button("All High-Quality", help="Gold + Silver tiers"):
            st.session_state.preset_filter = 'high_quality'
    with preset_col4:
        if st.button("Clear Presets"):
            st.session_state.preset_filter = None
    
    preset = st.session_state.preset_filter
    default_tiers = []
    default_efo = False
    default_grch38 = False
    default_grch37 = False
    
    if preset == 'ark':
        default_tiers = ['Gold']
        default_efo = True
        default_grch38 = True
    elif preset == 'kraken':
        default_tiers = ['Gold', 'Silver', 'Bronze']
        default_efo = True
        default_grch37 = True
        default_grch38 = True
    elif preset == 'high_quality':
        default_tiers = ['Gold', 'Silver']
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        search = st.text_input("Search", placeholder="PGS ID, trait, author...")
        
        st.write("**Quality Tier**")
        quality_tiers = st.multiselect(
            "Select tiers",
            options=['Gold', 'Silver', 'Bronze', 'Unrated'],
            default=default_tiers if default_tiers else None,
            help="Gold: LD-aware + 2+ evals + multi-ancestry. Silver: LD-aware + eval. Bronze: Has eval. Unrated: No evals.",
            label_visibility="collapsed"
        )
        
        method_classes = st.multiselect(
            "Method Classification",
            options=['High (LD-aware)', 'Moderate (C+T)', 'Other', 'Unknown'],
            help="High: PRS-CS, LDpred2, lassosum, etc. Moderate: C+T, PRSice, etc."
        )
        
        st.write("**Ontology Mapping**")
        ontology_options = ['EFO only', 'MONDO only', 'HP only', 'Multiple', 'No mapping']
        default_ont = ['EFO only', 'MONDO only', 'HP only', 'Multiple'] if default_efo else []
        ontology_filters = st.multiselect(
            "Select ontology types",
            options=ontology_options,
            default=default_ont if default_ont else None,
            help="EFO/MONDO/HP mappings enable PRS→Disease edges in Kraken. 'No mapping' excluded from Kraken.",
            label_visibility="collapsed"
        )
        
        st.write("**Harmonized Files**")
        has_grch37 = st.checkbox("GRCh37 available", value=default_grch37)
        has_grch38 = st.checkbox("GRCh38 available", value=default_grch38)
        
        st.write("**Variant Count**")
        max_variants_val = int(scores_df['n_variants'].max()) if scores_df['n_variants'].max() > 0 else 1000000
        min_var, max_var = st.slider(
            "Range",
            min_value=0,
            max_value=max_variants_val,
            value=(0, max_variants_val)
        )
        
        st.write("**GWAS Sample Size**")
        gwas_max_val = int(scores_df['gwas_sample_n'].max()) if 'gwas_sample_n' in scores_df.columns and scores_df['gwas_sample_n'].max() > 0 else 5000000
        gwas_min, gwas_max = st.slider(
            "GWAS N Range",
            min_value=0,
            max_value=gwas_max_val,
            value=(0, gwas_max_val),
            help="Filter by total GWAS source sample size",
            label_visibility="collapsed"
        )
        
        st.write("**Publication Year**")
        if 'pub_year' in scores_df.columns:
            valid_years = scores_df['pub_year'].dropna()
            yr_min_val = int(valid_years.min()) if len(valid_years) > 0 else 2008
            yr_max_val = int(valid_years.max()) if len(valid_years) > 0 else 2026
        else:
            yr_min_val, yr_max_val = 2008, 2026
        pub_yr_min, pub_yr_max = st.slider(
            "Year Range",
            min_value=yr_min_val,
            max_value=yr_max_val,
            value=(yr_min_val, yr_max_val),
            label_visibility="collapsed"
        )
        
        gwas_filter_min = gwas_min if gwas_min > 0 else None
        gwas_filter_max = gwas_max if gwas_max < gwas_max_val else None
        pub_yr_filter_min = pub_yr_min if pub_yr_min > yr_min_val else None
        pub_yr_filter_max = pub_yr_max if pub_yr_max < yr_max_val else None
        
        filtered_df = filter_scores(
            scores_df,
            search_query=search if search else None,
            method_classes=method_classes if method_classes else None,
            quality_tiers=quality_tiers if quality_tiers else None,
            ontology_filters=ontology_filters if ontology_filters else None,
            has_grch37=has_grch37,
            has_grch38=has_grch38,
            min_variants=min_var,
            max_variants=max_var,
            gwas_n_min=gwas_filter_min,
            gwas_n_max=gwas_filter_max,
            pub_year_min=pub_yr_filter_min,
            pub_year_max=pub_yr_filter_max,
        )
    
    with col2:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Scores", len(filtered_df))
        with metric_cols[1]:
            gold_count = len(filtered_df[filtered_df['quality_tier'] == 'Gold']) if 'quality_tier' in filtered_df.columns else 0
            st.metric("Gold Tier", gold_count)
        with metric_cols[2]:
            silver_count = len(filtered_df[filtered_df['quality_tier'] == 'Silver']) if 'quality_tier' in filtered_df.columns else 0
            st.metric("Silver Tier", silver_count)
        with metric_cols[3]:
            bronze_count = len(filtered_df[filtered_df['quality_tier'] == 'Bronze']) if 'quality_tier' in filtered_df.columns else 0
            st.metric("Bronze Tier", bronze_count)
        
        if not filtered_df.empty:
            download_col1, download_col2 = st.columns(2)
            with download_col1:
                csv_data = export_scores_csv(filtered_df)
                st.download_button(
                    label="Download Filtered Results (CSV)",
                    data=csv_data,
                    file_name="pgs_catalog_scores.csv",
                    mime="text/csv"
                )
            with download_col2:
                kraken_csv = export_kraken_ingest_csv(filtered_df)
                st.download_button(
                    label="Download Kraken Ingest Plan (CSV)",
                    data=kraken_csv,
                    file_name="kraken_ingest_plan.csv",
                    mime="text/csv"
                )
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                method_counts = filtered_df['method_class'].value_counts()
                fig = px.pie(
                    values=method_counts.values,
                    names=method_counts.index,
                    title="Method Classification Distribution",
                    color=method_counts.index,
                    color_discrete_map=get_method_class_colors()
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                if 'quality_tier' in filtered_df.columns:
                    tier_order = ['Gold', 'Silver', 'Bronze', 'Unrated']
                    tier_counts = filtered_df['quality_tier'].value_counts()
                    tier_counts = tier_counts.reindex(tier_order).dropna()
                    fig = px.pie(
                        values=tier_counts.values,
                        names=tier_counts.index,
                        title="Quality Tier Distribution (All Filtered Scores)",
                        color=tier_counts.index,
                        color_discrete_map=get_quality_tier_colors()
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Charts reflect ALL scores matching current filters, not just the displayed 100 rows.")
            
            st.subheader("Variant Count Distribution")
            
            if 'n_variants' in filtered_df.columns and not filtered_df['n_variants'].isna().all():
                variant_data = filtered_df['n_variants'].dropna()
                
                if len(variant_data) > 0:
                    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
                    with stat_col1:
                        st.metric("Total Variants", f"{int(variant_data.sum()):,}")
                    with stat_col2:
                        st.metric("Mean", f"{int(variant_data.mean()):,}")
                    with stat_col3:
                        st.metric("Median", f"{int(variant_data.median()):,}")
                    with stat_col4:
                        st.metric("Min", f"{int(variant_data.min()):,}")
                    with stat_col5:
                        st.metric("Max", f"{int(variant_data.max()):,}")
                    
                    use_log_scale = st.toggle("Use log scale (recommended for wide range)", value=True, key="variant_log_toggle")
                    
                    if use_log_scale:
                        import numpy as np
                        log_data = np.log10(variant_data[variant_data > 0])
                        fig = px.histogram(
                            x=log_data,
                            nbins=30,
                            title=f"Variant Count Distribution (log₁₀ scale) - {len(variant_data):,} scores",
                            labels={'x': 'log₁₀(Variants)', 'y': 'Number of Scores'}
                        )
                        tick_vals = list(range(int(log_data.min()), int(log_data.max()) + 2))
                        tick_text = [f"10^{v}" for v in tick_vals]
                        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text)
                    else:
                        fig = px.histogram(
                            x=variant_data,
                            nbins=30,
                            title=f"Variant Count Distribution (linear scale) - {len(variant_data):,} scores",
                            labels={'x': 'Variants', 'y': 'Number of Scores'}
                        )
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            display_df = filtered_df.head(100).copy()
            
            display_df['pgs_link'] = display_df['pgs_id'].apply(
                lambda x: f"https://www.pgscatalog.org/score/{x}/" if x else ""
            )
            
            if 'has_efo' in display_df.columns:
                display_df['efo_mapped'] = display_df['has_efo'].apply(lambda x: '✓' if x else '✗')
            if 'has_mondo' in display_df.columns:
                display_df['mondo_mapped'] = display_df['has_mondo'].apply(lambda x: '✓' if x else '✗')
            if 'has_hp' in display_df.columns:
                display_df['hp_mapped'] = display_df['has_hp'].apply(lambda x: '✓' if x else '✗')
            
            if 'gwas_ids' in display_df.columns:
                display_df['gwas_link'] = display_df['gwas_ids'].apply(
                    lambda x: f"https://www.ebi.ac.uk/gwas/studies/{x.split(',')[0].strip()}" if x else ''
                )
                display_df['gwas_display'] = display_df['gwas_ids'].apply(
                    lambda x: x if x else 'Not reported'
                )
            
            display_cols = ['pgs_id', 'pgs_link', 'name', 'trait_names', 'quality_tier', 'method_class', 
                          'n_evaluations', 'n_variants', 'gwas_sample_n', 'gwas_link', 'pub_year', 'efo_mapped', 'mondo_mapped', 'hp_mapped', 'grch38_available', 'first_author']
            available_display = [c for c in display_cols if c in display_df.columns]
            
            display_df = display_df[available_display]
            
            if 'quality_tier' in display_df.columns:
                tier_emoji = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉', 'Unrated': '⚫'}
                display_df['quality_tier'] = display_df['quality_tier'].apply(
                    lambda x: f"{tier_emoji.get(x, '')} {x}"
                )
            
            column_config = {
                'pgs_id': st.column_config.TextColumn("PGS ID"),
                'pgs_link': st.column_config.LinkColumn("Link", display_text="View"),
                'gwas_sample_n': st.column_config.NumberColumn("GWAS N", format="%d"),
                'gwas_link': st.column_config.LinkColumn(
                    "GWAS ID",
                    display_text="https://www\\.ebi\\.ac\\.uk/gwas/studies/(.+)"
                ),
                'pub_year': st.column_config.NumberColumn("Year", format="%d"),
            }
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
            
            if len(filtered_df) > 100:
                st.info(f"Showing first 100 of {len(filtered_df)} scores. Use filters to narrow down.")
            
            with st.expander("Kraken Ingest Estimator"):
                render_kraken_estimator(filtered_df)
            
            with st.expander("Score Details"):
                selected_pgs = st.selectbox(
                    "Select a score to view details",
                    options=filtered_df['pgs_id'].tolist()[:50],
                    key="score_detail_select"
                )
                
                if selected_pgs:
                    render_score_details(selected_pgs, filtered_df)


def render_kraken_estimator(df: pd.DataFrame):
    """Render Kraken Ingest Estimator panel."""
    stats = compute_kraken_stats(df)
    
    st.markdown("### Kraken Ingest Estimate")
    st.markdown("---")
    
    st.markdown(f"**Scores matching filters:** {stats['total_scores']:,}")
    
    st.markdown("**Ontology mapping coverage:**")
    st.markdown(f"""
└─ EFO mapped: **{stats['efo_only']:,}**  
└─ MONDO mapped: **{stats['mondo_only']:,}**  
└─ HP mapped: **{stats['hp_only']:,}**  
└─ Multiple ontologies: **{stats['multiple_ontologies']:,}**  
└─ No mapping: **{stats['no_mapping']:,}** *(excluded from Kraken)*  
└─ **Total mappable: {stats['total_mappable']:,}**
""")
    
    st.markdown(f"- With harmonized files: **{stats['with_harmonized']:,}** (required for Kraken)")
    st.markdown(f"- **Kraken-eligible:** **{stats['kraken_eligible']:,}**")
    
    st.markdown("---")
    st.markdown("**Estimated graph impact:**")
    st.markdown(f"""
- PRS nodes: **{stats['kraken_eligible']:,}**
- PRS → Disease edges: **{stats['kraken_eligible']:,}** (via EFO/MONDO/HP)
- PRS → Gene edges: **~{stats['min_gene_edges']:,} - {stats['max_gene_edges']:,}**  
  *(tiered: <1K variants=50, 1K-100K=200, >100K=500 genes/PRS)*
- If storing all variant edges: **~{stats['total_variants']:,}** ⚠️
""")


def render_score_details(pgs_id: str, scores_df: pd.DataFrame):
    score_row = scores_df[scores_df['pgs_id'] == pgs_id].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Information**")
        st.write(f"- **PGS ID:** {score_row['pgs_id']}")
        st.write(f"- **Name:** {score_row.get('name', 'N/A')}")
        st.write(f"- **Traits:** {score_row.get('trait_names', 'N/A')}")
        st.write(f"- **Method:** {score_row.get('method_name', 'N/A')}")
        st.write(f"- **Classification:** {score_row.get('method_class', 'N/A')}")
        st.write(f"- **Variants:** {score_row.get('n_variants', 'N/A'):,}")
    
    with col2:
        st.write("**Publication**")
        st.write(f"- **Author:** {score_row.get('first_author', 'N/A')}")
        st.write(f"- **Date:** {score_row.get('publication_date', 'N/A')}")
        st.write(f"- **Journal:** {score_row.get('journal', 'N/A')}")
        if score_row.get('doi'):
            st.write(f"- **DOI:** [{score_row['doi']}](https://doi.org/{score_row['doi']})")
    
    st.write("**Scoring Files**")
    
    st.write(f"- **Original:** [Download]({score_row.get('ftp_scoring_file', '#')})")
    
    if score_row.get('grch37_available'):
        st.write(f"- **GRCh37 Harmonized:** [Download]({score_row.get('grch37_url', '#')})")
    else:
        st.write("- **GRCh37 Harmonized:** Not available")
    
    if score_row.get('grch38_available'):
        st.write(f"- **GRCh38 Harmonized:** [Download]({score_row.get('grch38_url', '#')})")
    else:
        st.write("- **GRCh38 Harmonized:** Not available")
    
    st.write("**Source GWAS**")
    gwas_ids = score_row.get('gwas_ids', '')
    if gwas_ids:
        gwas_links = ', '.join(f'[{gid.strip()}](https://www.ebi.ac.uk/gwas/studies/{gid.strip()})' for gid in gwas_ids.split(',') if gid.strip())
        st.write(f"- **GWAS Catalog:** {gwas_links}")
    else:
        st.write("- **GWAS Catalog:** Not reported")
    
    st.write("**Ancestry Coverage**")
    dev_anc = score_row.get('dev_ancestry', '')
    eval_anc = score_row.get('eval_ancestry', '')
    st.write(f"- **Development:** {translate_ancestry_codes(dev_anc) if dev_anc else 'Not reported'}")
    st.write(f"- **Evaluation:** {translate_ancestry_codes(eval_anc) if eval_anc else 'Not reported'}")


def render_traits_tab(scores_df):
    st.header("Traits Browser")
    
    traits_df = data_source.get_traits()
    categories_df = data_source.get_trait_categories()
    
    if traits_df.empty:
        st.warning("No traits loaded. Please check your internet connection.")
        return
    
    trait_tier_stats = compute_trait_tier_stats(scores_df) if not scores_df.empty else {}
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        search = st.text_input("Search traits", placeholder="EFO ID, name, description...")
        
        category_options = ['All'] + categories_df['category'].tolist() if not categories_df.empty else ['All']
        selected_category = st.selectbox("Category", options=category_options)
        
        min_scores = st.number_input("Minimum scores", min_value=0, value=0)
        
        st.write("**Best Quality Tier**")
        tier_filter = st.multiselect(
            "Filter by best tier",
            options=['Gold', 'Silver', 'Bronze', 'Unrated'],
            help="Filter traits by the best quality tier among their associated scores",
            label_visibility="collapsed",
            key="trait_tier_filter"
        )
        
        filtered_traits = filter_traits(
            traits_df,
            search_query=search if search else None,
            category=selected_category if selected_category != 'All' else None,
            min_scores=min_scores if min_scores > 0 else None
        )
        
        if trait_tier_stats and tier_filter:
            filtered_traits = filtered_traits[filtered_traits['trait_id'].apply(
                lambda x: trait_tier_stats.get(x, {}).get('best_tier', 'Unrated') in tier_filter
            )]
    
    with col2:
        st.metric("Traits Found", len(filtered_traits))
        
        if not filtered_traits.empty:
            csv_data = export_traits_csv(filtered_traits)
            st.download_button(
                label="Download Filtered Traits (CSV)",
                data=csv_data,
                file_name="pgs_catalog_traits.csv",
                mime="text/csv"
            )
            
            top_traits = filtered_traits.nlargest(15, 'n_scores')
            fig = px.bar(
                top_traits,
                x='n_scores',
                y='label',
                orientation='h',
                title="Top 15 Traits by Number of Associated Scores",
                labels={'n_scores': 'Number of Scores', 'label': 'Trait'}
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            display_df = filtered_traits.head(100).copy()
            if trait_tier_stats:
                tier_emoji = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉', 'Unrated': '⚫'}
                display_df['best_tier'] = display_df['trait_id'].apply(
                    lambda x: f"{tier_emoji.get(trait_tier_stats.get(x, {}).get('best_tier', 'Unrated'), '')} {trait_tier_stats.get(x, {}).get('best_tier', 'Unrated')}"
                )
                display_df['tier_breakdown'] = display_df['trait_id'].apply(
                    lambda x: trait_tier_stats.get(x, {}).get('breakdown', '')
                )
            
            display_cols = ['trait_id', 'label', 'n_scores', 'associated_pgs_ids', 'best_tier', 'tier_breakdown', 'categories']
            available_display = [c for c in display_cols if c in display_df.columns]
            
            st.dataframe(
                display_df[available_display],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'associated_pgs_ids': st.column_config.TextColumn("PGS IDs", width="large"),
                }
            )
    
    if not categories_df.empty:
        with st.expander("Trait Categories Overview"):
            fig = px.treemap(
                categories_df,
                path=['category'],
                values='n_traits',
                title="Trait Categories"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def render_pgs_publications(pgs_id: str):
    """Show development and all external evaluation publications for a PGS ID.
    
    Uses direct API calls - no dependency on cached scores data.
    """
    st.subheader(f"Publications for {pgs_id}")
    
    with st.spinner(f"Fetching {pgs_id} from API..."):
        score_info = data_source.get_score_by_id(pgs_id)
    
    if not score_info:
        st.warning(f"Score {pgs_id} not found in PGS Catalog.")
        return
    
    dev_pgp = score_info.get('pgp_id', '')
    dev_author = score_info.get('first_author', '')
    dev_date = score_info.get('publication_date', '')
    dev_doi = score_info.get('doi', '')
    trait_name = score_info.get('trait_names', 'Unknown trait')
    method_name = score_info.get('method_name', '')
    
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("Score", pgs_id)
    with info_cols[1]:
        st.metric("Method", method_name[:25] + "..." if len(method_name) > 25 else method_name if method_name else "Unknown")
    with info_cols[2]:
        st.metric("Trait", trait_name[:30] + "..." if len(trait_name) > 30 else trait_name)
    
    with st.spinner(f"Fetching evaluations for {pgs_id}..."):
        metrics_df = data_source.get_performance_metrics(pgs_id)
    
    publications = []
    
    if dev_pgp:
        publications.append({
            'pgp_id': dev_pgp,
            'source_type': 'Development',
            'first_author': dev_author,
            'publication_date': dev_date,
            'doi': dev_doi,
            'n_evaluations': 0,
            'best_auc': None,
            'ancestries': ''
        })
    
    if not metrics_df.empty:
        required_cols = ['pgp_id', 'first_author', 'publication_date', 'ppm_id']
        if all(c in metrics_df.columns for c in required_cols):
            agg_dict = {
                'first_author': 'first',
                'publication_date': 'first',
                'ppm_id': 'count',
            }
            if 'auc' in metrics_df.columns:
                agg_dict['auc'] = lambda x: x.dropna().max() if x.notna().any() else None
            if 'ancestry' in metrics_df.columns:
                agg_dict['ancestry'] = lambda x: '; '.join(sorted(set(a for anc in x.dropna() for a in str(anc).split('; ') if a)))
            if 'doi' in metrics_df.columns:
                agg_dict['doi'] = 'first'
            
            eval_pubs = metrics_df.groupby('pgp_id').agg(agg_dict).reset_index()
            
            for _, row in eval_pubs.iterrows():
                pgp = row['pgp_id']
                is_dev = pgp == dev_pgp
                best_auc = row.get('auc') if 'auc' in row else None
                ancestries = row.get('ancestry', '') if 'ancestry' in row else ''
                eval_doi = row.get('doi', '') if 'doi' in row else ''
                
                if is_dev:
                    for pub in publications:
                        if pub['pgp_id'] == dev_pgp:
                            pub['n_evaluations'] = row['ppm_id']
                            pub['best_auc'] = best_auc
                            pub['ancestries'] = ancestries
                            pub['source_type'] = 'Development + Evaluation'
                else:
                    publications.append({
                        'pgp_id': pgp,
                        'source_type': 'External Evaluation',
                        'first_author': row['first_author'],
                        'publication_date': row['publication_date'],
                        'doi': eval_doi,
                        'n_evaluations': row['ppm_id'],
                        'best_auc': best_auc,
                        'ancestries': ancestries
                    })
    
    if not publications:
        st.warning(f"No publication data found for {pgs_id}")
        return
    
    pub_df = pd.DataFrame(publications)
    
    source_emoji = {'Development': '📝', 'External Evaluation': '🔬', 'Development + Evaluation': '📝🔬'}
    pub_df['source'] = pub_df['source_type'].apply(lambda x: f"{source_emoji.get(x, '')} {x}")
    
    if 'doi' in pub_df.columns:
        pub_df['doi_link'] = pub_df['doi'].apply(lambda x: f"https://doi.org/{x}" if x else '')
    
    col1, col2 = st.columns(2)
    with col1:
        dev_count = len(pub_df[pub_df['source_type'].str.contains('Development')])
        st.metric("Development Publications", dev_count)
    with col2:
        eval_count = len(pub_df[pub_df['source_type'] == 'External Evaluation'])
        st.metric("External Evaluation Publications", eval_count)
    
    display_df = pub_df.copy()
    display_df['best_auc'] = display_df['best_auc'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else '-')
    
    display_cols = ['pgp_id', 'source', 'first_author', 'doi_link', 'n_evaluations', 'best_auc', 'ancestries']
    available_cols = [c for c in display_cols if c in display_df.columns]
    
    st.dataframe(
        display_df[available_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            'pgp_id': st.column_config.TextColumn("PGP ID"),
            'source': st.column_config.TextColumn("Source Type"),
            'first_author': st.column_config.TextColumn("First Author"),
            'doi_link': st.column_config.LinkColumn("DOI", display_text="https://doi\\.org/(.+)"),
            'n_evaluations': st.column_config.NumberColumn("# Evals"),
            'best_auc': st.column_config.TextColumn("Best AUC"),
            'ancestries': st.column_config.TextColumn("Ancestry Coverage")
        }
    )


def render_publications_tab_independent():
    """Publications tab that works independently of main data load.
    
    PGS ID search uses direct API calls, browse mode uses publications API.
    No dependency on cached scores data for the search functionality.
    """
    st.header("Publications Browser")
    
    pgs_search = st.text_input(
        "Search by PGS ID",
        placeholder="e.g., PGS000013 - shows development + all evaluation publications",
        key="pub_pgs_search"
    )
    
    if pgs_search:
        pgs_id = pgs_search.upper().strip()
        if not pgs_id.startswith("PGS"):
            pgs_id = f"PGS{pgs_id.zfill(6)}"
        
        render_pgs_publications(pgs_id)
        st.divider()
    
    st.subheader("Browse All Publications")
    st.info("The browse section shows publications from the PGS Catalog. For detailed score lookups, use the PGS ID search above.")
    
    with st.spinner("Loading publications..."):
        publications_df = data_source.get_publications()
    
    if publications_df.empty:
        st.warning("Publications are loading. This may take a moment...")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("**Filters**")
        
        search = st.text_input("Search publications", placeholder="PGP ID, author, title...", key="pub_browse_search")
        
        years = sorted(publications_df['date_publication'].str[:4].dropna().unique(), reverse=True)
        year_options = ['All'] + list(years)
        selected_year = st.selectbox("Publication Year", options=year_options, key="pub_browse_year")
        
        has_dev = st.checkbox("Has PGS development", key="pub_browse_dev")
        has_eval = st.checkbox("Has PGS evaluation", key="pub_browse_eval")
        
        filtered_pubs = filter_publications(
            publications_df,
            search_query=search if search else None,
            year=int(selected_year) if selected_year != 'All' else None,
            has_development=has_dev,
            has_evaluation=has_eval
        )
    
    with col2:
        st.metric("Publications Found", len(filtered_pubs))
        
        if not filtered_pubs.empty:
            csv_data = export_publications_csv(filtered_pubs)
            st.download_button(
                label="Download Filtered Publications (CSV)",
                data=csv_data,
                file_name="pgs_catalog_publications.csv",
                mime="text/csv",
                key="pub_browse_download"
            )
            
            pubs_by_year = publications_df.copy()
            pubs_by_year['year'] = pubs_by_year['date_publication'].str[:4]
            year_counts = pubs_by_year['year'].value_counts().sort_index()
            
            fig = px.bar(
                x=year_counts.index,
                y=year_counts.values,
                title="Publications per Year",
                labels={'x': 'Year', 'y': 'Number of Publications'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="pub_browse_chart")
            
            display_df = filtered_pubs.head(100).copy()
            
            if 'doi' in display_df.columns:
                display_df['doi_link'] = display_df.apply(
                    lambda row: f"https://doi.org/{row['doi']}" if row.get('doi') else (row.get('url') or ''),
                    axis=1
                )
            
            display_cols = ['pgp_id', 'first_author', 'title', 'doi_link', 'journal', 
                          'date_publication', 'n_development', 'n_evaluation']
            available_display = [c for c in display_cols if c in display_df.columns]
            
            column_config = {
                'title': st.column_config.TextColumn("Title", width="large"),
                'doi_link': st.column_config.LinkColumn(
                    "DOI",
                    display_text="https://doi\\.org/(.+)"
                )
            }
            
            st.dataframe(
                display_df[available_display],
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                key="pub_browse_table"
            )


def render_publications_tab(scores_df):
    st.header("Publications Browser")
    
    pgs_search = st.text_input(
        "Search by PGS ID",
        placeholder="e.g., PGS000013 - shows development + all evaluation publications",
        key="pub_pgs_search_legacy"
    )
    
    if pgs_search:
        pgs_id = pgs_search.upper().strip()
        if not pgs_id.startswith("PGS"):
            pgs_id = f"PGS{pgs_id.zfill(6)}"
        
        render_pgs_publications(pgs_id)
        st.divider()
        st.subheader("Browse All Publications")
    
    publications_df = data_source.get_publications()
    
    if publications_df.empty:
        st.warning("No publications loaded. Please check your internet connection.")
        return
    
    pub_tier_stats = compute_publication_tier_stats(scores_df) if not scores_df.empty else {}
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        search = st.text_input("Search publications", placeholder="PGP ID, author, title...")
        
        years = sorted(publications_df['date_publication'].str[:4].dropna().unique(), reverse=True)
        year_options = ['All'] + list(years)
        selected_year = st.selectbox("Publication Year", options=year_options)
        
        has_dev = st.checkbox("Has PGS development")
        has_eval = st.checkbox("Has PGS evaluation")
        
        st.write("**Best Quality Tier**")
        tier_filter = st.multiselect(
            "Filter by best tier",
            options=['Gold', 'Silver', 'Bronze', 'Unrated'],
            help="Filter publications by the best quality tier among their scores",
            label_visibility="collapsed",
            key="pub_tier_filter"
        )
        
        filtered_pubs = filter_publications(
            publications_df,
            search_query=search if search else None,
            year=int(selected_year) if selected_year != 'All' else None,
            has_development=has_dev,
            has_evaluation=has_eval
        )
        
        if pub_tier_stats and tier_filter:
            filtered_pubs = filtered_pubs[filtered_pubs['pgp_id'].apply(
                lambda x: pub_tier_stats.get(x, {}).get('best_tier', 'Unrated') in tier_filter
            )]
    
    with col2:
        st.metric("Publications Found", len(filtered_pubs))
        
        if not filtered_pubs.empty:
            csv_data = export_publications_csv(filtered_pubs)
            st.download_button(
                label="Download Filtered Publications (CSV)",
                data=csv_data,
                file_name="pgs_catalog_publications.csv",
                mime="text/csv"
            )
            
            pubs_by_year = publications_df.copy()
            pubs_by_year['year'] = pubs_by_year['date_publication'].str[:4]
            year_counts = pubs_by_year['year'].value_counts().sort_index()
            
            fig = px.bar(
                x=year_counts.index,
                y=year_counts.values,
                title="Publications per Year",
                labels={'x': 'Year', 'y': 'Number of Publications'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            display_df = filtered_pubs.head(100).copy()
            if pub_tier_stats:
                tier_emoji = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉', 'Unrated': '⚫'}
                display_df['best_tier'] = display_df['pgp_id'].apply(
                    lambda x: f"{tier_emoji.get(pub_tier_stats.get(x, {}).get('best_tier', 'Unrated'), '')} {pub_tier_stats.get(x, {}).get('best_tier', 'Unrated')}"
                )
            
            if 'doi' in display_df.columns:
                display_df['doi_link'] = display_df.apply(
                    lambda row: f"https://doi.org/{row['doi']}" if row.get('doi') else (row.get('url') or ''),
                    axis=1
                )
            
            display_cols = ['pgp_id', 'first_author', 'title', 'doi_link', 'best_tier', 'journal', 
                          'date_publication', 'n_development', 'n_evaluation']
            available_display = [c for c in display_cols if c in display_df.columns]
            
            column_config = {
                'title': st.column_config.TextColumn("Title", width="large"),
                'doi_link': st.column_config.LinkColumn(
                    "DOI",
                    display_text="https://doi\\.org/(.+)"
                )
            }
            
            st.dataframe(
                display_df[available_display],
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )


def render_performance_tab_independent(scores_df=None):
    st.header("Performance Metrics")
    
    st.info("Performance metrics show how well a PGS predicts the trait in different populations. "
            "Context matters: ancestry and sample size significantly affect metric interpretation.")
    
    pgs_id = st.text_input("Enter PGS ID to view performance metrics", placeholder="e.g., PGS000001", key="perf_pgs_search")
    
    if pgs_id:
        pgs_id = pgs_id.upper().strip()
        
        if not pgs_id.startswith("PGS"):
            pgs_id = f"PGS{pgs_id.zfill(6)}"
        
        with st.spinner(f"Fetching performance metrics for {pgs_id}..."):
            metrics_df = data_source.get_performance_metrics(pgs_id)
        
        if metrics_df.empty:
            st.warning(f"No performance metrics found for {pgs_id}")
            return
        
        quality_tier = 'Unknown'
        score_info_dict = data_source.get_score_by_id(pgs_id)
        if score_info_dict:
            method_class = classify_method(score_info_dict.get('method_name', ''))
            n_evaluations = len(metrics_df)
            ancestry_groups = set()
            if 'ancestry' in metrics_df.columns:
                for anc in metrics_df['ancestry'].dropna():
                    for a in str(anc).split('; '):
                        if a.strip():
                            ancestry_groups.add(a.strip())
            n_ancestry_groups = len(ancestry_groups)
            quality_tier = compute_quality_tier(method_class, n_evaluations, n_ancestry_groups)
        
        tier_emoji = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉', 'Unrated': '⚫', 'Unknown': '❓'}
        
        st.subheader(f"Performance Metrics for {pgs_id}")
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("Number of Evaluations", len(metrics_df))
        with metric_cols[1]:
            st.metric("Quality Tier", f"{tier_emoji.get(quality_tier, '')} {quality_tier}")
        
        if 'ancestry' in metrics_df.columns:
            ancestry_counts = metrics_df['ancestry'].str.split('; ').explode().value_counts()
            if not ancestry_counts.empty:
                translated_index = [translate_ancestry_codes(c) for c in ancestry_counts.index]
                fig = px.pie(
                    values=ancestry_counts.values,
                    names=translated_index,
                    title="Evaluation Ancestry Distribution",
                    color=translated_index,
                    color_discrete_map={translate_ancestry_codes(k): v for k, v in get_ancestry_colors().items()}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Evaluation Details")
        st.write("Each row shows ancestry, sample size, and individual metrics for proper context.")
        
        has_auc = metrics_df['auc'].notna().any() if 'auc' in metrics_df.columns else False
        has_r2 = metrics_df['r2'].notna().any() if 'r2' in metrics_df.columns else False
        has_or = metrics_df['or_val'].notna().any() if 'or_val' in metrics_df.columns else False
        has_hr = metrics_df['hr'].notna().any() if 'hr' in metrics_df.columns else False
        has_beta = metrics_df['beta'].notna().any() if 'beta' in metrics_df.columns else False
        
        display_df = metrics_df.copy()
        if 'ancestry' in display_df.columns:
            display_df['ancestry'] = display_df['ancestry'].apply(lambda x: translate_ancestry_codes(str(x)) if pd.notna(x) else x)
        
        display_cols = ['ppm_id', 'ancestry', 'sample_size']
        if has_auc:
            display_cols.extend(['auc', 'auc_ci'])
        if has_r2:
            display_cols.extend(['r2', 'r2_ci'])
        if has_or:
            display_cols.extend(['or_val', 'or_ci'])
        if has_hr:
            display_cols.extend(['hr', 'hr_ci'])
        if has_beta:
            display_cols.extend(['beta', 'beta_ci'])
        display_cols.extend(['cohorts', 'phenotyping_reported', 'first_author', 'publication_date'])
        
        available_display = [c for c in display_cols if c in display_df.columns]
        
        column_config = {
            'auc': st.column_config.NumberColumn("AUC", format="%.3f"),
            'auc_ci': st.column_config.TextColumn("AUC CI"),
            'r2': st.column_config.NumberColumn("R²", format="%.3f"),
            'r2_ci': st.column_config.TextColumn("R² CI"),
            'or_val': st.column_config.NumberColumn("OR", format="%.2f"),
            'or_ci': st.column_config.TextColumn("OR CI"),
            'hr': st.column_config.NumberColumn("HR", format="%.2f"),
            'hr_ci': st.column_config.TextColumn("HR CI"),
            'beta': st.column_config.NumberColumn("Beta", format="%.3f"),
            'beta_ci': st.column_config.TextColumn("Beta CI"),
            'sample_size': st.column_config.NumberColumn("Sample Size", format="%d"),
        }
        
        st.dataframe(
            display_df[available_display],
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        
        csv_data = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Performance Metrics (CSV)",
            data=csv_data,
            file_name=f"pgs_{pgs_id}_performance.csv",
            mime="text/csv"
        )


def render_sidebar_info(eval_summary_df):
    st.header("Data Status")
    
    cache_meta = st.session_state.get('cache_metadata', {})
    scores_count = cache_meta.get('scores_count', 0)
    evals_count = cache_meta.get('evals_count', 0)
    scores_loaded = cache_meta.get('scores_loaded', 0)
    expected_scores = cache_meta.get('expected_scores', 0)
    last_checked = cache_meta.get('last_checked')
    is_fresh = cache_meta.get('is_fresh', True)
    is_complete = cache_meta.get('is_complete', True)
    
    if scores_loaded > 0 or scores_count > 0:
        if not is_complete and expected_scores > scores_loaded:
            st.warning(f"⚠️ **Data Status: Incomplete**")
            st.markdown(f"Scores: **{scores_loaded:,}** / {expected_scores:,}")
            st.caption("API error during load - some scores missing")
            if st.button("🔄 Retry Full Load", type="primary", use_container_width=True):
                st.session_state.force_refresh = True
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Scores Loaded", f"{scores_loaded:,}" if scores_loaded > 0 else f"{scores_count:,}")
            with col2:
                st.metric("Evaluations", f"{evals_count:,}")
            
            if last_checked:
                try:
                    checked_dt = datetime.fromisoformat(last_checked)
                    st.caption(f"Last checked: {checked_dt.strftime('%b %d, %Y %H:%M')}")
                except:
                    pass
            
            if is_fresh:
                st.success("✓ Up to date with PGS Catalog", icon="✅")
    
    st.divider()
    
    st.header("About")
    st.markdown("""
    This app explores the [PGS Catalog](https://www.pgscatalog.org/), 
    an open database of polygenic scores.
    
    **Features:**
    - Browse and filter scores by method, ancestry, traits
    - Quality tier classification for score assessment
    - Kraken ingest estimation for graph planning
    - Explore trait categories and ontologies
    - View publication history
    - Analyze performance metrics with ancestry context
    
    **Data Source:**  
    PGS Catalog REST API (cached up to 30 days, auto-refreshes when new data available)
    
    **Links:**
    - [PGS Catalog](https://www.pgscatalog.org/)
    - [API Documentation](https://www.pgscatalog.org/rest/)
    - [pgscatalog-utils](https://pypi.org/project/pgscatalog-utils/)
    """)
    
    st.divider()
    
    st.header("Quality Tiers")
    st.markdown("""
    **🥇 Gold (ARK-ready):**  
    LD-aware method + 2+ evaluations + multi-ancestry (2+ groups)
    
    **🥈 Silver (Research-grade):**  
    LD-aware method + at least 1 evaluation
    
    **🥉 Bronze:**  
    Moderate (C+T) method + at least 1 evaluation
    
    **⚫ Unrated:**  
    Missing evaluations or Other/Unknown method
    """)
    
    st.divider()
    
    st.header("Method Classification")
    st.markdown("""
    **High (LD-aware):**  
    PRS-CS, PRS-CSx, LDpred, LDpred2, lassosum, SBayesR, MegaPRS
    
    **Moderate (C+T):**  
    C+T, P+T, PRSice, PRSice2
    
    **Other:**  
    Unclassified methods
    """)
    
    st.divider()
    
    st.header("Ancestry Coverage")
    st.caption("Scores with evaluations per ancestry group")
    
    ancestry_counts = {}
    if not eval_summary_df.empty and 'ancestry_groups' in eval_summary_df.columns:
        for groups in eval_summary_df['ancestry_groups'].dropna():
            if groups:
                for ancestry in groups.split('; '):
                    ancestry = ancestry.strip()
                    if ancestry:
                        ancestry_counts[ancestry] = ancestry_counts.get(ancestry, 0) + 1
    
    standard_ancestries = [
        ("European", "EUR"),
        ("East Asian", "EAS"),
        ("African", "AFR"),
        ("South Asian", "SAS"),
        ("Hispanic or Latin American", "HIS"),
        ("Greater Middle Eastern", "GME"),
        ("Oceanian", "OTH"),
        ("Native American", "AMR"),
        ("Multi-ancestry", "MAO"),
    ]
    
    if ancestry_counts:
        for label, abbrev in standard_ancestries:
            count = ancestry_counts.get(label, 0)
            if count > 0:
                st.write(f"**{abbrev}**: {label} ({count:,})")
            else:
                st.write(f"**{abbrev}**: {label}")
    else:
        st.caption("Loading ancestry data...")
        for label, abbrev in standard_ancestries:
            st.write(f"**{abbrev}**: {label}")


def render_compare_tab():
    st.header("PGS Pairwise Comparison")

    result = load_comparison_data()
    if result is None or result[0] is None:
        st.warning(
            "Comparison data not yet available. Run the pairwise comparison pipeline to generate data files in the `data/` directory."
        )
        st.markdown("""
        **Required files:**
        - `data/pgs_pairwise_stats.parquet` — Summary statistics for all PGS pairs
        - `data/pgs_pairwise_variants_sample.json.gz` — Sample variant data for plotting (or full variants file)
        - `data/pipeline_metadata.json` — Pipeline metadata
        """)
        return

    stats_df, metadata = result

    if metadata:
        meta_parts = []
        if metadata.get("generated_date"):
            meta_parts.append(f"Data generated: {metadata['generated_date']}")
        if metadata.get("genome_build"):
            meta_parts.append(f"Genome build: {metadata['genome_build']}")
        if metadata.get("n_pairs"):
            meta_parts.append(f"{metadata['n_pairs']} pairs")
        if metadata.get("n_traits"):
            meta_parts.append(f"{metadata['n_traits']} traits")
        if meta_parts:
            st.caption(" | ".join(meta_parts))

    trait_options = ["All traits"] + sorted(stats_df["trait_label"].unique().tolist())

    st.subheader("Filters")
    fcol1, fcol2, fcol3 = st.columns(3)

    with fcol1:
        selected_trait = st.selectbox(
            "Trait",
            options=trait_options,
            key="compare_trait_filter",
        )
    with fcol2:
        max_shared = int(stats_df["n_shared"].max()) if not stats_df.empty else 10000
        min_shared = st.slider(
            "Min Shared Variants",
            min_value=0,
            max_value=max_shared,
            value=0,
            step=max(1, max_shared // 100),
        )
    with fcol3:
        min_corr = st.slider(
            "Min Correlation (r)",
            min_value=-1.0,
            max_value=1.0,
            value=-1.0,
            step=0.05,
        )

    filtered_stats = filter_comparison_data(
        stats_df,
        trait=selected_trait,
        min_shared=min_shared,
        min_correlation=min_corr,
    )

    st.subheader("Pairwise Comparison Statistics")

    if filtered_stats.empty:
        st.info("No pairs match the current filters.")
    else:
        st.caption(f"Showing {len(filtered_stats)} of {len(stats_df)} pairs")

        display_df = filtered_stats.copy()
        display_df["pgs_link_1"] = display_df["pgs_id_1"].apply(
            lambda x: f"https://www.pgscatalog.org/score/{x}/"
        )
        display_df["pgs_link_2"] = display_df["pgs_id_2"].apply(
            lambda x: f"https://www.pgscatalog.org/score/{x}/"
        )
        display_df["pct_overlap_display"] = display_df["pct_overlap_1"].apply(
            lambda x: f"{x:.1f}%"
        )
        display_df["correlation_display"] = display_df["pearson_r"].apply(
            lambda x: f"{x:.3f}"
        )
        display_df["concordance_display"] = display_df["pct_concordant_sign"].apply(
            lambda x: f"{x:.1f}%"
        )

        table_cols = [
            "pgs_id_1", "pgs_link_1", "pgs_id_2", "pgs_link_2",
            "trait_label", "n_variants_1", "n_variants_2", "n_shared",
            "pct_overlap_display", "correlation_display", "concordance_display",
        ]
        available_cols = [c for c in table_cols if c in display_df.columns]

        column_config = {
            "pgs_id_1": st.column_config.TextColumn("PGS 1"),
            "pgs_link_1": st.column_config.LinkColumn("Link 1", display_text="View"),
            "pgs_id_2": st.column_config.TextColumn("PGS 2"),
            "pgs_link_2": st.column_config.LinkColumn("Link 2", display_text="View"),
            "trait_label": st.column_config.TextColumn("Trait"),
            "n_variants_1": st.column_config.NumberColumn("N1", format="%d"),
            "n_variants_2": st.column_config.NumberColumn("N2", format="%d"),
            "n_shared": st.column_config.NumberColumn("Shared", format="%d"),
            "pct_overlap_display": st.column_config.TextColumn("Overlap %"),
            "correlation_display": st.column_config.TextColumn("Correlation"),
            "concordance_display": st.column_config.TextColumn("Concordance"),
        }

        st.dataframe(
            display_df[available_cols],
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

        csv_data = export_comparison_csv(filtered_stats)
        st.download_button(
            label="Download Filtered Pairs (CSV)",
            data=csv_data,
            file_name="pgs_pairwise_comparison.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Comparison Visualization")
    st.markdown("Enter two PGS IDs to visualize their variant overlap and effect weight correlation.")

    all_pgs_ids = sorted(set(stats_df["pgs_id_1"]) | set(stats_df["pgs_id_2"]))

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        pgs1_input = st.selectbox(
            "PGS 1",
            options=[""] + all_pgs_ids,
            key="compare_pgs1",
        )
    with vcol2:
        pgs2_input = st.selectbox(
            "PGS 2",
            options=[""] + all_pgs_ids,
            key="compare_pgs2",
        )

    if pgs1_input and pgs2_input and pgs1_input != pgs2_input:
        pair_row = stats_df[
            ((stats_df["pgs_id_1"] == pgs1_input) & (stats_df["pgs_id_2"] == pgs2_input)) |
            ((stats_df["pgs_id_1"] == pgs2_input) & (stats_df["pgs_id_2"] == pgs1_input))
        ]

        if pair_row.empty:
            st.warning(f"No comparison data found for {pgs1_input} and {pgs2_input}. These scores may not share the same trait.")
        else:
            pair_stats = pair_row.iloc[0].to_dict()
            actual_pgs1 = pair_stats["pgs_id_1"]
            actual_pgs2 = pair_stats["pgs_id_2"]

            n_shared = pair_stats["n_shared"]
            if n_shared == 0:
                st.warning("No overlapping variants between these scores. Scatterplot is not available.")
            else:
                with st.spinner("Loading variant data..."):
                    variant_data = load_variant_data(actual_pgs1, actual_pgs2)

                if variant_data is None:
                    # File not available (production mode)
                    st.info(
                        "**Scatterplot data not available in production.**\n\n"
                        "The variant-level data file (4.6GB) is not deployed to the web server. "
                        "To view scatterplots, run the Jupyter notebook locally with the full dataset."
                    )
                elif not variant_data:
                    # File exists but pair not found
                    st.warning(f"No shared variants found between {actual_pgs1} and {actual_pgs2}")
                else:
                    # Normal flow - render scatterplot
                    fig = create_scatterplot(variant_data, actual_pgs1, actual_pgs2, pair_stats)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Summary Statistics:**")
            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                st.metric(f"Variants in {actual_pgs1}", f"{pair_stats['n_variants_1']:,}")
            with scol2:
                st.metric(f"Variants in {actual_pgs2}", f"{pair_stats['n_variants_2']:,}")
            with scol3:
                overlap_text = f"{pair_stats['n_shared']:,}"
                overlap_detail = f"{pair_stats['pct_overlap_1']:.1f}% of {actual_pgs1}, {pair_stats['pct_overlap_2']:.1f}% of {actual_pgs2}"
                st.metric("Shared Variants", overlap_text, help=overlap_detail)
            with scol4:
                p_val = pair_stats.get("pearson_p", 0)
                p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
                st.metric("Pearson r", f"{pair_stats['pearson_r']:.3f}", help=p_str)

            st.markdown(f"**Sign concordance:** {pair_stats['pct_concordant_sign']:.1f}%")

            if pair_stats.get("n_sampled") and pair_stats["n_sampled"] < pair_stats["n_shared"]:
                st.caption(f"Showing {pair_stats['n_sampled']:,} of {pair_stats['n_shared']:,} shared variants (sampled for performance)")

            st.markdown("**Interpretation:**")
            st.info(get_interpretation(pair_stats))

    elif pgs1_input and pgs2_input and pgs1_input == pgs2_input:
        st.warning("Please select two different PGS IDs.")

    st.divider()

    st.subheader("PRS Correlation Network")

    network_trait_options = sorted(stats_df["trait_label"].unique().tolist())
    if not network_trait_options:
        st.info("No trait data available for network view.")
        return

    ncol1, ncol2 = st.columns(2)
    with ncol1:
        network_trait = st.selectbox(
            "Trait",
            options=network_trait_options,
            key="network_trait",
        )
    with ncol2:
        metric = st.selectbox(
            "Edge Metric",
            options=["pearson_r", "pct_concordant_sign", "jaccard_index"],
            format_func=lambda x: {
                "pearson_r": "Pearson Correlation (r)",
                "pct_concordant_sign": "Sign Concordance (%)",
                "jaccard_index": "Jaccard Index",
            }.get(x, x),
            key="network_metric",
        )

    default_threshold = 0.5
    if metric == "pct_concordant_sign":
        default_threshold = 70.0
        threshold = st.slider(
            f"Edge Threshold (show edges where {metric} >= threshold)",
            min_value=0.0,
            max_value=100.0,
            value=default_threshold,
            step=1.0,
            key="network_threshold",
        )
    else:
        threshold = st.slider(
            f"Edge Threshold (show edges where {metric} >= threshold)",
            min_value=0.0 if metric != "pearson_r" else -1.0,
            max_value=1.0,
            value=default_threshold,
            step=0.05,
            key="network_threshold",
        )

    if network_trait:
        G = build_network(stats_df, network_trait, metric, threshold)

        if G.number_of_nodes() == 0:
            st.warning("No PRS found for this trait.")
        elif G.number_of_nodes() == 1:
            st.info("Only 1 PRS for this trait — no comparisons possible.")
        else:
            fig = plot_network(G, f"PRS Network: {network_trait}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            net_stats = get_network_stats(G)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("PRS (nodes)", net_stats["n_nodes"])
            mcol2.metric("Edges", f"{net_stats['n_edges']} / {net_stats['n_possible_edges']}")
            mcol3.metric("Clusters", net_stats["n_clusters"])
            mcol4.metric("Redundant (r>0.95)", net_stats["redundant_pairs"])

            if net_stats["isolated_nodes"] > 0:
                st.caption(f"{net_stats['isolated_nodes']} PRS have no connections above the threshold")

    with st.expander("How to interpret the network"):
        st.markdown("""
        **What the network shows:**
        - Each **node** is a PRS for the selected trait
        - **Edges** connect PRS pairs with the selected metric above the threshold
        - **Node size** reflects number of variants in the PRS
        - **Node color** reflects how many connections (darker = more connected)
        - **Edge color** reflects correlation strength (green = high, yellow = moderate, red = low)

        **Patterns to look for:**

        | Pattern | Interpretation |
        |---------|----------------|
        | Tight cluster with thick green edges | Highly correlated scores — likely redundant, pick best-validated one |
        | Multiple separate clusters | Different "families" of PRS using different approaches or GWAS sources |
        | Isolated nodes | PRS that don't share much signal with others — may capture unique variance |
        | Red/yellow edges | Moderate correlation — scores may complement each other |

        **Recommended workflow:**
        1. Start with threshold ~0.5 to see overall structure
        2. Raise threshold to 0.8+ to identify near-redundant pairs
        3. Lower threshold to 0.3 to see weaker connections
        4. Select specific pairs above to compare in the scatterplot
        """)


def render_supplemental_tab():
    """Render supplemental information tab with PDF document."""
    st.header("Supplemental Information")
    
    pdf_path = "prs-quality-assessment-framework.pdf"
    
    import os
    if os.path.exists(pdf_path):
        st.markdown("### PRS Quality Assessment Framework")
        st.markdown("This document describes the quality assessment framework used for evaluating polygenic risk scores.")
        
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name="prs-quality-assessment-framework.pdf",
            mime="application/pdf"
        )
        
        import base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.warning("PDF document not found. Please ensure 'prs-quality-assessment-framework.pdf' is in the project directory.")
        st.info("To add the document, upload a PDF file named 'prs-quality-assessment-framework.pdf' to the project root.")


if __name__ == "__main__":
    main()
