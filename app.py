import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_layer import get_data_source
from utils import (
    add_method_classification, add_quality_tiers, filter_scores, filter_traits, filter_publications,
    export_scores_csv, export_traits_csv, export_publications_csv, export_kraken_ingest_csv,
    get_method_class_colors, get_ancestry_colors, get_quality_tier_colors, 
    classify_method, compute_kraken_stats, compute_trait_tier_stats, compute_publication_tier_stats
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


@st.cache_data(ttl=86400, show_spinner="Loading enriched scores data...")
def get_enriched_scores():
    """Get scores dataframe enriched with method classification and quality tiers."""
    scores_df = data_source.get_scores()
    eval_summary_df = data_source.get_evaluation_summary()
    
    if scores_df.empty:
        return scores_df, eval_summary_df
    
    scores_df = add_method_classification(scores_df)
    scores_df = add_quality_tiers(scores_df, eval_summary_df)
    return scores_df, eval_summary_df


def main():
    st.markdown('<p class="main-header">PGS Catalog Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Browse and explore polygenic scores, traits, and publications from the PGS Catalog</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scores", "Traits", "Publications", "Performance Metrics", "Supplemental Info"])
    
    with tab1:
        render_scores_tab()
    
    with tab2:
        render_traits_tab()
    
    with tab3:
        render_publications_tab()
    
    with tab4:
        render_performance_tab()
    
    with tab5:
        render_supplemental_tab()
    
    with st.sidebar:
        render_sidebar_info()


def render_scores_tab():
    st.header("Polygenic Scores Browser")
    
    scores_df, eval_summary_df = get_enriched_scores()
    
    if scores_df.empty:
        st.warning("No scores loaded. Please check your internet connection.")
        return
    
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
        
        filtered_df = filter_scores(
            scores_df,
            search_query=search if search else None,
            method_classes=method_classes if method_classes else None,
            quality_tiers=quality_tiers if quality_tiers else None,
            ontology_filters=ontology_filters if ontology_filters else None,
            has_grch37=has_grch37,
            has_grch38=has_grch38,
            min_variants=min_var,
            max_variants=max_var
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
            kraken_stats = compute_kraken_stats(filtered_df)
            st.metric("Kraken-Eligible", kraken_stats['kraken_eligible'])
        
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
            
            display_cols = ['pgs_id', 'pgs_link', 'name', 'trait_names', 'quality_tier', 'method_class', 
                          'n_evaluations', 'n_variants', 'efo_mapped', 'mondo_mapped', 'hp_mapped', 'grch38_available', 'first_author']
            available_display = [c for c in display_cols if c in display_df.columns]
            
            display_df = display_df[available_display]
            
            if 'quality_tier' in display_df.columns:
                tier_emoji = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉', 'Unrated': '⚫'}
                display_df['quality_tier'] = display_df['quality_tier'].apply(
                    lambda x: f"{tier_emoji.get(x, '')} {x}"
                )
            
            column_config = {
                'pgs_id': st.column_config.TextColumn("PGS ID"),
                'pgs_link': st.column_config.LinkColumn("Link", display_text="View")
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
    
    st.write("**Ancestry Coverage**")
    st.write(f"- **Development:** {score_row.get('dev_ancestry', 'N/A')}")
    st.write(f"- **Evaluation:** {score_row.get('eval_ancestry', 'N/A')}")


def render_traits_tab():
    st.header("Traits Browser")
    
    traits_df = data_source.get_traits()
    categories_df = data_source.get_trait_categories()
    scores_df, _ = get_enriched_scores()
    
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
            
            display_cols = ['trait_id', 'label', 'n_scores', 'best_tier', 'tier_breakdown', 'categories']
            available_display = [c for c in display_cols if c in display_df.columns]
            
            st.dataframe(
                display_df[available_display],
                use_container_width=True,
                hide_index=True
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


def render_publications_tab():
    st.header("Publications Browser")
    
    publications_df = data_source.get_publications()
    scores_df, _ = get_enriched_scores()
    
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


def render_performance_tab():
    st.header("Performance Metrics")
    
    st.info("Performance metrics show how well a PGS predicts the trait in different populations. "
            "Context matters: ancestry and sample size significantly affect metric interpretation.")
    
    scores_df, _ = get_enriched_scores()
    
    if scores_df.empty:
        st.warning("Load scores first to search for performance metrics.")
        return
    
    pgs_id = st.text_input("Enter PGS ID to view performance metrics", placeholder="e.g., PGS000001")
    
    if pgs_id:
        pgs_id = pgs_id.upper().strip()
        
        if not pgs_id.startswith("PGS"):
            pgs_id = f"PGS{pgs_id.zfill(6)}"
        
        metrics_df = data_source.get_performance_metrics(pgs_id)
        
        if metrics_df.empty:
            st.warning(f"No performance metrics found for {pgs_id}")
            return
        
        score_info = scores_df[scores_df['pgs_id'] == pgs_id]
        quality_tier = score_info['quality_tier'].iloc[0] if not score_info.empty and 'quality_tier' in score_info.columns else 'Unknown'
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
                fig = px.pie(
                    values=ancestry_counts.values,
                    names=ancestry_counts.index,
                    title="Evaluation Ancestry Distribution",
                    color=ancestry_counts.index,
                    color_discrete_map=get_ancestry_colors()
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Evaluation Details")
        st.write("Each row shows ancestry, sample size, and metrics for proper context.")
        
        display_cols = ['ppm_id', 'ancestry', 'sample_size', 'cohorts', 'metrics', 
                       'phenotyping_reported', 'first_author', 'publication_date']
        available_display = [c for c in display_cols if c in metrics_df.columns]
        
        st.dataframe(
            metrics_df[available_display],
            use_container_width=True,
            hide_index=True
        )
        
        csv_data = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Performance Metrics (CSV)",
            data=csv_data,
            file_name=f"pgs_{pgs_id}_performance.csv",
            mime="text/csv"
        )


def render_sidebar_info():
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
    PGS Catalog REST API (cached 24hrs)
    
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
    
    _, eval_summary_df = get_enriched_scores()
    
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
