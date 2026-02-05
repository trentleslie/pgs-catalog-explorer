import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_layer import get_data_source
from utils import (
    add_method_classification, filter_scores, filter_traits, filter_publications,
    export_scores_csv, export_traits_csv, export_publications_csv,
    get_method_class_colors, get_ancestry_colors, classify_method
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


def main():
    st.markdown('<p class="main-header">PGS Catalog Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Browse and explore polygenic scores, traits, and publications from the PGS Catalog</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Scores", "Traits", "Publications", "Performance Metrics"])
    
    with tab1:
        render_scores_tab()
    
    with tab2:
        render_traits_tab()
    
    with tab3:
        render_publications_tab()
    
    with tab4:
        render_performance_tab()
    
    with st.sidebar:
        render_sidebar_info()


def render_scores_tab():
    st.header("Polygenic Scores Browser")
    
    scores_df = data_source.get_scores()
    
    if scores_df.empty:
        st.warning("No scores loaded. Please check your internet connection.")
        return
    
    scores_df = add_method_classification(scores_df)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        search = st.text_input("Search", placeholder="PGS ID, trait, author...")
        
        method_classes = st.multiselect(
            "Method Classification",
            options=['High (LD-aware)', 'Moderate (C+T)', 'Other', 'Unknown'],
            help="High: PRS-CS, LDpred2, lassosum, etc. Moderate: C+T, PRSice, etc."
        )
        
        has_efo = st.checkbox("Has EFO/MONDO mapping only", help="Filter to scores with ontology mappings")
        
        st.write("**Harmonized Files**")
        has_grch37 = st.checkbox("GRCh37 available")
        has_grch38 = st.checkbox("GRCh38 available")
        
        st.write("**Variant Count**")
        min_var, max_var = st.slider(
            "Range",
            min_value=0,
            max_value=int(scores_df['n_variants'].max()) if scores_df['n_variants'].max() > 0 else 1000000,
            value=(0, int(scores_df['n_variants'].max()) if scores_df['n_variants'].max() > 0 else 1000000)
        )
        
        filtered_df = filter_scores(
            scores_df,
            search_query=search if search else None,
            method_classes=method_classes if method_classes else None,
            has_efo_only=has_efo,
            has_grch37=has_grch37,
            has_grch38=has_grch38,
            min_variants=min_var,
            max_variants=max_var
        )
    
    with col2:
        st.metric("Total Scores", len(filtered_df))
        
        if not filtered_df.empty:
            csv_data = export_scores_csv(filtered_df)
            st.download_button(
                label="Download Filtered Results (CSV)",
                data=csv_data,
                file_name="pgs_catalog_scores.csv",
                mime="text/csv"
            )
            
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
            
            display_cols = ['pgs_id', 'name', 'trait_names', 'method_class', 'n_variants', 
                          'has_efo_mapping', 'grch37_available', 'grch38_available', 'first_author']
            available_display = [c for c in display_cols if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_display].head(100),
                use_container_width=True,
                hide_index=True
            )
            
            if len(filtered_df) > 100:
                st.info(f"Showing first 100 of {len(filtered_df)} scores. Use filters to narrow down.")
            
            with st.expander("Score Details"):
                selected_pgs = st.selectbox(
                    "Select a score to view details",
                    options=filtered_df['pgs_id'].tolist()[:50],
                    key="score_detail_select"
                )
                
                if selected_pgs:
                    render_score_details(selected_pgs, filtered_df)


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
    
    if traits_df.empty:
        st.warning("No traits loaded. Please check your internet connection.")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        search = st.text_input("Search traits", placeholder="EFO ID, name, description...")
        
        category_options = ['All'] + categories_df['category'].tolist() if not categories_df.empty else ['All']
        selected_category = st.selectbox("Category", options=category_options)
        
        min_scores = st.number_input("Minimum scores", min_value=0, value=0)
        
        filtered_traits = filter_traits(
            traits_df,
            search_query=search if search else None,
            category=selected_category if selected_category != 'All' else None,
            min_scores=min_scores if min_scores > 0 else None
        )
    
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
            
            display_cols = ['trait_id', 'label', 'n_scores', 'categories', 'description']
            available_display = [c for c in display_cols if c in filtered_traits.columns]
            
            st.dataframe(
                filtered_traits[available_display].head(100),
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
    
    if publications_df.empty:
        st.warning("No publications loaded. Please check your internet connection.")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        search = st.text_input("Search publications", placeholder="PGP ID, author, title...")
        
        years = sorted(publications_df['date_publication'].str[:4].dropna().unique(), reverse=True)
        year_options = ['All'] + list(years)
        selected_year = st.selectbox("Publication Year", options=year_options)
        
        has_dev = st.checkbox("Has PGS development")
        has_eval = st.checkbox("Has PGS evaluation")
        
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
            
            display_cols = ['pgp_id', 'first_author', 'title', 'journal', 
                          'date_publication', 'n_development', 'n_evaluation']
            available_display = [c for c in display_cols if c in filtered_pubs.columns]
            
            st.dataframe(
                filtered_pubs[available_display].head(100),
                use_container_width=True,
                hide_index=True
            )


def render_performance_tab():
    st.header("Performance Metrics")
    
    st.info("Performance metrics show how well a PGS predicts the trait in different populations. "
            "Context matters: ancestry and sample size significantly affect metric interpretation.")
    
    scores_df = data_source.get_scores()
    
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
        
        st.subheader(f"Performance Metrics for {pgs_id}")
        st.metric("Number of Evaluations", len(metrics_df))
        
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
    
    ancestry_cats = data_source.get_ancestry_categories()
    if ancestry_cats:
        st.header("Ancestry Categories")
        if isinstance(ancestry_cats, dict):
            categories = ancestry_cats.get('categories', [])
            if categories:
                for cat in categories[:10]:
                    if isinstance(cat, dict):
                        st.write(f"**{cat.get('symbol', '')}**: {cat.get('display_category', cat.get('label', ''))}")


if __name__ == "__main__":
    main()
