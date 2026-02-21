import pandas as pd
import numpy as np
import json
import gzip
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


@st.cache_data
def load_comparison_data():
    stats_path = Path("data/pgs_pairwise_stats.parquet")

    if not stats_path.exists():
        return None, None

    stats_df = pd.read_parquet(stats_path)

    metadata_path = Path("data/pipeline_metadata.json")
    metadata = json.load(open(metadata_path)) if metadata_path.exists() else {}

    return stats_df, metadata


def load_variant_data(pgs_id_1: str, pgs_id_2: str) -> list[dict]:
    variants_path = Path("data/pgs_pairwise_variants.json.gz")

    if not variants_path.exists():
        return []

    with gzip.open(variants_path, "rt") as f:
        all_variants = json.load(f)

    pair_key = f"{pgs_id_1}_{pgs_id_2}"
    if pair_key not in all_variants:
        pair_key = f"{pgs_id_2}_{pgs_id_1}"

    return all_variants.get(pair_key, [])


def filter_comparison_data(
    stats_df: pd.DataFrame,
    trait: str = None,
    min_shared: int = 0,
    min_correlation: float = -1.0,
) -> pd.DataFrame:
    filtered = stats_df.copy()

    if trait and trait != "All traits":
        filtered = filtered[filtered["trait_label"] == trait]

    if min_shared > 0:
        filtered = filtered[filtered["n_shared"] >= min_shared]

    filtered = filtered[filtered["pearson_r"] >= min_correlation]

    return filtered


def get_interpretation(stats: dict) -> str:
    r = stats.get("pearson_r", 0)
    overlap_1 = stats.get("pct_overlap_1", 0)
    overlap_2 = stats.get("pct_overlap_2", 0)
    concordance = stats.get("pct_concordant_sign", 0)

    parts = []

    if r > 0.9:
        parts.append(
            "These scores are highly correlated — they may be essentially redundant. "
            "Choose based on validation performance or ancestry coverage."
        )
    elif r > 0.7:
        parts.append(
            "These scores show strong agreement in their effect estimates for shared variants."
        )
    elif r > 0.3:
        parts.append(
            "These scores show moderate agreement. They may capture partially different genetic signals."
        )
    elif r > 0:
        parts.append(
            "These scores show weak positive correlation despite targeting the same trait."
        )
    else:
        parts.append(
            "These scores are negatively correlated — this may indicate effect allele "
            "coding differences or fundamentally different modeling approaches."
        )

    if overlap_1 < 20 and overlap_2 < 20:
        parts.append(
            "Very low variant overlap suggests these scores use substantially different variant sets."
        )
    elif abs(overlap_1 - overlap_2) > 50:
        larger = "PGS 1" if overlap_1 > overlap_2 else "PGS 2"
        parts.append(
            f"Asymmetric overlap — {larger} contains most of the other score's variants plus additional ones."
        )

    if concordance < 70:
        parts.append(
            "Low sign concordance — a substantial fraction of shared variants have "
            "opposite effect directions between the two scores."
        )

    return " ".join(parts)


def create_scatterplot(
    variant_data: list[dict],
    pgs_id_1: str,
    pgs_id_2: str,
    stats: dict,
) -> go.Figure:
    if not variant_data:
        return None

    df = pd.DataFrame(variant_data)

    if len(df) > 10000:
        sampled = True
        original_len = len(df)
        df = df.sample(n=10000, random_state=42)
    else:
        sampled = False
        original_len = len(df)

    df["concordant"] = (np.sign(df["weight_1"]) == np.sign(df["weight_2"]))

    concordant = df[df["concordant"]]
    discordant = df[~df["concordant"]]

    use_gl = len(df) > 5000
    scatter_type = go.Scattergl if use_gl else go.Scatter

    fig = go.Figure()

    if len(concordant) > 0:
        fig.add_trace(scatter_type(
            x=concordant["weight_1"],
            y=concordant["weight_2"],
            mode="markers",
            marker=dict(color="#2ecc71", size=5, opacity=0.6),
            name="Concordant",
            text=concordant.get("variant_id", None),
            hovertemplate=(
                "Variant: %{text}<br>"
                f"{pgs_id_1} weight: %{{x:.4f}}<br>"
                f"{pgs_id_2} weight: %{{y:.4f}}<extra></extra>"
            ),
        ))

    if len(discordant) > 0:
        fig.add_trace(scatter_type(
            x=discordant["weight_1"],
            y=discordant["weight_2"],
            mode="markers",
            marker=dict(color="#e74c3c", size=5, opacity=0.6),
            name="Discordant",
            text=discordant.get("variant_id", None),
            hovertemplate=(
                "Variant: %{text}<br>"
                f"{pgs_id_1} weight: %{{x:.4f}}<br>"
                f"{pgs_id_2} weight: %{{y:.4f}}<extra></extra>"
            ),
        ))

    all_weights = pd.concat([df["weight_1"], df["weight_2"]])
    w_min, w_max = all_weights.min(), all_weights.max()
    padding = (w_max - w_min) * 0.05
    line_min = w_min - padding
    line_max = w_max + padding

    fig.add_trace(go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="y = x",
        showlegend=True,
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="lightgray", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="lightgray", line_width=1)

    r_val = stats.get("pearson_r", 0)
    trait = stats.get("trait_label", "")
    n_shared = stats.get("n_shared", len(df))
    subtitle = f"r = {r_val:.3f} | {n_shared:,} shared variants | {trait}"

    title_text = f"{pgs_id_1} vs {pgs_id_2}"
    if sampled:
        subtitle += f" (showing {len(df):,} of {original_len:,})"

    fig.update_layout(
        title=dict(text=f"{title_text}<br><sup>{subtitle}</sup>"),
        xaxis_title=f"{pgs_id_1} effect weight",
        yaxis_title=f"{pgs_id_2} effect weight",
        hovermode="closest",
        plot_bgcolor="white",
        height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    return fig


def build_network(
    stats_df: pd.DataFrame,
    trait: str,
    metric: str = "pearson_r",
    threshold: float = 0.5,
) -> nx.Graph:
    if "trait_label" in stats_df.columns:
        trait_df = stats_df[stats_df["trait_label"] == trait].copy()
    else:
        trait_df = stats_df[stats_df["trait_efo"] == trait].copy()

    all_pgs = set(trait_df["pgs_id_1"]) | set(trait_df["pgs_id_2"])

    G = nx.Graph()

    for pgs_id in all_pgs:
        row = trait_df[trait_df["pgs_id_1"] == pgs_id]
        if len(row) > 0:
            n_variants = row.iloc[0]["n_variants_1"]
        else:
            row = trait_df[trait_df["pgs_id_2"] == pgs_id]
            n_variants = row.iloc[0]["n_variants_2"] if len(row) > 0 else 0

        G.add_node(pgs_id, n_variants=int(n_variants))

    effective_metric = metric
    if metric == "jaccard_index" and "jaccard_index" not in trait_df.columns:
        trait_df["jaccard_index"] = trait_df["n_shared"] / (
            trait_df["n_variants_1"] + trait_df["n_variants_2"] - trait_df["n_shared"]
        )

    for _, row in trait_df.iterrows():
        metric_value = row.get(effective_metric, None)
        if pd.notna(metric_value) and metric_value >= threshold:
            G.add_edge(
                row["pgs_id_1"],
                row["pgs_id_2"],
                weight=float(metric_value),
                pearson_r=float(row["pearson_r"]),
                n_shared=int(row["n_shared"]),
                pct_overlap_1=float(row.get("pct_overlap_1", 0)),
                pct_concordant_sign=float(row.get("pct_concordant_sign", 0)),
            )

    return G


def plot_network(G: nx.Graph, title: str = "PRS Correlation Network") -> go.Figure:
    if G.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        r = d.get("pearson_r", 0.5)

        if r > 0.7:
            color = "#2ecc71"
        elif r > 0.3:
            color = "#f39c12"
        else:
            color = "#e74c3c"

        width = max(1, d.get("weight", 0.5) * 5)

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=f"r = {r:.2f}, shared = {d.get('n_shared', 0):,}",
            showlegend=False,
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    max_variants = max((G.nodes[n].get("n_variants", 1000) for n in G.nodes()), default=1000)
    node_sizes = [
        max(12, G.nodes[n].get("n_variants", 1000) / max(max_variants, 1) * 40 + 8)
        for n in G.nodes()
    ]
    node_text = [
        f"{n}<br>{G.nodes[n].get('n_variants', 0):,} variants<br>{G.degree(n)} connections"
        for n in G.nodes()
    ]
    node_colors = [G.degree(n) for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Connections", thickness=15),
            line=dict(width=2, color="white"),
        ),
        text=[n for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=600,
    )

    return fig


def get_network_stats(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {
            "n_nodes": 0, "n_edges": 0, "n_possible_edges": 0,
            "n_clusters": 0, "largest_cluster": 0, "isolated_nodes": 0,
            "redundant_pairs": 0, "density": 0,
        }

    components = list(nx.connected_components(G))

    redundant_pairs = sum(
        1 for _, _, d in G.edges(data=True)
        if d.get("pearson_r", 0) > 0.95
    )

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_possible_edges": G.number_of_nodes() * (G.number_of_nodes() - 1) // 2,
        "n_clusters": len(components),
        "largest_cluster": max(len(c) for c in components) if components else 0,
        "isolated_nodes": sum(1 for c in components if len(c) == 1),
        "redundant_pairs": redundant_pairs,
        "density": nx.density(G),
    }


def correlation_color(r: float) -> str:
    if r > 0.7:
        return "#2ecc71"
    elif r > 0.3:
        return "#f39c12"
    else:
        return "#e74c3c"


def export_comparison_csv(df: pd.DataFrame) -> str:
    export_cols = [
        "pgs_id_1", "pgs_id_2", "trait_label", "trait_efo",
        "n_variants_1", "n_variants_2", "n_shared",
        "pct_overlap_1", "pct_overlap_2",
        "pearson_r", "pearson_p", "pct_concordant_sign", "n_sampled",
    ]
    available = [c for c in export_cols if c in df.columns]
    return df[available].to_csv(index=False)
