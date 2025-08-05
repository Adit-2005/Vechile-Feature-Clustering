# -*- coding: utf-8 -*-
"""
Enhanced Vehicle Clustering Dashboard with Improved Visibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Advanced Vehicle Clustering",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced visibility
st.markdown("""
<style>
    :root {
        --primary: #166088;
        --secondary: #4a6fa5;
        --accent: #4fc3f7;
        --background: #f8f9fa;
        --card: #ffffff;
        --text: #333333;
        --text-light: #555555;
    }
    
    .main {
        background-color: var(--background);
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--card);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    .stSelectbox, .stSlider, .stFileUploader {
        margin-bottom: 1.5rem;
    }
    
    .cluster-header {
        color: var(--primary);
        font-size: 1.5rem !important;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background-color: var(--card);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .metric-title {
        font-size: 1rem;
        color: var(--text-light);
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 2rem;
        color: var(--primary);
        font-weight: 700;
        margin: 0;
    }
    
    .instruction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    .instruction-title {
        color: var(--primary);
        margin-top: 0;
        font-size: 1.25rem;
    }
    
    .instruction-list {
        font-size: 1rem;
        line-height: 1.6;
        color: var(--text);
    }
    
    .instruction-list li {
        margin-bottom: 0.5rem;
    }
    
    .tip-box {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    
    .tip-text {
        margin: 0;
        font-size: 0.95rem;
        color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# App title with improved visibility
st.title("🚗 Advanced Vehicle Clustering Dashboard")
st.markdown("""
<div style="background-color: #166088; padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
    <h3 style="margin:0; color:white; font-weight:500;">Cluster vehicles based on their specifications using machine learning algorithms</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar controls with improved visibility
with st.sidebar:
    st.markdown("""
    <div style="background-color: #166088; padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1.5rem;">
        <h3 style="margin:0; color:white; font-weight:500;">Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Data upload section
    with st.expander("📁 Data Upload", expanded=True):
        uploaded_file = st.file_uploader("Upload vehicle data (CSV)", type=["csv"])
        st.markdown("<p style='color: var(--text-light); font-size: 0.9rem;'>Sample dataset will be used if no file uploaded</p>", unsafe_allow_html=True)
    
    # Clustering parameters
    with st.expander("⚙️ Clustering Parameters", expanded=True):
        algorithm = st.selectbox(
            "Algorithm",
            ["Hierarchical", "K-Means", "DBSCAN"],
            index=0,
            help="Select clustering algorithm"
        )
        
        if algorithm == "DBSCAN":
            eps = st.slider("EPS", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 1, 20, 5)
            n_clusters = None
        else:
            n_clusters = st.slider(
                "Number of clusters",
                2, 10, 4, 1,
                help="Select the number of clusters to create"
            )
    
    # Visualization options
    with st.expander("📊 Visualization Options", expanded=True):
        show_dendrogram = st.checkbox("Show dendrogram", value=True, disabled=(algorithm != "Hierarchical"))
        show_pca = st.checkbox("Show PCA visualization", value=True)
        pca_3d = st.checkbox("Enable 3D PCA", value=False)
        show_pairplot = st.checkbox("Show pair plot", value=False)
        show_heatmaps = st.checkbox("Show heatmaps", value=True)

# Load data with improved caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
        df = pd.read_csv(url)
    
    # Improved feature handling
    numerical_features = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    
    if all(col in df.columns for col in numerical_features):
        df = df[numerical_features].copy()
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        df = df.dropna()
    else:
        st.warning("Default columns not found. Using all numerical columns.")
        df = df.select_dtypes(include=['number']).dropna()
    
    return df

# Show loading state while processing
with st.spinner('Loading and processing data...'):
    df = load_data(uploaded_file)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(X_scaled, columns=df.columns)

# Clustering function with DBSCAN support
@st.cache_data
def perform_clustering(X, n_clusters, algorithm, eps=None, min_samples=None):
    if algorithm == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    elif algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:  # DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    labels = model.fit_predict(X)
    
    # For DBSCAN, count actual clusters (excluding noise if present)
    if algorithm == "DBSCAN":
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    return labels, n_clusters

# Perform clustering with progress indicator
with st.spinner('Performing clustering...'):
    if algorithm == "DBSCAN":
        labels, actual_clusters = perform_clustering(X_scaled, None, algorithm, eps, min_samples)
    else:
        labels, actual_clusters = perform_clustering(X_scaled, n_clusters, algorithm)

df['Cluster'] = labels

# PCA for visualization
with st.spinner('Calculating PCA...'):
    pca = PCA(n_components=3 if pca_3d else 2)
    X_pca = pca.fit_transform(X_scaled)
    
    if pca_3d:
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    else:
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    
    df_pca['Cluster'] = labels

# Main content with improved visibility
st.header("🔍 Clustering Results")

# Metrics cards with improved visibility
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Vehicles</div>
        <div class="metric-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Number of Clusters</div>
        <div class="metric-value">{actual_clusters}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    sil_score = silhouette_score(X_scaled, labels)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Silhouette Score</div>
        <div class="metric-value">{sil_score:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if len(set(labels)) > 1:
        ch_score = calinski_harabasz_score(X_scaled, labels)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Calinski-Harabasz</div>
            <div class="metric-value">{ch_score:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Calinski-Harabasz</div>
            <div class="metric-value">N/A</div>
        </div>
        """, unsafe_allow_html=True)

# Visualizations with improved visibility
if show_dendrogram and algorithm == "Hierarchical":
    st.markdown('<div class="cluster-header">Dendrogram</div>', unsafe_allow_html=True)
    with st.container():
        fig, ax = plt.subplots(figsize=(12, 6))
        Z = linkage(X_scaled, method='ward')
        dendrogram(Z, truncate_mode='lastp', p=15, show_leaf_counts=True, ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14)
        ax.set_xlabel('Vehicle Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.axhline(y=10, color='r', linestyle='--', label='Suggested Cutoff')
        st.pyplot(fig)

if show_pca:
    st.markdown('<div class="cluster-header">PCA Visualization</div>', unsafe_allow_html=True)
    
    if pca_3d:
        # Interactive 3D Plot with improved visibility
        with st.container():
            fig = px.scatter_3d(
                df_pca,
                x='PC1',
                y='PC2',
                z='PC3',
                color='Cluster',
                hover_name=df.index,
                title='3D PCA Cluster Visualization',
                width=900,
                height=700,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            # Customize layout for better visibility
            fig.update_layout(
                scene=dict(
                    xaxis_title=f'PC1 ({round(pca.explained_variance_ratio_[0]*100, 1)}%)',
                    yaxis_title=f'PC2 ({round(pca.explained_variance_ratio_[1]*100, 1)}%)',
                    zaxis_title=f'PC3 ({round(pca.explained_variance_ratio_[2]*100, 1)}%)',
                    xaxis_title_font=dict(size=12),
                    yaxis_title_font=dict(size=12),
                    zaxis_title_font=dict(size=12),
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                font=dict(size=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        # 2D Visualization with tabs
        tab1, tab2 = st.tabs(["Interactive Plot", "Static Plot"])
        
        with tab1:
            fig = px.scatter(
                df_pca,
                x='PC1',
                y='PC2',
                color='Cluster',
                hover_name=df.index,
                title='PCA Cluster Visualization',
                width=800,
                height=600,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            fig.update_layout(
                xaxis_title=f'PC1 ({round(pca.explained_variance_ratio_[0]*100, 1)}%)',
                yaxis_title=f'PC2 ({round(pca.explained_variance_ratio_[1]*100, 1)}%)',
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                x='PC1',
                y='PC2',
                hue='Cluster',
                data=df_pca,
                palette='viridis',
                s=100,
                ax=ax
            )
            ax.set_title('PCA: Vehicle Clusters', fontsize=14)
            ax.set_xlabel(f'PC1 ({round(pca.explained_variance_ratio_[0]*100, 1)}%)', fontsize=12)
            ax.set_ylabel(f'PC2 ({round(pca.explained_variance_ratio_[1]*100, 1)}%)', fontsize=12)
            st.pyplot(fig)

if show_pairplot:
    st.markdown('<div class="cluster-header">Pair Plot</div>', unsafe_allow_html=True)
    st.warning("Note: Pair plots may take longer to generate with larger datasets.")
    
    with st.spinner('Generating pair plot...'):
        fig = sns.pairplot(
            pd.concat([df.iloc[:, :-1], df['Cluster']], axis=1),
            hue='Cluster',
            palette='viridis',
            corner=True,
            plot_kws={'alpha': 0.6, 's': 30}
        )
        st.pyplot(fig)

if show_heatmaps:
    st.markdown('<div class="cluster-header">Heatmaps</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("Cluster Averages")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                df.groupby('Cluster').mean().T,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                ax=ax,
                cbar_kws={'label': 'Average Value'}
            )
            plt.title('Average Feature Values by Cluster', fontsize=14)
            st.pyplot(fig)
    
    with col2:
        with st.container():
            st.subheader("Feature Correlations")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                df.corr(),
                annot=True,
                cmap='coolwarm',
                center=0,
                ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'}
            )
            plt.title('Feature Correlation Heatmap', fontsize=14)
            st.pyplot(fig)

# Cluster profiles with improved visibility
st.markdown('<div class="cluster-header">📊 Cluster Profiles</div>', unsafe_allow_html=True)
with st.container():
    cluster_stats = df.groupby('Cluster').agg(['mean', 'std', 'count'])
    st.dataframe(
        cluster_stats.style
        .background_gradient(cmap='YlGnBu', subset=cluster_stats.columns.get_level_values(1).isin(['mean']))
        .format("{:.2f}", subset=cluster_stats.columns.get_level_values(1).isin(['mean', 'std']))
        .set_properties(**{'font-size': '12pt'})
    )

# Download section with improved visibility
st.markdown('<div class="cluster-header">📥 Export Results</div>', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download clustered data (CSV)",
            data=csv,
            file_name='clustered_vehicles.csv',
            mime='text/csv',
            help="Download the complete dataset with cluster assignments"
        )
    
    with col2:
        if show_pca and pca_3d:
            pca_csv = df_pca.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download PCA coordinates (CSV)",
                data=pca_csv,
                file_name='pca_coordinates.csv',
                mime='text/csv',
                help="Download the PCA coordinates for further analysis"
            )

# Footer with improved visibility
st.markdown("---")
# Customizable color variables
bg_color = "#E3F2FD"  # Light blue background
header_color = "#0D47A1"  # Dark blue header
text_color = "#212121"  # Dark gray text
border_color = "#2196F3"  # Blue border
icon_color = "#FF9800"  # Orange icon

st.markdown(f"""
<div style="background-color: {bg_color}; 
            padding: 1.5rem; 
            border-radius: 10px;
            border-left: 4px solid {border_color};
            margin-top: 2rem;">
    <h3 style="color: {header_color}; margin-top: 0;">
        <span style="color: {icon_color};">📌</span> How to Use This Dashboard
    </h3>
    <ol style="color: {text_color};">
        <li>Upload your data</li>
        <li>Select algorithm</li>
        <li>Explore results</li>
    </ol>
</div>
""", unsafe_allow_html=True)
