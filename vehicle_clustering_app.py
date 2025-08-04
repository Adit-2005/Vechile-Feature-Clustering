# -*- coding: utf-8 -*-
"""
Vehicle Clustering Dashboard with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import plotly.express as px
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Vehicle Clustering Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox, .stSlider, .stFileUploader {
        margin-bottom: 20px;
    }
    .cluster-header {
        color: #2E86C1;
        font-size: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🚗 Vehicle Feature Clustering Dashboard")
st.markdown("""
Cluster vehicles based on their specifications using machine learning algorithms.
""")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    # Data upload
    uploaded_file = st.file_uploader("Upload your vehicle data (CSV)", type=["csv"])
    
    # Clustering parameters
    st.subheader("Clustering Parameters")
    algorithm = st.selectbox(
        "Algorithm",
        ["Hierarchical", "K-Means"],
        index=0
    )
    
    n_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )
    
    # Visualization options
    st.subheader("Visualization Options")
    show_dendrogram = st.checkbox("Show dendrogram", value=True)
    show_pca = st.checkbox("Show PCA plot", value=True)
    show_pairplot = st.checkbox("Show pair plot", value=True)
    show_heatmaps = st.checkbox("Show heatmaps", value=True)

# Load data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
        df = pd.read_csv(url)
    
    numerical_features = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    if all(col in df.columns for col in numerical_features):
        df = df[numerical_features].dropna()
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce').dropna()
    else:
        st.warning("Default columns not found. Using all numerical columns.")
        df = df.select_dtypes(include=['number']).dropna()
    
    return df

df = load_data(uploaded_file)

# Show raw data
with st.expander("View Raw Data"):
    st.dataframe(df.head(10))

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(X_scaled, columns=df.columns)

# Clustering
@st.cache_data
def perform_clustering(X, n_clusters, algorithm):
    if algorithm == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    else:  # K-Means
        model = KMeans(n_clusters=n_clusters, random_state=42)
    
    labels = model.fit_predict(X)
    return labels

labels = perform_clustering(X_scaled, n_clusters, algorithm)
df['Cluster'] = labels

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = labels

# Main content
st.header("Clustering Results")

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Vehicles", len(df))
with col2:
    st.metric("Number of Clusters", n_clusters)
with col3:
    sil_score = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{sil_score:.3f}")

# Visualizations
if show_dendrogram and algorithm == "Hierarchical":
    st.markdown('<p class="cluster-header">Dendrogram</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    Z = linkage(X_scaled, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=15, show_leaf_counts=True, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Vehicle Index')
    ax.set_ylabel('Distance')
    ax.axhline(y=10, color='r', linestyle='--', label='Suggested Cutoff')
    st.pyplot(fig)

if show_pca:
    st.markdown('<p class="cluster-header">PCA Visualization</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Matplotlib", "Plotly"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, 
                       palette='viridis', s=100, ax=ax)
        ax.set_title('PCA: Vehicle Clusters')
        st.pyplot(fig)
    
    with tab2:
        fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                         hover_name=df.index, 
                         title='Interactive Cluster Visualization',
                         width=800, height=600)
        st.plotly_chart(fig)

if show_pairplot:
    st.markdown('<p class="cluster-header">Pair Plot</p>', unsafe_allow_html=True)
    st.info("This may take a moment for larger datasets...")
    
    fig = sns.pairplot(pd.concat([df.iloc[:, :-1], df['Cluster']], axis=1), 
                      hue='Cluster', palette='viridis', corner=True)
    st.pyplot(fig)

if show_heatmaps:
    st.markdown('<p class="cluster-header">Heatmaps</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Averages")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.groupby('Cluster').mean().T, annot=True, fmt=".1f", 
                    cmap="YlGnBu", ax=ax)
        plt.title('Average Feature Values by Cluster')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Feature Correlation Heatmap')
        st.pyplot(fig)

# Cluster profiles
st.markdown('<p class="cluster-header">Cluster Profiles</p>', unsafe_allow_html=True)
st.dataframe(df.groupby('Cluster').mean().style.background_gradient(cmap='YlGnBu'))

# Download results
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download clustered data as CSV",
    data=csv,
    file_name='clustered_vehicles.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("""
**Instructions:**
1. Upload your vehicle data (or use the default dataset)
2. Adjust clustering parameters in the sidebar
3. Explore the visualizations
4. Download your clustered data
""")