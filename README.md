# 🚗 Vehicle Clustering Analytics Dashboard

![Dashboard Preview](./screenshots/dashboard.png)

A professional machine learning dashboard for analyzing vehicle characteristics through advanced clustering techniques.

## ✨ Features

- **Dual Algorithm Support**: Choose between Hierarchical or K-Means clustering
- **Interactive 3D Visualizations**: Rotatable PCA plots with cluster highlighting
- **Smart Data Handling**:
  - Automatic data preprocessing
  - Missing value handling
  - Feature scaling
- **Export Results**: Download clustered datasets as CSV
- **Responsive Design**: Works on desktop and mobile

## 🛠️ Tech Stack

| Category       | Technologies |
|----------------|--------------|
| Frontend       | Streamlit |
| Visualization  | Plotly, Matplotlib, Seaborn |
| ML Backend     | scikit-learn, SciPy |
| Data Processing| Pandas, NumPy |

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vehicle-clustering-dashboard.git
cd vehicle-clustering-dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
