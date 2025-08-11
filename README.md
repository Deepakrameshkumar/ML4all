# ML4all: Machine Learning Clustering Web Interface

ML4all is a Streamlit-based web application that allows users to perform clustering on text and categorical data using various machine learning algorithms. It supports KMeans, DBSCAN, OPTICS, and BIRCH, and provides descriptive summaries and visualizations for each cluster.

## 🚀 Features

- Upload CSV or Excel files
- Select text and categorical columns
- Choose clustering algorithms with customizable parameters
- View cluster summaries using TextRank or TF-IDF fallback
- Interactive treemap visualization
- Preview cluster data
- Download clustered results

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Deepakrameshkumar/ML4all.git
cd ML4all
```

### 2. Create a python virtual environment (recommended) or skip this step  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
``` bash
streamlit run app.py
```


