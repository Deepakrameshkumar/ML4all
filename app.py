import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch
#import hdbscan
import io
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer



# 🔹 Setup for Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔹 App Layout
st.title("ML4all: Machine Learning Clustering Web Interface")

# 🔹 Upload
st.header("Upload Data")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.write("📊 Data Preview:")
    st.dataframe(df.head())

    # 🔹 Feature Selection
    st.header("Select Features")
    all_columns = df.columns.tolist()
    text_cols = st.multiselect("Select text columns for clustering (at least one required)", all_columns)
    categorical_cols = st.multiselect("Select categorical columns (optional)", all_columns)

    # 🔹 Algorithm Selection
    st.header("Clustering Algorithm")
    algorithm = st.selectbox("Choose clustering algorithm", [ "DBSCAN","KMeans", "OPTICS", "BIRCH"])

    params = {}
    if algorithm == "DBSCAN":
        params['eps'] = st.slider("Epsilon (eps)", 0.1, 2.0, 0.7, step=0.1)
        params['min_samples'] = st.slider("Minimum samples", 5, 50, 30)
        st.warning("DBSCAN is recommended for most clustering needs. Adjust eps and min_samples for optimal results.")

    elif algorithm == "KMeans":
            params['n_clusters'] = st.slider("Number of clusters", 2, 50, 3)
            params['random_state'] = 42
            st.warning("You define the number of clusters. KMeans is suitable for well-separated clusters.")

    #elif algorithm == "HDBSCAN":
    #    params['min_cluster_size'] = st.slider("Minimum cluster size", 5, 50, 15)

    elif algorithm == "OPTICS":
        params['min_samples'] = st.slider("Minimum samples", 5, 50, 30)
        params['xi'] = st.slider("Xi", 0.01, 0.1, 0.05, step=0.01)
        st.warning("OPTICS requires dense matrices, which may consume significant memory for large datasets.")
    elif algorithm == "BIRCH":
        params['n_clusters'] = st.slider("Number of clusters", 2, 50, 3)
        params['threshold'] = st.slider("Threshold", 0.1, 1.0, 0.5, step=0.1)
        st.warning("You define the number of clusters. BIRCH is suitable for large datasets but may not perform well with small clusters.")

    # 🔹 Run Button
    if st.button("Run Clustering"):
        if not text_cols:
            text_cols = ["N/A"]
        if text_cols == ["N/A"] and not categorical_cols:
            st.error("Please select at least one text or categorical column.")
        else:
            try:
                # 🔸 Convert text to string
                for col in text_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

                # 🔸 Combine text columns
                if text_cols != ["N/A"]:
                    df['combined_text'] = df[text_cols].fillna('').agg(lambda row: ' '.join(str(val) for val in row), axis=1)
                    text_col_for_processing = 'combined_text'
                else:
                    text_col_for_processing = None

                # 🔸 Preprocessing
                transformers = []
                if text_col_for_processing:
                    transformers.append(('text', TfidfVectorizer(stop_words='english'), text_col_for_processing))
                if categorical_cols:
                    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))

                preprocessor = ColumnTransformer(transformers=transformers)
                X = preprocessor.fit_transform(df)

                if algorithm == "OPTICS" and hasattr(X, 'toarray'):
                    X = X.toarray()

                # 🔸 Initialize clustering
                if algorithm == "KMeans":
                    clusterer = KMeans(n_clusters=params['n_clusters'], random_state=params['random_state'])
                elif algorithm == "DBSCAN":
                    clusterer = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
#                elif algorithm == "HDBSCAN":
#                    clusterer = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'])
                elif algorithm == "OPTICS":
                    clusterer = OPTICS(min_samples=params['min_samples'], xi=params['xi'])
                elif algorithm == "BIRCH":
                    clusterer = Birch(n_clusters=params['n_clusters'], threshold=params['threshold'])

                # 🔸 Run Clustering
                with st.spinner("Running clustering..."):
                    clusters = clusterer.fit_predict(X)
                df['cluster_label'] = clusters

                # 🔸 Generate cluster summaries
                def generate_cluster_summaries(df, text_col='combined_text', label_col='cluster_label', sentence_count=3, top_n=5):
                    summaries = {}
                    for label in sorted(df[label_col].unique()):
                        subset = df[df[label_col] == label]
                        all_text = ' '.join(subset[text_col].dropna())

                        try:
                            parser = PlaintextParser.from_string(all_text, Tokenizer("english"))
                            summarizer = TextRankSummarizer()
                            summary_sentences = summarizer(parser.document, sentence_count)

                            summary = ' '.join(str(sentence) for sentence in summary_sentences)
                            if not summary.strip():
                                raise ValueError("Too sparse")

                        except Exception:
                            # Fallback to TF-IDF keywords without commas
                            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
                            tfidf_matrix = vectorizer.fit_transform([all_text])
                            top_indices = tfidf_matrix.toarray()[0].argsort()[-top_n:][::-1]
                            keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
                            summary = ' '.join(keywords)

                        summaries[label] = summary
                    return summaries



                summaries = generate_cluster_summaries(df)
                # Format long descriptions with line breaks every ~8 words
                def format_description(description, words_per_line=8):
                    words = description.split()
                    lines = [' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
                    return '<br>'.join(lines)
                
                # 🔹 Treemap with Descriptions
                df['cluster_description'] = df['cluster_label'].map(lambda x: f"{x}: {summaries.get(x, 'N/A')}")
                treemap_data = df.groupby('cluster_description').size().reset_index(name='count')

                st.header("Clustering Results")
                st.dataframe(df.head())

                # Format descriptions for line breaks
                treemap_data['cluster_description'] = treemap_data['cluster_description'].apply(format_description)

                fig = px.treemap(
                    treemap_data,
                    path=['cluster_description'],
                    values='count',
                    title="Cluster Distribution with Descriptive Labels"
                )
                fig.update_traces(textinfo='label+value', textfont_size=18)  # Font size boost

                st.plotly_chart(fig, key="treemap")

                # 🔹 Preview Pane
                st.header("Cluster Preview")
                selected_cluster = None
                if st.session_state.get('treemap'):
                    click_data = st.session_state['treemap'].get('clickData')
                    if click_data and click_data['points']:
                        label_text = click_data['points'][0]['label']
                        cluster_id = label_text.split(":")[0].replace("Cluster", "").strip()
                        try:
                            selected_cluster = int(cluster_id)
                        except:
                            selected_cluster = cluster_id

                if selected_cluster is not None:
                    st.write(f"Showing data for cluster: {selected_cluster}")
                    cluster_data = df[df['cluster_label'] == selected_cluster]
                    st.dataframe(cluster_data)
                else:
                    st.info("Click a cluster in the treemap to preview its data.")

                # 🔹 Download
                st.header("Download Results")
                output = io.BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)
                st.download_button(
                    label="Download clustered data as CSV",
                    data=output,
                    file_name="clustered_output.csv",
                    mime="text/csv"
                )

                st.success("✅ Clustering complete!")

            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")
else:
    st.info("📁 Please upload a CSV or Excel file to start.")