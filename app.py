import streamlit as st
import pandas as pd

st.set_page_config(page_title="Netflix Clustering App", layout="wide")

st.title("🎬 Netflix Movies & TV Shows Clustering")

# Load datasets
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_original.csv")
    tv = pd.read_csv("tv_original.csv")
    return movies, tv

movies_df, tv_df = load_data()

# Select dataset
data_type = st.sidebar.selectbox("Select Content Type", ["Movies", "TV Shows"])

if data_type == "Movies":
    df = movies_df
else:
    df = tv_df

st.write(f"### Showing {data_type} Dataset")

# Select cluster
cluster_list = sorted(df['cluster_label'].unique())
selected_cluster = st.sidebar.selectbox("Select Cluster", cluster_list)

# Filter data
filtered_df = df[df['cluster_label'] == selected_cluster]

st.write(f"### Cluster {selected_cluster} Data")
st.dataframe(filtered_df)

# Show basic stats
st.write("### Cluster Summary")
st.write("Total Titles:", filtered_df.shape[0])
st.write("Average Release Year:", round(filtered_df['release_year'].mean(), 2))