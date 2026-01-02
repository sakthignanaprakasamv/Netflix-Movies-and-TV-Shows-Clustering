import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# --------------------------------
# Load data
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv")

df = load_data()

# --------------------------------
# STEP 1: Fill missing ratings using MODE per listed_in
# --------------------------------
not_missing_rating = df[df['rating'].notnull()]
missing_rating = df[df['rating'].isnull()]

# Most repeated rating (MODE) for each listed_in
most_common_rating = (
    not_missing_rating
    .groupby('listed_in')['rating']
    .agg(lambda x: x.mode().iloc[0])
)

# Fill missing ratings
df.loc[df['rating'].isnull(), 'rating'] = (
    df.loc[df['rating'].isnull(), 'listed_in']
    .map(most_common_rating)
)

# Safety fallback (if any still missing)
df['rating'].fillna('TV-MA', inplace=True)

# --------------------------------
# STEP 2: Prepare data for clustering
# --------------------------------
df_cluster = df[['title', 'listed_in', 'duration', 'rating']].copy()

# Convert duration to numeric
df_cluster['duration'] = (
    df_cluster['duration']
    .str.extract('(\d+)')
    .astype(float)
)

df_cluster.dropna(inplace=True)

# --------------------------------
# STEP 3: Encoding + Scaling + K-Means
# --------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ['listed_in', 'rating']),
        ('num', StandardScaler(), ['duration'])
    ]
)

kmeans_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('kmeans', KMeans(n_clusters=5, random_state=42))
])

df_cluster['cluster'] = kmeans_pipeline.fit_predict(
    df_cluster[['listed_in', 'duration', 'rating']]
)

X_transformed = kmeans_pipeline.named_steps['preprocessing'].transform(
    df_cluster[['listed_in', 'duration', 'rating']]
)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_transformed)

# Add PCA coordinates to the dataframe
df_cluster['pca_1'] = X_pca[:, 0]
df_cluster['pca_2'] = X_pca[:, 1]

# --------------------------------
# STREAMLIT UI
# --------------------------------

st.logo("netflix_logo.png")
# Set full-width layout
st.set_page_config(
    layout="wide",
    page_title="Netflix Recommendation System"
)

# Full-width title at the top
st.markdown(
    "<h1 style='text-align: center;'>Netflix Movie & TV Show Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Recommendations are generated using K-Means clustering based on genre, rating, and duration.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Create two columns for layout
left_col, right_col = st.columns([1, 3])

with left_col:
    st.subheader("Select a Movie / TV Show")
    movie_name = st.selectbox(
        "",
        sorted(df_cluster['title'].unique())
    )

    if movie_name:
        selected_movie = df_cluster[df_cluster['title'] == movie_name].iloc[0]
        
        st.subheader("Selected Title Details")
        st.write("**Title:**", selected_movie['title'])
        st.write("**Genre:**", selected_movie['listed_in'])
        st.write("**Rating:**", selected_movie['rating'])
        st.write("**Duration:**", int(selected_movie['duration']))

# ... (previous code remains the same until we get recommendations) ...

with right_col:
    if movie_name:
        cluster_id = selected_movie['cluster']
        recommendations = df_cluster[
            (df_cluster['cluster'] == cluster_id) &
            (df_cluster['title'] != movie_name)
        ].copy()
        
        # Get the PCA coordinates of the selected movie
        selected_pca = df_cluster.loc[df_cluster['title'] == movie_name, ['pca_1', 'pca_2']].iloc[0]

        # Calculate distance in PCA space
        recommendations['pca_1'] = df_cluster.loc[recommendations.index, 'pca_1'].values
        recommendations['pca_2'] = df_cluster.loc[recommendations.index, 'pca_2'].values
        recommendations['pca_distance'] = ((recommendations['pca_1'] - selected_pca['pca_1'])**2 + (recommendations['pca_2'] - selected_pca['pca_2'])**2)**0.5

        # Sort by PCA distance
        recommendations = recommendations.sort_values(by='pca_distance').drop(columns=['pca_1', 'pca_2', 'pca_distance'])

        # Reset index for custom S.No
        recommendations = recommendations.reset_index(drop=True)
        recommendations.index = recommendations.index + 1
        recommendations.index.name = "S.No"

        st.subheader("Recommended Titles")
        st.caption(f"Total Recommendations: {len(recommendations)}")

        st.dataframe(recommendations[['title', 'listed_in', 'rating', 'duration']], use_container_width=True, height=700)


if movie_name:

    st.subheader("K-Means Clusters (2D Visualization)")

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df_cluster['pca_1'],
        df_cluster['pca_2'],
        c=df_cluster['cluster'],
        cmap='tab10',
        s=40,
        alpha=0.7
    )

    # Highlight selected movie (important)
    ax.scatter(
        selected_movie['pca_1'],
        selected_movie['pca_2'],
        color='red',
        s=120,
        edgecolors='black',
        label='Selected Movie'
    )

    ax.set_title("K-Means Clusters (2D Visualization)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Create legend manually (cluster colors)
    legend1 = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="upper right"
    )
    ax.add_artist(legend1)

    ax.legend(loc="lower right")

    st.pyplot(fig)
