# Netflix Movies and TV Shows Clustering 🎬📊

## 📌 Project Title

**Netflix Movies and TV Shows Clustering**

---

## 📖 Domain

**Entertainment | Data Science | Machine Learning**

---

## 🧠 Project Overview

Netflix hosts a vast catalog of movies and TV shows across multiple genres, languages, and regions. Understanding patterns within this content is essential for improving recommendations and identifying content gaps.

This project applies **unsupervised machine learning (clustering)** techniques to group similar Netflix content based on metadata such as genre, rating, duration, and release year.

---

## 🎯 Objective

* To cluster Netflix movies and TV shows into meaningful groups based on similarity.
* To help improve **content discovery, recommendation systems, and business insights** using clustering results.

---

## 🗂 Dataset Information

* **Dataset Name:** Netflix Movies and TV Shows
* **Format:** CSV
* **Total Records:** 7,787
* **Total Columns:** 12

### 📄 Dataset Columns

| Column Name  | Description                         |
| ------------ | ----------------------------------- |
| show_id      | Unique ID for each title            |
| type         | Movie or TV Show                    |
| title        | Name of the content                 |
| director     | Director name (may contain nulls)   |
| cast         | Actors involved (may contain nulls) |
| country      | Country of production               |
| date_added   | Date added to Netflix               |
| release_year | Year of release                     |
| rating       | Content rating                      |
| duration     | Movie runtime / TV seasons          |
| listed_in    | Genres                              |
| description  | Short summary                       |

---

## 🔄 Project Workflow

### 1️⃣ Data Collection & Exploration

* Loaded dataset and inspected structure
* Checked data types, missing values, and duplicates

### 2️⃣ Data Cleaning & Preprocessing

* Handled missing values in **director, cast, and country**
* Converted categorical features using **encoding techniques**
* Standardized numerical features like `release_year` and `duration`
* Extracted meaningful information from text features

### 3️⃣ Exploratory Data Analysis (EDA)

* Analyzed content distribution by type, genre, and rating
* Identified dominant genres and release trends
* Studied correlations between numerical features

### 4️⃣ Feature Engineering

* Created new features such as:

  * **Content Age** = Current Year − Release Year
  * **Genre Count** = Number of genres per title
* Transformed categorical data into machine-readable format

### 5️⃣ Clustering Techniques Used

* **K-Means Clustering**
* **Hierarchical Clustering**
* **DBSCAN (Density-Based Clustering)**

### 6️⃣ Dimensionality Reduction

* **PCA (Principal Component Analysis)**
* **t-SNE** for visualizing clusters in 2D space

### 7️⃣ Model Evaluation

* **Silhouette Score**
* **Davies–Bouldin Index**
* **Inertia (Elbow Method)**

---

## 📊 Results & Insights

* Successfully grouped Netflix content into meaningful clusters
* Identified clusters dominated by specific genres and ratings
* Helped uncover niche content segments
* Provided visual insights into content similarity

---

## 💡 Business Use Cases

* Personalized content recommendations
* Identifying underrepresented genres
* Supporting Netflix’s recommendation engine
* Helping production houses identify content demand gaps
* Targeted marketing and advertisement strategies

---

## 🖥️ Streamlit Web Application

The project includes an **interactive Streamlit application** where users can explore clustering results visually.

### 🔗 Live App URL

👉 **[Click here to visit the Streamlit App](https://netflix-movies-and-tv-shows-clustering.streamlit.app/)**

> *Note: Replace the above link with your actual deployed Streamlit URL.*

### ⚙️ Streamlit Features

* Interactive cluster visualization
* 2D scatter plots using dimensionality reduction
* Cluster-wise content analysis

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Scikit-learn
  * Matplotlib
  * Seaborn
* **Machine Learning:** Unsupervised Learning
* **Visualization:** PCA, t-SNE
* **Web App:** Streamlit

---

## 📁 Project Structure

```
├── data/
│   └── NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv
├── notebooks/
│   └── Netflix_EDA.ipynb
├── app.py                # Streamlit application
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run the Project Locally

```bash
# Clone the repository
git clone <your-repository-url>

# Navigate to the project directory
cd netflix-clustering

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## ✅ Conclusion

This project demonstrates how **unsupervised machine learning** can be effectively used to analyze and cluster entertainment content. The insights derived can enhance recommendation systems and support strategic business decisions in the streaming industry.

---

## 📌 Future Enhancements

* Integrate user watch history for hybrid recommendations
* Use NLP models on descriptions for deeper semantic clustering
* Deploy using cloud platforms with CI/CD

---

## 👤 Author

**Sakthi Gnana Prakasam**
*Data Science & Machine Learning Enthusiast*

---

If you want, I can also:

* Validate this README against **GUVI evaluation checklist**
* Customize it for **resume / portfolio**
* Add **screenshots section**
* Adjust wording to match **live evaluation expectations**

Just tell me what you want next.
