# 🎬 PySpark Movie Recommendation System

This project demonstrates how to build a **Movie Recommendation System** using **PySpark** and the **Alternating Least Squares (ALS)** algorithm on a dataset of user-movie ratings.

---

## 🔧 What is Apache Spark?

**Apache Spark** is a powerful open-source distributed computing system designed for large-scale data processing. It supports in-memory processing, making it significantly faster than traditional big data tools like Hadoop.

---

## 🐍 What is PySpark?

**PySpark** is the Python API for Apache Spark. It allows you to harness the full power of Spark with Python code, including:
- Distributed data processing
- Machine learning with MLlib
- SQL querying with SparkSQL

---

## 📘 Project Overview

This notebook walks through the process of building a **collaborative filtering-based recommender system** using the ALS (Alternating Least Squares) algorithm from Spark MLlib.

---

## 🧠 Model: ALS (Alternating Least Squares)

The ALS model is widely used for **collaborative filtering**. It works well with large sparse matrices like user-item ratings by factorizing them into user and item latent features.

---

## 📂 Dataset

The dataset used is assumed to be a CSV file named `merged_movie_df.csv` which contains:
- `userId`: Unique ID for each user
- `movieId`: Unique ID for each movie
- `rating`: Rating given by a user to a movie
- `title`: Title of the movie (used for joining final recommendations)

---

## 📈 Notebook Workflow

### 1. **Environment Setup**
```python
!pip install pyspark
```
---

### 2. **Importing Libraries**
Standard Python libraries, Scikit-learn (for similarity), and PySpark MLlib are imported.

---

### 3. **Initialize Spark Session**
```python
spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()
```
A Spark session is started to interact with Spark’s APIs.

---

### 4. **Data Loading & Cleaning**
```python
def load_data():
    df = spark.read.option("header", True).csv("/content/merged_movie_df.csv")
    ...
```
- Reads the CSV
- Casts types (`userId`, `movieId`, `rating`)
- Drops any missing values

---

### 5. **Train ALS Model**
```python
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    implicitPrefs=False
)
```
- Trains ALS on the user-movie rating matrix.
- `coldStartStrategy="drop"` ensures unseen users/items are ignored during prediction.

---

### 6. **Load Movie Metadata**
```python
movies_df = spark.read.option("header", True).csv("/content/merged_movie_df.csv") \
    .select("movieId", "title") \
    .dropna().dropDuplicates()
```
Loads movie titles for mapping movieId to human-readable output.

---

### 7. **Generate Recommendations**
```python
def get_recommendations_for_user(model, movies_df, user_id, top_n=10):
    ...
```
- Creates a DataFrame for the target user
- Uses `recommendForUserSubset()` from Spark ALS
- Explodes nested recommendations into flat format
- Joins with movie metadata for final titles

---

## ✅ Output

Returns the **Top-N movie recommendations** for a given user, including:
- Movie ID
- Rating score (predicted by ALS)
- Movie Title

---

## 🧪 Example Usage

```python
user_id = 123
top_recs = get_recommendations_for_user(model, movies_df, user_id, top_n=10)
top_recs.show()
```

---

