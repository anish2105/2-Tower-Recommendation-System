# 📌 Case Study: YouTube Recommendation System — Candidate Retrieval via Penultimate Embedding

## 📈 Overview

This case study explains the **candidate retrieval stage** of YouTube’s large-scale recommendation system using a **deep learning-based two-tower architecture**. This stage is crucial in narrowing down billions of videos to a manageable set of top-N candidates that a downstream ranking model can evaluate.

The architecture leverages user and video embeddings, combined with a nearest neighbor search over the penultimate layer embeddings, to efficiently serve personalized video recommendations.

---

## 🔧 System Architecture

The recommendation model consists of two main components:

1. **User Tower**: Generates a dense embedding vector representing the user.
2. **Video Tower**: Generates dense embedding vectors for all candidate videos.

These embeddings are compared using approximate nearest neighbor (ANN) search to retrieve the top-N most relevant videos for a given user.

---

## 🧠 Input Features

The model takes in several types of input features which are embedded and concatenated:

| Feature Type             | Description                                             |
|--------------------------|---------------------------------------------------------|
| Embedded Video Watches   | Historical user watch data embedded and averaged        |
| Embedded Search Tokens   | Tokens from user search history embedded and averaged   |
| Geographic Embedding     | Encoded location-based features                         |
| Age                      | Scalar age feature and its square (e.g., `x`, `x²`)     |
| Gender                   | Gender represented using categorical embeddings         |

---

## 🏗 Embedding & Transformation Pipeline

The pipeline works as follows:

1. **Embedding Layer**: 
   - Categorical inputs like video watches, search tokens, and gender are passed through embedding layers.
   - Continuous inputs like age are represented with raw and squared terms.
2. **Averaging Layer**:
   - The user’s watch history and search tokens are averaged to get a fixed-size "watch vector" and "search vector".
3. **Concatenation**:
   - All embedded and transformed features are concatenated into a single dense vector.
4. **Fully Connected (ReLU) Layers**:
   - This concatenated vector is passed through multiple **ReLU-activated dense layers**, producing the final **user embedding**.

---

## 🧮 Training vs Serving

- **Training Phase**:
  - The user vector `u` is compared to video vectors `v_j`.
  - A softmax layer is used to compute **class probabilities** over all candidate videos.
  - The loss function encourages the dot product `u·v_j` to be high for watched videos and low for others.

- **Serving Phase**:
  - The trained user vector is matched against precomputed video vectors using a **Nearest Neighbor Index** (e.g., ScaNN, Faiss) to fetch the **top-N recommendations**.

---

## ⚙️ Objective Function

The model optimizes a **categorical cross-entropy loss** using dot product similarity between user and video embeddings:

\[
P(v_j|u) = \frac{e^{u \cdot v_j}}{\sum_{k} e^{u \cdot v_k}}
\]

Where:
- \( u \): user embedding
- \( v_j \): embedding of video \( j \)
- \( P(v_j|u) \): probability that the user watches video \( j \)

---

## 📤 Output

- **Training**: Class probability distribution across all candidate videos.
- **Serving**: Top-N video recommendations based on ANN retrieval using user vector.

---

## 🛠 Technologies & Tools

- TensorFlow or PyTorch for modeling
- Faiss or ScaNN for nearest neighbor retrieval
- TFRecords or similar format for dataset ingestion
- GPUs for training, and ANN indices for real-time serving

---

## ✅ Benefits

- Scales to billions of videos with real-time retrieval
- Modular design: Embedding + ANN allows independent training and serving
- Fast retrieval (in milliseconds) using precomputed ANN indices

---

## 📚 References

- YouTube’s deep learning recommendation model: [Covington et al., 2016](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
- ScaNN: Efficient vector search from Google Research
- TensorFlow Recommenders: TFRS framework for two-tower models

---
