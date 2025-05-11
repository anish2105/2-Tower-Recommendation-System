## Recommendation System

Recommendation systems are crucial for helping users discover relevant items within massive collections, such as the millions of products available on Amazon or the vast library of music on Spotify (with new content added constantly). While search functionality is helpful, these systems can highlight interesting items that users might not have otherwise encountered.

### Architecture for recommendation systems consists of the following components:

#### Candidate generation
In this first stage, the system starts from a potentially huge corpus and generates a much smaller subset of candidates. For example, the candidate generator in YouTube reduces billions of videos down to hundreds or thousands. The model needs to evaluate queries quickly given the enormous size of the corpus. A given model may provide multiple candidate generators, each nominating a different subset of candidates.

#### Scoring
Next, another model scores and ranks the candidates in order to select the set of items (on the order of 10) to display to the user. Since this model evaluates a relatively small subset of items, the system can use a more precise model relying on additional queries.

#### Re-ranking
Finally, the system must take into account additional constraints for the final ranking. For example, the system removes items that the user explicitly disliked or boosts the score of fresher content. Re-ranking can also help ensure diversity, freshness, and fairness.

---

## 1) Candidate generation

### Content based filtering

- Uses similarity between items to recommend items similar to what the user likes

---

### Collaborative filtering 

- Uses similarities between queries and items simultaneously to provide recommendations.

---

### Matrix Factorization

A technique that learns user and item embeddings from a feedback matrix to predict preferences.

- Matrix factorization is a straightforward embedding model used in recommendation systems. It takes a feedback matrix, where rows represent users and columns represent items, and decomposes it into two lower-dimensional matrices: a user embedding matrix and an item embedding matrix.
- Each row in these embedding matrices represents a latent vector capturing the characteristics of a user or an item, respectively. The goal is to learn these embeddings by minimizing the objective function, which measures the difference between the predicted and actual feedback.
- Common optimization algorithms include Stochastic Gradient Descent (SGD) and Weighted Alternating Least Squares (WALS), with WALS often preferred due to its reliance on least squares, parallelizability, faster convergence, and easier handling of unobserved interactions.

---

Due to limitations in handling new items (the cold-start problem) and other scaling issues inherent in traditional Collaborative and Content-based filtering, Deep Neural Network-based recommender systems were being employed.

## 2 Tower Model

The Two-Tower Model is a deep learning architecture designed to enhance recommendation systems by separately processing user and item information. This separation allows for more flexible and scalable recommendations, especially when dealing with new users or items

### Architecture Diagram

User Features ──▶ User Tower ──▶ User Embedding

                                          │
                                          ▼
                                    Similarity Score
                                          ▲
                                          │
Item Features ──▶ Item Tower ──▶ Item Embedding

### Example - Movie Recommendation System

🎯 User Tower
Input: Features about the user, such as:

1) User ID
2) Age
3) Gender
4) Viewing history
5) Preferred genres​

- Process: These features are passed through a neural network to produce a user embedding, a numerical representation capturing the user's preferences.​

🎬 Item Tower

Input: Features about the movie, such as:

1) Movie ID
2) Genre
3) Director
4) Cast
5) Release year​

- Process: These features are passed through another neural network to produce an item embedding, a numerical representation capturing the movie's characteristics.​

🔗 Matching

- Similarity Score: The system computes the similarity between the user and item embeddings, often using a dot product. A higher score indicates a higher likelihood that the user will enjoy the movie.

#### Negative examples 
Items labeled "irrelevant" to a given query. Showing the model negative examples during training teaches the model that embeddings of different groups should be pushed away from each other.

The 2-tower model produces 2 embeddings model: query and item embeddings.

---

## 2) Retrieval

After obtaining the query embedding q, the next step involves identifying item embeddings Vj that are located nearby in the embedding space. This constitutes a nearest neighbor search. For instance, the top k items can be retrieved based on their similarity score, often calculated using a softmax function applied to the query and item embeddings: softmax(query,Item).

## 3) Scoring

After candidate generation, another model scores and ranks the generated candidates to select the set of items to display. The recommendation system may have multiple candidate generators that use different sources, such as the following:
- User features that account for personalization.
- Popular or trending items.
- A social graph; that is, items liked or recommended by friends.

## 4) Re-ranker

Recommendation systems can be improved by re-ranking candidates using filters or score transformations based on criteria like video age or click-bait detection.