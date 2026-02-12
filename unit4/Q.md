# Unit-4: Cluster Analysis 

**Approximate Weightage Distribution:**
- **Partitioning Methods (K-Means/K-Medoids):** ~25% (Compulsory numerical or 10-mark descriptive).
- **Hierarchical Methods (Agglomerative/Divisive):** ~20% (Dendrogram and Linkage questions).
- **Density & Grid-Based (DBSCAN/STING/CLIQUE):** ~30% (Highest weightage for "Explain with diagram" questions).
- **Evaluation & Advanced Techniques (BIRCH/Evaluation):** ~25% (Metrics and scalability).

---

## **Section A: Introduction & Partitioning Methods**
1. **Define Cluster Analysis.** What are the requirements of a good clustering algorithm?
2. **Explain Partitioning Methods.** List the two main requirements for a valid partitioning.
3. **Describe the K-Means algorithm in detail.** Provide the step-by-step iterative process.
4. **What is K-Medoids?** Why is it considered more robust than K-Means?
5. **Differentiate between a Centroid and a Medoid.**
6. **What is the "Elbow Method"?** How is it used to determine the optimal value of $k$?



---

## **Section B: Hierarchical Methods**
7. **Explain Hierarchical Clustering.** Differentiate between Agglomerative and Divisive approaches.
8. **What is a Dendrogram?** Explain its role in visualizing hierarchical relationships.
9. **Describe the following linkage metrics with diagrams:**
    * Single Linkage (Nearest Neighbor)
    * Complete Linkage (Farthest Neighbor)
    * Average Linkage
10. **What are the limitations of hierarchical clustering?** Why is it considered "irreversible"?



---

## **Section C: Density, Grid, and Model-Based Methods**
11. **Explain the core concept of DBSCAN.** Define Core points, Border points, and Noise.
12. **Describe the two parameters of DBSCAN:** Epsilon ($\epsilon$) and MinPts.
13. **What is BIRCH?** Explain the Clustering Feature (CF) and the structure of a CF-Tree.
14. **Discuss the STING algorithm.** How does it use a hierarchical grid for statistical queries?
15. **What is CLIQUE?** Explain how it handles high-dimensional data using subspace clustering.
16. **Explain Probabilistic Model-Based Clustering.** How does it differ from "hard" clustering?



---

## **Section D: Evaluation of Clustering**
17. **Why is cluster evaluation necessary?** List the three main categories of evaluation.
18. **Explain Internal vs. External evaluation metrics.**
19. **Define the Silhouette Coefficient.** What does a score of +1, 0, or -1 indicate?
20. **What are Cohesion and Separation?** How are they used to measure cluster quality?

---

# **MCQs on Cluster Analysis (Unit 4)**

1. Which algorithm is most sensitive to outliers?
   **A. K-Means** B. K-Medoids  C. DBSCAN  D. BIRCH

2. A Dendrogram is used in which type of clustering?
   A. Partitioning  **B. Hierarchical** C. Density-based  D. Grid-based

3. Which parameter in DBSCAN defines the scanning radius?
   **A. Epsilon** B. MinPts  C. K  D. Threshold

4. Which algorithm is specifically designed for very large datasets (Big Data)?
   A. K-Means  B. PAM  **C. BIRCH** D. AGNES

5. In STING, what does 'n' represent in a grid cell?
   **A. Number of points** B. Mean  C. Standard Deviation  D. Distribution

6. Hard clustering means an object belongs to:
   **A. Exactly one cluster** B. Multiple clusters  C. Zero clusters  D. A probability distribution

7. The "Chaining Effect" is a common problem in:
   **A. Single Linkage** B. Complete Linkage  C. K-Means  D. CLIQUE

8. Which metric measures how similar an object is to its own cluster compared to others?
   A. SSE  **B. Silhouette Score** C. Entropy  D. Gini Index

9. CLIQUE is a hybrid of which two methods?
   **A. Grid and Density** B. Partitioning and Hierarchical  C. Model and Density  D. Grid and Model

10. Which algorithm uses the "Medoid" as a representative object?
    A. K-Means  **B. PAM** C. DBSCAN  D. STING

---

# **Mixed / Important Long-Answer Questions (10/16 Marks)**

1. **Partitioning vs. Hierarchical:** Compare K-Means and Agglomerative Hierarchical clustering in terms of complexity, data size, and cluster shape. (10 marks)
2. **DBSCAN Deep Dive:** Explain the concepts of density-reachability and density-connectivity with suitable diagrams. (16 marks)
3. **Big Data Clustering:** Discuss the BIRCH algorithm, explaining its four phases and why it is memory efficient. (16 marks)
4. **Grid-Based Methods:** Compare STING and CLIQUE. Why is CLIQUE preferred for high-dimensional data? (10 marks)
5. **Numerical Problem:** Given a set of 1D or 2D points and initial centroids, perform one iteration of the K-Means algorithm. (16 marks)

---

**Preparation Tips for Exam:**
* **Diagrams are Compulsory:** For questions on DBSCAN, Hierarchical (Dendrogram), or Grid-based methods, you **must** draw the diagrams to get full marks.
* **Comparison Tables:** Use the tables provided in the study guide for K-Means vs. K-Medoids and Agglomerative vs. Divisive.
* **Memorize Parameters:** Know exactly what $\epsilon$, MinPts, Threshold ($T$), and Branching Factor ($B$) mean.
* **Real-World Examples:** Use "Customer Segmentation" for K-Means and "Satellite Mapping" for STING to illustrate your points.