# Lazy Learners - K-Nearest Neighbors (KNN)

Unlike Decision Trees or Neural Networks, which spend a lot of time "learning" and building a model before they can make a prediction, **Lazy Learners** do almost no work during the training phase. Instead, they wait until they see a new test tuple and then "act" by searching through the stored data. **K-Nearest Neighbors (KNN)** is the most famous example of this approach.

---

## 1. The Concept: "Show Me Your Friends"
The philosophy of KNN is simple: **"Birds of a feather flock together."** If you want to know what class a new data point belongs to, look at its $K$ closest neighbors. If the majority of its neighbors belong to Class A, then the new point likely belongs to Class A too.

* **Eager Learners:** Build a general model (like a rule or a formula) that fits the whole data space.
* **Lazy Learners:** Don't build a model. They simply store the training data and perform "local" reasoning when a query arrives.



---

## 2. How the KNN Algorithm Works
The process is straightforward and involves these steps:

1.  **Store the Data:** All training tuples are stored in memory.
2.  **Calculate Distance:** When a new tuple arrives, calculate its distance to every single point in the training set.
3.  **Identify Neighbors:** Pick the $K$ points that are closest to the new tuple.
4.  **Vote:**  **For Classification:** Use "Majority Voting." The class that appears most often among the $K$ neighbors is assigned to the new point.
    * **For Regression:** Calculate the **average** value of the $K$ neighbors.


![knn](./imgs/KNN.png)

---

## 3. Measuring "Closeness": Distance Metrics
To find the "nearest" neighbors, we need to calculate the distance between points. The choice of metric depends on the type of data:

### **A. Euclidean Distance (Most Common)**
The straight-line distance between two points $(x_1, y_1)$ and $(x_2, y_2)$.
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

### **B. Manhattan Distance**
The distance measured along axes at right angles (like walking through city blocks).
$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

### **C. Minkowski Distance**
A generalized formula where you can adjust a parameter $p$ to switch between Euclidean ($p=2$) and Manhattan ($p=1$).

[Image comparing Euclidean vs Manhattan distance logic]

---

## 4. Choosing the "K" Value
The parameter **$K$** is the most important setting in this algorithm.

* **Small K (e.g., $K=1$):** The model is very sensitive to "noise" and outliers. It might overfit because it only looks at the single closest neighbor.
* **Large K:** The model becomes "smoother" and more stable, but if $K$ is too large, it might include points from other classes, leading to "underfitting."
* **The "Odd Number" Rule:** Always pick an **odd number** for $K$ (e.g., 3, 5, 7) to avoid a "tie" during voting.

---

## 5. Pros and Cons of Lazy Learning

| Feature | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **Training Time** | Zero. It just stores the data. | **Prediction Time** | Very slow for large datasets because it calculates distances for every query. |
| **Complexity** | Simple to understand and implement. | **Memory Usage** | High. Must keep the entire training set in RAM. |
| **Adaptability** | Naturally handles new data (just add it to the set). | **Sensitive to Scale** | Features with large ranges (like Salary) will dominate small ones (like Age) unless data is normalized. |
| **Non-Linear** | Can learn very complex boundaries. | **Outlier Sensitive** | Noisy data can easily misguide the "vote." |

---

## 6. Real-World Applications
1.  **Recommender Systems:** "Users who liked this movie also liked..." (KNN finds similar users).
2.  **Credit Scoring:** Comparing a new applicant to similar historical profiles to predict risk.
3.  **Pattern Recognition:** Used in initial stages of optical character recognition (OCR).
4.  **Medical Detection:** Finding similar past cases of patients with specific symptoms.

---

## 7. Comparison: Eager vs. Lazy Learning

| Category | Eager Learners (Decision Tree, SVM, ANN) | Lazy Learners (KNN, Case-Based Reasoning) |
| :--- | :--- | :--- |
| **Learning Step** | Slow (Builds a global model). | Fast (Just stores data). |
| **Classification Step**| Fast (Just uses the model). | Slow (Must compare to all data). |
| **Model Storage** | Small (Only the rules/weights). | Large (Must store all training tuples). |
| **Generalization** | Commits to a single global hypothesis. | Performs local, instance-specific reasoning. |

---

# Algorithm: k-Nearest Neighbor (k-NN)
**Topic: Classification and Prediction (Unit 3)**

k-NN is a non-parametric, "lazy learning" algorithm that classifies a data point based on how its neighbors are classified. It stores the entire training dataset and performs a calculation only when a new query is made.

---

## 1. Basic Working Steps
1.  **Select 'k':** Choose the number of nearest neighbors (usually an odd number to avoid ties).
2.  **Calculate Distance:** Compute the distance between the new test point and all points in the training dataset.
3.  **Find Neighbors:** Sort the distances in ascending order and pick the top $k$ closest points.
4.  **Vote:** Count the number of points in each category among the $k$ neighbors.
5.  **Classify:** Assign the test point to the class that appears most frequently.

---

## 2. Key Formulas

### **A. Euclidean Distance**
The most common metric for calculating distance between two points $P_1(x_1, y_1)$ and $P_2(x_2, y_2)$ in 2D space:
$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

### **B. Rule of Thumb for 'k'**
A common practice to choose the initial value of $k$ based on the total number of samples ($n$):
$$k \approx \sqrt{n}$$



---

## 3. Practical Example

**Dataset: Predict if a "Book" is "Heavy" or "Light" based on its "Number of Pages" and "Cost".**

| Book | Pages ($x_1$) | Cost ($x_2$) | Class |
| :--- | :--- | :--- | :--- |
| A | 167 | 51 | Light |
| B | 176 | 69 | Heavy |
| C | 174 | 56 | Light |
| D | 173 | 64 | Heavy |

**Test Case (X):** {Pages = 170, Cost = 57}, **Set $k = 3$**

### **Step 1: Calculate Euclidean Distance from X**
* **Dist(X, A):** $\sqrt{(170-167)^2 + (57-51)^2} = \sqrt{3^2 + 6^2} = \sqrt{45} \approx \mathbf{6.71}$
* **Dist(X, B):** $\sqrt{(170-176)^2 + (57-69)^2} = \sqrt{(-6)^2 + (-12)^2} = \sqrt{180} \approx \mathbf{13.42}$
* **Dist(X, C):** $\sqrt{(170-174)^2 + (57-56)^2} = \sqrt{(-4)^2 + 1^2} = \sqrt{17} \approx \mathbf{4.12}$
* **Dist(X, D):** $\sqrt{(170-173)^2 + (57-64)^2} = \sqrt{(-3)^2 + (-7)^2} = \sqrt{58} \approx \mathbf{7.62}$

### **Step 2: Identify 3 Nearest Neighbors**
1. Book C (Dist = 4.12)
2. Book A (Dist = 6.71)
3. Book D (Dist = 7.62)

### **Step 3: Majority Vote**
* Neighbors: {Light, Light, Heavy}
* Most frequent class: **Light**

**Decision:** The new book is classified as **Light**.
