# Tree Pruning

In machine learning, "bigger is not always better." While a fully grown decision tree can achieve 100% accuracy on training data, it often fails miserably on new data because it has captured the **noise** (random fluctuations) instead of the **signal** (the actual pattern). **Tree Pruning** is the essential process of simplifying a tree to make it "smarter" and more robust.

---

## 1. The Core Conflict: Overfitting vs. Underfitting
Pruning is the act of finding the "Sweet Spot" in the complexity of a model.

* **No Pruning (Overfitting):** The tree is too deep. It has a branch for every single outlier. It "memorizes" the past but cannot predict the future.
* **Too Much Pruning (Underfitting):** The tree is too shallow. It misses important patterns and is too simple to be useful.
* **Optimal Tree (Pruned):** The tree is just deep enough to capture the general rules while ignoring the random noise.



---

## 2. Pre-Pruning: The "Preventative" Approach
Pre-pruning acts like a set of rules that stop the tree-building process before it goes too far. This is a **Top-Down** approach.

### **Common Stopping Thresholds:**
1.  **Maximum Depth:** Limits how many "levels" the tree can have.
2.  **Minimum Samples per Leaf:** Ensures that a final decision (leaf) is based on a significant number of data points (e.g., "Don't make a decision based on just 2 people").
3.  **Minimum Impurity Decrease:** Only split a node if it improves the "purity" (Gini/Entropy) by a significant amount (e.g., > 0.05).
4.  **Max Features:** Limits the number of attributes the tree can consider at each split.

**The Risk:** Pre-pruning can suffer from the **"Horizon Effect."** It might stop a split that looks bad now, but that split could have led to a very important discovery 2 levels deeper.

---

## 3. Post-Pruning: The "Curative" Approach
Post-pruning is a **Bottom-Up** approach. It is generally more successful because it looks at the whole tree before making cuts.

### **Cost-Complexity Pruning (The CART Logic)**
This is the most famous mathematical method for pruning. It uses a parameter called **Alpha ($\alpha$)**.

The goal is to minimize the following "Cost" function:
$$R_\alpha(T) = R(T) + \alpha|T|$$

* **$R(T)$:** The misclassification error of the tree.
* **$|T|$:** The number of terminal nodes (leaves).
* **$\alpha$:** The complexity parameter. 
    * If $\alpha$ is 0, the tree stays large.
    * As $\alpha$ increases, the "penalty" for having more leaves grows, forcing the tree to become smaller.



---

## 4. Pruning Techniques: A Closer Look

### **Reduced Error Pruning (REP)**
1.  Divide your data into a **Training Set** and a **Validation Set**.
2.  Grow the tree fully using the Training Set.
3.  For every internal node, temporarily replace it with a leaf and check the accuracy on the **Validation Set**.
4.  If the accuracy stays the same or improves, **cut the branch permanently.**
5.  Repeat until no more cuts improve the accuracy.

### **Rule Post-Pruning**
1.  Convert the tree into a set of **IF-THEN rules**.
2.  Each path from the root to a leaf becomes one rule.
3.  Analyze each rule and remove any "IF" conditions that don't help accuracy.
4.  Sort the simplified rules by accuracy and use them for classification.

![treepurning](./imgs/Before_after_pruning.png)
---

## 5. Comparison: Why Post-Pruning is Often Better

| Feature | Pre-Pruning | Post-Pruning |
| :--- | :--- | :--- |
| **Strategy** | Stop early. | Grow full, then cut. |
| **Visibility** | Limited (cannot see future splits). | Full (sees the entire tree structure). |
| **Computation** | Very fast and efficient. | Slower (requires building the full tree). |
| **Accuracy** | Risk of "Underfitting." | Usually higher accuracy and better generalization. |
| **Standard Use** | Real-time / Large datasets. | High-precision scientific models. |

---

## 6. How to Explain Pruning in Exams
When asked about Pruning, focus on these three keywords:
1.  **Complexity:** Reducing the number of nodes.
2.  **Generalization:** Improving performance on new, unseen data.
3.  **Noise:** Removing branches that were created due to errors or outliers in the training set.



---

### **Summary Table**
| If the tree is... | Problem | Action |
| :--- | :--- | :--- |
| Too Deep | Overfitting | Apply Post-Pruning |
| Too Shallow | Underfitting | Reduce Pruning constraints |
| Perfect | Balanced | Model generalizes well |