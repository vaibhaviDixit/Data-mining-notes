# Unit-3: Classification and Prediction 

**Approximate Weightage Distribution:**
* **Basic Concepts & Decision Tree Induction:** ~25% (ID3, CART, and Attribute Selection are frequent 10/16 mark questions).
* **Bayes Classification Methods:** ~20% (Bayes’ Theorem and Naive Bayes numericals/theory).
* **Neural Networks & SVM:** ~20% (Complex but high-value for long answers).
* **Evaluation Metrics:** ~15% (Confusion Matrix, Precision, Recall—compulsory short/medium questions).
* **Techniques to Improve Accuracy & Regression:** ~20% (Ensemble methods like Bagging/Boosting).

---

## **Section A: Basic Concepts & Decision Trees**
1. **Define Classification and Prediction.** Differentiate between them with suitable examples.
2. **Explain the two-step process of Classification** (Model Construction and Model Usage).
3. **What is Decision Tree Induction?** Describe the general algorithm for building a tree.
4. **Compare ID3 and CART algorithms.** List their selection measures and branching styles.
5. **What is Tree Pruning?** Differentiate between pre-pruning and post-pruning. Why is it necessary?
6. **Explain the "Greedy" nature of decision tree algorithms.**



---

## **Section B: Attribute Selection Measures (High Weightage)**
7. **What are Attribute Selection Measures (ASM)?** Why are they called the "brain" of the tree?
8. **Explain Information Gain.** Define Entropy and provide the mathematical formula:
   $$Info(D) = -\sum_{i=1}^{m} p_i \log_2(p_i)$$
9. **Discuss Gain Ratio.** How does it address the bias of Information Gain toward many-valued attributes?
10. **Define Gini Index.** How is it used in the CART algorithm to find the best binary split?
    $$Gini(D) = 1 - \sum_{i=1}^{m} p_i^2$$
11. **Numerical Challenge:** Practice calculating Information Gain for a small dataset (e.g., the "Weather/Play Tennis" dataset).



---

## **Section C: Bayesian Classification**
12. **State and explain Bayes’ Theorem.** Define Posterior probability, Likelihood, and Prior probability.
    $$P(H|X) = \frac{P(X|H) P(H)}{P(X)}$$
13. **Explain Naive Bayesian Classification.** Why is the assumption of "class conditional independence" considered "naive"?
14. **What is Laplacian Correction (Smoothing)?** Explain its importance in handling zero-probability issues.
15. **Discuss the advantages and disadvantages of Naive Bayes**, especially regarding high-dimensional text data.

---

## **Section D: Neural Networks & SVM**
16. **Explain Classification by Backpropagation.** Describe the architecture of a Multilayer Feedforward Neural Network.
17. **Describe the Forward Pass and Backward Pass** in the Backpropagation algorithm.
18. **What are Activation Functions?** Explain the role of Sigmoid or ReLU in introducing non-linearity.
19. **Define Support Vector Machines (SVM).** What are "Support Vectors" and why are they critical?
20. **Explain the "Kernel Trick" in SVM.** How does it help in classifying non-linearly separable data?
21. **What is the Maximum Margin Hyperplane?** Illustrate how it separates two classes.



---

## **Section E: Evaluation & Accuracy Improvement**
22. **Define a Confusion Matrix.** Explain the four outcomes: TP, TN, FP, and FN.
23. **Explain the following metrics with formulas:**
    * **Accuracy:** $(TP + TN) / \text{Total}$
    * **Precision:** $TP / (TP + FP)$
    * **Recall:** $TP / (TP + FN)$
    * **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
24. **Why is Accuracy not always a reliable metric?** Justify using the "Imbalanced Data" example.
25. **What are Ensemble Methods?** Explain Bagging and Boosting.
26. **Differentiate between Random Forest (Bagging) and AdaBoost (Boosting).**
27. **Explain K-Fold Cross-Validation.** How does it ensure the reliability of a model?



---

## **Section F: Prediction & Regression**
28. **Define Regression Analysis.** How does it differ from Classification?
29. **Explain Simple Linear Regression vs. Multiple Linear Regression.**
30. **What is the Method of Least Squares?** Explain how it minimizes the sum of squared residuals.
31. **List the evaluation metrics for Regression:** MAE, MSE, RMSE, and $R^2$.

---

# **MCQs on Classification and Prediction (Unit 3)**

1. Which of the following is a "Lazy Learner"?
   A. Decision Tree  **B. K-Nearest Neighbor** C. SVM  D. Neural Network

2. In Information Gain, a perfectly pure node has an Entropy of:
   **A. 0** B. 1  C. 0.5  D. Infinity

3. Which algorithm is strictly restricted to binary splits?
   A. ID3  **B. CART** C. C4.5  D. Naive Bayes

4. The problem of a model performing well on training data but poorly on test data is called:
   A. Underfitting  **B. Overfitting** C. Pruning  D. Scaling

5. Which metric is most important for a medical test where we cannot afford to miss a sick patient?
   A. Precision  **B. Recall (Sensitivity)** C. Accuracy  D. Specificity

6. SVM finds the hyperplane that:
   A. Minimizes the error  **B. Maximizes the margin** C. Minimizes the margin  D. Ignores outliers

7. Laplacian correction is used to:
   A. Reduce noise  **B. Avoid zero probabilities** C. Prune trees  D. Normalize data

8. Random Forest is an example of:
   **A. Bagging** B. Boosting  C. Pruning  D. Regression

9. The "Kernel Trick" is associated with:
   A. Naive Bayes  B. ID3  **C. SVM** D. KNN

10. Which activation function squashes values between 0 and 1?
    **A. Sigmoid** B. ReLU  C. Tanh  D. Linear

---

# **Mixed / Important Long-Answer Questions**

1.  **Decision Tree Induction:** Explain the building process, attribute selection (Information Gain), and the role of pruning. (16 marks)
2.  **Naive Bayes Numericals:** Given a dataset table, predict the class of a new tuple $X$ using Bayesian probabilities. (16 marks)
3.  **Neural Network Architecture:** Draw and explain the Backpropagation algorithm, including the weight update rule. (16 marks)
4.  **Evaluation Metrics:** Discuss the Confusion Matrix and derive formulas for Accuracy, Precision, Recall, and F1-score. (10 marks)
5.  **Ensemble Methods:** Compare Bagging and Boosting in detail. Explain how Random Forest improves results. (16 marks)

---

**Preparation Tips for Exam:**
* **Focus on ASM:** Expect a 16-mark question combining Information Gain and Gini Index logic.
* **Formulas are Key:** Memorize the Bayes' formula and the evaluation metrics (Precision/Recall). 
* **Diagrams:** Always draw the Decision Tree structure and the SVM Hyperplane.
* **Practice Numericals:** Be ready to calculate Entropy and Gain for a small table.