---


---

<h1 id="unit-3-classification-and-prediction">Unit-3: Classification and Prediction</h1>
<p><strong>Approximate Weightage Distribution:</strong></p>
<ul>
<li><strong>Basic Concepts &amp; Decision Tree Induction:</strong> ~25% (ID3, CART, and Attribute Selection are frequent 10/16 mark questions).</li>
<li><strong>Bayes Classification Methods:</strong> ~20% (Bayes’ Theorem and Naive Bayes numericals/theory).</li>
<li><strong>Neural Networks &amp; SVM:</strong> ~20% (Complex but high-value for long answers).</li>
<li><strong>Evaluation Metrics:</strong> ~15% (Confusion Matrix, Precision, Recall—compulsory short/medium questions).</li>
<li><strong>Techniques to Improve Accuracy &amp; Regression:</strong> ~20% (Ensemble methods like Bagging/Boosting).</li>
</ul>
<hr>
<h2 id="section-a-basic-concepts--decision-trees"><strong>Section A: Basic Concepts &amp; Decision Trees</strong></h2>
<ol>
<li><strong>Define Classification and Prediction.</strong> Differentiate between them with suitable examples.</li>
<li><strong>Explain the two-step process of Classification</strong> (Model Construction and Model Usage).</li>
<li><strong>What is Decision Tree Induction?</strong> Describe the general algorithm for building a tree.</li>
<li><strong>Compare ID3 and CART algorithms.</strong> List their selection measures and branching styles.</li>
<li><strong>What is Tree Pruning?</strong> Differentiate between pre-pruning and post-pruning. Why is it necessary?</li>
<li><strong>Explain the “Greedy” nature of decision tree algorithms.</strong></li>
</ol>
<hr>
<h2 id="section-b-attribute-selection-measures-high-weightage"><strong>Section B: Attribute Selection Measures (High Weightage)</strong></h2>
<ol start="7">
<li><strong>What are Attribute Selection Measures (ASM)?</strong> Why are they called the “brain” of the tree?</li>
<li><strong>Explain Information Gain.</strong> Define Entropy and provide the mathematical formula:<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>I</mi><mi>n</mi><mi>f</mi><mi>o</mi><mo stretchy="false">(</mo><mi>D</mi><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msub><mi>p</mi><mi>i</mi></msub><msub><mrow><mi>log</mi><mo>⁡</mo></mrow><mn>2</mn></msub><mo stretchy="false">(</mo><msub><mi>p</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">Info(D) = -\sum_{i=1}^{m} p_i \log_2(p_i)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.07847em;">I</span><span class="mord mathnormal">n</span><span class="mord mathnormal" style="margin-right: 0.10764em;">f</span><span class="mord mathnormal">o</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.02778em;">D</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 2.92907em; vertical-align: -1.27767em;"></span><span class="mord">−</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.6514em;"><span class="" style="top: -1.87233em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mrel mtight">=</span><span class="mord mtight">1</span></span></span></span><span class="" style="top: -3.05001em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span><span class="" style="top: -4.30001em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">m</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.27767em;"><span class=""></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop"><span class="mop">lo<span style="margin-right: 0.01389em;">g</span></span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.206968em;"><span class="" style="top: -2.45586em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.24414em;"><span class=""></span></span></span></span></span></span><span class="mopen">(</span><span class="mord"><span class="mord mathnormal">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span></span></li>
<li><strong>Discuss Gain Ratio.</strong> How does it address the bias of Information Gain toward many-valued attributes?</li>
<li><strong>Define Gini Index.</strong> How is it used in the CART algorithm to find the best binary split?<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>G</mi><mi>i</mi><mi>n</mi><mi>i</mi><mo stretchy="false">(</mo><mi>D</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mo>−</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msubsup><mi>p</mi><mi>i</mi><mn>2</mn></msubsup></mrow><annotation encoding="application/x-tex">Gini(D) = 1 - \sum_{i=1}^{m} p_i^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">G</span><span class="mord mathnormal">ini</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.02778em;">D</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.72777em; vertical-align: -0.08333em;"></span><span class="mord">1</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 2.92907em; vertical-align: -1.27767em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.6514em;"><span class="" style="top: -1.87233em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mrel mtight">=</span><span class="mord mtight">1</span></span></span></span><span class="" style="top: -3.05001em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span><span class="" style="top: -4.30001em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">m</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.27767em;"><span class=""></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.864108em;"><span class="" style="top: -2.453em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.247em;"><span class=""></span></span></span></span></span></span></span></span></span></span></span></li>
<li><strong>Numerical Challenge:</strong> Practice calculating Information Gain for a small dataset (e.g., the “Weather/Play Tennis” dataset).</li>
</ol>
<hr>
<h2 id="section-c-bayesian-classification"><strong>Section C: Bayesian Classification</strong></h2>
<ol start="12">
<li><strong>State and explain Bayes’ Theorem.</strong> Define Posterior probability, Likelihood, and Prior probability.<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mi mathvariant="normal">∣</mi><mi>X</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mi mathvariant="normal">∣</mi><mi>H</mi><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo></mrow><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo></mrow></mfrac></mrow><annotation encoding="application/x-tex">P(H|X) = \frac{P(X|H) P(H)}{P(X)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.13889em;">P</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.08125em;">H</span><span class="mord">∣</span><span class="mord mathnormal" style="margin-right: 0.07847em;">X</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 2.363em; vertical-align: -0.936em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.427em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.13889em;">P</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.07847em;">X</span><span class="mclose">)</span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.13889em;">P</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.07847em;">X</span><span class="mord">∣</span><span class="mord mathnormal" style="margin-right: 0.08125em;">H</span><span class="mclose">)</span><span class="mord mathnormal" style="margin-right: 0.13889em;">P</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.08125em;">H</span><span class="mclose">)</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.936em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span></span></li>
<li><strong>Explain Naive Bayesian Classification.</strong> Why is the assumption of “class conditional independence” considered “naive”?</li>
<li><strong>What is Laplacian Correction (Smoothing)?</strong> Explain its importance in handling zero-probability issues.</li>
<li><strong>Discuss the advantages and disadvantages of Naive Bayes</strong>, especially regarding high-dimensional text data.</li>
</ol>
<hr>
<h2 id="section-d-neural-networks--svm"><strong>Section D: Neural Networks &amp; SVM</strong></h2>
<ol start="16">
<li><strong>Explain Classification by Backpropagation.</strong> Describe the architecture of a Multilayer Feedforward Neural Network.</li>
<li><strong>Describe the Forward Pass and Backward Pass</strong> in the Backpropagation algorithm.</li>
<li><strong>What are Activation Functions?</strong> Explain the role of Sigmoid or ReLU in introducing non-linearity.</li>
<li><strong>Define Support Vector Machines (SVM).</strong> What are “Support Vectors” and why are they critical?</li>
<li><strong>Explain the “Kernel Trick” in SVM.</strong> How does it help in classifying non-linearly separable data?</li>
<li><strong>What is the Maximum Margin Hyperplane?</strong> Illustrate how it separates two classes.</li>
</ol>
<hr>
<h2 id="section-e-evaluation--accuracy-improvement"><strong>Section E: Evaluation &amp; Accuracy Improvement</strong></h2>
<ol start="22">
<li><strong>Define a Confusion Matrix.</strong> Explain the four outcomes: TP, TN, FP, and FN.</li>
<li><strong>Explain the following metrics with formulas:</strong>
<ul>
<li><strong>Accuracy:</strong> <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false">(</mo><mi>T</mi><mi>P</mi><mo>+</mo><mi>T</mi><mi>N</mi><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mtext>Total</mtext></mrow><annotation encoding="application/x-tex">(TP + TN) / \text{Total}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.13889em;">TP</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.10903em;">TN</span><span class="mclose">)</span><span class="mord">/</span><span class="mord text"><span class="mord">Total</span></span></span></span></span></span></li>
<li><strong>Precision:</strong> <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi><mi>P</mi><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mi>T</mi><mi>P</mi><mo>+</mo><mi>F</mi><mi>P</mi><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">TP / (TP + FP)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.13889em;">TP</span><span class="mord">/</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.13889em;">TP</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.13889em;">FP</span><span class="mclose">)</span></span></span></span></span></li>
<li><strong>Recall:</strong> <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi><mi>P</mi><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mi>T</mi><mi>P</mi><mo>+</mo><mi>F</mi><mi>N</mi><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">TP / (TP + FN)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.13889em;">TP</span><span class="mord">/</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.13889em;">TP</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.10903em;">FN</span><span class="mclose">)</span></span></span></span></span></li>
<li><strong>F1-Score:</strong> <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2</mn><mo>×</mo><mfrac><mrow><mtext>Precision</mtext><mo>×</mo><mtext>Recall</mtext></mrow><mrow><mtext>Precision</mtext><mo>+</mo><mtext>Recall</mtext></mrow></mfrac></mrow><annotation encoding="application/x-tex">2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.72777em; vertical-align: -0.08333em;"></span><span class="mord">2</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.28344em; vertical-align: -0.403331em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.880108em;"><span class="" style="top: -2.655em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord text mtight"><span class="mord mtight">Precision</span></span><span class="mbin mtight">+</span><span class="mord text mtight"><span class="mord mtight">Recall</span></span></span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.394em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord text mtight"><span class="mord mtight">Precision</span></span><span class="mbin mtight">×</span><span class="mord text mtight"><span class="mord mtight">Recall</span></span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.403331em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span></li>
</ul>
</li>
<li><strong>Why is Accuracy not always a reliable metric?</strong> Justify using the “Imbalanced Data” example.</li>
<li><strong>What are Ensemble Methods?</strong> Explain Bagging and Boosting.</li>
<li><strong>Differentiate between Random Forest (Bagging) and AdaBoost (Boosting).</strong></li>
<li><strong>Explain K-Fold Cross-Validation.</strong> How does it ensure the reliability of a model?</li>
</ol>
<hr>
<h2 id="section-f-prediction--regression"><strong>Section F: Prediction &amp; Regression</strong></h2>
<ol start="28">
<li><strong>Define Regression Analysis.</strong> How does it differ from Classification?</li>
<li><strong>Explain Simple Linear Regression vs. Multiple Linear Regression.</strong></li>
<li><strong>What is the Method of Least Squares?</strong> Explain how it minimizes the sum of squared residuals.</li>
<li><strong>List the evaluation metrics for Regression:</strong> MAE, MSE, RMSE, and <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi>R</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">R^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.814108em; vertical-align: 0em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.00773em;">R</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.814108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span>.</li>
</ol>
<hr>
<h1 id="mcqs-on-classification-and-prediction-unit-3"><strong>MCQs on Classification and Prediction (Unit 3)</strong></h1>
<ol>
<li>
<p>Which of the following is a “Lazy Learner”?<br>
A. Decision Tree  <strong>B. K-Nearest Neighbor</strong> C. SVM  D. Neural Network</p>
</li>
<li>
<p>In Information Gain, a perfectly pure node has an Entropy of:<br>
<strong>A. 0</strong> B. 1  C. 0.5  D. Infinity</p>
</li>
<li>
<p>Which algorithm is strictly restricted to binary splits?<br>
A. ID3  <strong>B. CART</strong> C. C4.5  D. Naive Bayes</p>
</li>
<li>
<p>The problem of a model performing well on training data but poorly on test data is called:<br>
A. Underfitting  <strong>B. Overfitting</strong> C. Pruning  D. Scaling</p>
</li>
<li>
<p>Which metric is most important for a medical test where we cannot afford to miss a sick patient?<br>
A. Precision  <strong>B. Recall (Sensitivity)</strong> C. Accuracy  D. Specificity</p>
</li>
<li>
<p>SVM finds the hyperplane that:<br>
A. Minimizes the error  <strong>B. Maximizes the margin</strong> C. Minimizes the margin  D. Ignores outliers</p>
</li>
<li>
<p>Laplacian correction is used to:<br>
A. Reduce noise  <strong>B. Avoid zero probabilities</strong> C. Prune trees  D. Normalize data</p>
</li>
<li>
<p>Random Forest is an example of:<br>
<strong>A. Bagging</strong> B. Boosting  C. Pruning  D. Regression</p>
</li>
<li>
<p>The “Kernel Trick” is associated with:<br>
A. Naive Bayes  B. ID3  <strong>C. SVM</strong> D. KNN</p>
</li>
<li>
<p>Which activation function squashes values between 0 and 1?<br>
<strong>A. Sigmoid</strong> B. ReLU  C. Tanh  D. Linear</p>
</li>
</ol>
<hr>
<h1 id="mixed--important-long-answer-questions"><strong>Mixed / Important Long-Answer Questions</strong></h1>
<ol>
<li><strong>Decision Tree Induction:</strong> Explain the building process, attribute selection (Information Gain), and the role of pruning. (16 marks)</li>
<li><strong>Naive Bayes Numericals:</strong> Given a dataset table, predict the class of a new tuple <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>X</mi></mrow><annotation encoding="application/x-tex">X</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.07847em;">X</span></span></span></span></span> using Bayesian probabilities. (16 marks)</li>
<li><strong>Neural Network Architecture:</strong> Draw and explain the Backpropagation algorithm, including the weight update rule. (16 marks)</li>
<li><strong>Evaluation Metrics:</strong> Discuss the Confusion Matrix and derive formulas for Accuracy, Precision, Recall, and F1-score. (10 marks)</li>
<li><strong>Ensemble Methods:</strong> Compare Bagging and Boosting in detail. Explain how Random Forest improves results. (16 marks)</li>
</ol>
<hr>
<p><strong>Preparation Tips for Exam:</strong></p>
<ul>
<li><strong>Focus on ASM:</strong> Expect a 16-mark question combining Information Gain and Gini Index logic.</li>
<li><strong>Formulas are Key:</strong> Memorize the Bayes’ formula and the evaluation metrics (Precision/Recall).</li>
<li><strong>Diagrams:</strong> Always draw the Decision Tree structure and the SVM Hyperplane.</li>
<li><strong>Practice Numericals:</strong> Be ready to calculate Entropy and Gain for a small table.</li>
</ul>

