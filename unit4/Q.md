---


---

<h1 id="unit-4-cluster-analysis">Unit-4: Cluster Analysis</h1>
<p><strong>Approximate Weightage Distribution:</strong></p>
<ul>
<li><strong>Partitioning Methods (K-Means/K-Medoids):</strong> ~25% (Compulsory numerical or 10-mark descriptive).</li>
<li><strong>Hierarchical Methods (Agglomerative/Divisive):</strong> ~20% (Dendrogram and Linkage questions).</li>
<li><strong>Density &amp; Grid-Based (DBSCAN/STING/CLIQUE):</strong> ~30% (Highest weightage for “Explain with diagram” questions).</li>
<li><strong>Evaluation &amp; Advanced Techniques (BIRCH/Evaluation):</strong> ~25% (Metrics and scalability).</li>
</ul>
<hr>
<h2 id="section-a-introduction--partitioning-methods"><strong>Section A: Introduction &amp; Partitioning Methods</strong></h2>
<ol>
<li><strong>Define Cluster Analysis.</strong> What are the requirements of a good clustering algorithm?</li>
<li><strong>Explain Partitioning Methods.</strong> List the two main requirements for a valid partitioning.</li>
<li><strong>Describe the K-Means algorithm in detail.</strong> Provide the step-by-step iterative process.</li>
<li><strong>What is K-Medoids?</strong> Why is it considered more robust than K-Means?</li>
<li><strong>Differentiate between a Centroid and a Medoid.</strong></li>
<li><strong>What is the “Elbow Method”?</strong> How is it used to determine the optimal value of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span>?</li>
</ol>
<hr>
<h2 id="section-b-hierarchical-methods"><strong>Section B: Hierarchical Methods</strong></h2>
<ol start="7">
<li><strong>Explain Hierarchical Clustering.</strong> Differentiate between Agglomerative and Divisive approaches.</li>
<li><strong>What is a Dendrogram?</strong> Explain its role in visualizing hierarchical relationships.</li>
<li><strong>Describe the following linkage metrics with diagrams:</strong>
<ul>
<li>Single Linkage (Nearest Neighbor)</li>
<li>Complete Linkage (Farthest Neighbor)</li>
<li>Average Linkage</li>
</ul>
</li>
<li><strong>What are the limitations of hierarchical clustering?</strong> Why is it considered “irreversible”?</li>
</ol>
<hr>
<h2 id="section-c-density-grid-and-model-based-methods"><strong>Section C: Density, Grid, and Model-Based Methods</strong></h2>
<ol start="11">
<li><strong>Explain the core concept of DBSCAN.</strong> Define Core points, Border points, and Noise.</li>
<li><strong>Describe the two parameters of DBSCAN:</strong> Epsilon (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span>) and MinPts.</li>
<li><strong>What is BIRCH?</strong> Explain the Clustering Feature (CF) and the structure of a CF-Tree.</li>
<li><strong>Discuss the STING algorithm.</strong> How does it use a hierarchical grid for statistical queries?</li>
<li><strong>What is CLIQUE?</strong> Explain how it handles high-dimensional data using subspace clustering.</li>
<li><strong>Explain Probabilistic Model-Based Clustering.</strong> How does it differ from “hard” clustering?</li>
</ol>
<hr>
<h2 id="section-d-evaluation-of-clustering"><strong>Section D: Evaluation of Clustering</strong></h2>
<ol start="17">
<li><strong>Why is cluster evaluation necessary?</strong> List the three main categories of evaluation.</li>
<li><strong>Explain Internal vs. External evaluation metrics.</strong></li>
<li><strong>Define the Silhouette Coefficient.</strong> What does a score of +1, 0, or -1 indicate?</li>
<li><strong>What are Cohesion and Separation?</strong> How are they used to measure cluster quality?</li>
</ol>
<hr>
<h1 id="mcqs-on-cluster-analysis-unit-4"><strong>MCQs on Cluster Analysis (Unit 4)</strong></h1>
<ol>
<li>
<p>Which algorithm is most sensitive to outliers?<br>
<strong>A. K-Means</strong> B. K-Medoids  C. DBSCAN  D. BIRCH</p>
</li>
<li>
<p>A Dendrogram is used in which type of clustering?<br>
A. Partitioning  <strong>B. Hierarchical</strong> C. Density-based  D. Grid-based</p>
</li>
<li>
<p>Which parameter in DBSCAN defines the scanning radius?<br>
<strong>A. Epsilon</strong> B. MinPts  C. K  D. Threshold</p>
</li>
<li>
<p>Which algorithm is specifically designed for very large datasets (Big Data)?<br>
A. K-Means  B. PAM  <strong>C. BIRCH</strong> D. AGNES</p>
</li>
<li>
<p>In STING, what does ‘n’ represent in a grid cell?<br>
<strong>A. Number of points</strong> B. Mean  C. Standard Deviation  D. Distribution</p>
</li>
<li>
<p>Hard clustering means an object belongs to:<br>
<strong>A. Exactly one cluster</strong> B. Multiple clusters  C. Zero clusters  D. A probability distribution</p>
</li>
<li>
<p>The “Chaining Effect” is a common problem in:<br>
<strong>A. Single Linkage</strong> B. Complete Linkage  C. K-Means  D. CLIQUE</p>
</li>
<li>
<p>Which metric measures how similar an object is to its own cluster compared to others?<br>
A. SSE  <strong>B. Silhouette Score</strong> C. Entropy  D. Gini Index</p>
</li>
<li>
<p>CLIQUE is a hybrid of which two methods?<br>
<strong>A. Grid and Density</strong> B. Partitioning and Hierarchical  C. Model and Density  D. Grid and Model</p>
</li>
<li>
<p>Which algorithm uses the “Medoid” as a representative object?<br>
A. K-Means  <strong>B. PAM</strong> C. DBSCAN  D. STING</p>
</li>
</ol>
<hr>
<h1 id="mixed--important-long-answer-questions-1016-marks"><strong>Mixed / Important Long-Answer Questions (10/16 Marks)</strong></h1>
<ol>
<li><strong>Partitioning vs. Hierarchical:</strong> Compare K-Means and Agglomerative Hierarchical clustering in terms of complexity, data size, and cluster shape. (10 marks)</li>
<li><strong>DBSCAN Deep Dive:</strong> Explain the concepts of density-reachability and density-connectivity with suitable diagrams. (16 marks)</li>
<li><strong>Big Data Clustering:</strong> Discuss the BIRCH algorithm, explaining its four phases and why it is memory efficient. (16 marks)</li>
<li><strong>Grid-Based Methods:</strong> Compare STING and CLIQUE. Why is CLIQUE preferred for high-dimensional data? (10 marks)</li>
<li><strong>Numerical Problem:</strong> Given a set of 1D or 2D points and initial centroids, perform one iteration of the K-Means algorithm. (16 marks)</li>
</ol>
<hr>
<p><strong>Preparation Tips for Exam:</strong></p>
<ul>
<li><strong>Diagrams are Compulsory:</strong> For questions on DBSCAN, Hierarchical (Dendrogram), or Grid-based methods, you <strong>must</strong> draw the diagrams to get full marks.</li>
<li><strong>Comparison Tables:</strong> Use the tables provided in the study guide for K-Means vs. K-Medoids and Agglomerative vs. Divisive.</li>
<li><strong>Memorize Parameters:</strong> Know exactly what <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span>, MinPts, Threshold (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi></mrow><annotation encoding="application/x-tex">T</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span></span></span></span></span>), and Branching Factor (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>B</mi></mrow><annotation encoding="application/x-tex">B</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.05017em;">B</span></span></span></span></span>) mean.</li>
<li><strong>Real-World Examples:</strong> Use “Customer Segmentation” for K-Means and “Satellite Mapping” for STING to illustrate your points.</li>
</ul>

