---


---

<h1 id="support-vector-machines-svm">Support Vector Machines (SVM)</h1>
<p><strong>Support Vector Machines (SVM)</strong> are among the most robust and mathematically elegant classification algorithms in data mining. While other algorithms try to find any boundary that works, SVM is obsessed with finding the <strong>Optimal Hyperplane</strong>—the one that provides the greatest “safety buffer” between classes.</p>
<hr>
<h2 id="the-geometry-of-the-safety-buffer">1. The Geometry of the “Safety Buffer”</h2>
<p>The power of SVM lies in its objective: maximizing the <strong>Margin</strong>.</p>
<ul>
<li><strong>The Hyperplane:</strong> In a 2D space, this is a line. In 3D, it’s a plane. In N-dimensions, it’s a hyperplane. It is defined by the equation:<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>w</mi><mo>⋅</mo><mi>x</mi><mo>+</mo><mi>b</mi><mo>=</mo><mn>0</mn></mrow><annotation encoding="application/x-tex">w \cdot x + b = 0</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.44445em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.02691em;">w</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.66666em; vertical-align: -0.08333em;"></span><span class="mord mathnormal">x</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal">b</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.64444em; vertical-align: 0em;"></span><span class="mord">0</span></span></span></span></span></span></li>
<li><strong>Support Vectors:</strong> These are the “VIP” data points. They are the observations that lie exactly on the marginal boundaries. If you remove any other data points, the hyperplane stays the same; but if you move a Support Vector, the entire model changes.</li>
<li><strong>Maximum Margin:</strong> The distance between the hyperplane and the nearest data point from either class. A larger margin leads to better generalization on new data.</li>
</ul>
<hr>
<h2 id="linear-svm-the-mathematical-goal">2. Linear SVM: The Mathematical Goal</h2>
<p>For a dataset that is linearly separable, SVM tries to solve a constrained optimization problem. We want to minimize the weight vector <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>w</mi></mrow><annotation encoding="application/x-tex">w</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.02691em;">w</span></span></span></span></span> (which maximizes the margin) subject to the condition that all points are correctly classified:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Minimize&nbsp;</mtext><mfrac><mn>1</mn><mn>2</mn></mfrac><mi mathvariant="normal">∣</mi><mi mathvariant="normal">∣</mi><mi>w</mi><mi mathvariant="normal">∣</mi><msup><mi mathvariant="normal">∣</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">\text{Minimize } \frac{1}{2} ||w||^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 2.00744em; vertical-align: -0.686em;"></span><span class="mord text"><span class="mord">Minimize&nbsp;</span></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.32144em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord">2</span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.686em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span><span class="mord">∣∣</span><span class="mord mathnormal" style="margin-right: 0.02691em;">w</span><span class="mord">∣</span><span class="mord"><span class="mord">∣</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.864108em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span></span><br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Subject&nbsp;to:&nbsp;</mtext><msub><mi>y</mi><mi>i</mi></msub><mo stretchy="false">(</mo><mi>w</mi><mo>⋅</mo><msub><mi>x</mi><mi>i</mi></msub><mo>+</mo><mi>b</mi><mo stretchy="false">)</mo><mo>≥</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">\text{Subject to: } y_i(w \cdot x_i + b) \ge 1</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord text"><span class="mord">Subject&nbsp;to:&nbsp;</span></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03588em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.02691em;">w</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.73333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathnormal">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">b</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">≥</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.64444em; vertical-align: 0em;"></span><span class="mord">1</span></span></span></span></span></span></p>
<p>To solve this, SVM uses <strong>Lagrange Multipliers</strong>, converting the problem into a “Dual Form” that only depends on the dot product of the input vectors.</p>
<hr>
<h2 id="the-kernel-trick-moving-to-higher-dimensions">3. The Kernel Trick: Moving to Higher Dimensions</h2>
<p>Real-world data is rarely a straight line. Sometimes, data points of Class A are surrounded by Class B.</p>
<p><strong>The Logic:</strong> If we cannot separate data in 2D, we project it into 3D (or higher). In this new space, we can pass a flat “sheet” (hyperplane) through the data.</p>
<h3 id="common-kernel-functions"><strong>Common Kernel Functions:</strong></h3>
<ol>
<li><strong>Linear Kernel:</strong> No transformation. Used when data is already separable.</li>
<li><strong>Polynomial Kernel:</strong> Represents the similarity of vectors in a feature space over polynomials of the original variables.</li>
<li><strong>Radial Basis Function (RBF/Gaussian):</strong> The most popular kernel. It can handle infinite-dimensional spaces and creates complex, circular boundaries.</li>
<li><strong>Sigmoid Kernel:</strong> Acts similarly to the activation functions in Neural Networks.</li>
</ol>
<hr>
<h2 id="tuning-the-svm-hyperparameters">4. Tuning the SVM: Hyperparameters</h2>
<p>To get the best performance, two parameters must be tuned carefully:</p>
<h3 id="a.-the-c-parameter-regularization"><strong>A. The C Parameter (Regularization)</strong></h3>
<ul>
<li><strong>Small C:</strong> Prioritizes a large margin, even if it means some training points are misclassified (Soft Margin). This prevents overfitting.</li>
<li><strong>Large C:</strong> Prioritizes classifying all training points correctly, even if the margin becomes very small. This can lead to overfitting.</li>
</ul>
<h3 id="b.-gamma-gamma"><strong>B. Gamma (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>γ</mi></mrow><annotation encoding="application/x-tex">\gamma</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.05556em;">γ</span></span></span></span></span>)</strong></h3>
<ul>
<li>Only used with the RBF kernel.</li>
<li><strong>Low Gamma:</strong> The “influence” of a support vector reaches far. The boundary is smooth.</li>
<li><strong>High Gamma:</strong> The “influence” is local. The boundary is very wiggly and tries to “hug” every single data point.</li>
</ul>
<hr>
<h2 id="multi-class-classification">5. Multi-Class Classification</h2>
<p>SVM is naturally a binary classifier (Class A vs Class B). To handle multiple classes (e.g., Apple vs Orange vs Banana), it uses two strategies:</p>
<ol>
<li><strong>One-vs-One (OvO):</strong> It builds a classifier for every possible pair of classes.</li>
<li><strong>One-vs-All (OvA):</strong> It builds a classifier for each class against all the others combined.</li>
</ol>
<hr>
<h2 id="comparison-table-svm-vs.-the-rest">6. Comparison Table: SVM vs. The Rest</h2>

<table>
<thead>
<tr>
<th align="left">Feature</th>
<th align="left">Naive Bayes</th>
<th align="left">Decision Trees</th>
<th align="left">SVM</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Foundation</strong></td>
<td align="left">Probability</td>
<td align="left">Logic/Rules</td>
<td align="left">Geometry/Optimization</td>
</tr>
<tr>
<td align="left"><strong>Boundary</strong></td>
<td align="left">Linear/Simple</td>
<td align="left">Axis-Parallel Steps</td>
<td align="left">Smooth/Complex Curves</td>
</tr>
<tr>
<td align="left"><strong>Outliers</strong></td>
<td align="left">Robust</td>
<td align="left">Sensitive (without pruning)</td>
<td align="left">Robust (due to Margin)</td>
</tr>
<tr>
<td align="left"><strong>Small Data</strong></td>
<td align="left">Good</td>
<td align="left">Moderate</td>
<td align="left">Excellent</td>
</tr>
</tbody>
</table><hr>
<h2 id="when-to-choose-svm">7. When to Choose SVM?</h2>
<ul>
<li>Use SVM when you have a <strong>clear margin of separation</strong>.</li>
<li>Use SVM when your data has <strong>high dimensions</strong> (e.g., gene sequences or text features).</li>
<li>Avoid SVM if your dataset is <strong>extremely large</strong> (millions of rows) because the training time increases cubically (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false">(</mo><msup><mi>n</mi><mn>3</mn></msup><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">O(n^3)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.06411em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.02778em;">O</span><span class="mopen">(</span><span class="mord"><span class="mord mathnormal">n</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.814108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">3</span></span></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span>).</li>
</ul>
<hr>
<h3 id="key-takeaways"><strong>Key Takeaways</strong></h3>
<blockquote>
<p>SVM doesn’t just look for <em>a</em> solution; it looks for the <strong>strongest</strong> solution by maximizing the gap between classes. Through the <strong>Kernel Trick</strong>, it can solve problems that appear impossible in lower dimensions.</p>
</blockquote>

