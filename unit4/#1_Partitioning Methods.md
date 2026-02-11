---


---

<h1 id="partitioning-methods">Partitioning Methods</h1>
<p>In data mining, partitioning methods are the most intuitive clustering techniques. They work by creating <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span> groups from a set of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>n</mi></mrow><annotation encoding="application/x-tex">n</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">n</span></span></span></span></span> objects, where each group represents a cluster. These methods are “iterative” because they constantly relocate objects between clusters to reach an optimal configuration.</p>
<hr>
<h2 id="fundamental-concept--mathematical-basis"><strong>1. Fundamental Concept &amp; Mathematical Basis</strong></h2>
<p>Partitioning algorithms aim to minimize a <strong>Cost Function</strong> (also known as the Squared Error Criterion). The goal is to make the objects within a cluster as close to each other as possible.</p>
<h3 id="distance-metrics"><strong>Distance Metrics</strong></h3>
<p>To decide which cluster an object belongs to, the algorithm calculates the distance between the object (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>p</mi></mrow><annotation encoding="application/x-tex">p</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal">p</span></span></span></span></span>) and the cluster center (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>c</mi></mrow><annotation encoding="application/x-tex">c</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">c</span></span></span></span></span>).</p>
<ul>
<li><strong>Euclidean Distance:</strong> The most common metric for continuous data.</li>
<li><strong>Manhattan Distance:</strong> Often used when data is less “straight-line” or has many outliers.</li>
<li><strong>Minkowski Distance:</strong> A generalized version of both Euclidean and Manhattan distances.</li>
</ul>
<hr>
<h2 id="k-means-the-centroid-approach"><strong>2. K-Means: The Centroid Approach</strong></h2>
<p>K-Means defines the center of a cluster as the <strong>Centroid</strong> (the mean value of all points in the cluster).</p>
<h3 id="the-algorithm-logic"><strong>The Algorithm Logic</strong></h3>
<ol>
<li><strong>Initialize:</strong> Select <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span> points randomly as initial centroids.</li>
<li><strong>Assign:</strong> Calculate the distance of each object to these <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span> centroids and assign it to the nearest one.</li>
<li><strong>Update:</strong> Compute the new mean for each cluster to find the new centroid.</li>
<li><strong>Converge:</strong> Repeat until the assignments no longer change.</li>
</ol>
<hr>
<h2 id="k-medoids-the-representative-object-approach"><strong>3. K-Medoids: The Representative Object Approach</strong></h2>
<p>K-Medoids is a more robust alternative to K-Means. It uses an actual data point from the dataset, called a <strong>Medoid</strong>, to represent the cluster.</p>
<h3 id="why-use-k-medoids"><strong>Why use K-Medoids?</strong></h3>
<ul>
<li><strong>Outlier Sensitivity:</strong> K-Means is easily “pulled” by extreme values because the mean is affected by outliers.</li>
<li><strong>Robustness:</strong> Since a medoid must be an actual data point, it is less influenced by random noise or extreme values.</li>
</ul>
<h3 id="pam-partitioning-around-medoids-algorithm"><strong>PAM (Partitioning Around Medoids) Algorithm</strong></h3>
<ol>
<li><strong>Select:</strong> Randomly pick <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span> objects as medoids.</li>
<li><strong>Associate:</strong> Assign each remaining object to the nearest medoid.</li>
<li><strong>Swap:</strong> Randomly select a non-medoid object and calculate the cost of swapping it with an existing medoid.</li>
<li><strong>Accept:</strong> If the total cost of the new configuration is lower, the swap is permanent.</li>
</ol>
<hr>
<h2 id="comparison-k-means-vs.-k-medoids"><strong>4. Comparison: K-Means vs. K-Medoids</strong></h2>

<table>
<thead>
<tr>
<th align="left">Feature</th>
<th align="left">K-Means</th>
<th align="left">K-Medoids (PAM)</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Cluster Center</strong></td>
<td align="left">Calculated Mean (Centroid)</td>
<td align="left">Actual Data Point (Medoid)</td>
</tr>
<tr>
<td align="left"><strong>Efficiency</strong></td>
<td align="left">Faster (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false">(</mo><mi>t</mi><mi>k</mi><mi>n</mi><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">O(tkn)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.02778em;">O</span><span class="mopen">(</span><span class="mord mathnormal">t</span><span class="mord mathnormal">kn</span><span class="mclose">)</span></span></span></span></span>) - Good for large data</td>
<td align="left">Slower (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mi>k</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">O(k(n-k)^2)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.02778em;">O</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span><span class="mopen">(</span><span class="mord mathnormal">n</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.06411em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span><span class="mclose"><span class="mclose">)</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.814108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span>) - Best for small data</td>
</tr>
<tr>
<td align="left"><strong>Sensitivity</strong></td>
<td align="left">Highly sensitive to outliers</td>
<td align="left">Robust to noise and outliers</td>
</tr>
<tr>
<td align="left"><strong>Result</strong></td>
<td align="left">Local optimum</td>
<td align="left">More stable, representative results</td>
</tr>
</tbody>
</table><hr>
<h2 id="how-to-choose-the-value-of-k"><strong>5. How to Choose the Value of ‘K’?</strong></h2>
<p>Since <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span> must be specified by the user, we often use the <strong>Elbow Method</strong>.</p>
<ul>
<li>We run the algorithm for different values of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span> (e.g., 1 to 10).</li>
<li>We plot the <strong>Sum of Squared Errors (SSE)</strong> against <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span>.</li>
<li>The point where the curve “bends” like an elbow is considered the optimal number of clusters.</li>
</ul>
<hr>
<h2 id="summary-of-partitioning-methods"><strong>6. Summary of Partitioning Methods</strong></h2>
<ul>
<li><strong>Best for:</strong> Finding spherical-shaped clusters that are well-separated.</li>
<li><strong>Weakness:</strong> It struggles with clusters of different sizes, densities, or non-spherical shapes.</li>
<li><strong>Memory:</strong> Requires the data to be kept in memory, which can be a challenge for massive datasets.</li>
</ul>

