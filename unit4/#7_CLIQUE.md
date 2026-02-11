---


---

<h1 id="clique-clustering-in-quest">CLIQUE: Clustering In QUEst</h1>
<p>While <strong>STING</strong> is great for looking at the big picture, it struggles when data has many dimensions (features). <strong>CLIQUE</strong> (Clustering In QUEst) is a hybrid algorithm that combines <strong>Grid-based</strong> and <strong>Density-based</strong> clustering to find clusters in high-dimensional data.</p>
<hr>
<h2 id="the-scenario-the-high-dimensional-shadow">1. The Scenario: The High-Dimensional Shadow</h2>
<p>Imagine you are a detective trying to find a criminal in a massive 50-story building.</p>
<ul>
<li><strong>The Problem:</strong> Searching the whole building at once is impossible.</li>
<li><strong>The CLIQUE Solution:</strong> You look at the building from the side (2D) and notice a shadow on the ground floor. You then look from the front and see the same shadow.</li>
<li><strong>The Logic:</strong> If a “crowded” cluster exists in 50 dimensions, it <strong>must</strong> also look crowded when you look at it in 1 dimension or 2 dimensions. CLIQUE starts with 1D, finds the crowded spots, and then only looks for 2D clusters in those specific spots.</li>
</ul>
<hr>
<h2 id="core-concept-subspace-clustering">2. Core Concept: Subspace Clustering</h2>
<p>In high-dimensional spaces, data points are often very spread out (the “curse of dimensionality”). CLIQUE solves this by finding <strong>Subspace Clusters</strong>.</p>
<ul>
<li>It partitions each dimension into non-overlapping intervals (a grid).</li>
<li>It identifies “dense” units—cells that contain more than a certain number of data points (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>t</mi><mi>a</mi><mi>u</mi></mrow><annotation encoding="application/x-tex">tau</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.61508em; vertical-align: 0em;"></span><span class="mord mathnormal">t</span><span class="mord mathnormal">a</span><span class="mord mathnormal">u</span></span></span></span></span>).</li>
</ul>
<hr>
<h2 id="the-apriori-property-of-clique">3. The Apriori Property of CLIQUE</h2>
<p>CLIQUE uses a logic similar to the <strong>Apriori Algorithm</strong> you learned in Unit 2:</p>
<ul>
<li><strong>The Rule:</strong> If a <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span>-dimensional cell is dense, then all of its <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false">(</mo><mi>k</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">(k-1)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord">1</span><span class="mclose">)</span></span></span></span></span>-dimensional projections must also be dense.</li>
<li><strong>How it works:</strong> 1. Find all dense 1D intervals.<br>
2. Combine dense 1D intervals to find potential dense 2D cells.<br>
3. Combine dense 2D cells to find 3D cells, and so on.<br>
4. This “pruning” saves the computer from checking billions of empty cells.</li>
</ul>
<hr>
<h2 id="how-the-algorithm-works">4. How the Algorithm Works</h2>
<ol>
<li><strong>Partitioning:</strong> Divide the data space into a grid of units.</li>
<li><strong>Identification:</strong> Find all dense units in all subspaces using the bottom-up (Apriori) approach.</li>
<li><strong>Clustering:</strong> Connect adjacent dense units to form larger clusters.</li>
<li><strong>Generation:</strong> Generate a minimal description (rules) for each cluster.</li>
</ol>
<hr>
<h2 id="advantages">5. Advantages</h2>
<ul>
<li><strong>High-Dimensional Expert:</strong> It is specifically designed to find clusters that are “hidden” in specific dimensions of huge datasets.</li>
<li><strong>No Pre-set K:</strong> Like DBSCAN, it automatically finds the number of clusters.</li>
<li><strong>Interpretability:</strong> It provides a simple “if-then” description of the clusters it finds.</li>
<li><strong>Scalability:</strong> It scales linearly with the number of data points.</li>
</ul>
<hr>
<h2 id="the-limitations-the-catch">6. The Limitations (The Catch)</h2>
<ul>
<li><strong>Accuracy vs. Grid Size:</strong> Just like STING, if the grid intervals are too large, you lose accuracy; if they are too small, you increase computation.</li>
<li><strong>Complexity:</strong> As the number of dimensions increases, the number of potential subspaces to check can still become very large.</li>
</ul>
<hr>
<h2 id="comparison-table-sting-vs.-clique">7. Comparison Table: STING vs. CLIQUE</h2>

<table>
<thead>
<tr>
<th align="left">Feature</th>
<th align="left">STING</th>
<th align="left">CLIQUE</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Category</strong></td>
<td align="left">Purely Grid-based</td>
<td align="left">Hybrid (Grid + Density)</td>
</tr>
<tr>
<td align="left"><strong>Strategy</strong></td>
<td align="left">Top-Down (Querying)</td>
<td align="left">Bottom-Up (Subspace building)</td>
</tr>
<tr>
<td align="left"><strong>Best For</strong></td>
<td align="left">Fast spatial queries</td>
<td align="left">High-dimensional data</td>
</tr>
<tr>
<td align="left"><strong>Logic</strong></td>
<td align="left">Statistical Summaries</td>
<td align="left">Dense Unit Projections</td>
</tr>
</tbody>
</table>
