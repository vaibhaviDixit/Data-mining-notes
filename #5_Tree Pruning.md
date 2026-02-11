---


---

<h1 id="tree-pruning">Tree Pruning</h1>
<p>In machine learning, “bigger is not always better.” While a fully grown decision tree can achieve 100% accuracy on training data, it often fails miserably on new data because it has captured the <strong>noise</strong> (random fluctuations) instead of the <strong>signal</strong> (the actual pattern). <strong>Tree Pruning</strong> is the essential process of simplifying a tree to make it “smarter” and more robust.</p>
<hr>
<h2 id="the-core-conflict-overfitting-vs.-underfitting">1. The Core Conflict: Overfitting vs. Underfitting</h2>
<p>Pruning is the act of finding the “Sweet Spot” in the complexity of a model.</p>
<ul>
<li><strong>No Pruning (Overfitting):</strong> The tree is too deep. It has a branch for every single outlier. It “memorizes” the past but cannot predict the future.</li>
<li><strong>Too Much Pruning (Underfitting):</strong> The tree is too shallow. It misses important patterns and is too simple to be useful.</li>
<li><strong>Optimal Tree (Pruned):</strong> The tree is just deep enough to capture the general rules while ignoring the random noise.</li>
</ul>
<hr>
<h2 id="pre-pruning-the-preventative-approach">2. Pre-Pruning: The “Preventative” Approach</h2>
<p>Pre-pruning acts like a set of rules that stop the tree-building process before it goes too far. This is a <strong>Top-Down</strong> approach.</p>
<h3 id="common-stopping-thresholds"><strong>Common Stopping Thresholds:</strong></h3>
<ol>
<li><strong>Maximum Depth:</strong> Limits how many “levels” the tree can have.</li>
<li><strong>Minimum Samples per Leaf:</strong> Ensures that a final decision (leaf) is based on a significant number of data points (e.g., “Don’t make a decision based on just 2 people”).</li>
<li><strong>Minimum Impurity Decrease:</strong> Only split a node if it improves the “purity” (Gini/Entropy) by a significant amount (e.g., &gt; 0.05).</li>
<li><strong>Max Features:</strong> Limits the number of attributes the tree can consider at each split.</li>
</ol>
<p><strong>The Risk:</strong> Pre-pruning can suffer from the <strong>“Horizon Effect.”</strong> It might stop a split that looks bad now, but that split could have led to a very important discovery 2 levels deeper.</p>
<hr>
<h2 id="post-pruning-the-curative-approach">3. Post-Pruning: The “Curative” Approach</h2>
<p>Post-pruning is a <strong>Bottom-Up</strong> approach. It is generally more successful because it looks at the whole tree before making cuts.</p>
<h3 id="cost-complexity-pruning-the-cart-logic"><strong>Cost-Complexity Pruning (The CART Logic)</strong></h3>
<p>This is the most famous mathematical method for pruning. It uses a parameter called <strong>Alpha (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>α</mi></mrow><annotation encoding="application/x-tex">\alpha</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.0037em;">α</span></span></span></span></span>)</strong>.</p>
<p>The goal is to minimize the following “Cost” function:<br>
<span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>R</mi><mi>α</mi></msub><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo><mo>=</mo><mi>R</mi><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo><mo>+</mo><mi>α</mi><mi mathvariant="normal">∣</mi><mi>T</mi><mi mathvariant="normal">∣</mi></mrow><annotation encoding="application/x-tex">R_\alpha(T) = R(T) + \alpha|T|</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.00773em;">R</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: -0.00773em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.0037em;">α</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.00773em;">R</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.0037em;">α</span><span class="mord">∣</span><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span><span class="mord">∣</span></span></span></span></span></span></p>
<ul>
<li><strong><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>R</mi><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">R(T)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.00773em;">R</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span><span class="mclose">)</span></span></span></span></span>:</strong> The misclassification error of the tree.</li>
<li><strong><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">∣</mi><mi>T</mi><mi mathvariant="normal">∣</mi></mrow><annotation encoding="application/x-tex">|T|</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord">∣</span><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span><span class="mord">∣</span></span></span></span></span>:</strong> The number of terminal nodes (leaves).</li>
<li><strong><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>α</mi></mrow><annotation encoding="application/x-tex">\alpha</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.0037em;">α</span></span></span></span></span>:</strong> The complexity parameter.
<ul>
<li>If <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>α</mi></mrow><annotation encoding="application/x-tex">\alpha</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.0037em;">α</span></span></span></span></span> is 0, the tree stays large.</li>
<li>As <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>α</mi></mrow><annotation encoding="application/x-tex">\alpha</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.0037em;">α</span></span></span></span></span> increases, the “penalty” for having more leaves grows, forcing the tree to become smaller.</li>
</ul>
</li>
</ul>
<hr>
<h2 id="pruning-techniques-a-closer-look">4. Pruning Techniques: A Closer Look</h2>
<h3 id="reduced-error-pruning-rep"><strong>Reduced Error Pruning (REP)</strong></h3>
<ol>
<li>Divide your data into a <strong>Training Set</strong> and a <strong>Validation Set</strong>.</li>
<li>Grow the tree fully using the Training Set.</li>
<li>For every internal node, temporarily replace it with a leaf and check the accuracy on the <strong>Validation Set</strong>.</li>
<li>If the accuracy stays the same or improves, <strong>cut the branch permanently.</strong></li>
<li>Repeat until no more cuts improve the accuracy.</li>
</ol>
<h3 id="rule-post-pruning"><strong>Rule Post-Pruning</strong></h3>
<ol>
<li>Convert the tree into a set of <strong>IF-THEN rules</strong>.</li>
<li>Each path from the root to a leaf becomes one rule.</li>
<li>Analyze each rule and remove any “IF” conditions that don’t help accuracy.</li>
<li>Sort the simplified rules by accuracy and use them for classification.</li>
</ol>
<hr>
<h2 id="comparison-why-post-pruning-is-often-better">5. Comparison: Why Post-Pruning is Often Better</h2>

<table>
<thead>
<tr>
<th align="left">Feature</th>
<th align="left">Pre-Pruning</th>
<th align="left">Post-Pruning</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Strategy</strong></td>
<td align="left">Stop early.</td>
<td align="left">Grow full, then cut.</td>
</tr>
<tr>
<td align="left"><strong>Visibility</strong></td>
<td align="left">Limited (cannot see future splits).</td>
<td align="left">Full (sees the entire tree structure).</td>
</tr>
<tr>
<td align="left"><strong>Computation</strong></td>
<td align="left">Very fast and efficient.</td>
<td align="left">Slower (requires building the full tree).</td>
</tr>
<tr>
<td align="left"><strong>Accuracy</strong></td>
<td align="left">Risk of “Underfitting.”</td>
<td align="left">Usually higher accuracy and better generalization.</td>
</tr>
<tr>
<td align="left"><strong>Standard Use</strong></td>
<td align="left">Real-time / Large datasets.</td>
<td align="left">High-precision scientific models.</td>
</tr>
</tbody>
</table><hr>
<h2 id="how-to-explain-pruning-in-exams">6. How to Explain Pruning in Exams</h2>
<p>When asked about Pruning, focus on these three keywords:</p>
<ol>
<li><strong>Complexity:</strong> Reducing the number of nodes.</li>
<li><strong>Generalization:</strong> Improving performance on new, unseen data.</li>
<li><strong>Noise:</strong> Removing branches that were created due to errors or outliers in the training set.</li>
</ol>
<hr>
<h3 id="summary-table"><strong>Summary Table</strong></h3>

<table>
<thead>
<tr>
<th align="left">If the tree is…</th>
<th align="left">Problem</th>
<th align="left">Action</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left">Too Deep</td>
<td align="left">Overfitting</td>
<td align="left">Apply Post-Pruning</td>
</tr>
<tr>
<td align="left">Too Shallow</td>
<td align="left">Underfitting</td>
<td align="left">Reduce Pruning constraints</td>
</tr>
<tr>
<td align="left">Perfect</td>
<td align="left">Balanced</td>
<td align="left">Model generalizes well</td>
</tr>
</tbody>
</table>
