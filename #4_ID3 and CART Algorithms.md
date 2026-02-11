---


---

<h1 id="id3-and-cart-algorithms">ID3 and CART Algorithms</h1>
<p>In the world of Decision Trees, <strong>ID3</strong> and <strong>CART</strong> are like two different master architects. They both want to build a “house” (the model), but they use different tools and blueprints to get there.</p>
<hr>
<h2 id="id3-iterative-dichotomiser-3">1. ID3 (Iterative Dichotomiser 3)</h2>
<p>Developed in 1986, ID3 is the “original” algorithm that brought decision trees into the spotlight.</p>
<h3 id="the-multi-way-splitter"><strong>The “Multi-Way” Splitter</strong></h3>
<p>Imagine you are splitting a dataset by the attribute <strong>“Color”</strong> which has three values: <em>Red, Blue, and Green</em>.</p>
<ul>
<li><strong>ID3’s Approach:</strong> It will immediately grow <strong>three</strong> branches.</li>
<li><strong>The Logic:</strong> ID3 believes that if an attribute has multiple categories, we should explore all of them simultaneously.</li>
</ul>
<h3 id="how-id3-thinks-step-by-step"><strong>How ID3 Thinks (Step-by-Step)</strong></h3>
<ol>
<li><strong>Entropy Check:</strong> It looks at the data and asks, “How messy is this?”</li>
<li><strong>Gain Calculation:</strong> It calculates <strong>Information Gain</strong> for every attribute.</li>
<li><strong>The Winner:</strong> The attribute that cleans up the “mess” (Entropy) the most becomes the next node.</li>
<li><strong>Repeat:</strong> It keeps doing this until every branch leads to a “Pure” leaf (where all data belongs to one class).</li>
</ol>
<p><strong>The Weakness:</strong> ID3 is like a perfectionist who doesn’t know when to stop. It often creates very deep, complex trees that work perfectly on training data but fail in the real world (Overfitting).</p>
<hr>
<h2 id="cart-classification-and-regression-trees">2. CART (Classification and Regression Trees)</h2>
<p>Introduced in 1984, CART is the modern “powerhouse” used in almost all professional Data Science libraries (like Scikit-Learn).</p>
<h3 id="the-binary-specialist"><strong>The “Binary” Specialist</strong></h3>
<p>CART is strictly binary. Even if the attribute <strong>“Color”</strong> has <em>Red, Blue, and Green</em>, CART will only split it into <strong>two</strong> branches at a time (e.g., “Is it Red?” vs. “Is it Not Red?”).</p>
<h3 id="how-cart-thinks-step-by-step"><strong>How CART Thinks (Step-by-Step)</strong></h3>
<ol>
<li><strong>Gini Index:</strong> Instead of Entropy, CART uses the <strong>Gini Index</strong> to measure “Impurity.”</li>
<li><strong>Binary Search:</strong> It tests every possible two-way split for every attribute.</li>
<li><strong>Regression Power:</strong> Unlike ID3, if you want to predict a <strong>Number</strong> (like the price of a house), CART can do it using “Regression Trees.”</li>
<li><strong>Pruning:</strong> CART is smarter about stopping. It uses “Cost-Complexity Pruning” to snip off useless branches after the tree is built.</li>
</ol>
<hr>
<h2 id="key-technical-differences">3. Key Technical Differences</h2>

<table>
<thead>
<tr>
<th align="left">Feature</th>
<th align="left">ID3</th>
<th align="left">CART</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Mathematical Brain</strong></td>
<td align="left">Information Gain (Entropy)</td>
<td align="left">Gini Index (Impurity)</td>
</tr>
<tr>
<td align="left"><strong>Branching Factor</strong></td>
<td align="left">Multi-way (As many as you need)</td>
<td align="left">Always Binary (Exactly 2)</td>
</tr>
<tr>
<td align="left"><strong>Handles Numbers?</strong></td>
<td align="left">No (Categorical only)</td>
<td align="left">Yes (Categorical &amp; Continuous)</td>
</tr>
<tr>
<td align="left"><strong>Handles Missing Data?</strong></td>
<td align="left">No</td>
<td align="left">Yes (Uses “Surrogate Splits”)</td>
</tr>
<tr>
<td align="left"><strong>Best For…</strong></td>
<td align="left">Simple, labeled datasets</td>
<td align="left">Complex, real-world big data</td>
</tr>
</tbody>
</table><hr>
<h2 id="why-greedy-algorithms">4. Why “Greedy” Algorithms?</h2>
<p>Students often see the term <strong>“Greedy”</strong> in textbooks. Here is what it actually means in this context:</p>
<ul>
<li>The algorithm looks for the <strong>best split right now</strong>.</li>
<li>It does <strong>not</strong> look ahead to see if a different split now would make the tree better 5 steps later.</li>
<li>It’s like eating the best-looking piece of candy in a box immediately, rather than saving it for later.</li>
</ul>
<hr>
<h2 id="summary-of-stopping-conditions">5. Summary of Stopping Conditions</h2>
<p>A student must know when these algorithms decide to stop growing a branch:</p>
<ol>
<li><strong>Pure Node:</strong> Every record in the branch belongs to the same class (e.g., all are “Spam”).</li>
<li><strong>No Attributes Left:</strong> There are no more questions left to ask.</li>
<li><strong>Threshold Reached:</strong> The “Gain” from the next split is too small to be worth the effort.</li>
<li><strong>Empty Subset:</strong> There are no more data points to classify in that branch.</li>
</ol>
<hr>
<h2 id="common-confusion-which-one-is-better">6. Common Confusion: Which one is “Better”?</h2>
<p>There is no “perfect” algorithm, but:</p>
<ul>
<li><strong>CART</strong> is generally preferred in modern industries because it handles numbers and is less likely to overfit due to its pruning logic.</li>
<li><strong>ID3</strong> is excellent for learning the fundamentals of Information Theory and works well for small, purely categorical datasets.</li>
</ul>

