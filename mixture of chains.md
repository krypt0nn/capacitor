# Domain-specific Markov chains: MoC, Mixture of Chains

Markov chains imply a single table of token transitions and their probabilities which is not enough to produce coherent results on larger scale. Mixture of Chains architecture attempts to solve this problem by introducing different chains for automatically clusterized documents of different fields, naturally embedding contexts of these fields into the transition matrices.

At creation time, all the known documents are clusterized into several groups, each assigned to its own expert. Clusterization happens based on the chosen "key words" and ranking function. After several (optional) optimization steps these documents are used to produce transition matrices for each expert. Depending on many factors it's expected that each expert will have different matrices for different types of documents.

At query time, the input vector is processed to identify the "key words" and, using the ranking function, choose top-$E$ experts and the base model to produce the output token. After each token recalculate the vector's rank and update used experts.

## Definitions

- $D^n = \left\{ x \in N \right\}, \left| D^n \right| = n$ - document of length $n$.
- $S = \left\{ D_i \right\}, i \in N$ - set of variable size documents.
- $\left| S \right|$ - amount of documents in set $S$.
- $M \leq \left| S \right|$ - amount of experts.
- $E \leq M$ - amount of active experts.
- $p_i(D^n) = \frac{\left| D^n \cap \left\{ i \right\} \right|}{n}$ - probability of token $i$ in document $D^n$.
- $f^0_i(S) = \frac{\sum^{\left| S \right|}_{j = 1}{p_i(D_j)}}{\left| S \right|}$ - frequency of token $i$ in the whole documents space $S$.
- $f^j_i(S) = p_i(D_j), D_j \in S$ - frequency of token $i$ in the document $D_j$.
- $r^j_i(S) = \frac{f^j_i(S)}{f_i(S)}$ - rank of token $i$ in the document $D_j \in S$.
- $s_{D_i}(D_j) = \sum_{k \in D_i}{r^j_k}, D_i \in S_{s_i}, D_j \in S_{s_j}, i \neq j$ - similarity between documents $D_i$ and $D_j$.

## Clusterization

Each expert $i$ gets a subset $S_i = \left\{ D_j \right\} \in S$ of the whole documents set $S$. Sizes of the subsets can be very different, depending on initial distribution and hyperparameters.

To split the documents set into clusters and assign them to the experts we will choose $M$ "centroids" far from each other based on tokens they contain. Once centroids are determined we can assign all the documents to each cluster based on to which of them a document is closer.

0. Assign $S' = S, i = 1$. During clusterization, documents from the $S'$ set will be *taken*, which means they will be removed from the set $S'$.
1. Randomly take any document $D_j \in S'$ and put it to the $S_{i = 1}$ set.
2. Calculate similarities $s_{D_j}(D_k)$ for $k = \overline{1, \left| S \right|}, k \neq j$.
3. Randomly take any document $D_j \in S'$ with probability proportional to the geometric mean of all the already known similarities to this document $\frac{1}{\sqrt[i]{\prod^{i}_{k = 1}{d_{D_{s_k}}(D_j)}}}$, so documents least similar to all the already taken documents have higher probability of being taken. Put taken document to the $S_{i + 1}$ set.
4. If $i \lt M$, then assign $i = i + 1$ and go to step 2. Otherwise initial documents for all the experts were taken and we can continue.
5. Using already calculated geometric means of similarities from all the experts' documents to all the other documents assign each document $D_i \in S'$ to the expert $j$ and put it to the appropriate set $S_j$.

Originally taken documents for each expert ("centroids") will be used in future and should be remembered.

> Instead of a single randomly taken document for each expert several ones could be used, depending on size of the documents in the set. The logic will be the same.

## Evaluation

1. For given vector $T = \left\{ x \in N \right\}$ of input tokens identify top-$E$ experts based on the similarities with their centroids.
2. Use base model and all the active experts to predict the next token, where probability is proportional to frequency of token in the base model plus sum of frequencies of this token in each active expert model $\left[ f^0_i(S) + \sum^E_{k = 1}{\frac{\sum^{\left| S_{s_k} \right|}_{j = 1}{f^j_i(S_{s_k})}}{\left|| S_{s_k} \right||}} \right]$. Append this token to the $T$ vector.
3. Repeat from step 1 until stop condition is met.