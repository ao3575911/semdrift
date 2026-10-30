# Embedding geometry

Install the extra: `pip install "semdrift[vec]"` (gensim).

```
semdrift vec train --slice early --seeds 5 --dim 100 --min-count 5
semdrift vec nn broadcast --slice early -k 15          # mean cosine, sd, replicates, stability
semdrift vec nn broadcast --slice early --csls
semdrift vec analogy man king woman
semdrift vec axis migrant refugee expat --pos welcome --neg threat
```

- **Replicates.** One SGNS model is trained per seed (single worker, with a seed-stable hash), so runs are reproducible.
  - Neighbours are aggregated across replicates and ranked by how many replicates agree.
  - Stability is the mean pairwise Jaccard of the top-k sets (Hellrich & Hahn 2016; Antoniak & Mimno 2018). Below 0.5 a warning is shown.
- **CSLS** (Conneau et al. 2018): 2·cos(x,y) − r(x) − r(y), with r the mean similarity to the 10 nearest neighbours. It reduces hubness.
- **Axes** (An et al. 2018): cos(w, mean(pos) − mean(neg)), given as the mean and sd over replicates.
- **Search** is exact (a dense matmul). An ANN backend (hnswlib/FAISS) is planned for vocabularies of more than 100k words.
