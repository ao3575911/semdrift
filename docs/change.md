# Semantic change

```bash
semdrift vec train -s early --seeds 3
semdrift vec train -s late --seeds 3
semdrift change score --slices early,late
semdrift change drift pivot --slices early,late
```

How a word is scored:

1. Each slice is trained with several seeds. Replicate stability (top-k neighbour Jaccard) is reported.
2. Slices are aligned with orthogonal Procrustes on their shared vocabulary.
3. A **shuffled-slice null** (Dubossarsky et al. 2017) retrains on random splits of the pooled documents.
4. Distances are divided by their run's median word distance. This removes shifts that every word shares, so the test is for word-specific change.
5. Each word is tested against its own null mean with a predictive t test. The spread is pooled within frequency bins. `q` is Benjamini–Hochberg.
6. An alignment-free check, neighbour overlap, is reported alongside.

`change` is measured in median-word-distance units above that word's null. Flag words that have both `q < 0.05` and low neighbour overlap.
