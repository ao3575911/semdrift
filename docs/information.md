# Information measures

```
semdrift info summary --slice early            # H1, H(X,Y), H(Y|X), TTR, MATTR, Zipf s, Heaps beta
semdrift info divergence early late             # JS divergence + the words that drive it
semdrift --json info divergence early late --measure kl --alpha 0.5
```

| Measure | Definition | Reference |
|---|---|---|
| H1 | −Σ p(w) log₂ p(w) | Shannon 1948 |
| H(Y\|X) | H(X,Y) − H(X), over adjacent pairs within a document | Cover & Thomas 2006 |
| MATTR | mean TTR over all windows of W tokens; O(N) | Covington & McFadden 2010 |
| Zipf s | MLE of f(r) ∝ r^−s | Zipf 1949; Clauset et al. 2009 |
| Heaps β | log-space least squares for V = K·N^β | Heaps 1978 |
| KL | D(A‖B), add-α smoothed | Kullback & Leibler 1951 |
| JS | ½D(A‖M)+½D(B‖M), in [0, 1] bits | Lin 1991 |

**Intervals.**
- Intervals are 95% delete-a-group jackknife intervals, with documents split into `--groups` groups (default 20).
- semdrift doesn't bootstrap here on purpose. Resampling with replacement duplicates documents, which biases type-based measures (TTR, H1, Zipf) downward.
- With fewer than 10 documents a warning is shown.

TTR, H1 and the Zipf/Heaps fits depend on sample size. Compare slices of similar size, or use MATTR.
