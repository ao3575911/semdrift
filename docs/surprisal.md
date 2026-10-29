# LM surprisal

```bash
pip install "langsuite[lm]"
langsuite lm pull                      # distilbert/distilgpt2, Apache-2.0, once
langsuite lm score --slices shakespeare,austen,fitzgerald
langsuite lm top -s fitzgerald -n 10   # most surprising words, in context
langsuite lm words -s austen           # types surprising beyond their frequency
```

Example run: 25 documents of about 250 words each, under distilgpt2:

| slice | bits/word | 95% CI | perplexity |
|---|---|---|---|
| Shakespeare (1597) | 8.54 | 8.27–8.81 | 372 |
| Austen (1813) | 6.13 | 6.06–6.21 | 70 |
| Shelley (1818) | 6.38 | 6.24–6.52 | 83 |
| Fitzgerald (1925) | 6.54 | 6.35–6.73 | 93 |

How it works:

- **Offline.** `lm pull` checks the model card licence against an open list (`--accept-licence` overrides it) and stores safetensors in the langSuite cache. After that, every command loads with `local_files_only`.
- **Word surprisal.** Text is detokenised (so `word,` rather than `word ,`). Subword surprisals in bits are summed into the corpus word that contains the token's last character, so the total number of bits is conserved.
- **Long documents** are scored with overlapping windows (`--stride`). After the first window, each token is scored only once, with the maximum available context.
- **Intervals.** Bits per word is a ratio estimator (sum of bits over sum of words), with a delete-a-group jackknife over documents.
- **Caching.** Results are cached in `.langsuite/lm/` per corpus, model and stride.
- **`words`** reports each word's residual from a log-frequency regression, which separates "rare" from "unexpected".
