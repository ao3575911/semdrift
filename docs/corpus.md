# Workspaces, ingest and slices

```
semdrift init study && cd study
semdrift ingest letters.jsonl --pipeline basic          # or spacy:en_core_web_sm, stanza:en
semdrift ingest diaries/*.txt --meta year=1851 --meta register=diary
semdrift slice define early "year < 1900"
semdrift slice define late  "year >= 1900 AND register = 'press'"
semdrift corpus info
```

- **Formats:** `txt` (one file is one document), `jsonl` and `csv` (a `text` field or column; the other fields become metadata; an `id` field becomes the document id), and `conllu` (`# newdoc id` splits documents; multiword ranges are skipped).
- **Pipelines:**
  - `auto` uses spaCy `en_core_web_sm` if it's installed, and otherwise `basic` (a Unicode regex tokeniser with no dependencies).
  - The pipeline id is recorded with the corpus.
- **Cache:** tokenised output is keyed by `xxh3-128(file bytes, format, pipeline id, text field)` under the user cache dir. Set `SEMDRIFT_CACHE_DIR` to change it. Re-ingesting is a cache hit, and already-present document ids are skipped.
- **Slices:** a SQL `WHERE` clause over the metadata columns (evaluated by Polars SQL).
