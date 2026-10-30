# semdrift

[![CI](https://github.com/ao3575911/semdrift/actions/workflows/ci.yml/badge.svg)](https://github.com/ao3575911/semdrift/actions/workflows/ci.yml)
[![Docs](https://github.com/ao3575911/semdrift/actions/workflows/docs.yml/badge.svg)](https://ao3575911.github.io/semdrift/)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A measurement lab for meaning, surprisal and change in text, from the terminal.

![Semantic change: train, score, drift](docs/assets/semdrift-change.gif)

## Install

Pre-release; install from source with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ao3575911/semdrift && cd semdrift
uv sync --extra vec --extra lm      # embeddings + local language model
```

Extras: `vec` (gensim), `lm` (torch, transformers), `spacy`, `stanza`. CPU-only works; nothing calls a remote API.

## Quickstart

```bash
semdrift init study && cd study
semdrift ingest letters.jsonl                 # txt, csv, jsonl, CoNLL-U
semdrift slice define early "year < 1850"
semdrift slice define late  "year >= 1850"
semdrift vec train -s early --seeds 3
semdrift vec train -s late  --seeds 3
semdrift change score --slices early,late     # change vs a shuffled-slice null
semdrift change drift pivot --slices early,late
```

The GIF above runs this on a synthetic corpus where one word, `pivot`, moves between two clusters. It is the only word flagged at q < 0.05.

## Common workflows

**Compare surprisal across authors.** With the default model, `distilbert/distilgpt2`, four Gutenberg excerpts each measured 8.87 bits/word for Shakespeare and 5.84 for Austen.

```bash
semdrift lm pull                              # once; then offline
semdrift lm score --slices shakespeare,austen
semdrift lm top -s shakespeare -n 10          # most surprising words, in context
```

**Write an evidence card for one word.** Frequency with intervals, concordance, change, surprisal, a methods paragraph and provenance, as Markdown and JSON.

```bash
semdrift card love --slices shakespeare,austen -m distilbert/distilgpt2
```

![Surprisal and an evidence card](docs/assets/semdrift-lm.gif)

**Profile information.** `semdrift info summary --slice early` and `semdrift info divergence early late`.

**Explore geometry.** `semdrift vec nn WORD --slice early --csls`, `vec analogy`, `vec axis`.

Every command takes `--json` for pipelines.

## Concepts

- **Slice**: a named subcorpus defined by a metadata filter.
- **Surprisal**: −log₂ p(word | context) under a causal language model.
- **Change score**: displacement of a word between aligned slice embeddings, scaled against seed noise and a shuffled-slice null.
- **Evidence card**: one word's measurements with the methods and provenance needed to cite them.

## Docs

[ao3575911.github.io/semdrift](https://ao3575911.github.io/semdrift/) · [SPEC.md](SPEC.md)

Sister project: [glossa](https://github.com/ao3575911/glossa) is the corpus workbench (import, index, concordance). semdrift is the model-driven measurement side.

## Contributing

Issues and PRs welcome. Run `uv run pytest`, `uv run ruff check .` and `uv run mypy src` before opening a PR.

## Licence

[MIT](LICENSE)
