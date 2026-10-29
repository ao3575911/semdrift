# langSuite

Semantic change you can defend: model-driven measurement of meaning, surprisal and change.

`langsuite` is a CLI and TUI for linguistics research, built CPU-first and offline-capable. It joins three lenses on one corpus:

- **Information:** entropy, MATTR, Zipf and Heaps fits, KL/JS divergence, LM surprisal.
- **Geometry:** embeddings, nearest neighbours, CSLS, semantic axes.
- **Time and variety:** aligned slices, with change scores that carry intervals, nulls and cross-lens evidence.

Sister project: [glossa](https://github.com/ao3575911/glossa) is the corpus workbench (index, KWIC, collocations, parsing). langSuite is the model lab: it reads glossa exports. See [SPEC.md](SPEC.md#langsuite-vs-glossa).

Status: pre-release, built milestone by milestone. See [SPEC.md](SPEC.md).

MIT licence.
