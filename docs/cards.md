# Evidence cards

```bash
langsuite card pivot --slices early,late -m distilbert/distilgpt2
```

A card collects every measurement langSuite has for one word:

- frequency per slice, with jackknife intervals;
- examples in context;
- the calibrated change test, with the neighbour timeline;
- LM surprisal.

It ends with plain-language findings, a methods paragraph with citations, and provenance: corpus hash, slice definitions, parameters and package versions.

A missing source doesn't make the card fail. If there are no vectors or no pulled model, that section says why it was skipped.

Cards are written to `.langsuite/cards/WORD.md` and `.json`, or to the path given with `--out`.

A finding about meaning change is reported as *supported* only when all three agree:

- q < 0.05;
- neighbour overlap < 0.5;
- the word is stable across seeds.

If it's significant but not corroborated, the card says to treat it with caution.
